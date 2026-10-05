# Copyright 2026 The LLaMA Factory Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import io
import json
import math
import runpy
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import yaml

from planner_val.models import ContractSnapshot, JSONValue


CONTEXT_TOOL_NAMES = (
    "HumanAction",
    "RobotGesture",
    "RobotDance",
    "get_weather",
    "web_search",
    "rag_query",
    "robot_status",
    "get_visual_info",
)
ACTION_TOOLS = frozenset({"HumanAction", "RobotGesture", "RobotDance"})
QUERY_TOOLS = frozenset({"get_weather", "web_search", "rag_query", "robot_status", "get_visual_info"})
PROMPT_CONSTANTS = (
    "SYSTEM_PROMPT",
    "USER_TOOLS_HEADER",
    "USER_LIMIT_HEADER",
    "USER_INSTRUCTION_HEADER",
    "STEP_LIMIT_TEMPLATE",
)
MESSAGE_CONFIG_KEYS = {
    "USER_TOOLS_HEADER": "user_tools_header",
    "USER_LIMIT_HEADER": "user_limit_header",
    "USER_INSTRUCTION_HEADER": "user_instruction_header",
    "STEP_LIMIT_TEMPLATE": "step_limit_template",
}


def _read_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(inner_value) for key, inner_value in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonicalize(value: Any) -> str:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint_contract(
    *,
    name: Literal["controlled", "production"],
    system_prompt: str,
    tools_context: JSONValue,
    tool_schemas: Mapping[str, Mapping[str, Any]],
    tool_kinds: Mapping[str, Literal["query", "action"]],
    max_steps: int,
    message_config: Mapping[str, JSONValue],
) -> str:
    payload = {
        "name": name,
        "system_prompt": system_prompt,
        "tools_context": _thaw_json(tools_context),
        "tool_schemas": _thaw_json(tool_schemas),
        "tool_kinds": dict(tool_kinds),
        "max_steps": max_steps,
        "message_config": _thaw_json(message_config),
    }
    digest = hashlib.sha256(_canonicalize(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_json_value(value: Any, *, context: str) -> JSONValue:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context}: non-finite float")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, list):
        return tuple(_normalize_json_value(item, context=context) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_json_value(item, context=context) for item in value)
    if isinstance(value, Mapping):
        normalized: dict[str, JSONValue] = {}
        for key, inner_value in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context}: non-string JSON key")
            normalized[key] = _normalize_json_value(inner_value, context=context)
        return normalized
    raise ValueError(f"{context}: not JSON-compatible")


def _require_str(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: expected non-empty string")
    return value


def _find_message(messages: Any, role: str, *, package_path: Path, row_index: int) -> Mapping[str, Any]:
    if not isinstance(messages, list):
        raise ValueError(f"{package_path}: row {row_index}: malformed messages")
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == role:
            return message
    raise ValueError(f"{package_path}: row {row_index}: missing {role} message")


def _validate_tool_entry(tool: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise ValueError(f"{context}: malformed tool entry")
    name = _require_str(tool.get("name"), context=f"{context}: tool name")
    kind = tool.get("kind")
    if kind not in {"query", "action"}:
        raise ValueError(f"{context}: malformed tool kind for {name}")
    arguments_schema = tool.get("arguments_schema")
    if not isinstance(arguments_schema, Mapping):
        raise ValueError(f"{context}: malformed tool schema for {name}")
    return {"name": name, "kind": kind, "arguments_schema": _thaw_json(arguments_schema)}


def _parse_controlled_user_message(
    content: Any, *, package_path: Path, row_index: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(content, str):
        raise ValueError(f"{package_path}: row {row_index}: malformed user JSON")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{package_path}: row {row_index}: malformed user JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{package_path}: row {row_index}: malformed user JSON")
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{package_path}: row {row_index}: missing tools")
    return [_validate_tool_entry(tool, context=f"{package_path}: row {row_index}") for tool in tools], dict(payload)


def _collect_row_hash(metadata: Any, key: str, *, package_path: Path, row_index: int) -> str | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{package_path}: row {row_index}: malformed metadata")
    value = metadata.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{package_path}: row {row_index}: malformed {key}")
    return value


def load_controlled_contract(package_path: Path) -> ContractSnapshot:
    try:
        with zipfile.ZipFile(package_path) as archive:
            member_bytes = archive.read("action_sequence_sft_train.parquet")
    except KeyError as exc:
        raise ValueError(f"{package_path}: missing action_sequence_sft_train.parquet") from exc
    except FileNotFoundError as exc:
        raise ValueError(f"{package_path}: package not found") from exc

    try:
        rows = pq.read_table(io.BytesIO(member_bytes)).to_pylist()
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{package_path}: failed to read action_sequence_sft_train.parquet") from exc
    if not rows:
        raise ValueError(f"{package_path}: no training rows found")

    first_system_prompt: str | None = None
    first_tools: list[dict[str, Any]] | None = None
    system_hashes: set[str] = set()
    tool_hashes: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{package_path}: row {row_index}: malformed row")
        system_message = _find_message(row.get("messages"), "system", package_path=package_path, row_index=row_index)
        user_message = _find_message(row.get("messages"), "user", package_path=package_path, row_index=row_index)
        system_prompt = _require_str(system_message.get("content"), context=f"{package_path}: row {row_index}: system")
        tools, _ = _parse_controlled_user_message(
            user_message.get("content"), package_path=package_path, row_index=row_index
        )
        if first_system_prompt is None:
            first_system_prompt = system_prompt
        elif system_prompt != first_system_prompt:
            raise ValueError(f"{package_path}: row {row_index}: mixed system content")
        if first_tools is None:
            first_tools = tools
        elif _canonicalize(tools) != _canonicalize(first_tools):
            raise ValueError(f"{package_path}: row {row_index}: mixed tools")

        system_hash = _collect_row_hash(
            row.get("metadata"), "system_prompt_sha256", package_path=package_path, row_index=row_index
        )
        if system_hash is not None:
            system_hashes.add(system_hash)
        tool_hash = _collect_row_hash(
            row.get("metadata"), "tool_contracts_sha256", package_path=package_path, row_index=row_index
        )
        if tool_hash is not None:
            tool_hashes.add(tool_hash)

    if len(system_hashes) > 1:
        raise ValueError(f"{package_path}: mixed system_prompt_sha256 values")
    if len(tool_hashes) > 1:
        raise ValueError(f"{package_path}: mixed tool_contracts_sha256 values")

    assert first_system_prompt is not None
    assert first_tools is not None
    tool_schemas = {tool["name"]: tool["arguments_schema"] for tool in first_tools}
    tool_kinds = {tool["name"]: tool["kind"] for tool in first_tools}
    source = {"package_path": str(package_path)}
    if system_hashes:
        source["system_prompt_sha256"] = next(iter(system_hashes))
    if tool_hashes:
        source["tool_contracts_sha256"] = next(iter(tool_hashes))
    fingerprint = _fingerprint_contract(
        name="controlled",
        system_prompt=first_system_prompt,
        tools_context=tuple(first_tools),
        tool_schemas=tool_schemas,
        tool_kinds=tool_kinds,
        max_steps=12,
        message_config={},
    )
    return ContractSnapshot(
        name="controlled",
        system_prompt=first_system_prompt,
        tools_context=tuple(first_tools),
        tool_schemas=tool_schemas,
        tool_kinds=tool_kinds,
        source=source,
        fingerprint=fingerprint,
        message_config={},
    )


def _load_prompt_module(prompts_path: Path) -> dict[str, str]:
    if not prompts_path.is_file():
        raise ValueError(f"{prompts_path}: missing prompts file")
    namespace = runpy.run_path(str(prompts_path))
    prompts: dict[str, str] = {}
    for name in PROMPT_CONSTANTS:
        value = namespace.get(name)
        if not isinstance(value, str):
            raise ValueError(f"{prompts_path}: missing prompt constant {name}")
        prompts[name] = value
    return prompts


def _load_descriptions(descriptions_path: Path) -> Mapping[str, JSONValue]:
    if not descriptions_path.is_file():
        raise ValueError(f"{descriptions_path}: missing YAML file")
    try:
        payload = yaml.safe_load(descriptions_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{descriptions_path}: malformed YAML") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{descriptions_path}: malformed YAML")
    descriptions = payload.get("tools")
    if not isinstance(descriptions, Mapping):
        raise ValueError(f"{descriptions_path}: missing tools mapping")
    normalized: dict[str, JSONValue] = {}
    for name, value in descriptions.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{descriptions_path}: malformed tools mapping")
        if not isinstance(value, Mapping):
            raise ValueError(f"{descriptions_path}: malformed description for tool {name}")
        normalized[name] = _normalize_json_value(
            value, context=f"{descriptions_path}: malformed description for tool {name}"
        )
    return normalized


def _load_capture_schemas(capture_path: Path) -> dict[str, Mapping[str, Any]]:
    if not capture_path.is_file():
        raise ValueError(f"{capture_path}: missing capture file")
    try:
        payload = json.loads(capture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{capture_path}: malformed capture JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{capture_path}: malformed capture JSON")
    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"{capture_path}: malformed capture JSON")
    schemas: dict[str, Mapping[str, Any]] = {}
    for entry_index, entry in enumerate(tools):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{capture_path}: tools[{entry_index}] malformed")
        function = entry.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(f"{capture_path}: tools[{entry_index}] missing function")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, Mapping):
            raise ValueError(f"{capture_path}: tools[{entry_index}] malformed function")
        schemas[name] = parameters
    return schemas


def load_production_contract(agent_root: Path, tool_capture_path: Path, max_steps: int) -> ContractSnapshot:
    prompts_path = agent_root / "src" / "omni" / "subagents" / "action_sequence" / "prompts.py"
    descriptions_path = agent_root / "src" / "omni" / "tools" / "tool_descriptions.yaml"
    prompts = _load_prompt_module(prompts_path)
    descriptions = _load_descriptions(descriptions_path)
    capture_schemas = _load_capture_schemas(tool_capture_path)

    context_map = {name: descriptions[name] for name in CONTEXT_TOOL_NAMES if name in descriptions}
    tools_context = json.dumps(context_map, ensure_ascii=False, indent=2)

    tool_schemas: dict[str, Mapping[str, Any]] = {}
    tool_kinds: dict[str, Literal["query", "action"]] = {}
    for name in context_map:
        if name not in capture_schemas:
            raise ValueError(f"{tool_capture_path}: missing schema for context tool {name}")
        tool_schemas[name] = _thaw_json(capture_schemas[name])
        if name in ACTION_TOOLS:
            tool_kinds[name] = "action"
        elif name in QUERY_TOOLS:
            tool_kinds[name] = "query"
        else:
            raise ValueError(f"{tool_capture_path}: unknown tool kind for {name}")

    message_config = {
        MESSAGE_CONFIG_KEYS[key]: prompts[key]
        for key in ("USER_TOOLS_HEADER", "USER_LIMIT_HEADER", "USER_INSTRUCTION_HEADER", "STEP_LIMIT_TEMPLATE")
    }
    source = {
        "agent_root": str(agent_root),
        "prompts_path": str(prompts_path),
        "prompts_sha256": _read_sha256(prompts_path),
        "tool_descriptions_path": str(descriptions_path),
        "tool_descriptions_sha256": _read_sha256(descriptions_path),
        "tool_capture_path": str(tool_capture_path),
        "tool_capture_sha256": _read_sha256(tool_capture_path),
    }
    fingerprint = _fingerprint_contract(
        name="production",
        system_prompt=prompts["SYSTEM_PROMPT"],
        tools_context=tools_context,
        tool_schemas=tool_schemas,
        tool_kinds=tool_kinds,
        max_steps=max_steps,
        message_config=message_config,
    )
    return ContractSnapshot(
        name="production",
        system_prompt=prompts["SYSTEM_PROMPT"],
        tools_context=tools_context,
        tool_schemas=tool_schemas,
        tool_kinds=tool_kinds,
        source=source,
        fingerprint=fingerprint,
        max_steps=max_steps,
        message_config=message_config,
    )


def build_messages(contract: ContractSnapshot, query: str) -> tuple[dict[str, str], dict[str, str]]:
    system = {"role": "system", "content": contract.system_prompt}
    if contract.name == "controlled":
        user_content = json.dumps(
            {"tools": _thaw_json(contract.tools_context), "instruction": query},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return system, {"role": "user", "content": user_content}

    user_tools_header = _require_str(
        contract.message_config.get("user_tools_header"),
        context="production message_config.user_tools_header",
    )
    user_limit_header = _require_str(
        contract.message_config.get("user_limit_header"),
        context="production message_config.user_limit_header",
    )
    user_instruction_header = _require_str(
        contract.message_config.get("user_instruction_header"),
        context="production message_config.user_instruction_header",
    )
    step_limit_template = _require_str(
        contract.message_config.get("step_limit_template"),
        context="production message_config.step_limit_template",
    )
    user_content = (
        f"{user_tools_header}\n"
        f"{contract.tools_context}\n\n"
        f"{user_limit_header}\n"
        f"{step_limit_template.format(max_steps=contract.max_steps)}\n\n"
        f"{user_instruction_header}\n"
        f"{query}"
    )
    return system, {"role": "user", "content": user_content}
