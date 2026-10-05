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

import io
import json
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from planner_val.contracts import build_messages, load_controlled_contract, load_production_contract
from planner_val.models import ContractSnapshot


def _write_training_package(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("action_sequence_sft_train.parquet", buffer.getvalue())
    return path


def _make_controlled_row(
    *,
    system_prompt: str = "system prompt",
    tools: list[dict[str, Any]] | None = None,
    instruction: str = "training instruction",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    tool_list = tools or [
        {
            "name": "get_weather",
            "kind": "query",
            "arguments_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        {
            "name": "HumanAction",
            "kind": "action",
            "arguments_schema": {
                "type": "object",
                "properties": {"action": {"type": "string"}},
                "required": ["action"],
            },
        },
    ]
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps({"tools": tool_list, "instruction": instruction}, ensure_ascii=False),
            },
        ],
        "metadata": metadata
        or {
            "system_prompt_sha256": "sha256:sys",
            "tool_contracts_sha256": "sha256:tools",
        },
    }


def _write_prompts_file(path: Path, *, system_prompt: str = "prod system") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"SYSTEM_PROMPT = {system_prompt!r}",
                'USER_TOOLS_HEADER = "Tools Header"',
                'USER_LIMIT_HEADER = "Limit Header"',
                'USER_INSTRUCTION_HEADER = "Instruction Header"',
                'STEP_LIMIT_TEMPLATE = "At most {max_steps} steps"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _nested_description(label: str) -> dict[str, object]:
    return {
        "summary": f"{label} summary",
        "examples": [f"{label} example"],
        "arguments": {
            "value": {
                "type": "string",
                "description": f"{label} argument",
            }
        },
    }


def _write_production_files(
    root: Path,
    *,
    descriptions: dict[str, object] | None = None,
    capture_tools: list[dict[str, object]] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    agent_root = root / "agent"
    prompts_path = agent_root / "src" / "omni" / "subagents" / "action_sequence" / "prompts.py"
    _write_prompts_file(prompts_path)
    if headers is not None:
        prompts_path.write_text(
            "\n".join(
                [
                    'SYSTEM_PROMPT = "prod system"',
                    f"USER_TOOLS_HEADER = {headers['USER_TOOLS_HEADER']!r}",
                    f"USER_LIMIT_HEADER = {headers['USER_LIMIT_HEADER']!r}",
                    f"USER_INSTRUCTION_HEADER = {headers['USER_INSTRUCTION_HEADER']!r}",
                    f"STEP_LIMIT_TEMPLATE = {headers['STEP_LIMIT_TEMPLATE']!r}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    descriptions_path = agent_root / "src" / "omni" / "tools" / "tool_descriptions.yaml"
    descriptions_path.parent.mkdir(parents=True, exist_ok=True)
    descriptions_payload = descriptions or {
        "HumanAction": _nested_description("human action"),
        "RobotGesture": _nested_description("robot gesture"),
        "RobotDance": _nested_description("robot dance"),
        "get_weather": _nested_description("weather"),
        "web_search": _nested_description("web search"),
        "rag_query": _nested_description("rag query"),
        "robot_status": _nested_description("robot status"),
        "get_visual_info": _nested_description("visual info"),
        "EndChat": _nested_description("ignored"),
    }
    descriptions_path.write_text(yaml.safe_dump({"tools": descriptions_payload}, sort_keys=False), encoding="utf-8")

    capture_path = root / "capture.json"
    tools_payload = capture_tools or [
        {
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        }
        for name in (
            "HumanAction",
            "RobotGesture",
            "RobotDance",
            "get_weather",
            "web_search",
            "rag_query",
            "robot_status",
            "get_visual_info",
            "EndChat",
            "ActionSequence",
        )
    ]
    capture_path.write_text(json.dumps({"tools": tools_payload}, ensure_ascii=False, indent=2), encoding="utf-8")
    return agent_root, capture_path


def test_controlled_extraction_rebuilds_compact_training_user_json(tmp_path: Path) -> None:
    package_path = _write_training_package(tmp_path / "train.zip", [_make_controlled_row(), _make_controlled_row()])

    contract = load_controlled_contract(package_path)
    system_message, user_message = build_messages(contract, "fresh query")

    assert system_message == {"role": "system", "content": "system prompt"}
    assert user_message == {
        "role": "user",
        "content": json.dumps(
            {
                "tools": [
                    {
                        "name": "get_weather",
                        "kind": "query",
                        "arguments_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                    {
                        "name": "HumanAction",
                        "kind": "action",
                        "arguments_schema": {
                            "type": "object",
                            "properties": {"action": {"type": "string"}},
                            "required": ["action"],
                        },
                    },
                ],
                "instruction": "fresh query",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    assert contract.source["package_path"] == str(package_path)
    assert contract.source["system_prompt_sha256"] == "sha256:sys"
    assert contract.source["tool_contracts_sha256"] == "sha256:tools"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        pytest.param(
            [_make_controlled_row(), _make_controlled_row(system_prompt="other prompt")],
            "row 1",
            id="mixed-system-content",
        ),
        pytest.param(
            [
                _make_controlled_row(),
                _make_controlled_row(
                    metadata={"system_prompt_sha256": "sha256:other", "tool_contracts_sha256": "sha256:tools"}
                ),
            ],
            "system_prompt_sha256",
            id="mixed-system-hash",
        ),
        pytest.param(
            [
                _make_controlled_row(),
                _make_controlled_row(
                    metadata={"system_prompt_sha256": "sha256:sys", "tool_contracts_sha256": "sha256:other"}
                ),
            ],
            "tool_contracts_sha256",
            id="mixed-tool-hash",
        ),
        pytest.param(
            [
                _make_controlled_row(),
                _make_controlled_row(
                    tools=[
                        {
                            "name": "get_weather",
                            "kind": "query",
                            "arguments_schema": {"type": "object", "properties": {"city": {"type": "number"}}},
                        }
                    ]
                ),
            ],
            "row 1",
            id="mixed-tools",
        ),
        pytest.param(
            [{"messages": [{"role": "user", "content": "{}"}], "metadata": None}],
            "system",
            id="missing-system-message",
        ),
        pytest.param(
            [{"messages": [{"role": "system", "content": "system only"}], "metadata": None}],
            "user",
            id="missing-user-message",
        ),
        pytest.param(
            [
                {
                    "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "{"}],
                    "metadata": None,
                }
            ],
            "malformed user JSON",
            id="malformed-user-json",
        ),
        pytest.param(
            [_make_controlled_row(tools=[{"name": "", "kind": "query", "arguments_schema": {"type": "object"}}])],
            "tool",
            id="malformed-tool-entry",
        ),
    ],
)
def test_controlled_extraction_rejects_invalid_training_contracts(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    package_path = _write_training_package(tmp_path / "train.zip", rows)

    with pytest.raises(ValueError, match=message):
        load_controlled_contract(package_path)


def test_fingerprint_is_canonical_and_changes_for_semantic_differences(tmp_path: Path) -> None:
    tool_a = {
        "name": "get_weather",
        "kind": "query",
        "arguments_schema": {
            "required": ["city"],
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "type": "object",
        },
    }
    tool_b = {
        "arguments_schema": {
            "type": "object",
            "properties": {"unit": {"type": "string"}, "city": {"type": "string"}},
            "required": ["city"],
        },
        "kind": "query",
        "name": "get_weather",
    }
    base = load_controlled_contract(
        _write_training_package(tmp_path / "base.zip", [_make_controlled_row(tools=[tool_a], metadata={})])
    )
    reordered = load_controlled_contract(
        _write_training_package(tmp_path / "reordered.zip", [_make_controlled_row(tools=[tool_b], metadata={})])
    )
    different_prompt = load_controlled_contract(
        _write_training_package(
            tmp_path / "prompt.zip",
            [_make_controlled_row(system_prompt="other prompt", tools=[tool_a], metadata={})],
        )
    )
    different_schema = load_controlled_contract(
        _write_training_package(
            tmp_path / "schema.zip",
            [
                _make_controlled_row(
                    tools=[
                        {
                            "name": "get_weather",
                            "kind": "query",
                            "arguments_schema": {"type": "object", "properties": {"city": {"type": "number"}}},
                        }
                    ],
                    metadata={},
                )
            ],
        )
    )
    different_kind = load_controlled_contract(
        _write_training_package(
            tmp_path / "kind.zip",
            [
                _make_controlled_row(
                    tools=[
                        {
                            "name": "get_weather",
                            "kind": "action",
                            "arguments_schema": tool_a["arguments_schema"],
                        }
                    ],
                    metadata={},
                )
            ],
        )
    )

    agent_root, capture_path = _write_production_files(tmp_path / "production")
    max_steps_12 = load_production_contract(agent_root, capture_path, 12)
    max_steps_3 = load_production_contract(agent_root, capture_path, 3)

    assert base.fingerprint == reordered.fingerprint
    assert base.fingerprint != different_prompt.fingerprint
    assert base.fingerprint != different_schema.fingerprint
    assert base.fingerprint != different_kind.fingerprint
    assert max_steps_12.fingerprint != max_steps_3.fingerprint


def test_source_path_changes_do_not_change_semantic_fingerprint(tmp_path: Path) -> None:
    first = load_controlled_contract(_write_training_package(tmp_path / "a.zip", [_make_controlled_row(metadata={})]))
    second = load_controlled_contract(
        _write_training_package(tmp_path / "nested" / "b.zip", [_make_controlled_row(metadata={})])
    )

    assert first.fingerprint == second.fingerprint
    assert first.source["package_path"] != second.source["package_path"]


def test_controlled_allows_rows_with_different_instruction_when_system_and_tools_match(tmp_path: Path) -> None:
    package_path = _write_training_package(
        tmp_path / "train.zip",
        [
            _make_controlled_row(instruction="first instruction"),
            _make_controlled_row(instruction="second instruction"),
        ],
    )

    contract = load_controlled_contract(package_path)

    assert contract.system_prompt == "system prompt"
    assert tuple(contract.tool_schemas) == ("get_weather", "HumanAction")


def test_production_message_layout_headers_and_context_order(tmp_path: Path) -> None:
    descriptions = {
        "web_search": _nested_description("search first"),
        "HumanAction": _nested_description("action first"),
        "unrelated": _nested_description("ignore me"),
        "robot_status": _nested_description("status"),
        "get_visual_info": _nested_description("vision"),
        "RobotDance": _nested_description("dance"),
        "rag_query": _nested_description("rag"),
        "RobotGesture": _nested_description("gesture"),
        "get_weather": _nested_description("weather"),
    }
    capture_tools = [
        {"function": {"name": "EndChat", "parameters": {"type": "object"}}},
        {
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        },
        {
            "function": {
                "name": "HumanAction",
                "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
            }
        },
        {
            "function": {
                "name": "RobotGesture",
                "parameters": {"type": "object", "properties": {"gesture": {"type": "string"}}},
            }
        },
        {
            "function": {
                "name": "RobotDance",
                "parameters": {"type": "object", "properties": {"dance": {"type": "string"}}},
            }
        },
        {
            "function": {
                "name": "web_search",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        },
        {"function": {"name": "rag_query", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
        {
            "function": {
                "name": "robot_status",
                "parameters": {"type": "object", "properties": {"detail": {"type": "boolean"}}},
            }
        },
        {
            "function": {
                "name": "get_visual_info",
                "parameters": {"type": "object", "properties": {"frame": {"type": "integer"}}},
            }
        },
        {"function": {"name": "ActionSequence", "parameters": {"type": "object"}}},
    ]
    agent_root, capture_path = _write_production_files(
        tmp_path / "production-layout", descriptions=descriptions, capture_tools=capture_tools
    )

    contract = load_production_contract(agent_root, capture_path, 5)
    system_message, user_message = build_messages(contract, "do the thing")

    expected_context = json.dumps(
        {
            "HumanAction": descriptions["HumanAction"],
            "RobotGesture": descriptions["RobotGesture"],
            "RobotDance": descriptions["RobotDance"],
            "get_weather": descriptions["get_weather"],
            "web_search": descriptions["web_search"],
            "rag_query": descriptions["rag_query"],
            "robot_status": descriptions["robot_status"],
            "get_visual_info": descriptions["get_visual_info"],
        },
        ensure_ascii=False,
        indent=2,
    )
    assert contract.tools_context == expected_context
    assert tuple(contract.tool_schemas) == (
        "HumanAction",
        "RobotGesture",
        "RobotDance",
        "get_weather",
        "web_search",
        "rag_query",
        "robot_status",
        "get_visual_info",
    )
    assert tuple(contract.tool_kinds) == tuple(contract.tool_schemas)
    assert system_message == {"role": "system", "content": "prod system"}
    assert user_message == {
        "role": "user",
        "content": (
            f"Tools Header\n{expected_context}\n\nLimit Header\nAt most 5 steps\n\nInstruction Header\ndo the thing"
        ),
    }
    assert user_message["content"].count("Tools Header") == 1
    assert user_message["content"].count("Limit Header") == 1
    assert user_message["content"].count("Instruction Header") == 1
    assert "EndChat" not in contract.tool_schemas
    assert "ActionSequence" not in contract.tool_schemas


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(lambda root, capture: capture.unlink(), "capture", id="missing-capture"),
        pytest.param(
            lambda root, capture: (root / "src" / "omni" / "subagents" / "action_sequence" / "prompts.py").unlink(),
            "prompts",
            id="missing-prompts",
        ),
        pytest.param(
            lambda root, capture: (root / "src" / "omni" / "tools" / "tool_descriptions.yaml").write_text(
                "tools: [", encoding="utf-8"
            ),
            "YAML",
            id="malformed-yaml",
        ),
        pytest.param(
            lambda root, capture: (root / "src" / "omni" / "tools" / "tool_descriptions.yaml").write_text(
                "tools:\n  HumanAction: !!set\n    ? nope\n", encoding="utf-8"
            ),
            "description",
            id="non-json-compatible-description",
        ),
        pytest.param(
            lambda root, capture: capture.write_text("{", encoding="utf-8"),
            "JSON",
            id="malformed-capture",
        ),
        pytest.param(
            lambda root, capture: (root / "src" / "omni" / "subagents" / "action_sequence" / "prompts.py").write_text(
                'SYSTEM_PROMPT = "prod system"\nUSER_TOOLS_HEADER = "Tools Header"\n', encoding="utf-8"
            ),
            "USER_LIMIT_HEADER",
            id="missing-prompt-constants",
        ),
        pytest.param(
            lambda root, capture: capture.write_text(
                json.dumps(
                    {"tools": [{"function": {"name": "HumanAction", "parameters": {"type": "object"}}}]}, indent=2
                ),
                encoding="utf-8",
            ),
            "RobotGesture",
            id="missing-context-schema",
        ),
    ],
)
def test_production_rejects_missing_or_malformed_inputs(tmp_path: Path, mutate: Any, message: str) -> None:
    agent_root, capture_path = _write_production_files(tmp_path / "invalid-production")
    mutate(agent_root, capture_path)

    with pytest.raises(ValueError, match=message):
        load_production_contract(agent_root, capture_path, 7)


@pytest.mark.parametrize(
    ("yaml_value", "snippet"),
    [
        pytest.param(".nan", "non-finite float", id="nan"),
        pytest.param(".inf", "non-finite float", id="inf"),
        pytest.param("-.inf", "non-finite float", id="neg-inf"),
    ],
)
def test_production_rejects_non_finite_description_values(tmp_path: Path, yaml_value: str, snippet: str) -> None:
    agent_root, capture_path = _write_production_files(tmp_path / yaml_value.replace(".", "_"))
    descriptions_path = agent_root / "src" / "omni" / "tools" / "tool_descriptions.yaml"
    descriptions_path.write_text(
        "\n".join(
            [
                "tools:",
                "  HumanAction:",
                f"    summary: {yaml_value}",
                "    examples:",
                "      - example",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=snippet):
        load_production_contract(agent_root, capture_path, 7)


def test_production_accepts_finite_numeric_description_values(tmp_path: Path) -> None:
    agent_root, capture_path = _write_production_files(tmp_path / "finite-numeric")
    descriptions_path = agent_root / "src" / "omni" / "tools" / "tool_descriptions.yaml"
    descriptions_path.write_text(
        "\n".join(
            [
                "tools:",
                "  HumanAction:",
                "    summary: 1.25",
                "    count: 3",
                "    examples:",
                "      - example",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    contract = load_production_contract(agent_root, capture_path, 7)

    assert '"count": 3' in contract.tools_context
    assert '"summary": 1.25' in contract.tools_context


def test_production_fingerprint_changes_when_description_or_schema_changes(tmp_path: Path) -> None:
    first_root, first_capture = _write_production_files(tmp_path / "first")
    second_root, second_capture = _write_production_files(
        tmp_path / "second",
        descriptions={
            "HumanAction": _nested_description("changed description"),
            "RobotGesture": _nested_description("robot gesture"),
            "RobotDance": _nested_description("robot dance"),
            "get_weather": _nested_description("weather"),
            "web_search": _nested_description("web search"),
            "rag_query": _nested_description("rag query"),
            "robot_status": _nested_description("robot status"),
            "get_visual_info": _nested_description("visual info"),
        },
    )
    third_root, third_capture = _write_production_files(tmp_path / "third")
    third_capture.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "function": {
                            "name": "HumanAction",
                            "parameters": {"type": "object", "properties": {"action": {"type": "integer"}}},
                        }
                    },
                    *[
                        {
                            "function": {
                                "name": name,
                                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
                            }
                        }
                        for name in (
                            "RobotGesture",
                            "RobotDance",
                            "get_weather",
                            "web_search",
                            "rag_query",
                            "robot_status",
                            "get_visual_info",
                        )
                    ],
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    base = load_production_contract(first_root, first_capture, 7)
    changed_description = load_production_contract(second_root, second_capture, 7)
    changed_schema = load_production_contract(third_root, third_capture, 7)

    assert base.fingerprint != changed_description.fingerprint
    assert base.fingerprint != changed_schema.fingerprint


def test_production_fingerprint_changes_for_each_message_header_and_template(tmp_path: Path) -> None:
    base_headers = {
        "USER_TOOLS_HEADER": "Tools Header",
        "USER_LIMIT_HEADER": "Limit Header",
        "USER_INSTRUCTION_HEADER": "Instruction Header",
        "STEP_LIMIT_TEMPLATE": "At most {max_steps} steps",
    }
    base_root, base_capture = _write_production_files(tmp_path / "base", headers=base_headers)
    base = load_production_contract(base_root, base_capture, 7)

    for field, changed_value in (
        ("USER_TOOLS_HEADER", "Other Tools Header"),
        ("USER_LIMIT_HEADER", "Other Limit Header"),
        ("USER_INSTRUCTION_HEADER", "Other Instruction Header"),
        ("STEP_LIMIT_TEMPLATE", "Use <= {max_steps} steps"),
    ):
        changed_headers = dict(base_headers)
        changed_headers[field] = changed_value
        changed_root, changed_capture = _write_production_files(
            tmp_path / field.lower(),
            headers=changed_headers,
        )
        changed = load_production_contract(changed_root, changed_capture, 7)
        assert changed.fingerprint != base.fingerprint


def test_production_fingerprint_ignores_source_paths_and_file_hashes_when_semantics_match(tmp_path: Path) -> None:
    first_root, first_capture = _write_production_files(tmp_path / "first")
    second_root, second_capture = _write_production_files(tmp_path / "second")
    prompts_path = second_root / "src" / "omni" / "subagents" / "action_sequence" / "prompts.py"
    prompts_path.write_text(prompts_path.read_text(encoding="utf-8") + "# provenance only\n", encoding="utf-8")
    descriptions_path = second_root / "src" / "omni" / "tools" / "tool_descriptions.yaml"
    descriptions_data = yaml.safe_load(descriptions_path.read_text(encoding="utf-8"))
    descriptions_path.write_text(yaml.safe_dump(descriptions_data, sort_keys=False), encoding="utf-8")
    second_capture.write_text(
        json.dumps(json.loads(second_capture.read_text(encoding="utf-8")), indent=4),
        encoding="utf-8",
    )

    first = load_production_contract(first_root, first_capture, 7)
    second = load_production_contract(second_root, second_capture, 7)

    assert first.fingerprint == second.fingerprint
    assert first.source["prompts_path"] != second.source["prompts_path"]
    assert first.source["prompts_sha256"] != second.source["prompts_sha256"]


def test_build_messages_uses_message_config_not_source_provenance() -> None:
    snapshot = ContractSnapshot(
        name="production",
        system_prompt="sys",
        tools_context='{"tool":{"summary":"ok"}}',
        tool_schemas={"tool": {"type": "object"}},
        tool_kinds={"tool": "query"},
        message_config={
            "user_tools_header": "Tools Header",
            "user_limit_header": "Limit Header",
            "user_instruction_header": "Instruction Header",
            "step_limit_template": "At most {max_steps} steps",
        },
        source={
            "prompts_path": "/tmp/prompts.py",
            "USER_TOOLS_HEADER": "Wrong Tools Header",
            "USER_LIMIT_HEADER": "Wrong Limit Header",
            "USER_INSTRUCTION_HEADER": "Wrong Instruction Header",
            "STEP_LIMIT_TEMPLATE": "Wrong {max_steps}",
        },
        fingerprint="sha256:test",
        max_steps=4,
    )

    _, user_message = build_messages(snapshot, "query")

    assert (
        user_message["content"]
        == 'Tools Header\n{"tool":{"summary":"ok"}}\n\nLimit Header\nAt most 4 steps\n\nInstruction Header\nquery'
    )


def test_contract_snapshots_are_recursively_immutable(tmp_path: Path) -> None:
    controlled = load_controlled_contract(_write_training_package(tmp_path / "train.zip", [_make_controlled_row()]))

    assert isinstance(controlled.tools_context, tuple)
    assert isinstance(controlled.tools_context[0], MappingProxyType)
    assert isinstance(controlled.tool_schemas, MappingProxyType)
    assert isinstance(controlled.tool_schemas["get_weather"], MappingProxyType)
    assert isinstance(controlled.message_config, MappingProxyType)
    with pytest.raises(TypeError):
        controlled.tool_schemas["get_weather"] = {"type": "object"}
    with pytest.raises(TypeError):
        controlled.tool_schemas["get_weather"]["type"] = "array"
    with pytest.raises(AttributeError):
        controlled.tools_context.append("x")


def test_contract_snapshot_direct_construction_freezes_nested_values() -> None:
    snapshot = ContractSnapshot(
        name="controlled",
        system_prompt="sys",
        tools_context={"outer": [{"inner": [1, 2, 3]}]},
        tool_schemas={"tool": {"properties": {"value": {"type": "string"}}}},
        tool_kinds={"tool": "query"},
        message_config={"user_tools_header": "Tools Header"},
        source={"path": "/tmp/source"},
        fingerprint="sha256:test",
    )

    assert isinstance(snapshot.tools_context, MappingProxyType)
    assert isinstance(snapshot.tools_context["outer"], tuple)
    assert isinstance(snapshot.tools_context["outer"][0], MappingProxyType)
    assert isinstance(snapshot.tool_schemas["tool"], MappingProxyType)
    assert isinstance(snapshot.message_config, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot.tools_context["outer"][0]["inner"] = ()
