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

import ast
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BENCHMARK_20260913_REVISION = "3911119e607e1f7cd7baba92d67941c36871dead"
_BENCHMARK_PROFILE_NAME = "benchmark-20260913-production"
_BENCHMARK_MAX_STEPS = 12
_PROMPTS_PATH = "src/omni/subagents/action_sequence/prompts.py"
_PLANNER_PATH = "src/omni/subagents/action_sequence/planner.py"
_DESCRIPTIONS_PATH = "src/omni/tools/tool_descriptions.yaml"
_PROMPT_KEYS = (
    "SYSTEM_PROMPT",
    "USER_TOOLS_HEADER",
    "USER_LIMIT_HEADER",
    "USER_INSTRUCTION_HEADER",
    "STEP_LIMIT_TEMPLATE",
)
_SAFE_EXPRESSION_NODES = (
    ast.Add,
    ast.BinOp,
    ast.Constant,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Load,
    ast.Name,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_text(repo_path: Path, revision: str, object_path: str | None = None) -> str:
    command = ["git", "-C", str(repo_path)]
    if object_path is None:
        command.extend(["rev-parse", f"{revision}^{{commit}}"])
    else:
        command.extend(["show", f"{revision}:{object_path}"])
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        label = revision if object_path is None else f"{revision}:{object_path}"
        raise ValueError(f"unable to read production prompt object {label}") from exc


def _prompt_constants(source: str) -> dict[str, str]:
    tree = ast.parse(source, filename=_PROMPTS_PATH)
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if any(not isinstance(inner, _SAFE_EXPRESSION_NODES) for inner in ast.walk(node.value)):
            raise ValueError(f"{_PROMPTS_PATH}: unsupported expression for {node.targets[0].id}")
        try:
            values[node.targets[0].id] = eval(  # noqa: S307
                compile(ast.Expression(node.value), _PROMPTS_PATH, "eval"),
                {"__builtins__": {}},
                values,
            )
        except Exception as exc:
            raise ValueError(f"{_PROMPTS_PATH}: unable to evaluate {node.targets[0].id}") from exc

    prompts: dict[str, str] = {}
    for key in _PROMPT_KEYS:
        value = values.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{_PROMPTS_PATH}: missing string constant {key}")
        prompts[key] = value
    return prompts


def _context_tool_names(source: str) -> tuple[str, ...]:
    tree = ast.parse(source, filename=_PLANNER_PATH)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "_CONTEXT_TOOL_NAMES":
            value = ast.literal_eval(node.value)
            if isinstance(value, tuple) and value and all(isinstance(item, str) and item for item in value):
                return value
            break
    raise ValueError(f"{_PLANNER_PATH}: missing _CONTEXT_TOOL_NAMES")


def _tools_context(source: str, names: tuple[str, ...]) -> str:
    payload = yaml.safe_load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), dict):
        raise ValueError(f"{_DESCRIPTIONS_PATH}: missing tools mapping")
    tools = payload["tools"]
    missing = [name for name in names if name not in tools]
    if missing:
        raise ValueError(f"{_DESCRIPTIONS_PATH}: missing context tools {missing}")
    return json.dumps({name: tools[name] for name in names}, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class ProductionPromptProfile:
    revision: str
    system_prompt: str
    tools_context: str
    user_tools_header: str
    user_limit_header: str
    user_instruction_header: str
    step_limit_template: str
    max_steps: int
    prompt_source_sha256: str
    planner_source_sha256: str
    descriptions_source_sha256: str

    def build_user_prompt(self, query: str) -> str:
        return (
            f"{self.user_tools_header}\n{self.tools_context}\n\n"
            f"{self.user_limit_header}\n"
            f"{self.step_limit_template.format(max_steps=self.max_steps)}\n\n"
            f"{self.user_instruction_header}\n{query}"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": _BENCHMARK_PROFILE_NAME,
            "source_run": "20260913_151333_groupfiles238_range0-237",
            "agent_revision": self.revision,
            "max_steps": self.max_steps,
            "system_prompt_sha256": _sha256(self.system_prompt),
            "tools_context_sha256": _sha256(self.tools_context),
            "prompt_source_sha256": self.prompt_source_sha256,
            "planner_source_sha256": self.planner_source_sha256,
            "descriptions_source_sha256": self.descriptions_source_sha256,
        }

    @property
    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "system_prompt": self.system_prompt,
            "tools_context": self.tools_context,
            "user_prompt_template": self.build_user_prompt("{query}"),
        }


def load_benchmark_20260913_profile(repo_path: Path) -> ProductionPromptProfile:
    revision = _git_text(repo_path, BENCHMARK_20260913_REVISION).strip()
    if revision != BENCHMARK_20260913_REVISION:
        raise ValueError(f"unexpected production prompt revision: {revision}")

    prompt_source = _git_text(repo_path, revision, _PROMPTS_PATH)
    planner_source = _git_text(repo_path, revision, _PLANNER_PATH)
    descriptions_source = _git_text(repo_path, revision, _DESCRIPTIONS_PATH)
    prompts = _prompt_constants(prompt_source)
    tools_context = _tools_context(descriptions_source, _context_tool_names(planner_source))
    return ProductionPromptProfile(
        revision=revision,
        system_prompt=prompts["SYSTEM_PROMPT"],
        tools_context=tools_context,
        user_tools_header=prompts["USER_TOOLS_HEADER"],
        user_limit_header=prompts["USER_LIMIT_HEADER"],
        user_instruction_header=prompts["USER_INSTRUCTION_HEADER"],
        step_limit_template=prompts["STEP_LIMIT_TEMPLATE"],
        max_steps=_BENCHMARK_MAX_STEPS,
        prompt_source_sha256=_sha256(prompt_source),
        planner_source_sha256=_sha256(planner_source),
        descriptions_source_sha256=_sha256(descriptions_source),
    )
