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

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(inner_value) for key, inner_value in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(inner_value) for key, inner_value in value.items()})


@dataclass(frozen=True)
class ContractSnapshot:
    name: Literal["controlled", "production"]
    system_prompt: str
    tools_context: JSONValue
    tool_schemas: Mapping[str, Mapping[str, Any]]
    tool_kinds: Mapping[str, Literal["query", "action"]]
    source: Mapping[str, str]
    fingerprint: str
    max_steps: int = 12
    message_config: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools_context", _freeze_value(self.tools_context))
        object.__setattr__(self, "tool_schemas", _freeze_mapping(self.tool_schemas))
        object.__setattr__(self, "tool_kinds", _freeze_mapping(self.tool_kinds))
        object.__setattr__(self, "message_config", _freeze_mapping(self.message_config))
        object.__setattr__(self, "source", _freeze_mapping(self.source))


@dataclass(frozen=True)
class ExpectedTask:
    kind: Literal["reply", "query", "action"]
    instruction: str
    name: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True)
class ParsedTask:
    kind: Literal["reply", "query", "action"]
    instruction: str
    name: str | None
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True)
class ParseResult:
    tasks: tuple[ParsedTask, ...] = ()
    valid_json: bool = False
    contract_valid: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    reason: str
    strict_pass: bool
    valid_json: bool
    contract_valid: bool
    task_count_match: bool
    ordered_kind_name_match: bool
    parameter_match: bool | None
    forbidden_tool_violation: bool
    parsed_tasks: tuple[ParsedTask, ...]
    parameter_errors: tuple[str, ...]
    forbidden_calls: tuple[str, ...]
    raw_output: str | None


@dataclass(frozen=True)
class BenchmarkScoreResult:
    reason: str
    passed: bool
    parser_valid: bool
    tool_sequence_match: bool
    parameter_match: bool | None
    forbidden_tool_violation: bool
    reply_slots: int
    expected_tool_count: int
    predicted_tool_count: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class OverlapInfo:
    exact_train: bool = False
    exact_validation: bool = False
    all_atoms_seen_train: bool = False
    has_novel_atom: bool = False


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    expected: tuple[ExpectedTask, ...]
    forbidden_functions: frozenset[str]
    level: str
    language: str
    difficulty: str
    polarity: str
    source_atom_ids: frozenset[str]
    overlap: OverlapInfo = OverlapInfo()


@dataclass(frozen=True)
class TrainingCorpusIndex:
    train_instructions: frozenset[str]
    validation_instructions: frozenset[str]
    train_source_atom_ids: frozenset[str]
    validation_source_atom_ids: frozenset[str]
