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

"""Deterministic planning contract compilation for ActionSequence training data."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise, product
from typing import TYPE_CHECKING, Any, Literal

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from planner_data.common import is_range_spec, value_in_range


if TYPE_CHECKING:
    from pathlib import Path


PlanKind = Literal["reply", "query", "action"]
CallKind = Literal["query", "action"]


class PlannerTarget(BaseModel):
    """Planner target annotation attached to a canonical L0 atom."""

    model_config = ConfigDict(extra="forbid")

    kind: PlanKind
    instructions: dict[str, str]


class SubagentToolContract(BaseModel):
    """Tool contract used to validate executable call tasks."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: CallKind
    description: str
    arguments_schema: dict[str, Any]

    @field_validator("arguments_schema")
    @classmethod
    def _validate_arguments_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as exc:
            raise ValueError(f"invalid Draft 2020-12 JSON Schema: {exc.message}") from exc
        return value


class ContractSource(BaseModel):
    """Source and stable identity for a planner contract bundle."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["curated", "agent-exported"]
    identity: dict[str, str]

    @field_validator("identity")
    @classmethod
    def _validate_identity(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("source identity must not be empty")
        normalized: dict[str, str] = {}
        for key, raw in value.items():
            key_text = str(key).strip()
            value_text = str(raw).strip()
            if not key_text or not value_text:
                raise ValueError("source identity must contain non-empty string pairs")
            normalized[key_text] = value_text
        return normalized


class SystemPromptContract(BaseModel):
    """Stable system prompt contract used by the planner."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str


class ActionSequenceContractBundle(BaseModel):
    """Frozen planner contract bundle with display order and provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["action-sequence-contract/v1"]
    source: ContractSource
    system_prompt: SystemPromptContract
    tools: list[SubagentToolContract]
    display_order: list[str]

    @model_validator(mode="after")
    def _validate_display_order(self) -> ActionSequenceContractBundle:
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tools contain duplicate names")
        if len(self.display_order) != len(set(self.display_order)):
            raise ValueError("display_order must not contain duplicates")
        if set(tool_names) != set(self.display_order):
            raise ValueError("display_order must be an exact set match for tools")
        ordered = {tool.name: tool for tool in self.tools}
        self.tools = [ordered[name] for name in self.display_order]
        return self


class ReplyTask(BaseModel):
    """Non-executable reply task."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reply"]
    instruction: str


class CallTask(BaseModel):
    """Executable query/action task."""

    model_config = ConfigDict(extra="forbid")

    kind: CallKind
    instruction: str
    name: str
    arguments: dict[str, Any]


class PlanGroundTruth(BaseModel):
    """Compiled planning truth produced from canonical L0 atoms."""

    model_config = ConfigDict(extra="forbid")

    schema: Literal["action-sequence-plan/v1"]
    semantic_id: str
    language: str
    contract_ref: dict[str, str]
    provenance: Literal["generated", "legacy-reviewed"]
    tasks: list[ReplyTask | CallTask]
    lineage: list[dict[str, Any]]


class PlanningCompileError(ValueError):
    """Raised when plan compilation fails with a stable reason code."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {message}")


class _StepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_function: str | None = None
    expected_params: dict[str, Any] | None = None


class _AtomDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    test_input: str
    expected_steps: list[_StepDraft] = Field(default_factory=list)
    planner_target: PlannerTarget | None = None


_PLAN_SCHEMA_VERSION = "action-sequence-plan/v1"
_SEMANTIC_PROGRAM_SCHEMA_VERSION = "semantic-program/v1"
_RANGE_INSTANTIATION_VERSION = "range-instantiation/v2"
_MAX_RANGE_COMBINATIONS = 100_000


def _canonicalize(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Serialize a value into canonical JSON with stable key ordering."""
    return json.dumps(
        _canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_prefixed(value: object) -> str:
    blob = canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def sha256_identity(value: object) -> str:
    """Return a SHA-256 identity string for canonical JSON content."""
    blob = canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def load_contract_bundle(path: Path) -> ActionSequenceContractBundle:
    """Load and validate a frozen planner contract bundle from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load contract bundle: {path}") from exc
    try:
        return ActionSequenceContractBundle.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid contract bundle: {path}") from exc


def contract_ref(bundle: ActionSequenceContractBundle) -> dict[str, str]:
    """Stable full SHA-256 references for system prompt and tool contracts."""
    ordered_tools = sorted(bundle.tools, key=lambda tool: tool.name)
    return {
        "system_prompt_sha256": _sha256_prefixed(bundle.system_prompt.model_dump(mode="python")),
        "tool_contracts_sha256": _sha256_prefixed(
            [tool.model_dump(mode="python") for tool in ordered_tools]
        ),
    }


def render_planner_messages(
    *, system_prompt: str, tools: Sequence[SubagentToolContract], instruction: str
) -> list[dict[str, str]]:
    """Render the stable planner prompt as ordinary chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "tools": [tool.model_dump(mode="python") for tool in tools],
                    "instruction": instruction,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def render_assistant_target(tasks: Sequence[ReplyTask | CallTask]) -> str:
    """Render the assistant target as compact UTF-8 JSON."""
    return json.dumps(
        [task.model_dump(mode="python") for task in tasks],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def semantic_program_id(atom_ids: Sequence[str], slots: Sequence[Mapping[str, Any]]) -> str:
    """Return the stable semantic program id for ordered atoms and slots."""
    payload = {
        "schema": _SEMANTIC_PROGRAM_SCHEMA_VERSION,
        "atom_ids": list(atom_ids),
        "slots": [dict(slot) for slot in slots],
    }
    return sha256_identity(payload)


def _find_contract(
    name: str, tools: Sequence[SubagentToolContract]
) -> SubagentToolContract | None:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def validate_task_against_contract(
    task: CallTask, tools: Sequence[SubagentToolContract]
) -> None:
    """Validate an executable task against its matching tool contract."""
    contract = _find_contract(task.name, tools)
    if contract is None:
        raise PlanningCompileError("unknown_tool", f"unknown tool: {task.name}")
    if contract.kind != task.kind:
        raise PlanningCompileError(
            "tool_kind_mismatch",
            f"tool kind mismatch for {task.name}: expected {contract.kind}, got {task.kind}",
        )
    try:
        Draft202012Validator(contract.arguments_schema).validate(task.arguments)
    except ValidationError as exc:
        raise PlanningCompileError("contract_validation_failed", str(exc)) from exc
    except Exception as exc:
        raise PlanningCompileError("contract_validation_failed", str(exc)) from exc


def _range_paths(value: object, path: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    if is_range_spec(value):
        return [path]
    paths: list[tuple[str | int, ...]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            paths.extend(_range_paths(value[key], (*path, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_range_paths(item, (*path, index)))
    return paths


def _value_at_path(value: object, path: Sequence[str | int]) -> object:
    current: Any = value
    for part in path:
        if (
            (isinstance(part, int) and isinstance(current, list))
            or (isinstance(part, str) and isinstance(current, Mapping))
        ):
            current = current[part]
        else:
            raise PlanningCompileError("non_concrete_arguments", "invalid range path")
    return current


def _schemas_at_path(
    schema: Mapping[str, Any], path: Sequence[str | int]
) -> list[Mapping[str, Any]]:
    nodes: list[Mapping[str, Any]] = [schema]
    for part in path:
        expanded: list[Mapping[str, Any]] = []
        pending = list(nodes)
        while pending:
            node = pending.pop()
            expanded.append(node)
            for keyword in ("allOf", "anyOf", "oneOf"):
                branches = node.get(keyword)
                if isinstance(branches, list):
                    pending.extend(branch for branch in branches if isinstance(branch, Mapping))
        next_nodes: list[Mapping[str, Any]] = []
        for node in expanded:
            if isinstance(part, str):
                properties = node.get("properties")
                if isinstance(properties, Mapping) and isinstance(properties.get(part), Mapping):
                    next_nodes.append(properties[part])
            else:
                items = node.get("items")
                if isinstance(items, Mapping):
                    next_nodes.append(items)
                elif isinstance(items, list) and part < len(items) and isinstance(items[part], Mapping):
                    next_nodes.append(items[part])
        nodes = next_nodes
    return nodes


def _numeric_landmarks(schemas: Sequence[Mapping[str, Any]]) -> list[int | float]:
    values: list[int | float] = []
    pending = list(schemas)
    while pending:
        schema = pending.pop()
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "const"):
            value = schema.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
                values.append(value)
        enum = schema.get("enum")
        if isinstance(enum, list):
            values.extend(
                value
                for value in enum
                if isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        for keyword in ("allOf", "anyOf", "oneOf"):
            branches = schema.get(keyword)
            if isinstance(branches, list):
                pending.extend(branch for branch in branches if isinstance(branch, Mapping))
    return values


def _numeric_multiples(schemas: Sequence[Mapping[str, Any]]) -> list[int | float]:
    values: list[int | float] = []
    pending = list(schemas)
    while pending:
        schema = pending.pop()
        value = schema.get("multipleOf")
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            values.append(value)
        for keyword in ("allOf", "anyOf", "oneOf"):
            branches = schema.get(keyword)
            if isinstance(branches, list):
                pending.extend(branch for branch in branches if isinstance(branch, Mapping))
    return values


def _range_candidates(
    spec: Mapping[str, Any], schemas: Sequence[Mapping[str, Any]]
) -> list[int | float]:
    landmarks = [
        value
        for key in ("min", "max", "gte", "gt", "lte", "lt")
        if isinstance((value := spec.get(key)), int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    ]
    landmarks.extend(_numeric_landmarks(schemas))
    landmarks.extend((-1, 0, 1))
    probes: list[int | float] = list(landmarks)
    for value in landmarks:
        probes.extend(
            (
                value - 1,
                value + 1,
                value - 0.5,
                value + 0.5,
                math.nextafter(float(value), -math.inf),
                math.nextafter(float(value), math.inf),
            )
        )
    ordered = sorted(set(landmarks))
    midpoints = [(left + right) / 2 for left, right in pairwise(ordered)]
    probes.extend(midpoints)
    for multiple in _numeric_multiples(schemas):
        for anchor in [*landmarks, *midpoints, 0]:
            quotient = anchor / multiple
            for factor in range(math.floor(quotient) - 1, math.ceil(quotient) + 2):
                probes.append(factor * multiple)

    candidates: dict[str, int | float] = {}
    for raw in probes:
        if not math.isfinite(raw):
            continue
        value: int | float = int(raw) if float(raw).is_integer() else raw
        if value_in_range(dict(spec), value):
            candidates.setdefault(canonical_json(value), value)
    return [candidates[key] for key in sorted(candidates)]


def _replace_ranges(
    value: object,
    replacements: Mapping[tuple[str | int, ...], int | float],
    path: tuple[str | int, ...] = (),
) -> object:
    if path in replacements:
        return replacements[path]
    if isinstance(value, Mapping):
        return {
            str(key): _replace_ranges(item, replacements, (*path, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_ranges(item, replacements, (*path, index))
            for index, item in enumerate(value)
        ]
    return value


def _instantiation_preference(
    values: Sequence[int | float],
) -> tuple[tuple[int, int, float], ...]:
    """Rank one range instantiation by how sane its numbers look as a label.

    Range specs like ``{"lt": 0}`` have infinitely many valid instantiations;
    picking purely by hash (the original rule) is deterministic but happily
    lands on ``-5e-324`` or ``-0.9999999999999999``. Those are *correct* — and
    useless as supervision, since no planner should ever be taught to emit
    them. Ordering by (integer first, sane magnitude, smallest absolute value)
    puts ``-1`` ahead of the degenerate neighbours; the hash still breaks ties,
    so instantiation stays deterministic per atom.
    """
    ranked: list[tuple[int, int, float]] = []
    for value in values:
        number = float(value)
        degenerate = number != 0 and not (1e-3 <= abs(number) <= 1e6)
        ranked.append((0 if number.is_integer() else 1, 1 if degenerate else 0, abs(number)))
    return tuple(ranked)


def _instantiate_range_arguments(
    arguments: Mapping[str, Any],
    *,
    atom_id: str,
    contract: SubagentToolContract,
) -> dict[str, Any]:
    paths = _range_paths(arguments)
    if not paths:
        return dict(arguments)

    candidate_sets: list[list[int | float]] = []
    for path in paths:
        spec = _value_at_path(arguments, path)
        assert isinstance(spec, Mapping)
        candidates = _range_candidates(spec, _schemas_at_path(contract.arguments_schema, path))
        if not candidates:
            raise PlanningCompileError(
                "non_concrete_arguments",
                f"range at {list(path)} has no finite candidate",
            )
        candidate_sets.append(candidates)

    combination_count = math.prod(len(candidates) for candidates in candidate_sets)
    if combination_count > _MAX_RANGE_COMBINATIONS:
        raise PlanningCompileError(
            "non_concrete_arguments",
            f"range candidate space exceeds {_MAX_RANGE_COMBINATIONS}",
        )

    validator = Draft202012Validator(contract.arguments_schema)
    selected: tuple[tuple[tuple[int, int, float], ...], str, dict[str, Any]] | None = None
    for values in product(*candidate_sets):
        replacements = dict(zip(paths, values, strict=True))
        candidate = _replace_ranges(arguments, replacements)
        assert isinstance(candidate, dict)
        if not validator.is_valid(candidate):
            continue
        identity = {
            "schema": _RANGE_INSTANTIATION_VERSION,
            "atom_id": atom_id,
            "arguments": candidate,
        }
        ranked = (_instantiation_preference(values), sha256_identity(identity), candidate)
        if selected is None or ranked[:2] < selected[:2]:
            selected = ranked
    if selected is None:
        raise PlanningCompileError(
            "non_concrete_arguments",
            f"range arguments for atom {atom_id} do not satisfy tool contract",
        )
    return selected[2]


def _clean_instruction(instructions: Mapping[str, str], language: str) -> str:
    instruction = instructions.get(language)
    if instruction is None or not str(instruction).strip():
        raise PlanningCompileError(
            "missing_instruction", f"missing instruction for language: {language}"
        )
    return str(instruction)


def _task_slot(task: ReplyTask | CallTask) -> dict[str, Any]:
    if isinstance(task, ReplyTask):
        return {"kind": task.kind}
    return {"kind": task.kind, "name": task.name, "arguments": task.arguments}


def compile_plan(
    intents: Sequence[Mapping[str, Any]],
    *,
    language: str,
    tools: Sequence[SubagentToolContract],
    contract_ref: Mapping[str, str],
    provenance: Literal["generated", "legacy-reviewed"],
) -> PlanGroundTruth:
    """Compile canonical L0 atoms into deterministic planning ground truth."""
    if not intents:
        raise PlanningCompileError("empty_plan", "empty plan")

    atoms: list[_AtomDraft] = []
    for intent in intents:
        try:
            atom = _AtomDraft.model_validate(intent)
        except ValidationError as exc:
            raise PlanningCompileError("atom_validation_failed", str(exc)) from exc
        if atom.planner_target is None:
            raise PlanningCompileError("missing_planner_target", "missing planner_target")
        atoms.append(atom)

    tasks: list[ReplyTask | CallTask] = []
    lineage: list[dict[str, Any]] = []
    atom_ids: list[str] = []

    for position, atom in enumerate(atoms):
        target = atom.planner_target
        assert target is not None
        instruction = _clean_instruction(target.instructions, language)
        steps = atom.expected_steps
        if target.kind == "reply":
            if steps:
                raise PlanningCompileError(
                    "reply_has_executable_step", f"reply atom {atom.id} has executable step"
                )
            tasks.append(ReplyTask(kind="reply", instruction=instruction))
            lineage.append({"source_atom_id": atom.id, "task_index": position})
            atom_ids.append(atom.id)
            continue
        if not steps:
            raise PlanningCompileError(
                "call_missing_executable_step", f"call atom {atom.id} has no executable step"
            )
        if len(steps) > 1:
            raise PlanningCompileError(
                "call_has_multiple_executable_steps",
                f"call atom {atom.id} has multiple executable steps",
            )
        step = steps[0]
        if not step.expected_function:
            raise PlanningCompileError(
                "call_missing_executable_step", f"call atom {atom.id} has no executable step"
            )
        task_contract = _find_contract(step.expected_function, tools)
        if task_contract is None:
            raise PlanningCompileError(
                "unknown_tool", f"unknown tool: {step.expected_function}"
            )
        if target.kind != task_contract.kind:
            raise PlanningCompileError(
                "planner_target_kind_mismatch",
                f"planner target kind mismatch for {atom.id}: expected {task_contract.kind}, got {target.kind}",
            )
        task = CallTask(
            kind=task_contract.kind,
            instruction=instruction,
            name=step.expected_function,
            arguments=_instantiate_range_arguments(
                step.expected_params or {}, atom_id=atom.id, contract=task_contract
            ),
        )
        validate_task_against_contract(task, tools)
        tasks.append(task)
        lineage.append({"source_atom_id": atom.id, "task_index": position})
        atom_ids.append(atom.id)

    semantic_id = semantic_program_id(atom_ids, [_task_slot(task) for task in tasks])
    return PlanGroundTruth(
        schema=_PLAN_SCHEMA_VERSION,
        semantic_id=semantic_id,
        language=language,
        contract_ref=dict(contract_ref),
        provenance=provenance,
        tasks=tasks,
        lineage=lineage,
    )
