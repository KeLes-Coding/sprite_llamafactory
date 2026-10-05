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

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from jsonschema import Draft202012Validator

from planner_val.models import BenchmarkScoreResult, ContractSnapshot, EvalCase, ParsedTask, ParseResult, ScoreResult


PARAMETER_SKIPPED_TOOLS = frozenset({"get_weather", "web_search", "rag_query"})
DIRECTION_PATHS = frozenset({"parameters.x", "parameters.y", "parameters.yaw"})
_RANGE_OPERATORS = frozenset({"min", "max", "gt", "gte", "lt", "lte"})
_ACTION_TOOLS = frozenset({"HumanAction", "RobotGesture", "RobotDance"})
_BENCHMARK_EXEMPT_FIELDS = {
    "HumanAction": frozenset({"parameters.step", "parameters.distance", "parameters.degree"}),
    "RobotDance": frozenset({"repeat"}),
    "RobotGesture": frozenset({"repeat"}),
}


def _extract_array_text(raw: str) -> str | None:
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end == -1 and raw.strip() == "[":
        return "["
    if start == -1 or end == -1 or end < start:
        return None
    return raw[start : end + 1]


def _unexpected_keys(mapping: Mapping[str, Any], expected_keys: set[str]) -> tuple[str, ...]:
    return tuple(sorted(key for key in mapping if key not in expected_keys))


def _missing_keys(mapping: Mapping[str, Any], expected_keys: set[str]) -> tuple[str, ...]:
    return tuple(sorted(key for key in expected_keys if key not in mapping))


def _require_non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _task_schema_error(index: int, detail: str) -> ParseResult:
    return ParseResult(valid_json=True, contract_valid=False, error=f"task[{index}]: {detail}")


def _contract_error(index: int, detail: str, tasks: list[ParsedTask]) -> ParseResult:
    return ParseResult(tasks=tuple(tasks), valid_json=True, contract_valid=False, error=f"task[{index}]: {detail}")


def parse_plan(raw: str, contract: ContractSnapshot) -> ParseResult:
    array_text = _extract_array_text(raw)
    if array_text is None:
        return ParseResult(error="missing JSON array")

    try:
        payload = json.loads(array_text)
    except json.JSONDecodeError as exc:
        return ParseResult(error=f"JSON decode error: {exc.msg}")

    if not isinstance(payload, list):
        return ParseResult(valid_json=True, contract_valid=False, error="decoded plan must be a list")

    parsed_tasks: list[ParsedTask] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            return _task_schema_error(index, "must be an object")

        kind = item.get("kind")
        if kind == "reply":
            expected_keys = {"kind", "instruction"}
            unexpected = _unexpected_keys(item, expected_keys)
            missing = _missing_keys(item, expected_keys)
            if missing:
                return _task_schema_error(index, f"missing keys {list(missing)}")
            if unexpected:
                return _task_schema_error(index, f"unexpected keys {list(unexpected)}")
            try:
                instruction = _require_non_empty_string(item.get("instruction"), label="instruction")
            except ValueError as exc:
                return _task_schema_error(index, str(exc))
            parsed_tasks.append(ParsedTask(kind="reply", instruction=instruction, name=None, arguments={}))
            continue

        if kind not in {"query", "action"}:
            return _task_schema_error(index, f"invalid kind {kind!r}")

        expected_keys = {"kind", "instruction", "name", "arguments"}
        unexpected = _unexpected_keys(item, expected_keys)
        missing = _missing_keys(item, expected_keys)
        if missing:
            return _task_schema_error(index, f"missing keys {list(missing)}")
        if unexpected:
            return _task_schema_error(index, f"unexpected keys {list(unexpected)}")

        try:
            instruction = _require_non_empty_string(item.get("instruction"), label="instruction")
            name = _require_non_empty_string(item.get("name"), label="name")
        except ValueError as exc:
            return _task_schema_error(index, str(exc))

        arguments = item.get("arguments")
        if not isinstance(arguments, Mapping):
            return _task_schema_error(index, "arguments must be an object")

        contract_kind = contract.tool_kinds.get(name)
        if contract_kind is None:
            return _contract_error(index, f"unknown tool {name}", parsed_tasks)
        if kind != contract_kind:
            return _contract_error(
                index, f"kind {kind} does not match contract {contract_kind} for {name}", parsed_tasks
            )

        validator = Draft202012Validator(contract.tool_schemas[name])
        validation_errors = sorted(
            validator.iter_errors(_normalize_action_selector(name, arguments)),
            key=lambda err: list(err.absolute_path),
        )
        if validation_errors:
            return _contract_error(index, f"{name} arguments: {validation_errors[0].message}", parsed_tasks)

        parsed_tasks.append(ParsedTask(kind=kind, instruction=instruction, name=name, arguments=dict(arguments)))

    return ParseResult(tasks=tuple(parsed_tasks), valid_json=True, contract_valid=True, error=None)


def _is_contract_violation(error: str) -> bool:
    return "unknown tool" in error or "does not match contract" in error or " arguments:" in error


def _normalize_action_selector(name: str | None, arguments: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if name == "RobotGesture":
        if "gesture" not in normalized and "action" in normalized:
            normalized["gesture"] = normalized.pop("action")
        else:
            normalized.pop("action", None)
    if name == "RobotDance":
        if "dance" not in normalized and "action" in normalized:
            normalized["dance"] = normalized.pop("action")
        else:
            normalized.pop("action", None)
    return normalized


def _strip_task_prefix(path: str) -> str:
    _, _, remainder = path.partition(".")
    return remainder


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _is_range_expectation(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and set(value).issubset(_RANGE_OPERATORS)


def _range_spec_error(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None

    if not value:
        return None

    keys = set(value)
    if not (keys & _RANGE_OPERATORS):
        return None

    if not keys.issubset(_RANGE_OPERATORS):
        unsupported_keys = sorted(key for key in value if key not in _RANGE_OPERATORS)
        return f"invalid range spec: unsupported keys {unsupported_keys}"

    invalid_bounds = sorted(key for key, bound in value.items() if not _is_number(bound))
    if invalid_bounds:
        return f"invalid range spec: non-finite numeric bounds required for {invalid_bounds}"

    return None


def _compare_range(path: str, expected: Mapping[str, Any], actual: Any) -> list[str]:
    if not _is_number(actual):
        return [f"{path}: expected numeric value"]

    errors: list[str] = []
    operator_map = {
        "min": actual >= expected.get("min", actual),
        "max": actual <= expected.get("max", actual),
        "gt": actual > expected.get("gt", actual),
        "gte": actual >= expected.get("gte", actual),
        "lt": actual < expected.get("lt", actual),
        "lte": actual <= expected.get("lte", actual),
    }
    for operator in ("min", "max", "gt", "gte", "lt", "lte"):
        if operator in expected and not operator_map[operator]:
            errors.append(f"{path}: expected {operator} {expected[operator]!r}, got {actual!r}")
    return errors


def _compare_value(path: str, expected: Any, actual: Any) -> list[str]:
    range_spec_error = _range_spec_error(expected)
    if range_spec_error is not None:
        return [f"{path}: {range_spec_error}"]

    if _is_range_expectation(expected):
        return _compare_range(path, expected, actual)

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected object, got {actual!r}"]
        errors: list[str] = []
        for key, inner_expected in expected.items():
            inner_actual = actual.get(key) if key in actual else None
            inner_path = f"{path}.{key}" if path else str(key)
            if key == "repeat" and inner_expected == 1 and key not in actual:
                inner_actual = 1
            if key not in actual and not (key == "repeat" and inner_expected == 1):
                errors.append(f"{inner_path}: missing value")
                continue
            errors.extend(_compare_value(inner_path, inner_expected, inner_actual))
        return errors

    if _strip_task_prefix(path) in DIRECTION_PATHS and _is_number(expected):
        if not _is_number(actual):
            return [f"{path}: expected numeric value"]
        if _sign(float(expected)) != _sign(float(actual)):
            return [f"{path}: expected direction {_sign(float(expected))}, got {_sign(float(actual))}"]
        return []

    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def _compare_task_parameters(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, name: str, index: int
) -> tuple[int, tuple[str, ...]]:
    if name in PARAMETER_SKIPPED_TOOLS:
        return 0, ()

    normalized_expected = _normalize_action_selector(name, expected)
    normalized_actual = _normalize_action_selector(name, actual)
    if "repeat" in normalized_expected and normalized_expected["repeat"] == 1 and "repeat" not in normalized_actual:
        normalized_actual["repeat"] = 1

    checked_count = len(normalized_expected)
    errors = _compare_value(f"task[{index}]", normalized_expected, normalized_actual)
    return checked_count, tuple(errors)


def _normalize_tool_name(value: Any) -> str:
    return "_".join(str(value or "").lower().split())


def _benchmark_forbidden_names(forbidden_functions: frozenset[str]) -> tuple[set[str], bool]:
    action_names = {_normalize_tool_name(name) for name in _ACTION_TOOLS}
    forbidden_names = {_normalize_tool_name(name) for name in forbidden_functions}
    action_scope = bool(forbidden_names & action_names)
    if action_scope:
        forbidden_names |= action_names
    return forbidden_names, action_scope


def _remove_path(value: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    node: Any = value
    parents: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        parents.append((node, part))
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return
    if _is_range_expectation(node[parts[-1]]):
        return
    node.pop(parts[-1])
    for parent, key in reversed(parents):
        if isinstance(parent.get(key), dict) and not parent[key]:
            parent.pop(key)


def _mutable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_value(inner_value) for inner_value in value]
    return value


def _benchmark_expected_parameters(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    expected = _normalize_action_selector(name, _mutable_value(arguments))
    for path in _BENCHMARK_EXEMPT_FIELDS.get(name, frozenset()):
        _remove_path(expected, path)
    return expected


def _compare_benchmark_value(path: str, expected: Any, actual: Any, *, directional: bool) -> list[str]:
    range_spec_error = _range_spec_error(expected)
    if range_spec_error is not None:
        return [f"{path}: {range_spec_error}"]
    if _is_range_expectation(expected):
        return _compare_range(path, expected, actual)
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected object, got {actual!r}"]
        errors: list[str] = []
        for key, inner_expected in expected.items():
            inner_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                errors.append(f"{inner_path}: missing value")
                continue
            errors.extend(_compare_benchmark_value(inner_path, inner_expected, actual[key], directional=directional))
        return errors
    if directional and _strip_task_prefix(path) in DIRECTION_PATHS and _is_number(expected):
        if not _is_number(actual):
            return [f"{path}: expected numeric direction"]
        if not -1 <= float(actual) <= 1:
            return [f"{path}: expected value in [-1, 1], got {actual!r}"]
        if _sign(float(expected)) != _sign(float(actual)):
            return [f"{path}: expected direction {_sign(float(expected))}, got {_sign(float(actual))}"]
        return []
    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def _benchmark_parameter_errors(expected: ParsedTask | Any, actual: ParsedTask) -> tuple[str, ...]:
    name = expected.name
    if name is None or name in PARAMETER_SKIPPED_TOOLS:
        return ()
    expected_arguments = _benchmark_expected_parameters(name, expected.arguments)
    actual_arguments = _normalize_action_selector(name, actual.arguments)
    if not expected_arguments:
        return ()
    return tuple(
        _compare_benchmark_value(
            "parameters",
            expected_arguments,
            actual_arguments,
            directional=name == "HumanAction" and expected_arguments.get("action") == "walk",
        )
    )


def _align_benchmark_slots(
    expected: Sequence[Any], actual_tools: Sequence[ParsedTask], forbidden_functions: frozenset[str]
) -> tuple[tuple[int | None, ...] | None, bool]:
    forbidden_names, action_scope = _benchmark_forbidden_names(forbidden_functions)
    positive_names = {_normalize_tool_name(task.name) for task in expected if task.name is not None}
    effective_forbidden_names = forbidden_names - positive_names if action_scope else forbidden_names
    positive_suffix = [0] * (len(expected) + 1)
    for index in range(len(expected) - 1, -1, -1):
        positive_suffix[index] = positive_suffix[index + 1] + int(expected[index].name is not None)

    @cache
    def align_from(expected_index: int, actual_index: int) -> tuple[int, tuple[int | None, ...]] | None:
        if expected_index >= len(expected):
            return (0, ()) if actual_index >= len(actual_tools) else None
        remaining_slots = len(expected) - expected_index
        remaining_actual = len(actual_tools) - actual_index
        if remaining_actual < positive_suffix[expected_index] or remaining_actual > remaining_slots:
            return None

        expected_task = expected[expected_index]
        if expected_task.name is not None:
            if actual_index >= len(actual_tools):
                return None
            actual_task = actual_tools[actual_index]
            if _normalize_tool_name(expected_task.name) != _normalize_tool_name(actual_task.name):
                return None
            suffix = align_from(expected_index + 1, actual_index + 1)
            if suffix is None:
                return None
            suffix_score, suffix_assignments = suffix
            parameter_score = int(not _benchmark_parameter_errors(expected_task, actual_task))
            return (suffix_score + parameter_score, (actual_index, *suffix_assignments))

        candidates: list[tuple[int, tuple[int | None, ...]]] = []
        skipped = align_from(expected_index + 1, actual_index)
        if skipped is not None:
            candidates.append((skipped[0], (None, *skipped[1])))
        if actual_index < len(actual_tools) and forbidden_names:
            actual_name = _normalize_tool_name(actual_tools[actual_index].name)
            consumed = align_from(expected_index + 1, actual_index + 1)
            if actual_name not in forbidden_names and consumed is not None:
                candidates.append((consumed[0], (actual_index, *consumed[1])))
        if not candidates:
            return None
        return max(
            candidates, key=lambda candidate: (candidate[0], tuple(index is not None for index in candidate[1]))
        )

    alignment = align_from(0, 0)
    called_forbidden = False
    allowances = defaultdict(int)
    for task in expected:
        if task.name is not None:
            allowances[_normalize_tool_name(task.name)] += 1
    for task in actual_tools:
        name = _normalize_tool_name(task.name)
        if allowances[name] > 0:
            allowances[name] -= 1
        elif not forbidden_functions or name in effective_forbidden_names:
            called_forbidden = True
    return (alignment[1] if alignment is not None else None), called_forbidden


def score_benchmark_compatible(
    case: EvalCase,
    raw: str | None,
    contract: ContractSnapshot,
    backend_error: str | None = None,
) -> BenchmarkScoreResult:
    reply_slots = sum(task.kind == "reply" for task in case.expected)
    expected_tool_count = len(case.expected) - reply_slots
    if backend_error is not None or raw is None:
        return BenchmarkScoreResult(
            reason="backend_error" if backend_error is not None else "parser_rejected",
            passed=False,
            parser_valid=False,
            tool_sequence_match=False,
            parameter_match=None,
            forbidden_tool_violation=False,
            reply_slots=reply_slots,
            expected_tool_count=expected_tool_count,
            predicted_tool_count=0,
            errors=((backend_error or "missing output"),),
        )

    parsed = parse_plan(raw, contract)
    actual_tools = tuple(task for task in parsed.tasks if task.kind != "reply" and task.name is not None)
    if not parsed.contract_valid:
        return BenchmarkScoreResult(
            reason="parser_rejected",
            passed=False,
            parser_valid=False,
            tool_sequence_match=False,
            parameter_match=None,
            forbidden_tool_violation=False,
            reply_slots=reply_slots,
            expected_tool_count=expected_tool_count,
            predicted_tool_count=len(actual_tools),
            errors=((parsed.error or "parser rejected output"),),
        )

    assignments, called_forbidden = _align_benchmark_slots(case.expected, actual_tools, case.forbidden_functions)
    tool_sequence_match = assignments is not None and not called_forbidden
    parameter_errors: list[str] = []
    parameter_checked = False
    if assignments is not None:
        for expected_task, actual_index in zip(case.expected, assignments, strict=True):
            if expected_task.name is None or actual_index is None:
                continue
            if expected_task.name not in PARAMETER_SKIPPED_TOOLS:
                parameter_checked = True
            parameter_errors.extend(_benchmark_parameter_errors(expected_task, actual_tools[actual_index]))
    parameter_match = None if not parameter_checked else not parameter_errors
    passed = tool_sequence_match and not parameter_errors
    if called_forbidden:
        reason = "forbidden_tool"
    elif assignments is None:
        reason = "tool_sequence_mismatch"
    elif parameter_errors:
        reason = "parameter_mismatch"
    else:
        reason = "passed"
    return BenchmarkScoreResult(
        reason=reason,
        passed=passed,
        parser_valid=True,
        tool_sequence_match=tool_sequence_match,
        parameter_match=parameter_match,
        forbidden_tool_violation=called_forbidden,
        reply_slots=reply_slots,
        expected_tool_count=expected_tool_count,
        predicted_tool_count=len(actual_tools),
        errors=tuple(parameter_errors),
    )


def _has_checked_parameters(case: EvalCase) -> bool:
    return any(
        task.name not in PARAMETER_SKIPPED_TOOLS and task.kind != "reply" and bool(task.arguments)
        for task in case.expected
    )


def score_output(
    case: EvalCase,
    raw: str | None,
    contract: ContractSnapshot,
    backend_error: str | None = None,
) -> ScoreResult:
    if backend_error is not None:
        return ScoreResult(
            reason="backend_error",
            strict_pass=False,
            valid_json=False,
            contract_valid=False,
            task_count_match=False,
            ordered_kind_name_match=False,
            parameter_match=None,
            forbidden_tool_violation=False,
            parsed_tasks=(),
            parameter_errors=(backend_error,),
            forbidden_calls=(),
            raw_output=raw,
        )

    if raw is None:
        return ScoreResult(
            reason="invalid_json",
            strict_pass=False,
            valid_json=False,
            contract_valid=False,
            task_count_match=False,
            ordered_kind_name_match=False,
            parameter_match=None,
            forbidden_tool_violation=False,
            parsed_tasks=(),
            parameter_errors=("missing output",),
            forbidden_calls=(),
            raw_output=None,
        )

    parsed = parse_plan(raw, contract)
    if not parsed.valid_json:
        return ScoreResult(
            reason="invalid_json",
            strict_pass=False,
            valid_json=False,
            contract_valid=False,
            task_count_match=False,
            ordered_kind_name_match=False,
            parameter_match=None,
            forbidden_tool_violation=False,
            parsed_tasks=parsed.tasks,
            parameter_errors=((parsed.error,) if parsed.error is not None else ()),
            forbidden_calls=(),
            raw_output=raw,
        )

    if not parsed.contract_valid:
        reason = (
            "contract_violation" if parsed.error and _is_contract_violation(parsed.error) else "invalid_task_schema"
        )
        return ScoreResult(
            reason=reason,
            strict_pass=False,
            valid_json=True,
            contract_valid=False,
            task_count_match=False,
            ordered_kind_name_match=False,
            parameter_match=False if _has_checked_parameters(case) else None,
            forbidden_tool_violation=False,
            parsed_tasks=parsed.tasks,
            parameter_errors=((parsed.error,) if parsed.error is not None else ()),
            forbidden_calls=(),
            raw_output=raw,
        )

    expected_tasks = case.expected
    actual_tasks = parsed.tasks
    expected_names = {task.name for task in expected_tasks if task.name is not None}
    forbidden_calls = tuple(
        task.name for task in actual_tasks if task.name in case.forbidden_functions and task.name not in expected_names
    )
    task_count_match = len(expected_tasks) == len(actual_tasks)
    kind_match = len(expected_tasks) == len(actual_tasks) and all(
        expected.kind == actual.kind for expected, actual in zip(expected_tasks, actual_tasks)
    )
    name_match = len(expected_tasks) == len(actual_tasks) and all(
        expected.name == actual.name for expected, actual in zip(expected_tasks, actual_tasks)
    )

    parameter_errors: list[str] = []
    checked_parameter_count = 0
    if task_count_match and kind_match and name_match:
        for index, (expected, actual) in enumerate(zip(expected_tasks, actual_tasks)):
            if expected.kind == "reply" or expected.name is None:
                continue
            checked_count, errors = _compare_task_parameters(
                expected.arguments,
                actual.arguments,
                name=expected.name,
                index=index,
            )
            checked_parameter_count += checked_count
            parameter_errors.extend(errors)
    else:
        checked_parameter_count = 1 if _has_checked_parameters(case) else 0

    parameter_match = (
        None
        if checked_parameter_count == 0
        else not parameter_errors and task_count_match and kind_match and name_match
    )
    forbidden_tool_violation = bool(forbidden_calls)

    if forbidden_tool_violation:
        reason = "forbidden_tool"
    elif not task_count_match:
        reason = "task_count_mismatch"
    elif not kind_match:
        reason = "task_kind_mismatch"
    elif not name_match:
        reason = "task_name_mismatch"
    elif parameter_match is False:
        reason = "task_parameter_mismatch"
    else:
        reason = "passed"

    return ScoreResult(
        reason=reason,
        strict_pass=reason == "passed",
        valid_json=True,
        contract_valid=True,
        task_count_match=task_count_match,
        ordered_kind_name_match=kind_match and name_match,
        parameter_match=parameter_match,
        forbidden_tool_violation=forbidden_tool_violation,
        parsed_tasks=actual_tasks,
        parameter_errors=tuple(parameter_errors),
        forbidden_calls=forbidden_calls,
        raw_output=raw,
    )
