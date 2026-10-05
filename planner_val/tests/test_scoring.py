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

import pytest

from planner_val.models import ContractSnapshot, EvalCase, ExpectedTask
from planner_val.scoring import parse_plan, score_benchmark_compatible, score_output


def _make_contract() -> ContractSnapshot:
    tool_schemas = {
        "HumanAction": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "minLength": 1},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {},
                        "y": {},
                        "yaw": {},
                        "step": {"type": "integer"},
                        "degree": {"type": "number"},
                        "repeat": {"type": "integer", "minimum": 1},
                        "pose": {
                            "type": "object",
                            "properties": {
                                "arm": {"type": "string"},
                                "speed": {"type": "number"},
                            },
                            "additionalProperties": True,
                        },
                    },
                    "additionalProperties": True,
                },
            },
            "required": ["action", "parameters"],
            "additionalProperties": False,
        },
        "RobotGesture": {
            "type": "object",
            "properties": {
                "gesture": {"type": "string", "minLength": 1},
                "speed": {"type": "number"},
                "repeat": {"type": "integer", "minimum": 1},
            },
            "required": ["gesture"],
            "additionalProperties": False,
        },
        "RobotDance": {
            "type": "object",
            "properties": {
                "dance": {"type": "string", "minLength": 1},
                "beats": {"type": "integer", "minimum": 1},
                "repeat": {"type": "integer", "minimum": 1},
            },
            "required": ["dance"],
            "additionalProperties": False,
        },
        "get_weather": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "web_search": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "rag_query": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "namespace": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
    tool_kinds = {
        "HumanAction": "action",
        "RobotGesture": "action",
        "RobotDance": "action",
        "get_weather": "query",
        "web_search": "query",
        "rag_query": "query",
    }
    return ContractSnapshot(
        name="production",
        system_prompt="system",
        tools_context=(),
        tool_schemas=tool_schemas,
        tool_kinds=tool_kinds,
        source={"source": "test"},
        fingerprint="sha256:test",
        message_config={"reply_style": "strict"},
    )


def _replace_tool(contract: ContractSnapshot, name: str, *, kind: str, schema: dict[str, object]) -> ContractSnapshot:
    tool_schemas = dict(contract.tool_schemas)
    tool_kinds = dict(contract.tool_kinds)
    tool_schemas[name] = schema
    tool_kinds[name] = kind  # type: ignore[assignment]
    return ContractSnapshot(
        name=contract.name,
        system_prompt=contract.system_prompt,
        tools_context=contract.tools_context,
        tool_schemas=tool_schemas,
        tool_kinds=tool_kinds,
        source=contract.source,
        fingerprint=contract.fingerprint,
        max_steps=contract.max_steps,
        message_config=contract.message_config,
    )


def _make_case(
    *,
    expected: tuple[ExpectedTask, ...],
    forbidden_functions: frozenset[str] = frozenset(),
) -> EvalCase:
    return EvalCase(
        case_id="case-1",
        query="请规划动作",
        expected=expected,
        forbidden_functions=forbidden_functions,
        level="L1",
        language="zh",
        difficulty="easy",
        polarity="positive",
        source_atom_ids=frozenset({"atom-1"}),
    )


def _task(
    kind: str, instruction: str, *, name: str | None = None, arguments: dict[str, object] | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {"kind": kind, "instruction": instruction}
    if name is not None:
        payload["name"] = name
        payload["arguments"] = arguments or {}
    return payload


def test_score_output_accepts_fenced_valid_array_and_reports_all_success_metrics() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(kind="query", instruction="查天气", name="get_weather", arguments={"city": "北京"}),
            ExpectedTask(kind="reply", instruction="告诉用户我先查天气"),
            ExpectedTask(
                kind="action",
                instruction="挥手致意",
                name="RobotGesture",
                arguments={"action": "wave", "repeat": 1},
            ),
        )
    )
    raw = (
        "```json\n"
        + json.dumps(
            [
                _task("query", "先查询天气", name="get_weather", arguments={"city": "上海", "unit": "c"}),
                _task("reply", "我先帮你查一下天气"),
                _task("action", "执行挥手", name="RobotGesture", arguments={"gesture": "wave"}),
            ],
            ensure_ascii=False,
        )
        + "\n```"
    )

    result = score_output(case, raw, contract)

    assert result.reason == "passed"
    assert result.strict_pass is True
    assert result.valid_json is True
    assert result.contract_valid is True
    assert result.task_count_match is True
    assert result.ordered_kind_name_match is True
    assert result.parameter_match is True
    assert result.forbidden_tool_violation is False
    assert result.parameter_errors == ()
    assert result.forbidden_calls == ()
    assert tuple(task.kind for task in result.parsed_tasks) == ("query", "reply", "action")
    assert result.raw_output == raw


@pytest.mark.parametrize(
    ("raw", "backend_error", "reason", "valid_json", "contract_valid", "error_fragment"),
    [
        pytest.param("[]", "upstream timeout", "backend_error", False, False, "upstream timeout", id="backend-error"),
        pytest.param(None, None, "invalid_json", False, False, "missing output", id="missing-output"),
        pytest.param("prefix [", None, "invalid_json", False, False, "missing JSON array", id="missing-array"),
        pytest.param("[", None, "invalid_json", False, False, "JSON decode error", id="malformed-json"),
        pytest.param(
            json.dumps([{"kind": "reply", "instruction": "ok", "extra": True}], ensure_ascii=False),
            None,
            "invalid_task_schema",
            True,
            False,
            "unexpected keys",
            id="extra-key",
        ),
        pytest.param(
            json.dumps([{"kind": "action", "instruction": "go", "name": "HumanAction"}], ensure_ascii=False),
            None,
            "invalid_task_schema",
            True,
            False,
            "missing keys",
            id="missing-key",
        ),
        pytest.param(
            json.dumps([{"kind": "reply", "instruction": ""}], ensure_ascii=False),
            None,
            "invalid_task_schema",
            True,
            False,
            "instruction must be a non-empty string",
            id="empty-field",
        ),
        pytest.param(
            json.dumps([1], ensure_ascii=False),
            None,
            "invalid_task_schema",
            True,
            False,
            "must be an object",
            id="non-mapping-item",
        ),
    ],
)
def test_score_output_reports_backend_json_and_task_schema_failures(
    raw: str | None,
    backend_error: str | None,
    reason: str,
    valid_json: bool,
    contract_valid: bool,
    error_fragment: str,
) -> None:
    contract = _make_contract()
    case = _make_case(expected=(ExpectedTask(kind="reply", instruction="hi"),))

    result = score_output(case, raw, contract, backend_error=backend_error)

    assert result.reason == reason
    assert result.valid_json is valid_json
    assert result.contract_valid is contract_valid
    assert result.strict_pass is False
    assert result.parsed_tasks == ()
    assert (
        error_fragment in result.reason
        or error_fragment in (result.parameter_errors[0] if result.parameter_errors else "")
        or error_fragment in str(result)
    )


def test_parse_plan_distinguishes_parser_schema_from_contract_violations() -> None:
    contract = _make_contract()

    unknown_tool = parse_plan(
        json.dumps([_task("query", "查一下", name="not_exists", arguments={})], ensure_ascii=False),
        contract,
    )
    wrong_kind = parse_plan(
        json.dumps([_task("query", "跳舞", name="RobotDance", arguments={"dance": "pop"})], ensure_ascii=False),
        contract,
    )
    schema_invalid = parse_plan(
        json.dumps([_task("query", "查天气", name="get_weather", arguments={"unit": "c"})], ensure_ascii=False),
        contract,
    )

    assert unknown_tool.valid_json is True
    assert unknown_tool.contract_valid is False
    assert unknown_tool.error == "task[0]: unknown tool not_exists"
    assert wrong_kind.error == "task[0]: kind query does not match contract action for RobotDance"
    assert schema_invalid.error == "task[0]: get_weather arguments: 'city' is a required property"


def test_parse_plan_uses_real_jsonschema_contract_keywords() -> None:
    contract = _replace_tool(
        _make_contract(),
        "StrictMath",
        kind="action",
        schema={
            "type": "object",
            "properties": {
                "mode": {"const": "strict"},
                "level": {"enum": [1, 2]},
                "value": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
                "payload": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"tag": {"const": "a"}},
                            "required": ["tag"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {"tag": {"const": "b"}},
                            "required": ["tag"],
                            "additionalProperties": False,
                        },
                    ]
                },
            },
            "required": ["mode", "level", "value", "payload"],
            "additionalProperties": False,
        },
    )

    invalid_cases = [
        ({"mode": "loose", "level": 1, "value": 1, "payload": {"tag": "a"}}, "'strict' was expected"),
        ({"mode": "strict", "level": 3, "value": 1, "payload": {"tag": "a"}}, "is not one of"),
        ({"mode": "strict", "level": 1, "value": 11, "payload": {"tag": "a"}}, "greater than the maximum of 10"),
        (
            {"mode": "strict", "level": 1, "value": 0, "payload": {"tag": "a"}},
            "less than or equal to the minimum of 0",
        ),
        (
            {"mode": "strict", "level": 1, "value": 1, "payload": {"tag": "c"}},
            "is not valid under any of the given schemas",
        ),
    ]

    for arguments, fragment in invalid_cases:
        parsed = parse_plan(
            json.dumps(
                [_task("action", "strict", name="StrictMath", arguments=arguments)],
                ensure_ascii=False,
            ),
            contract,
        )

        assert parsed.valid_json is True
        assert parsed.contract_valid is False
        assert parsed.error is not None
        assert parsed.error.startswith("task[0]: StrictMath arguments:")
        assert fragment in parsed.error


def test_forbidden_tool_failure_precedes_count_and_order_checks() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(ExpectedTask(kind="reply", instruction="ok"),),
        forbidden_functions=frozenset({"web_search"}),
    )
    raw = json.dumps(
        [
            _task("query", "搜一下", name="web_search", arguments={"query": "news"}),
            _task("reply", "done"),
        ],
        ensure_ascii=False,
    )

    result = score_output(case, raw, contract)

    assert result.reason == "forbidden_tool"
    assert result.forbidden_tool_violation is True
    assert result.forbidden_calls == ("web_search",)
    assert result.task_count_match is False
    assert result.ordered_kind_name_match is False


def test_benchmark_compatible_ignores_reply_text_and_query_parameters() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(kind="action", instruction="挥手", name="RobotGesture", arguments={"gesture": "wave"}),
            ExpectedTask(kind="reply", instruction="介绍最近流行的舞蹈"),
            ExpectedTask(kind="query", instruction="搜索新闻", name="web_search", arguments={"query": "旧查询"}),
        ),
        forbidden_functions=frozenset({"RobotDance"}),
    )
    raw = json.dumps(
        [
            _task("action", "先打招呼", name="RobotGesture", arguments={"gesture": "wave", "repeat": 3}),
            _task("query", "查一下", name="web_search", arguments={"query": "完全不同的查询"}),
        ],
        ensure_ascii=False,
    )

    result = score_benchmark_compatible(case, raw, contract)

    assert result.passed is True
    assert result.reason == "passed"
    assert result.tool_sequence_match is True
    assert result.parameter_match is True
    assert result.reply_slots == 1


def test_benchmark_compatible_rejects_tool_for_negative_reply_slot() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(ExpectedTask(kind="reply", instruction="介绍舞蹈"),),
        forbidden_functions=frozenset({"RobotDance"}),
    )
    raw = json.dumps(
        [_task("action", "跳一支舞", name="RobotDance", arguments={"dance": "pop"})],
        ensure_ascii=False,
    )

    result = score_benchmark_compatible(case, raw, contract)

    assert result.passed is False
    assert result.reason == "forbidden_tool"
    assert result.forbidden_tool_violation is True


def test_benchmark_compatible_uses_historical_action_parameter_tolerance() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(
                kind="action",
                instruction="往前走",
                name="HumanAction",
                arguments={
                    "action": "walk",
                    "parameters": {"x": 0.6, "y": 0, "yaw": 0, "step": 1, "degree": 45},
                },
            ),
        )
    )
    tolerated = json.dumps(
        [
            _task(
                "action",
                "向前移动",
                name="HumanAction",
                arguments={
                    "action": "walk",
                    "parameters": {"x": 1, "y": 0, "yaw": 0, "step": 1, "degree": 360},
                },
            )
        ],
        ensure_ascii=False,
    )
    wrong_direction = json.dumps(
        [
            _task(
                "action",
                "向后移动",
                name="HumanAction",
                arguments={
                    "action": "walk",
                    "parameters": {"x": -1, "y": 0, "yaw": 0, "step": 1, "degree": 360},
                },
            )
        ],
        ensure_ascii=False,
    )

    assert score_benchmark_compatible(case, tolerated, contract).passed is True
    assert score_benchmark_compatible(case, wrong_direction, contract).reason == "parameter_mismatch"


def test_benchmark_compatible_keeps_production_parser_as_gate() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(ExpectedTask(kind="action", instruction="跳舞", name="RobotDance", arguments={"dance": "pop"}),)
    )
    wrong_kind = json.dumps(
        [_task("query", "跳舞", name="RobotDance", arguments={"dance": "pop"})],
        ensure_ascii=False,
    )

    result = score_benchmark_compatible(case, wrong_kind, contract)

    assert result.passed is False
    assert result.reason == "parser_rejected"
    assert result.parser_valid is False


@pytest.mark.parametrize("raw", ["{}", '"hello"', "1", "null"])
def test_non_array_json_protocol_inputs_score_as_invalid_json(raw: str) -> None:
    contract = _make_contract()
    case = _make_case(expected=(ExpectedTask(kind="reply", instruction="hi"),))

    parsed = parse_plan(raw, contract)
    result = score_output(case, raw, contract)

    assert parsed.valid_json is False
    assert parsed.contract_valid is False
    assert parsed.tasks == ()
    assert parsed.error == "missing JSON array"
    assert result.reason == "invalid_json"
    assert result.valid_json is False
    assert result.contract_valid is False
    assert result.parameter_errors == ("missing JSON array",)


@pytest.mark.parametrize(
    ("actual_tasks", "reason"),
    [
        pytest.param(
            [
                _task("action", "first", name="RobotGesture", arguments={"gesture": "wave"}),
            ],
            "task_count_mismatch",
            id="count",
        ),
        pytest.param(
            [
                _task("action", "first", name="RobotGesture", arguments={"gesture": "wave"}),
                _task("query", "second", name="get_weather", arguments={"city": "北京"}),
            ],
            "task_kind_mismatch",
            id="kind",
        ),
        pytest.param(
            [
                _task("action", "first", name="RobotGesture", arguments={"gesture": "wave"}),
                _task("action", "second", name="RobotDance", arguments={"dance": "pop"}),
            ],
            "task_name_mismatch",
            id="name",
        ),
    ],
)
def test_repeated_ordered_tools_report_count_kind_and_name_mismatches(
    actual_tasks: list[dict[str, object]],
    reason: str,
) -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(kind="action", instruction="first", name="RobotGesture", arguments={"gesture": "wave"}),
            ExpectedTask(kind="action", instruction="second", name="RobotGesture", arguments={"gesture": "bow"}),
        )
    )

    result = score_output(case, json.dumps(actual_tasks, ensure_ascii=False), contract)

    assert result.reason == reason
    assert result.strict_pass is False


def test_selector_normalization_and_repeat_equivalence_allow_strict_pass() -> None:
    contract = _make_contract()
    gesture_case = _make_case(
        expected=(
            ExpectedTask(
                kind="action", instruction="gesture", name="RobotGesture", arguments={"action": "wave", "repeat": 1}
            ),
        )
    )
    dance_case = _make_case(
        expected=(
            ExpectedTask(
                kind="action", instruction="dance", name="RobotDance", arguments={"dance": "hiphop", "repeat": 1}
            ),
        )
    )

    gesture_result = score_output(
        gesture_case,
        json.dumps(
            [_task("action", "gesture", name="RobotGesture", arguments={"gesture": "wave"})], ensure_ascii=False
        ),
        contract,
    )
    dance_result = score_output(
        dance_case,
        json.dumps([_task("action", "dance", name="RobotDance", arguments={"action": "hiphop"})], ensure_ascii=False),
        contract,
    )

    assert gesture_result.strict_pass is True
    assert dance_result.strict_pass is True


def test_human_action_direction_sign_exact_fields_and_nested_subset_are_checked() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(
                kind="action",
                instruction="move",
                name="HumanAction",
                arguments={
                    "action": "move",
                    "parameters": {
                        "x": -1,
                        "y": 0,
                        "yaw": 1,
                        "step": 3,
                        "degree": 45,
                        "pose": {"arm": "left"},
                    },
                },
            ),
        )
    )
    raw = json.dumps(
        [
            _task(
                "action",
                "move",
                name="HumanAction",
                arguments={
                    "action": "move",
                    "parameters": {
                        "x": -5,
                        "y": 0,
                        "yaw": 3.2,
                        "step": 3,
                        "degree": 45,
                        "pose": {"arm": "left", "speed": 0.8},
                        "extra": "ignored by subset",
                    },
                },
            )
        ],
        ensure_ascii=False,
    )

    result = score_output(case, raw, contract)

    assert result.strict_pass is True
    assert result.parameter_match is True


def test_range_operators_and_non_numeric_actual_values() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(
                kind="action",
                instruction="move",
                name="HumanAction",
                arguments={
                    "action": "move",
                    "parameters": {
                        "x": {"gte": 0, "lt": 10},
                        "y": {"min": -1, "max": 1},
                        "yaw": {"gt": -0.5, "lte": 1.5},
                    },
                },
            ),
        )
    )

    passing = score_output(
        case,
        json.dumps(
            [
                _task(
                    "action",
                    "move",
                    name="HumanAction",
                    arguments={"action": "move", "parameters": {"x": 2, "y": 1, "yaw": 1}},
                )
            ],
            ensure_ascii=False,
        ),
        contract,
    )
    failing = score_output(
        case,
        json.dumps(
            [
                _task(
                    "action",
                    "move",
                    name="HumanAction",
                    arguments={"action": "move", "parameters": {"x": True, "y": 1, "yaw": 1}},
                )
            ],
            ensure_ascii=False,
        ),
        contract,
    )

    assert passing.strict_pass is True
    assert failing.reason == "task_parameter_mismatch"
    assert failing.parameter_match is False
    assert failing.parameter_errors == ("task[0].parameters.x: expected numeric value",)


@pytest.mark.parametrize(
    ("bad_spec", "fragment"),
    [
        pytest.param({"gte": "x"}, "invalid range spec", id="non-numeric-bound"),
        pytest.param({"gte": True}, "invalid range spec", id="bool-bound"),
        pytest.param({"gte": math.nan}, "invalid range spec", id="nan-bound"),
        pytest.param({"lte": math.inf}, "invalid range spec", id="infinity-bound"),
        pytest.param({"gte": 1, "value": 2}, "invalid range spec", id="mixed-unsupported-keys"),
    ],
)
def test_invalid_expected_range_specs_never_raise_and_report_parameter_mismatch(
    bad_spec: dict[str, object],
    fragment: str,
) -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(
                kind="action",
                instruction="move",
                name="HumanAction",
                arguments={
                    "action": "move",
                    "parameters": {
                        "x": bad_spec,
                    },
                },
            ),
        )
    )
    raw = json.dumps(
        [
            _task(
                "action",
                "move",
                name="HumanAction",
                arguments={"action": "move", "parameters": {"x": 2}},
            )
        ],
        ensure_ascii=False,
    )

    result = score_output(case, raw, contract)

    assert result.reason == "task_parameter_mismatch"
    assert result.strict_pass is False
    assert result.parameter_match is False
    assert len(result.parameter_errors) == 1
    assert fragment in result.parameter_errors[0]


@pytest.mark.parametrize(
    ("actual_pose", "strict_pass", "error_fragment"),
    [
        pytest.param({"arm": "left", "speed": 0.8}, True, "", id="mapping"),
        pytest.param("standing", False, "task[0].parameters.pose: expected object, got 'standing'", id="non-mapping"),
    ],
)
def test_empty_expected_nested_object_matches_any_mapping_and_rejects_non_mapping(
    actual_pose: object,
    strict_pass: bool,
    error_fragment: str,
) -> None:
    contract = _replace_tool(
        _make_contract(),
        "HumanAction",
        kind="action",
        schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "minLength": 1},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pose": {},
                    },
                    "additionalProperties": True,
                },
            },
            "required": ["action", "parameters"],
            "additionalProperties": False,
        },
    )
    case = _make_case(
        expected=(
            ExpectedTask(
                kind="action",
                instruction="move",
                name="HumanAction",
                arguments={
                    "action": "move",
                    "parameters": {
                        "pose": {},
                    },
                },
            ),
        )
    )
    raw = json.dumps(
        [
            _task(
                "action",
                "move",
                name="HumanAction",
                arguments={
                    "action": "move",
                    "parameters": {"pose": actual_pose},
                },
            )
        ],
        ensure_ascii=False,
    )

    result = score_output(case, raw, contract)

    assert result.strict_pass is strict_pass
    if strict_pass:
        assert result.reason == "passed"
        assert result.parameter_match is True
        assert result.parameter_errors == ()
    else:
        assert result.reason == "task_parameter_mismatch"
        assert result.parameter_match is False
        assert error_fragment in result.parameter_errors[0]


def test_skipped_query_parameters_produce_none_and_still_pass_when_all_parameters_are_skipped() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(kind="query", instruction="天气", name="get_weather", arguments={"city": "北京"}),
            ExpectedTask(kind="query", instruction="搜索", name="web_search", arguments={"query": "机器人"}),
        )
    )
    raw = json.dumps(
        [
            _task("query", "天气", name="get_weather", arguments={"city": "上海"}),
            _task("query", "搜索", name="web_search", arguments={"query": "完全不同", "top_k": 9}),
        ],
        ensure_ascii=False,
    )

    result = score_output(case, raw, contract)

    assert result.strict_pass is True
    assert result.parameter_match is None
    assert result.parameter_errors == ()


def test_mixed_skipped_and_checked_tasks_only_score_checked_parameters() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(kind="query", instruction="天气", name="get_weather", arguments={"city": "北京"}),
            ExpectedTask(kind="action", instruction="gesture", name="RobotGesture", arguments={"gesture": "wave"}),
        )
    )
    raw = json.dumps(
        [
            _task("query", "天气", name="get_weather", arguments={"city": "广州"}),
            _task("action", "gesture", name="RobotGesture", arguments={"gesture": "wave", "speed": 2}),
        ],
        ensure_ascii=False,
    )

    result = score_output(case, raw, contract)

    assert result.strict_pass is True
    assert result.parameter_match is True


def test_score_output_preserves_raw_output_and_parameter_error_evidence_without_raising() -> None:
    contract = _make_contract()
    case = _make_case(
        expected=(
            ExpectedTask(kind="action", instruction="gesture", name="RobotGesture", arguments={"gesture": "wave"}),
        )
    )
    raw = json.dumps(
        [
            _task("action", "gesture", name="RobotGesture", arguments={"gesture": "bow"}),
        ],
        ensure_ascii=False,
    )

    result = score_output(case, raw, contract)

    assert result.reason == "task_parameter_mismatch"
    assert result.raw_output == raw
    assert result.parameter_errors == ("task[0].gesture: expected 'wave', got 'bow'",)
    assert result.parsed_tasks[0].arguments["gesture"] == "bow"
