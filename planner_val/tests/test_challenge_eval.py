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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from planner_val.challenge_eval import audit_challenge_cases, load_challenge_cases


def _messages(query: str, expected: list[dict[str, object]]) -> list[dict[str, str]]:
    tools = [
        {
            "name": "web_search",
            "kind": "query",
            "description": "Search public information.",
            "arguments_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "RobotDance",
            "kind": "action",
            "description": "Perform a dance.",
            "arguments_schema": {
                "type": "object",
                "properties": {"dance": {"type": "string", "enum": ["gee", "popping"]}},
                "required": ["dance"],
                "additionalProperties": False,
            },
        },
    ]
    return [
        {"role": "system", "content": "planner system"},
        {"role": "user", "content": json.dumps({"tools": tools, "instruction": query})},
        {"role": "assistant", "content": json.dumps(expected)},
    ]


def _write_package(path: Path) -> None:
    train = [
        {
            "messages": _messages(
                "seen query",
                [
                    {
                        "kind": "query",
                        "instruction": "seen query",
                        "name": "web_search",
                        "arguments": {"query": "seen query"},
                    }
                ],
            ),
            "metadata": {"fixture": "train"},
        }
    ]
    validation = [
        {
            "messages": _messages("validation query", [{"kind": "reply", "instruction": "answer"}]),
            "metadata": {"fixture": "validation"},
        }
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for member, rows in (
            ("action_sequence_sft_train.parquet", train),
            ("action_sequence_sft_validation.parquet", validation),
        ):
            buffer = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(rows), buffer)
            archive.writestr(member, buffer.getvalue())


def _write_challenges(path: Path, *, duplicate_id: bool = False) -> None:
    second_id = "challenge-001" if duplicate_id else "challenge-002"
    payload = {
        "schema_version": 1,
        "name": "test-challenges",
        "cases": [
            {
                "case_id": "challenge-001",
                "category": "unseen_value",
                "language": "en",
                "difficulty": "medium",
                "polarity": "positive",
                "query": "find a post-training topic",
                "expected": [
                    {
                        "kind": "query",
                        "instruction": "find a post-training topic",
                        "name": "web_search",
                        "arguments": {"query": "post-training topic"},
                    }
                ],
                "forbidden_functions": [],
                "novelty_tags": ["new_query"],
            },
            {
                "case_id": second_id,
                "category": "negative_constraint",
                "language": "zh",
                "difficulty": "medium",
                "polarity": "negative",
                "query": "不要跳舞，只回答收到",
                "expected": [{"kind": "reply", "instruction": "收到"}],
                "forbidden_functions": ["RobotDance"],
                "novelty_tags": ["negative_request"],
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_challenge_cases_uses_reference_contract_and_metadata(tmp_path: Path) -> None:
    package_path = tmp_path / "dataset.zip"
    challenge_path = tmp_path / "challenges.json"
    _write_package(package_path)
    _write_challenges(challenge_path)

    cases = load_challenge_cases(challenge_path, package_path)

    assert [case.case_id for case in cases] == ["challenge-001", "challenge-002"]
    assert cases[0].system == "planner system"
    assert json.loads(cases[0].user)["instruction"] == "find a post-training topic"
    assert cases[0].provenance == "unseen_value"
    assert cases[0].eval_case.difficulty == "medium"
    assert cases[1].eval_case.forbidden_functions == frozenset({"RobotDance"})


def test_audit_challenge_cases_reports_novelty_and_valid_ground_truth(tmp_path: Path) -> None:
    package_path = tmp_path / "dataset.zip"
    challenge_path = tmp_path / "challenges.json"
    _write_package(package_path)
    _write_challenges(challenge_path)

    audit = audit_challenge_cases(challenge_path, package_path)

    assert audit["total"] == 2
    assert audit["categories"] == {"negative_constraint": 1, "unseen_value": 1}
    assert audit["duplicate_case_ids"] == 0
    assert audit["duplicate_queries"] == 0
    assert audit["exact_query_overlap"] == 0
    assert audit["exact_tool_plan_overlap"] == 0
    assert audit["contract_valid_ground_truth"] == 2
    assert audit["strict_self_score_pass"] == 2
    assert audit["compatible_self_score_pass"] == 2
    assert audit["ready"] is True


def _weather_tool() -> dict[str, object]:
    return {
        "name": "get_weather",
        "kind": "query",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "minLength": 1},
                "query_type": {"type": "string", "enum": ["now", "7d"]},
            },
            "required": ["city", "query_type"],
            "additionalProperties": False,
        },
    }


def _jsonl_row(name: str, naming: str, aliases: dict[str, str]) -> dict[str, object]:
    tool = _weather_tool() | {"name": name}
    expected = [
        {"kind": "query", "instruction": "查天气", "name": name, "arguments": {"city": "长沙", "query_type": "now"}}
    ]
    return {
        "messages": [
            {"role": "system", "content": "planner system"},
            {"role": "user", "content": json.dumps({"tools": [tool], "instruction": "查天气"}, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)},
        ],
        "metadata": {"tool_naming": naming, "tool_aliases": aliases},
    }


def test_load_challenge_cases_uses_real_named_jsonl_reference_and_drops_removed_weather_coordinates(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "validation.jsonl"
    rows = [
        _jsonl_row("tool_04", "alias", {"get_weather": "tool_04"}),
        _jsonl_row("get_weather", "real", {}),
    ]
    package_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    challenge_path = tmp_path / "challenges.json"
    challenge = {
        "schema_version": 1,
        "name": "weather",
        "cases": [
            {
                "case_id": "challenge-001",
                "category": "unseen_value",
                "language": "zh",
                "difficulty": "easy",
                "polarity": "positive",
                "query": "长沙现在天气",
                "expected": [
                    {
                        "kind": "query",
                        "instruction": "查询长沙当前天气",
                        "name": "get_weather",
                        "arguments": {"city": "长沙", "latitude": 28.23, "longitude": 112.94, "query_type": "now"},
                    }
                ],
                "forbidden_functions": [],
                "novelty_tags": ["new_city"],
            }
        ],
    }
    challenge_path.write_text(json.dumps(challenge, ensure_ascii=False), encoding="utf-8")

    case = load_challenge_cases(challenge_path, package_path)[0]

    assert [tool["name"] for tool in json.loads(case.user)["tools"]] == ["get_weather"]
    assert json.loads(case.expected_raw)[0]["arguments"] == {"city": "长沙", "query_type": "now"}
    assert case.eval_case.expected[0].arguments == {"city": "长沙", "query_type": "now"}


def test_load_challenge_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    package_path = tmp_path / "dataset.zip"
    challenge_path = tmp_path / "challenges.json"
    _write_package(package_path)
    _write_challenges(challenge_path, duplicate_id=True)

    with pytest.raises(ValueError, match="duplicate challenge case_id"):
        load_challenge_cases(challenge_path, package_path)
