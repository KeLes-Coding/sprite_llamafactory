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
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

from planner_val.backends import GenerationResult
from planner_val.challenge_eval import load_challenge_cases
from planner_val.models import ContractSnapshot
from planner_val.scoring import score_output
from planner_val.tool_schema_v2 import adapt_case_to_schema_v2, enrich_tool_schema, main


def _tools() -> list[dict]:
    return [
        {
            "name": "RobotDance",
            "kind": "action",
            "description": "执行舞蹈",
            "arguments_schema": {
                "type": "object",
                "properties": {
                    "dance": {"type": "string", "enum": ["warm_up_dance", "egyptian_shake"]},
                    "repeat": {"const": 1},
                },
                "required": ["dance"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_weather",
            "kind": "query",
            "description": "查询天气",
            "arguments_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "minLength": 1},
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "query_type": {"type": "string", "enum": ["now", "24h", "7d", "now_7d"]},
                },
                "required": ["city", "latitude", "longitude", "query_type"],
                "additionalProperties": False,
            },
        },
    ]


def test_weather_schema_no_longer_accepts_or_requires_coordinates() -> None:
    tools = _tools()
    original = deepcopy(tools)
    enriched = enrich_tool_schema(tools)
    weather = enriched[1]["arguments_schema"]
    validator = Draft202012Validator(weather)
    assert validator.is_valid({"city": "长沙", "query_type": "now_7d"})
    assert not validator.is_valid({"city": "长沙", "query_type": "now_7d", "latitude": 28.23})
    assert weather["required"] == ["city", "query_type"]
    assert "当前" in weather["properties"]["query_type"]["description"]
    assert tools == original


def test_descriptions_cover_allowed_dance_values_without_changing_enum() -> None:
    tools = _tools()
    enriched = enrich_tool_schema(tools)
    dance = enriched[0]["arguments_schema"]["properties"]["dance"]
    assert dance["enum"] == ["warm_up_dance", "egyptian_shake"]
    assert "warm_up_dance: 热场舞" in dance["description"]
    assert "egyptian_shake: 埃及摇摆" in dance["description"]


def test_unknown_dance_value_requires_reviewed_description() -> None:
    tools = _tools()
    tools[0]["arguments_schema"]["properties"]["dance"]["enum"].append("new_dance")
    try:
        enrich_tool_schema(tools)
    except ValueError as exc:
        assert "new_dance" in str(exc)
    else:
        raise AssertionError("missing description must fail closed")


def test_accepts_frozen_contract_snapshot_without_mutation() -> None:
    source = _tools()
    snapshot = ContractSnapshot(
        name="controlled",
        system_prompt="system",
        tools_context=source,
        tool_schemas={tool["name"]: tool["arguments_schema"] for tool in source},
        tool_kinds={"RobotDance": "action", "get_weather": "query"},
        source={},
        fingerprint="test",
    )
    updated = enrich_tool_schema(snapshot.tools_context)
    assert list(updated[1]["arguments_schema"]["properties"]) == ["city", "query_type"]
    assert "latitude" in snapshot.tools_context[1]["arguments_schema"]["properties"]


def test_challenge_weather_target_matches_new_schema_without_editing_source() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "data/v3-2-0-1.0.0-134feaba-parquet.zip"
    cases = load_challenge_cases(root / "planner_val/challenges/challenge_50.json", package)
    original = cases[9]
    adjusted = adapt_case_to_schema_v2(original)
    assert json.loads(adjusted.expected_raw)[0]["arguments"] == {"city": "长沙", "query_type": "now_7d"}
    assert "latitude" in json.loads(original.expected_raw)[0]["arguments"]
    assert "latitude" not in json.loads(adjusted.user)["tools"][3]["arguments_schema"]["properties"]
    assert adjusted.contract.fingerprint != original.contract.fingerprint
    assert score_output(adjusted.eval_case, adjusted.expected_raw, adjusted.contract).strict_pass


def test_non_weather_challenge_target_stays_identical() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "data/v3-2-0-1.0.0-134feaba-parquet.zip"
    original = load_challenge_cases(root / "planner_val/challenges/challenge_50.json", package)[0]
    assert adapt_case_to_schema_v2(original).expected_raw == original.expected_raw


def test_all_challenge_targets_self_score_under_new_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "data/v3-2-0-1.0.0-134feaba-parquet.zip"
    cases = load_challenge_cases(root / "planner_val/challenges/challenge_50.json", package)
    adjusted = [adapt_case_to_schema_v2(case) for case in cases]
    assert len(adjusted) == 50
    assert all(score_output(case.eval_case, case.expected_raw, case.contract).strict_pass for case in adjusted)


def test_schema_v2_cli_runs_challenge_with_existing_scorer(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "data/v3-2-0-1.0.0-134feaba-parquet.zip"
    challenge = root / "planner_val/challenges/challenge_50.json"
    cases = [adapt_case_to_schema_v2(case) for case in load_challenge_cases(challenge, package)]
    targets = {case.case_id: case.expected_raw for case in cases}
    captured = []

    class FakeBackend:
        def __init__(self, **kwargs):
            pass

        def generate(self, requests):
            captured.extend(requests)
            return [GenerationResult(item.request_id, targets[item.request_id], 1.0, 10, 5, None) for item in requests]

        def close(self):
            pass

    monkeypatch.setattr("planner_val.tool_schema_v2.OpenAICompatibleBackend", FakeBackend)
    result_dir = tmp_path / "schema-v2"
    assert main(["--package", str(package), "--challenge", str(challenge), "--output-dir", str(result_dir)]) == 0
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    artifact = json.loads((result_dir / "tool_schema.json").read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "tool-schema-description/v2"
    assert artifact["tools"] == json.loads(captured[0].user)["tools"]
    assert summary["models"]["gemma3-270m-sft"]["strict_pass"] == 50
    assert len(captured) == 50
    assert "latitude" not in json.loads(captured[9].user)["tools"][3]["arguments_schema"]["properties"]
