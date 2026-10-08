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
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from planner_val.models import EvalCase
from planner_val.scoring import parse_plan, score_benchmark_compatible, score_output
from planner_val.validation_eval import ValidationCase, _expected_tasks, load_validation_cases


_DATASET_MEMBERS = ("action_sequence_sft_train.parquet", "action_sequence_sft_validation.parquet")


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid challenge file: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("challenge file must be a schema_version 1 object")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("challenge file must contain cases")
    return payload


def _required_string(item: Mapping[str, Any], key: str, *, case_id: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{case_id}: {key} must be a non-empty string")
    return value


def _task_metadata(expected: list[Mapping[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    kinds = tuple(str(task["kind"]) for task in expected)
    tools = tuple(str(task["name"]) for task in expected if isinstance(task.get("name"), str))
    actions: list[str] = []
    for task in expected:
        name = task.get("name")
        arguments = task.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        key = {"HumanAction": "action", "RobotGesture": "gesture", "RobotDance": "dance"}.get(name)
        if key is not None and isinstance(arguments.get(key), str):
            actions.append(arguments[key])
    return kinds, tools, tuple(actions)


def _project_removed_weather_coordinates(
    expected: list[Mapping[str, Any]], weather_schema: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    """Drop coordinates authored for v3.2.0 when the reference contract no longer has them."""
    if weather_schema is None or "latitude" in weather_schema.get("properties", {}):
        return expected
    projected = deepcopy(expected)
    for task in projected:
        if task.get("name") == "get_weather" and isinstance(task.get("arguments"), dict):
            task["arguments"].pop("latitude", None)
            task["arguments"].pop("longitude", None)
    return projected


def load_challenge_cases(challenge_path: Path, package_path: Path) -> list[ValidationCase]:
    payload = _load_payload(challenge_path)
    reference = next(
        (case for case in load_validation_cases(package_path) if case.tool_naming == "real"),
        None,
    )
    if reference is None:
        raise ValueError("challenge reference requires a validation row with real tool names")
    base_user = json.loads(reference.user)
    weather_schema = reference.contract.tool_schemas.get("get_weather")
    raw_cases = payload["cases"]
    case_ids = [item.get("case_id") for item in raw_cases if isinstance(item, Mapping)]
    if len(case_ids) != len(raw_cases) or len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate challenge case_id")

    cases: list[ValidationCase] = []
    known_tools = frozenset(reference.contract.tool_schemas)
    for item in raw_cases:
        case_id = _required_string(item, "case_id", case_id="challenge")
        query = _required_string(item, "query", case_id=case_id)
        category = _required_string(item, "category", case_id=case_id)
        language = _required_string(item, "language", case_id=case_id)
        difficulty = _required_string(item, "difficulty", case_id=case_id)
        polarity = _required_string(item, "polarity", case_id=case_id)
        expected = item.get("expected")
        forbidden = item.get("forbidden_functions", [])
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{case_id}: expected must be a non-empty list")
        if not isinstance(forbidden, list) or any(name not in known_tools for name in forbidden):
            raise ValueError(f"{case_id}: forbidden_functions contains an unknown tool")

        expected = _project_removed_weather_coordinates(expected, weather_schema)
        expected_raw = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        expected_tasks = _expected_tasks(expected_raw, case_id=case_id)
        kinds, tools, actions = _task_metadata(expected)
        user_payload = deepcopy(base_user)
        user_payload["instruction"] = query
        user = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
        cases.append(
            ValidationCase(
                case_id=case_id,
                system=reference.system,
                user=user,
                expected_raw=expected_raw,
                query=query,
                eval_case=EvalCase(
                    case_id=case_id,
                    query=query,
                    expected=expected_tasks,
                    forbidden_functions=frozenset(forbidden),
                    level=str(len(expected_tasks)),
                    language=language,
                    difficulty=difficulty,
                    polarity=polarity,
                    source_atom_ids=frozenset(),
                ),
                contract=reference.contract,
                level=str(len(expected_tasks)),
                language=language,
                polarity=polarity,
                provenance=category,
                kinds=kinds,
                tools=tools,
                actions=actions,
                source_atom_ids=(),
                source_item_id=case_id,
                example_id=case_id,
            )
        )
    return cases


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_plan_signature(tasks: Any) -> str:
    signature = []
    for task in tasks:
        if task.get("kind") == "reply":
            signature.append({"kind": "reply", "instruction": task.get("instruction")})
        else:
            signature.append(
                {"kind": task.get("kind"), "name": task.get("name"), "arguments": task.get("arguments", {})}
            )
    return _canonical_json(signature)


def _corpus_values(package_path: Path) -> tuple[set[str], set[str], set[str]]:
    queries: set[str] = set()
    expected_plans: set[str] = set()
    tool_plans: set[str] = set()
    with zipfile.ZipFile(package_path) as archive:
        for member in _DATASET_MEMBERS:
            rows = pq.read_table(io.BytesIO(archive.read(member))).to_pylist()
            for row in rows:
                messages = row.get("messages")
                if not isinstance(messages, list):
                    continue
                by_role = {
                    message.get("role"): message.get("content") for message in messages if isinstance(message, Mapping)
                }
                if isinstance(by_role.get("user"), str):
                    user = json.loads(by_role["user"])
                    if isinstance(user.get("instruction"), str):
                        queries.add(user["instruction"])
                if isinstance(by_role.get("assistant"), str):
                    expected = json.loads(by_role["assistant"])
                    expected_plans.add(_canonical_json(expected))
                    tool_plans.add(_tool_plan_signature(expected))
    return queries, expected_plans, tool_plans


def audit_challenge_cases(challenge_path: Path, package_path: Path) -> dict[str, Any]:
    payload = _load_payload(challenge_path)
    raw_cases = payload["cases"]
    cases = load_challenge_cases(challenge_path, package_path)
    corpus_queries, corpus_plans, corpus_tool_plans = _corpus_values(package_path)
    case_ids = [case.case_id for case in cases]
    queries = [case.query for case in cases]
    expected_plans = [_canonical_json(item["expected"]) for item in raw_cases]
    tool_plans = [_tool_plan_signature(item["expected"]) for item in raw_cases]
    contract_valid = 0
    strict_pass = 0
    compatible_pass = 0
    for case in cases:
        contract_valid += parse_plan(case.expected_raw, case.contract).contract_valid
        strict_pass += score_output(case.eval_case, case.expected_raw, case.contract).strict_pass
        compatible_pass += score_benchmark_compatible(case.eval_case, case.expected_raw, case.contract).passed

    total = len(cases)
    audit = {
        "total": total,
        "categories": dict(sorted(Counter(case.provenance for case in cases).items())),
        "languages": dict(sorted(Counter(case.language for case in cases).items())),
        "novelty_tags": dict(
            sorted(
                Counter(
                    str(tag) for item in raw_cases for tag in item.get("novelty_tags", []) if isinstance(tag, str)
                ).items()
            )
        ),
        "duplicate_case_ids": total - len(set(case_ids)),
        "duplicate_queries": total - len(set(queries)),
        "exact_query_overlap": sum(query in corpus_queries for query in queries),
        "exact_expected_plan_overlap": sum(plan in corpus_plans for plan in expected_plans),
        "exact_tool_plan_overlap": sum(plan in corpus_tool_plans for plan in tool_plans),
        "contract_valid_ground_truth": contract_valid,
        "strict_self_score_pass": strict_pass,
        "compatible_self_score_pass": compatible_pass,
    }
    audit["ready"] = all(
        (
            total > 0,
            audit["duplicate_case_ids"] == 0,
            audit["duplicate_queries"] == 0,
            audit["exact_query_overlap"] == 0,
            audit["exact_expected_plan_overlap"] == 0,
            audit["exact_tool_plan_overlap"] == 0,
            contract_valid == total,
            strict_pass == total,
            compatible_pass == total,
        )
    )
    return audit
