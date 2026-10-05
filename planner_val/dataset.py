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
import random
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from planner_val.models import EvalCase, ExpectedTask, OverlapInfo, TrainingCorpusIndex


ACTION_TOOLS = frozenset({"HumanAction", "RobotGesture", "RobotDance"})
QUERY_TOOLS = frozenset({"get_weather", "web_search", "rag_query", "robot_status", "get_visual_info"})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _freeze_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_freeze_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _as_non_empty_str(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected non-empty string")
    return value


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected object with data list")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected data list")
    return payload


def _iter_case_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("case/L*/*.json"), key=lambda path: str(path))


def _project_arguments(tool: str, intent: dict[str, Any]) -> dict[str, Any]:
    params = intent.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"intent params for {tool} must be an object")

    if tool == "HumanAction":
        action_value = intent.get("action")
        if not isinstance(action_value, str) or not action_value:
            raise ValueError("HumanAction intent requires a non-empty action")
        arguments: dict[str, Any] = {"action": action_value}
        nested_params = {key: _freeze_value(value) for key, value in params.items() if key != "action"}
        if nested_params:
            arguments["parameters"] = nested_params
        return arguments

    arguments = {key: _freeze_value(value) for key, value in params.items()}
    action_value = intent.get("action")
    if tool == "RobotGesture" and action_value is not None:
        arguments["gesture"] = action_value
    if tool == "RobotDance" and action_value is not None:
        arguments["dance"] = action_value
    return arguments


def _project_expected_tasks(case: dict[str, Any]) -> tuple[ExpectedTask, ...]:
    case_query = _as_non_empty_str(case.get("query"))
    source_query = case.get("source_query_zh")
    reply_instruction = source_query if isinstance(source_query, str) and source_query else case_query

    projected: list[ExpectedTask] = []
    for intent in case.get("intents") or []:
        if not isinstance(intent, dict):
            raise ValueError("intent must be an object")
        tool = intent.get("tool")
        if not isinstance(tool, str) or not tool:
            projected.append(ExpectedTask(kind="reply", instruction=reply_instruction))
            continue
        if tool in ACTION_TOOLS:
            kind = "action"
        elif tool in QUERY_TOOLS:
            kind = "query"
        else:
            kind = "reply"
        instruction = case_query if kind != "reply" else reply_instruction
        arguments = _project_arguments(tool, intent) if kind != "reply" else {}
        projected.append(
            ExpectedTask(
                kind=kind, instruction=instruction, name=tool if kind != "reply" else None, arguments=arguments
            )
        )
    return tuple(projected)


def _collect_source_atoms(case: dict[str, Any]) -> frozenset[str]:
    source_atom_ids: set[str] = set()
    for item in case.get("source_atom_ids") or []:
        if isinstance(item, str) and item:
            source_atom_ids.add(item)
    for intent in case.get("intents") or []:
        if not isinstance(intent, dict):
            continue
        for item in intent.get("source_atom_ids") or []:
            if isinstance(item, str) and item:
                source_atom_ids.add(item)
        item = intent.get("source_atom_id")
        if isinstance(item, str) and item:
            source_atom_ids.add(item)
    return frozenset(source_atom_ids)


def _collect_forbidden_functions(case: dict[str, Any]) -> frozenset[str]:
    return frozenset(item for item in case.get("forbidden_functions") or [] if isinstance(item, str) and item)


def _project_polarity(case: dict[str, Any]) -> str:
    if case.get("negative") or case.get("negative_sample"):
        return "negative"
    for intent in case.get("intents") or []:
        if isinstance(intent, dict) and (intent.get("negative") or intent.get("negative_sample")):
            return "negative"
    return "positive"


def load_v312_cases(dataset_root: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in _iter_case_files(dataset_root):
        payload = _load_json_file(path)
        level = path.parent.name
        for row_index, row in enumerate(payload["data"]):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: row {row_index}: expected object")
            case_id = row.get("case_id")
            query = row.get("query")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{path}: row {row_index}: missing case_id")
            if not isinstance(query, str) or not query:
                raise ValueError(f"{path}: row {row_index}: missing query")
            cases.append(
                EvalCase(
                    case_id=case_id,
                    query=query,
                    expected=_project_expected_tasks(row),
                    forbidden_functions=_collect_forbidden_functions(row),
                    level=level,
                    language=row.get("language") if isinstance(row.get("language"), str) else "",
                    difficulty=row.get("difficulty") if isinstance(row.get("difficulty"), str) else "unknown",
                    polarity=_project_polarity(row),
                    source_atom_ids=_collect_source_atoms(row),
                )
            )
    return cases


def _read_parquet_rows(member_bytes: bytes, member_name: str) -> list[dict[str, Any]]:
    try:
        table = pq.read_table(io.BytesIO(member_bytes))
    except Exception as exc:  # pragma: no cover - exercised through ValueError path in tests
        raise ValueError(f"{member_name}: failed to read parquet") from exc
    return table.to_pylist()


def _collect_lineage_source_atom_ids(metadata: Any) -> frozenset[str]:
    if not isinstance(metadata, dict):
        return frozenset()
    lineage = metadata.get("lineage")
    if not isinstance(lineage, list):
        return frozenset()
    source_atom_ids: set[str] = set()
    for lineage_entry in lineage:
        if not isinstance(lineage_entry, dict):
            continue
        source_atom_id = lineage_entry.get("source_atom_id")
        if isinstance(source_atom_id, str) and source_atom_id:
            source_atom_ids.add(source_atom_id)
    return frozenset(source_atom_ids)


def _collect_index_entries(rows: list[dict[str, Any]], member_name: str) -> tuple[frozenset[str], frozenset[str]]:
    instructions: set[str] = set()
    source_atom_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{member_name}: row {row_index}: malformed messages")
        user_message: dict[str, Any] | None = None
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user_message = message
                break
        if user_message is None:
            raise ValueError(f"{member_name}: row {row_index}: malformed messages")
        content = user_message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{member_name}: row {row_index}: malformed user JSON")
        try:
            user_payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{member_name}: row {row_index}: malformed user JSON") from exc
        instruction = user_payload.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"{member_name}: row {row_index}: missing instruction")
        instructions.add(instruction)
        row_source_atom_ids = _collect_lineage_source_atom_ids(row.get("metadata"))
        if row_source_atom_ids:
            source_atom_ids.update(row_source_atom_ids)
            continue
        source_atom_ids.update(_collect_lineage_source_atom_ids(user_message.get("metadata")))
    return frozenset(instructions), frozenset(source_atom_ids)


def load_training_corpus_index(package_path: Path) -> TrainingCorpusIndex:
    required_members = {
        "action_sequence_sft_train.parquet": "train",
        "action_sequence_sft_validation.parquet": "validation",
    }
    with zipfile.ZipFile(package_path) as archive:
        collected: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
        for member_name in required_members:
            try:
                member_bytes = archive.read(member_name)
            except KeyError as exc:
                raise ValueError(f"{package_path}: missing member {member_name}") from exc
            rows = _read_parquet_rows(member_bytes, member_name)
            collected[member_name] = _collect_index_entries(rows, member_name)
    train_instructions, train_source_atom_ids = collected["action_sequence_sft_train.parquet"]
    validation_instructions, validation_source_atom_ids = collected["action_sequence_sft_validation.parquet"]
    return TrainingCorpusIndex(
        train_instructions=train_instructions,
        validation_instructions=validation_instructions,
        train_source_atom_ids=train_source_atom_ids,
        validation_source_atom_ids=validation_source_atom_ids,
    )


def annotate_overlap(cases: Sequence[EvalCase], index: TrainingCorpusIndex) -> list[EvalCase]:
    annotated: list[EvalCase] = []
    for case in cases:
        overlap = OverlapInfo(
            exact_train=case.query in index.train_instructions,
            exact_validation=case.query in index.validation_instructions,
            all_atoms_seen_train=bool(case.source_atom_ids) and case.source_atom_ids <= index.train_source_atom_ids,
            has_novel_atom=bool(case.source_atom_ids - index.train_source_atom_ids),
        )
        annotated.append(
            EvalCase(
                case_id=case.case_id,
                query=case.query,
                expected=case.expected,
                forbidden_functions=case.forbidden_functions,
                level=case.level,
                language=case.language,
                difficulty=case.difficulty,
                polarity=case.polarity,
                source_atom_ids=case.source_atom_ids,
                overlap=overlap,
            )
        )
    return annotated


def _level_seed(seed: int, level: str) -> int:
    digest = hashlib.sha256(f"{seed}:{level}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def select_cases(cases: Sequence[EvalCase], sample_size: int | None, seed: int) -> list[EvalCase]:
    ordered_cases = sorted(cases, key=lambda case: case.case_id)
    if sample_size is None:
        return ordered_cases
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative")
    if sample_size > len(ordered_cases):
        raise ValueError("sample_size cannot be greater than the case count")
    if sample_size == 0:
        return []

    grouped: dict[str, list[EvalCase]] = defaultdict(list)
    for case in sorted(ordered_cases, key=lambda case: (case.level, case.case_id)):
        grouped[case.level].append(case)
    levels = sorted(grouped)
    if not levels:
        return []

    quotas = {level: sample_size // len(levels) for level in levels}
    for level in levels[: sample_size % len(levels)]:
        quotas[level] += 1

    selected: list[EvalCase] = []
    spare: list[tuple[str, list[EvalCase]]] = []
    leftover = 0
    for level in levels:
        level_cases = list(grouped[level])
        rng = random.Random(_level_seed(seed, level))
        rng.shuffle(level_cases)
        quota = min(len(level_cases), quotas[level])
        selected.extend(level_cases[:quota])
        if len(level_cases) > quota:
            spare.append((level, level_cases[quota:]))
        leftover += quotas[level] - quota

    if leftover:
        spare_map = {level: list(extra_cases) for level, extra_cases in spare}
        for level in levels:
            while leftover and spare_map.get(level):
                selected.append(spare_map[level].pop(0))
                leftover -= 1

    if len(selected) < sample_size:
        taken = {case.case_id for case in selected}
        for level in levels:
            for case in grouped[level]:
                if case.case_id not in taken:
                    selected.append(case)
                    taken.add(case.case_id)
                    if len(selected) == sample_size:
                        break
            if len(selected) == sample_size:
                break

    return sorted(selected, key=lambda case: case.case_id)
