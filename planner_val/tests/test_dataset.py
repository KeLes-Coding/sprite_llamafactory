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

import pytest

from planner_val.dataset import (
    ACTION_TOOLS,
    QUERY_TOOLS,
    annotate_overlap,
    load_training_corpus_index,
    load_v312_cases,
    select_cases,
)
from planner_val.models import EvalCase, ExpectedTask, OverlapInfo


def _write_case_file(root: Path, level: str, name: str, data: list[dict[str, object]]) -> Path:
    case_dir = root / "case" / level
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / name
    path.write_text(json.dumps({"data": data}), encoding="utf-8")
    return path


def _make_case(
    case_id: str,
    query: str,
    expected: tuple[ExpectedTask, ...],
    *,
    level: str = "L1",
    language: str = "zh",
    difficulty: str = "easy",
    polarity: str = "positive",
    source_atom_ids: frozenset[str] | None = None,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        query=query,
        expected=expected,
        forbidden_functions=frozenset(),
        level=level,
        language=language,
        difficulty=difficulty,
        polarity=polarity,
        source_atom_ids=source_atom_ids or frozenset(),
    )


def _make_training_zip(
    path: Path, *, train_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]
) -> Path:
    with zipfile.ZipFile(path, mode="w") as zf:
        for member_name, rows in (
            ("action_sequence_sft_train.parquet", train_rows),
            ("action_sequence_sft_validation.parquet", validation_rows),
        ):
            buffer = io.BytesIO()
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows)
            pq.write_table(table, buffer)
            zf.writestr(member_name, buffer.getvalue())
    return path


def test_projection_handles_query_action_and_reply_and_order(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_case_file(
        root,
        "L1",
        "001.json",
        [
            {
                "case_id": "case-action",
                "query": "full query text",
                "source_query_zh": "toolless query text",
                "language": "zh",
                "intents": [
                    {"tool": "get_weather", "params": {"city": "Beijing"}},
                    {"text": "plain reply"},
                    {"tool": "HumanAction", "action": "move", "params": {"x": 1, "yaw": -2}},
                ],
                "forbidden_functions": ["bad_fn"],
                "source_atom_ids": ["atom-a", "atom-b"],
            }
        ],
    )

    cases = load_v312_cases(root)

    assert cases == [
        EvalCase(
            case_id="case-action",
            query="full query text",
            expected=(
                ExpectedTask(
                    kind="query", instruction="full query text", name="get_weather", arguments={"city": "Beijing"}
                ),
                ExpectedTask(kind="reply", instruction="toolless query text"),
                ExpectedTask(
                    kind="action",
                    instruction="full query text",
                    name="HumanAction",
                    arguments={"action": "move", "parameters": {"x": 1, "yaw": -2}},
                ),
            ),
            forbidden_functions=frozenset({"bad_fn"}),
            level="L1",
            language="zh",
            difficulty="unknown",
            polarity="positive",
            source_atom_ids=frozenset({"atom-a", "atom-b"}),
        )
    ]

    def test_training_index_uses_row_level_metadata_lineage_with_message_fallback(tmp_path: Path) -> None:
        package_path = _make_training_zip(
            tmp_path / "train.zip",
            train_rows=[
                {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {
                            "role": "user",
                            "content": json.dumps({"instruction": "train exact"}),
                        },
                    ],
                    "metadata": {"lineage": [{"source_atom_id": "atom-row"}]},
                }
            ],
            validation_rows=[
                {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {
                            "role": "user",
                            "content": json.dumps({"instruction": "validation exact"}),
                            "metadata": {"lineage": [{"source_atom_id": "atom-msg"}]},
                        },
                    ]
                }
            ],
        )

        index = load_training_corpus_index(package_path)

        assert index.train_instructions == frozenset({"train exact"})
        assert index.validation_instructions == frozenset({"validation exact"})
        assert index.train_source_atom_ids == frozenset({"atom-row"})
        assert index.validation_source_atom_ids == frozenset({"atom-msg"})


def test_projection_preserves_repeated_tools_and_flat_selectors(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_case_file(
        root,
        "L2",
        "001.json",
        [
            {
                "case_id": "case-repeat",
                "query": "repeat query",
                "intents": [
                    {"tool": "RobotGesture", "action": "wave", "params": {"speed": 2}},
                    {"tool": "RobotGesture", "action": "bow", "params": {"speed": 1}},
                    {"tool": "RobotDance", "action": "dance", "params": {"beats": 4}},
                    {"tool": "RobotDance", "params": {"beats": 2}},
                ],
                "source_atom_ids": ["atom-x"],
            }
        ],
    )

    cases = load_v312_cases(root)

    assert cases[0].expected == (
        ExpectedTask(
            kind="action", instruction="repeat query", name="RobotGesture", arguments={"speed": 2, "gesture": "wave"}
        ),
        ExpectedTask(
            kind="action", instruction="repeat query", name="RobotGesture", arguments={"speed": 1, "gesture": "bow"}
        ),
        ExpectedTask(
            kind="action", instruction="repeat query", name="RobotDance", arguments={"beats": 4, "dance": "dance"}
        ),
        ExpectedTask(kind="action", instruction="repeat query", name="RobotDance", arguments={"beats": 2}),
    )


def test_training_index_uses_row_level_metadata_lineage_with_message_fallback(tmp_path: Path) -> None:
    package_path = _make_training_zip(
        tmp_path / "train.zip",
        train_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps({"instruction": "train exact"}),
                    },
                ],
                "metadata": {"lineage": [{"source_atom_id": "atom-row"}]},
            }
        ],
        validation_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps({"instruction": "validation exact"}),
                        "metadata": {"lineage": [{"source_atom_id": "atom-msg"}]},
                    },
                ]
            }
        ],
    )

    index = load_training_corpus_index(package_path)

    assert index.train_instructions == frozenset({"train exact"})
    assert index.validation_instructions == frozenset({"validation exact"})
    assert index.train_source_atom_ids == frozenset({"atom-row"})
    assert index.validation_source_atom_ids == frozenset({"atom-msg"})


def test_projection_preserves_forbidden_functions_and_negative_polarity(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_case_file(
        root,
        "L3",
        "001.json",
        [
            {
                "case_id": "case-negative",
                "query": "negative query",
                "negative": True,
                "forbidden_functions": ["f1", "f2"],
                "intents": [
                    {"tool": "web_search", "params": {"query": "x"}, "negative_sample": True},
                ],
                "source_atom_ids": ["atom-z"],
            }
        ],
    )

    case = load_v312_cases(root)[0]

    assert case.forbidden_functions == frozenset({"f1", "f2"})
    assert case.polarity == "negative"


def test_projection_preserves_numeric_ranges_and_nested_arguments(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_case_file(
        root,
        "L4",
        "001.json",
        [
            {
                "case_id": "case-ranges",
                "query": "range query",
                "intents": [
                    {
                        "tool": "HumanAction",
                        "action": "move",
                        "params": {"step": {"min": 1, "max": 3}, "degree": {"min": -5, "max": 5}},
                    }
                ],
            }
        ],
    )

    case = load_v312_cases(root)[0]

    assert case.expected[0].arguments == {
        "action": "move",
        "parameters": {
            "step": {"min": 1, "max": 3},
            "degree": {"min": -5, "max": 5},
        },
    }


def test_projection_reports_missing_fields_with_source_location(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_case_file(root, "L5", "001.json", [{"query": "missing id"}])

    with pytest.raises(ValueError, match=r"case/L5/001\.json.*row 0.*case_id"):
        load_v312_cases(root)


def test_training_index_and_overlap_flags(tmp_path: Path) -> None:
    package_path = _make_training_zip(
        tmp_path / "train.zip",
        train_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps({"instruction": "train exact"}),
                        "metadata": {"lineage": [{"source_atom_id": "atom-train"}, {"source_atom_id": ""}]},
                    },
                ]
            }
        ],
        validation_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps({"instruction": "validation exact"}),
                        "metadata": {"lineage": [{"source_atom_id": "atom-val"}]},
                    },
                ]
            }
        ],
    )

    index = load_training_corpus_index(package_path)
    cases = [
        _make_case("a", "train exact", (), source_atom_ids=frozenset({"atom-train"})),
        _make_case("b", "validation exact", (), source_atom_ids=frozenset({"atom-val"})),
        _make_case("c", "novel", (), source_atom_ids=frozenset({"atom-train", "atom-new"})),
        _make_case("d", "empty", (), source_atom_ids=frozenset()),
    ]

    annotated = annotate_overlap(cases, index)

    assert annotated[0].overlap == OverlapInfo(True, False, True, False)
    assert annotated[1].overlap == OverlapInfo(False, True, False, True)
    assert annotated[2].overlap == OverlapInfo(False, False, False, True)
    assert annotated[3].overlap == OverlapInfo(False, False, False, False)


def test_training_index_rejects_missing_zip_member(tmp_path: Path) -> None:
    package_path = tmp_path / "train.zip"
    with zipfile.ZipFile(package_path, mode="w") as zf:
        import pyarrow as pa
        import pyarrow.parquet as pq

        buffer = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "ok"},
                            {
                                "role": "user",
                                "content": json.dumps({"instruction": "train"}),
                                "metadata": {"lineage": [{"source_atom_id": "atom-train"}]},
                            },
                        ]
                    }
                ]
            ),
            buffer,
        )
        zf.writestr("action_sequence_sft_train.parquet", buffer.getvalue())

    with pytest.raises(ValueError, match=r"missing member action_sequence_sft_validation\.parquet"):
        load_training_corpus_index(package_path)


def test_training_index_rejects_malformed_messages(tmp_path: Path) -> None:
    package_path = tmp_path / "train.zip"
    with zipfile.ZipFile(package_path, mode="w") as zf:
        import pyarrow as pa
        import pyarrow.parquet as pq

        buffer = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist([{"messages": {"role": "user", "content": json.dumps({"instruction": "train"})}}]),
            buffer,
        )
        zf.writestr("action_sequence_sft_train.parquet", buffer.getvalue())
        zf.writestr("action_sequence_sft_validation.parquet", buffer.getvalue())

    with pytest.raises(ValueError, match=r"action_sequence_sft_train\.parquet: row 0: malformed messages"):
        load_training_corpus_index(package_path)


def test_training_index_rejects_malformed_user_json(tmp_path: Path) -> None:
    package_path = _make_training_zip(
        tmp_path / "train.zip",
        train_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "not-json"},
                ]
            }
        ],
        validation_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps({"instruction": "validation exact"}),
                    },
                ]
            }
        ],
    )

    with pytest.raises(ValueError, match=r"action_sequence_sft_train\.parquet: row 0: malformed user JSON"):
        load_training_corpus_index(package_path)


def test_training_index_rejects_missing_instruction(tmp_path: Path) -> None:
    package_path = _make_training_zip(
        tmp_path / "train.zip",
        train_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": json.dumps({})},
                ]
            }
        ],
        validation_rows=[
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps({"instruction": "validation exact"}),
                    },
                ]
            }
        ],
    )

    with pytest.raises(ValueError, match=r"action_sequence_sft_train\.parquet: row 0: missing instruction"):
        load_training_corpus_index(package_path)


def test_sampling_is_deterministic_and_balanced() -> None:
    cases = [
        _make_case("l1-a", "q1", (), level="L1"),
        _make_case("l2-a", "q2", (), level="L2"),
        _make_case("l2-b", "q3", (), level="L2"),
        _make_case("l2-c", "q4", (), level="L2"),
        _make_case("l2-d", "q5", (), level="L2"),
    ]

    sampled_one = select_cases(cases, sample_size=4, seed=7)
    sampled_two = select_cases(cases, sample_size=4, seed=7)

    assert sampled_one == sampled_two
    assert [case.case_id for case in sampled_one] == sorted(case.case_id for case in sampled_one)
    assert [case.level for case in sampled_one].count("L1") == 1
    assert [case.level for case in sampled_one].count("L2") == 3
    assert select_cases(cases, sample_size=None, seed=7) == sorted(cases, key=lambda case: case.case_id)


def test_sampling_rejects_invalid_sizes() -> None:
    cases = [_make_case("a", "q", ())]

    with pytest.raises(ValueError, match="negative"):
        select_cases(cases, sample_size=-1, seed=1)

    with pytest.raises(ValueError, match="greater"):
        select_cases(cases, sample_size=2, seed=1)


def test_nested_arguments_are_immutable() -> None:
    original = {"outer": {"inner": {"value": 1}}}
    task = ExpectedTask(
        kind="action", instruction="query", name="HumanAction", arguments={"action": "move", "parameters": original}
    )

    original["outer"]["inner"]["value"] = 2

    assert task.arguments["parameters"]["outer"]["inner"]["value"] == 1
    with pytest.raises(TypeError):
        task.arguments["parameters"]["outer"]["inner"]["value"] = 3  # type: ignore[index]


def test_public_tool_sets_are_defined() -> None:
    assert ACTION_TOOLS == frozenset({"HumanAction", "RobotGesture", "RobotDance"})
    assert QUERY_TOOLS == frozenset({"get_weather", "web_search", "rag_query", "robot_status", "get_visual_info"})
