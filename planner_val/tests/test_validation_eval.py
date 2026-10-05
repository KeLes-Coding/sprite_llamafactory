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
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from planner_val.backends import GenerationRequest, GenerationResult
from planner_val.production_prompt import ProductionPromptProfile
from planner_val.validation_eval import evaluate_validation, load_validation_cases, main, rescore_validation
from planner_val.validation_reporting import summarize_model


class _FakeBackend:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.requests: list[GenerationRequest] = []

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self.requests.extend(requests)
        return [
            GenerationResult(
                request_id=request.request_id,
                text=self.outputs[request.request_id],
                latency_ms=10.0,
                prompt_tokens=20,
                output_tokens=5,
                error=None,
            )
            for request in requests
        ]

    def close(self) -> None:
        pass


def _report_record(*, latency_ms: float, output_tokens: int | None, error: str | None) -> dict[str, object]:
    return {
        "benchmark_compatible_pass": error is None,
        "benchmark_compatible_reason": "passed" if error is None else "backend_error",
        "benchmark_tool_sequence_match": error is None,
        "benchmark_parameter_match": None,
        "benchmark_forbidden_tool_violation": False,
        "strict_pass": error is None,
        "valid_json": error is None,
        "contract_valid": error is None,
        "task_count_match": error is None,
        "ordered_kind_name_match": error is None,
        "parameter_match": None,
        "forbidden_tool_violation": False,
        "reason": "passed" if error is None else "backend_error",
        "expected_step_count": 1,
        "predicted_step_count": int(error is None),
        "aligned_kind_name_matches": int(error is None),
        "latency_ms": latency_ms,
        "prompt_tokens": 5 if output_tokens is not None else None,
        "output_tokens": output_tokens,
        "error": error,
        "metadata": {
            "level": "1",
            "language": "zh",
            "polarity": "positive",
            "provenance": "generated",
            "task_count": 1,
            "kinds": ["reply"],
            "tools": [],
            "actions": [],
        },
    }


def test_summary_reports_partial_token_coverage_and_effective_throughput() -> None:
    summary = summarize_model(
        [
            _report_record(latency_ms=100.0, output_tokens=10, error=None),
            _report_record(latency_ms=900.0, output_tokens=None, error="backend_error"),
        ],
        batch_wall_time_ms=1000.0,
    )

    assert summary["performance"]["requests"] == {
        "total": 2,
        "succeeded": 1,
        "failed": 1,
        "per_second": 2.0,
    }
    assert summary["performance"]["tokens"]["coverage"] == {"count": 1, "rate": 0.5}
    assert summary["performance"]["throughput"] == {
        "output_tokens_per_second": 10.0,
        "total_tokens_per_second": 15.0,
        "service_time_output_tokens_per_second": 10.0,
    }
    assert summary["benchmark_compatible_pass"] == 1
    assert summary["benchmark_compatible_accuracy"] == 0.5
    assert summary["quality"]["primary_metric"] == "benchmark_compatible_accuracy"


def test_summary_sorts_numeric_levels_numerically() -> None:
    records = []
    for level in ("10", "2", "1"):
        record = _report_record(latency_ms=10.0, output_tokens=1, error=None)
        record["metadata"]["level"] = level
        records.append(record)

    summary = summarize_model(records, batch_wall_time_ms=30.0)

    assert list(summary["slices"]["level"]) == ["1", "2", "10"]


def _write_package(path: Path) -> list[dict[str, object]]:
    rows = []
    for index in range(3):
        expected = [{"kind": "reply", "instruction": f"reply-{index}"}]
        metadata = {
            "actions": ["wave_greet_bye"] if index == 1 else [],
            "example_id": f"example-{index}",
            "kinds": ["action"] if index == 1 else ["reply"],
            "language": "en" if index == 1 else "zh",
            "level": 2 if index == 1 else 1,
            "lineage": [{"source_atom_id": f"atom-{index}", "task_index": 0}],
            "polarity": "negative" if index == 1 else "positive",
            "provenance": "generated",
            "source_item_id": f"source-{index}",
            "tools": ["RobotGesture"] if index == 1 else [],
        }
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "planner system"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "tools": [
                                    {
                                        "name": "noop",
                                        "kind": "query",
                                        "arguments_schema": {
                                            "type": "object",
                                            "properties": {},
                                            "additionalProperties": False,
                                        },
                                    }
                                ],
                                "instruction": f"query-{index}",
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)},
                ],
                "metadata": metadata,
            }
        )

    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("action_sequence_sft_validation.parquet", buffer.getvalue())
    return rows


def test_load_validation_cases_preserves_original_messages_and_ground_truth(tmp_path: Path) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)

    cases = load_validation_cases(package_path, limit=2)

    assert len(cases) == 2
    assert cases[0].case_id == "validation-0000"
    assert cases[0].system == rows[0]["messages"][0]["content"]
    assert cases[0].user == rows[0]["messages"][1]["content"]
    assert cases[0].expected_raw == rows[0]["messages"][2]["content"]
    assert cases[0].query == "query-0"
    assert cases[0].level == "1"
    assert cases[0].language == "zh"
    assert cases[0].polarity == "positive"
    assert cases[0].provenance == "generated"
    assert cases[0].kinds == ("reply",)
    assert cases[0].tools == ()
    assert cases[0].actions == ()
    assert cases[0].source_atom_ids == ("atom-0",)
    assert cases[0].source_item_id == "source-0"
    assert cases[0].example_id == "example-0"


def test_load_validation_cases_recovers_negative_tool_from_lineage(tmp_path: Path) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)
    rows[0]["metadata"]["lineage"] = [{"source_atom_id": "l0_neg_RobotDance_deadbeef", "task_index": 0}]
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.writestr("action_sequence_sft_validation.parquet", buffer.getvalue())

    case = load_validation_cases(package_path, limit=1)[0]

    assert case.eval_case.forbidden_functions == frozenset({"RobotDance"})


def test_evaluate_validation_writes_metrics_slices_and_timings(tmp_path: Path, monkeypatch) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)
    cases = load_validation_cases(package_path, limit=3)
    outputs = {
        "validation-0000": rows[0]["messages"][2]["content"],
        "validation-0001": rows[1]["messages"][2]["content"],
        "validation-0002": "not-json",
    }
    backend = _FakeBackend(outputs)
    ticks = iter([1100.0, 1600.0, 1800.0])
    monkeypatch.setattr("planner_val.validation_eval._now_ms", lambda: next(ticks))

    summary = evaluate_validation(
        cases,
        {"model-a": backend},
        tmp_path / "out",
        run_config={"concurrency": 2, "max_tokens": 1024, "temperature": 0},
        dataset_info={"member": "action_sequence_sft_validation.parquet", "package_sha256": "a" * 64},
        run_started_ms=500.0,
        run_started_at="2026-09-20T00:00:00+00:00",
    )

    assert [request.system for request in backend.requests] == ["planner system"] * 3
    assert [request.user for request in backend.requests] == [row["messages"][1]["content"] for row in rows]
    assert summary["models"]["model-a"]["total"] == 3
    assert summary["models"]["model-a"]["strict_pass"] == 2
    assert summary["models"]["model-a"]["strict_accuracy"] == 2 / 3
    assert summary["schema_version"] == 3
    assert summary["run"]["started_at"] == "2026-09-20T00:00:00+00:00"
    assert summary["run"]["wall_time_ms"] == 1300.0
    assert summary["run"]["wall_time_scope"] == "cli_start_through_results_serialization"
    assert summary["run"]["ttft"] == {
        "available": False,
        "reason": "non_streaming_planner_evaluation",
    }
    assert summary["run"]["config"]["concurrency"] == 2
    assert summary["dataset"]["cases"] == 3
    assert summary["dataset"]["package_sha256"] == "a" * 64
    model_summary = summary["models"]["model-a"]
    assert model_summary["quality"]["valid_json_rate"] == 2 / 3
    assert model_summary["quality"]["contract_valid_rate"] == 2 / 3
    assert model_summary["quality"]["step_kind_name"] == {
        "matched": 2,
        "expected": 3,
        "predicted": 2,
        "precision": 1.0,
        "recall": 2 / 3,
        "f1": 0.8,
    }
    assert model_summary["quality"]["forbidden_tool_violation_rate"] == 0.0
    assert model_summary["quality"]["failure_reasons"] == {"invalid_json": 1}
    assert model_summary["performance"] == {
        "batch_wall_time_ms": 500.0,
        "requests": {"total": 3, "succeeded": 3, "failed": 0, "per_second": 6.0},
        "latency_ms": {
            "min": 10.0,
            "mean": 10.0,
            "p50": 10.0,
            "p90": 10.0,
            "p95": 10.0,
            "p99": 10.0,
            "max": 10.0,
        },
        "tokens": {
            "coverage": {"count": 3, "rate": 1.0},
            "input": {"total": 60, "mean": 20.0},
            "output": {"total": 15, "mean": 5.0},
            "total": {"total": 75, "mean": 25.0},
        },
        "throughput": {
            "output_tokens_per_second": 30.0,
            "total_tokens_per_second": 150.0,
            "service_time_output_tokens_per_second": 500.0,
        },
    }
    assert model_summary["slices"]["level"]["1"]["quality"]["count"] == 2
    assert model_summary["slices"]["level"]["2"]["quality"]["count"] == 1
    assert model_summary["slices"]["language"]["en"]["quality"]["count"] == 1
    assert model_summary["slices"]["polarity"]["negative"]["quality"]["count"] == 1
    assert model_summary["slices"]["task_count"]["1"]["quality"]["count"] == 3
    assert model_summary["slices"]["kind"]["reply"]["quality"]["count"] == 2
    assert model_summary["slices"]["tool"]["RobotGesture"]["quality"]["count"] == 1
    assert model_summary["slices"]["action"]["wave_greet_bye"]["quality"]["count"] == 1
    records = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert records[2]["reason"] == "invalid_json"
    assert records[0]["timing"] == {"total_ms": 10.0}
    assert records[0]["tokens"] == {"input": 20, "output": 5, "total": 25}
    assert records[0]["metadata"]["source_atom_ids"] == ["atom-0"]
    assert records[0]["valid_json"] is True
    assert records[0]["contract_valid"] is True
    assert records[0]["predicted_step_count"] == 1
    assert json.loads((tmp_path / "out" / "summary.json").read_text()) == summary


def test_rescore_validation_reuses_raw_outputs_and_run_metadata(tmp_path: Path) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)
    cases = load_validation_cases(package_path, limit=2)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    records = [
        {
            "case_id": case.case_id,
            "model": "model-a",
            "output": rows[index]["messages"][2]["content"],
            "latency_ms": 10.0,
            "prompt_tokens": 20,
            "output_tokens": 5,
            "error": None,
        }
        for index, case in enumerate(cases)
    ]
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    original_summary = {
        "schema_version": 2,
        "cases": 2,
        "run": {"started_at": "2026-09-20T00:00:00+00:00", "wall_time_ms": 1234.0},
        "dataset": {"package_sha256": "a" * 64},
        "models": {"model-a": {"performance": {"batch_wall_time_ms": 500.0}}},
    }
    (output_dir / "summary.json").write_text(json.dumps(original_summary), encoding="utf-8")

    summary = rescore_validation(cases, output_dir)

    assert summary["run"] == original_summary["run"]
    assert summary["dataset"] == original_summary["dataset"]
    assert summary["models"]["model-a"]["performance"]["batch_wall_time_ms"] == 500.0
    assert summary["models"]["model-a"]["benchmark_compatible_pass"] == 2
    rescored = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]
    assert rescored[0]["benchmark_compatible_pass"] is True
    assert rescored[0]["metadata"]["source_atom_ids"] == ["atom-0"]


def test_cli_rescores_without_environment_or_backend(tmp_path: Path, monkeypatch) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "validation-0000",
                "model": "model-a",
                "output": rows[0]["messages"][2]["content"],
                "latency_ms": 10.0,
                "prompt_tokens": 20,
                "output_tokens": 5,
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"models": {"model-a": {"performance": {"batch_wall_time_ms": 10.0}}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "planner_val.validation_eval.OpenAICompatibleBackend",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("backend must not be created")),
    )

    exit_code = main(
        [
            "--package",
            str(package_path),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--rescore-results",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["models"]["model-a"]["benchmark_compatible_accuracy"] == 1.0


def test_cli_uses_cosa_key_for_qwen_and_vllm_endpoint(tmp_path: Path, monkeypatch) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)
    env_path = tmp_path / "env.local"
    env_path.write_text("DASHSCOPE_API_KEY='cosa-secret'\n", encoding="utf-8")
    created: list[dict[str, object]] = []

    class FakeOpenAIBackend(_FakeBackend):
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)
            super().__init__(
                {f"validation-{index:04d}": row["messages"][2]["content"] for index, row in enumerate(rows)}
            )

    monkeypatch.setattr("planner_val.validation_eval.OpenAICompatibleBackend", FakeOpenAIBackend)

    exit_code = main(
        [
            "--package",
            str(package_path),
            "--limit",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
            "--cosa-env",
            str(env_path),
            "--vllm-url",
            "http://127.0.0.1:8001/v1",
            "--vllm-model",
            "deployed-sft",
        ]
    )

    assert exit_code == 0
    assert created == [
        {
            "model": "deployed-sft",
            "base_url": "http://127.0.0.1:8001/v1",
            "api_key": "EMPTY",
            "concurrency": 2,
            "max_tokens": 1024,
        },
        {
            "model": "qwen-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "cosa-secret",
            "concurrency": 2,
            "max_tokens": 1024,
        },
    ]


def test_cli_qwen_only_does_not_create_vllm_backend(tmp_path: Path, monkeypatch) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)
    env_path = tmp_path / "env.local"
    env_path.write_text("DASHSCOPE_API_KEY='cosa-secret'\n", encoding="utf-8")
    created: list[dict[str, object]] = []

    class FakeOpenAIBackend(_FakeBackend):
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)
            super().__init__(
                {f"validation-{index:04d}": row["messages"][2]["content"] for index, row in enumerate(rows)}
            )

    monkeypatch.setattr("planner_val.validation_eval.OpenAICompatibleBackend", FakeOpenAIBackend)

    exit_code = main(
        [
            "--package",
            str(package_path),
            "--limit",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
            "--cosa-env",
            str(env_path),
            "--qwen-only",
        ]
    )

    assert exit_code == 0
    assert created == [
        {
            "model": "qwen-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "cosa-secret",
            "concurrency": 2,
            "max_tokens": 1024,
        }
    ]


def test_cli_qwen_production_profile_replaces_messages_and_records_provenance(tmp_path: Path, monkeypatch) -> None:
    package_path = tmp_path / "dataset.zip"
    rows = _write_package(package_path)
    env_path = tmp_path / "env.local"
    env_path.write_text("DASHSCOPE_API_KEY='cosa-secret'\n", encoding="utf-8")
    instances: list[_FakeBackend] = []

    class FakeOpenAIBackend(_FakeBackend):
        def __init__(self, **kwargs) -> None:
            super().__init__(
                {f"validation-{index:04d}": row["messages"][2]["content"] for index, row in enumerate(rows)}
            )
            instances.append(self)

    profile = ProductionPromptProfile(
        revision="3911119e607e1f7cd7baba92d67941c36871dead",
        system_prompt="production system",
        tools_context='{"tool":{"description":"ok"}}',
        user_tools_header="Tools",
        user_limit_header="Limit",
        user_instruction_header="Instruction",
        step_limit_template="At most {max_steps} tasks",
        max_steps=12,
        prompt_source_sha256="a" * 64,
        planner_source_sha256="b" * 64,
        descriptions_source_sha256="c" * 64,
    )
    monkeypatch.setattr("planner_val.validation_eval.OpenAICompatibleBackend", FakeOpenAIBackend)
    monkeypatch.setattr("planner_val.validation_eval.load_benchmark_20260913_profile", lambda _: profile)

    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "--package",
            str(package_path),
            "--limit",
            "2",
            "--output-dir",
            str(output_dir),
            "--cosa-env",
            str(env_path),
            "--qwen-only",
            "--qwen-production-profile",
            "--production-agent-repo",
            "/agent",
        ]
    )

    assert exit_code == 0
    assert len(instances) == 1
    assert [request.system for request in instances[0].requests] == ["production system"] * 2
    assert [request.user for request in instances[0].requests] == [
        'Tools\n{"tool":{"description":"ok"}}\n\nLimit\nAt most 12 tasks\n\nInstruction\nquery-0',
        'Tools\n{"tool":{"description":"ok"}}\n\nLimit\nAt most 12 tasks\n\nInstruction\nquery-1',
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["models"]) == ["qwen-flash-production-3911119"]
    assert summary["prompt_profile"]["name"] == profile.metadata["name"]
    assert summary["prompt_profile"]["agent_revision"] == profile.revision
    assert summary["prompt_profile"]["artifact"]["path"] == "prompt_profile.json"
    assert len(summary["prompt_profile"]["artifact"]["sha256"]) == 64
    prompt_artifact = json.loads((output_dir / "prompt_profile.json").read_text(encoding="utf-8"))
    assert prompt_artifact["system_prompt"] == "production system"
    assert prompt_artifact["tools_context"] == '{"tool":{"description":"ok"}}'
    assert prompt_artifact["user_prompt_template"].endswith("Instruction\n{query}")
    assert summary["run"]["config"] == {
        "concurrency": 2,
        "max_tokens": 1024,
        "models": {
            "qwen-flash-production-3911119": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-flash",
            }
        },
        "qwen_only": True,
        "stream": False,
        "temperature": 0,
    }
    assert summary["dataset"]["package"] == str(package_path)
    assert len(summary["dataset"]["package_sha256"]) == 64
