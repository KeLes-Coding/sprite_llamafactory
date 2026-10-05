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
from pathlib import Path
from typing import Any

import pytest

from planner_val.reporting import write_reports


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest() -> dict[str, Any]:
    return {
        "identity": {
            "schema_version": 1,
            "case_ids": ["case-1", "case-2", "case-3", "case-4"],
            "contract_fingerprints": ["fp-alpha", "fp-beta"],
            "variants": ["base", "qwen_flash", "sft"],
            "model_ids": {"base": "base-model", "qwen_flash": "qwen-model", "sft": "sft-model"},
            "max_tokens": 256,
            "seed": 9,
            "sample_size": None,
            "dataset_fingerprint": "dataset-fp",
            "training_package_fingerprint": "train-fp",
        },
        "contracts": [
            {
                "name": "alpha",
                "fingerprint": "fp-alpha",
                "max_steps": 4,
                "system_prompt": "system alpha",
                "tools_context": [],
                "tool_schemas": {},
                "tool_kinds": {},
                "message_config": {},
                "source": {"path": "alpha.json"},
            },
            {
                "name": "alpha",
                "fingerprint": "fp-beta",
                "max_steps": 4,
                "system_prompt": "system beta",
                "tools_context": [],
                "tool_schemas": {},
                "tool_kinds": {},
                "message_config": {},
                "source": {"path": "beta.json"},
            },
        ],
        "created_at": "2026-09-19T00:00:00Z",
        "environment": {"platform": "linux", "python": "3.12.0", "pid": 100},
    }


def _result_record(
    *,
    record_id: str,
    contract_name: str,
    contract_fingerprint: str,
    variant: str,
    case_id: str,
    exact_train: bool,
    all_atoms_seen_train: bool,
    has_novel_atom: bool,
    expected: list[dict[str, Any]],
    strict_pass: bool,
    valid_json: bool,
    contract_valid: bool,
    task_count_match: bool,
    ordered_kind_name_match: bool,
    parameter_match: bool | None,
    forbidden_tool_violation: bool,
    latency_ms: float,
    prompt_tokens: int | None,
    output_tokens: int | None,
    reason: str,
    raw_output: str | None,
    language: str = "zh",
    level: str = "L1",
    difficulty: str = "easy",
    polarity: str = "positive",
    forbidden_functions: list[str] | None = None,
    parsed_tasks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "contract": {
            "name": contract_name,
            "fingerprint": contract_fingerprint,
            "max_steps": 4,
            "system_prompt": f"system {contract_name}",
            "tools_context": [],
            "tool_schemas": {},
            "tool_kinds": {},
            "message_config": {},
            "source": {"path": f"{contract_fingerprint}.json"},
        },
        "case": {
            "case_id": case_id,
            "query": f"query {case_id}",
            "expected": expected,
            "forbidden_functions": forbidden_functions or [],
            "level": level,
            "language": language,
            "difficulty": difficulty,
            "polarity": polarity,
            "source_atom_ids": [f"atom-{case_id}", "shared"],
            "overlap": {
                "exact_train": exact_train,
                "exact_validation": False,
                "all_atoms_seen_train": all_atoms_seen_train,
                "has_novel_atom": has_novel_atom,
            },
        },
        "request": {"system": "system", "user": f"user {case_id}"},
        "variant": variant,
        "model_id": f"{variant}-model",
        "max_tokens": 256,
        "seed": 9,
        "sample_size": None,
        "generation": {
            "request_id": record_id,
            "text": raw_output,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "error": None,
        },
        "score": {
            "reason": reason,
            "strict_pass": strict_pass,
            "valid_json": valid_json,
            "contract_valid": contract_valid,
            "task_count_match": task_count_match,
            "ordered_kind_name_match": ordered_kind_name_match,
            "parameter_match": parameter_match,
            "forbidden_tool_violation": forbidden_tool_violation,
            "parsed_tasks": parsed_tasks or expected,
            "parameter_errors": [] if parameter_match is not False else ["arg mismatch"],
            "forbidden_calls": ["danger"] if forbidden_tool_violation else [],
            "raw_output": raw_output,
        },
    }


def _alpha_records() -> list[dict[str, Any]]:
    reply_only = [{"kind": "reply", "instruction": "done", "name": None, "arguments": None}]
    tool_repeat = [
        {"kind": "tool", "instruction": None, "name": "search", "arguments": {"q": "x"}},
        {"kind": "tool", "instruction": None, "name": "search", "arguments": {"q": "y"}},
        {"kind": "reply", "instruction": "done", "name": None, "arguments": None},
    ]
    mixed_tools = [
        {"kind": "tool", "instruction": None, "name": "calc", "arguments": {"expr": "1+1"}},
        {"kind": "tool", "instruction": None, "name": "search", "arguments": {"q": "z"}},
    ]
    return [
        _result_record(
            record_id="alpha-base-1",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="base",
            case_id="case-1",
            exact_train=True,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=reply_only,
            strict_pass=True,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=None,
            forbidden_tool_violation=False,
            latency_ms=100.0,
            prompt_tokens=10,
            output_tokens=5,
            reason="pass",
            raw_output='[{"kind":"reply","instruction":"done"}]',
        ),
        _result_record(
            record_id="alpha-sft-1",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="sft",
            case_id="case-1",
            exact_train=True,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=reply_only,
            strict_pass=True,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=None,
            forbidden_tool_violation=False,
            latency_ms=80.0,
            prompt_tokens=9,
            output_tokens=6,
            reason="pass",
            raw_output='[{"kind":"reply","instruction":"done"}]',
        ),
        _result_record(
            record_id="alpha-qwen-1",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="qwen_flash",
            case_id="case-1",
            exact_train=True,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=reply_only,
            strict_pass=False,
            valid_json=False,
            contract_valid=False,
            task_count_match=False,
            ordered_kind_name_match=False,
            parameter_match=None,
            forbidden_tool_violation=False,
            latency_ms=120.0,
            prompt_tokens=8,
            output_tokens=4,
            reason="malformed_output",
            raw_output="not-json",
        ),
        _result_record(
            record_id="alpha-base-2",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="base",
            case_id="case-2",
            exact_train=False,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=tool_repeat,
            strict_pass=False,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=False,
            forbidden_tool_violation=True,
            latency_ms=200.0,
            prompt_tokens=12,
            output_tokens=7,
            reason="parameter_mismatch",
            raw_output='[{"kind":"tool","name":"search"}]',
            language="en",
            level="L2",
            difficulty="medium",
            polarity="negative",
            forbidden_functions=["danger"],
            parsed_tasks=tool_repeat,
        ),
        _result_record(
            record_id="alpha-sft-2",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="sft",
            case_id="case-2",
            exact_train=False,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=tool_repeat,
            strict_pass=True,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=True,
            forbidden_tool_violation=False,
            latency_ms=150.0,
            prompt_tokens=11,
            output_tokens=8,
            reason="pass",
            raw_output='[{"kind":"tool","name":"search"}]',
            language="en",
            level="L2",
            difficulty="medium",
            polarity="negative",
            forbidden_functions=["danger"],
            parsed_tasks=tool_repeat,
        ),
        _result_record(
            record_id="alpha-qwen-2",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="qwen_flash",
            case_id="case-2",
            exact_train=False,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=tool_repeat,
            strict_pass=False,
            valid_json=True,
            contract_valid=True,
            task_count_match=False,
            ordered_kind_name_match=False,
            parameter_match=False,
            forbidden_tool_violation=False,
            latency_ms=300.0,
            prompt_tokens=10,
            output_tokens=3,
            reason="wrong_order",
            raw_output='[{"kind":"tool","name":"search"}]',
            language="en",
            level="L2",
            difficulty="medium",
            polarity="negative",
            forbidden_functions=["danger"],
            parsed_tasks=tool_repeat,
        ),
        _result_record(
            record_id="alpha-base-3",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="base",
            case_id="case-3",
            exact_train=False,
            all_atoms_seen_train=False,
            has_novel_atom=True,
            expected=mixed_tools,
            strict_pass=False,
            valid_json=True,
            contract_valid=False,
            task_count_match=False,
            ordered_kind_name_match=False,
            parameter_match=True,
            forbidden_tool_violation=False,
            latency_ms=400.0,
            prompt_tokens=15,
            output_tokens=9,
            reason="contract_violation",
            raw_output='[{"kind":"tool","name":"calc"}]',
            language="zh",
            level="L3",
            difficulty="hard",
            polarity="positive",
            parsed_tasks=mixed_tools,
        ),
        _result_record(
            record_id="alpha-sft-3",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="sft",
            case_id="case-3",
            exact_train=False,
            all_atoms_seen_train=False,
            has_novel_atom=True,
            expected=mixed_tools,
            strict_pass=False,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=True,
            forbidden_tool_violation=False,
            latency_ms=100.0,
            prompt_tokens=14,
            output_tokens=10,
            reason="runtime_error",
            raw_output='[{"kind":"tool","name":"calc"}]',
            language="zh",
            level="L3",
            difficulty="hard",
            polarity="positive",
            parsed_tasks=mixed_tools,
        ),
        _result_record(
            record_id="alpha-qwen-3",
            contract_name="alpha",
            contract_fingerprint="fp-alpha",
            variant="qwen_flash",
            case_id="case-3",
            exact_train=False,
            all_atoms_seen_train=False,
            has_novel_atom=True,
            expected=mixed_tools,
            strict_pass=True,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=True,
            forbidden_tool_violation=False,
            latency_ms=90.0,
            prompt_tokens=13,
            output_tokens=11,
            reason="pass",
            raw_output='[{"kind":"tool","name":"calc"}]',
            language="zh",
            level="L3",
            difficulty="hard",
            polarity="positive",
            parsed_tasks=mixed_tools,
        ),
    ]


def _beta_records() -> list[dict[str, Any]]:
    reply_only = [{"kind": "reply", "instruction": "done", "name": None, "arguments": None}]
    return [
        _result_record(
            record_id="beta-base-4",
            contract_name="alpha",
            contract_fingerprint="fp-beta",
            variant="base",
            case_id="case-4",
            exact_train=False,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=reply_only,
            strict_pass=True,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=None,
            forbidden_tool_violation=False,
            latency_ms=50.0,
            prompt_tokens=5,
            output_tokens=2,
            reason="pass",
            raw_output='[{"kind":"reply","instruction":"done"}]',
        ),
        _result_record(
            record_id="beta-sft-4",
            contract_name="alpha",
            contract_fingerprint="fp-beta",
            variant="sft",
            case_id="case-4",
            exact_train=False,
            all_atoms_seen_train=True,
            has_novel_atom=False,
            expected=reply_only,
            strict_pass=False,
            valid_json=True,
            contract_valid=True,
            task_count_match=True,
            ordered_kind_name_match=True,
            parameter_match=None,
            forbidden_tool_violation=False,
            latency_ms=60.0,
            prompt_tokens=6,
            output_tokens=1,
            reason="timeout",
            raw_output='[{"kind":"reply","instruction":"done"}]',
        ),
    ]


def _prepare_run_dir(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "manifest.json", _manifest())
    _write_jsonl(run_dir / "results.jsonl", records)
    return run_dir


def test_write_reports_aggregates_and_pairs_exactly(tmp_path: Path) -> None:
    run_dir = _prepare_run_dir(tmp_path, _alpha_records() + _beta_records())

    summary_json_path, summary_md_path = write_reports(run_dir, bootstrap_samples=50, bootstrap_seed=123)

    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary_md_path.name == "summary.md"
    assert summary["schema_version"] == 1
    assert summary["bootstrap"] == {"samples": 50, "seed": 123}

    contracts = summary["contracts"]
    assert [contract["contract_fingerprint"] for contract in contracts] == ["fp-alpha", "fp-beta"]

    alpha_contract = contracts[0]
    base_all = alpha_contract["variants"]["base"]["all"]
    assert base_all["count"] == 3
    assert base_all["strict_sequence_accuracy"] == pytest.approx(1 / 3)
    assert base_all["valid_json_rate"] == 1.0
    assert base_all["contract_valid_rate"] == pytest.approx(2 / 3)
    assert base_all["task_count_accuracy"] == pytest.approx(2 / 3)
    assert base_all["ordered_kind_name_accuracy"] == pytest.approx(2 / 3)
    assert base_all["parameter_accuracy"] == pytest.approx(0.5)
    assert base_all["parameter_accuracy_denominator"] == 2
    assert base_all["forbidden_tool_violation_rate"] == pytest.approx(1 / 3)
    assert base_all["latency_ms"] == {"mean": pytest.approx(700 / 3), "p50": 200.0, "p95": 400.0}
    assert base_all["tokens"] == {"prompt_total": 37, "output_total": 21}
    assert base_all["output_tokens_per_second"] == pytest.approx(30.0)
    assert base_all["failure_reasons"] == {"contract_violation": 1, "parameter_mismatch": 1}

    composition_slice = alpha_contract["variants"]["base"]["composition_seen_atoms"]
    assert composition_slice["count"] == 1
    assert composition_slice["strict_sequence_accuracy"] == 0.0

    novel_slice = alpha_contract["variants"]["qwen_flash"]["has_novel_atom"]
    assert novel_slice["count"] == 1
    assert novel_slice["strict_sequence_accuracy"] == 1.0

    by_expected_tool = alpha_contract["variants"]["base"]["by_expected_tool"]
    assert sorted(by_expected_tool) == ["calc", "reply", "search"]
    assert by_expected_tool["reply"]["count"] == 1
    assert by_expected_tool["search"]["count"] == 2
    assert by_expected_tool["calc"]["count"] == 1

    by_language = alpha_contract["variants"]["base"]["by_language"]
    assert sorted(by_language) == ["en", "zh"]
    assert by_language["en"]["count"] == 1
    assert by_language["zh"]["count"] == 2

    sft_vs_base = alpha_contract["pairwise"]["sft_vs_base"]["all"]
    assert sft_vs_base["paired_count"] == 3
    assert sft_vs_base["both_pass"] == 1
    assert sft_vs_base["left_only_pass"] == 1
    assert sft_vs_base["right_only_pass"] == 0
    assert sft_vs_base["both_fail"] == 1
    assert sft_vs_base["left_accuracy"] == pytest.approx(2 / 3)
    assert sft_vs_base["right_accuracy"] == pytest.approx(1 / 3)
    assert sft_vs_base["accuracy_delta"] == pytest.approx(1 / 3)
    assert sft_vs_base["accuracy_delta_ci95"][0] <= sft_vs_base["accuracy_delta"]
    assert sft_vs_base["accuracy_delta_ci95"][1] >= sft_vs_base["accuracy_delta"]

    base_vs_qwen = alpha_contract["pairwise"]["base_vs_qwen_flash"]["all"]
    assert base_vs_qwen["paired_count"] == 3
    assert base_vs_qwen["both_pass"] == 0
    assert base_vs_qwen["left_only_pass"] == 1
    assert base_vs_qwen["right_only_pass"] == 1
    assert base_vs_qwen["both_fail"] == 1
    assert base_vs_qwen["accuracy_delta_ci95"][0] <= base_vs_qwen["accuracy_delta"]
    assert base_vs_qwen["accuracy_delta_ci95"][1] >= base_vs_qwen["accuracy_delta"]

    qwen_vs_sft = alpha_contract["pairwise"]["sft_vs_qwen_flash"]["all"]
    assert qwen_vs_sft["paired_count"] == 3
    assert qwen_vs_sft["left_only_pass"] == 2
    assert qwen_vs_sft["right_only_pass"] == 1

    beta_contract = contracts[1]
    assert beta_contract["contract_name"] == "alpha"
    assert beta_contract["contract_fingerprint"] == "fp-beta"
    beta_pair = beta_contract["pairwise"]["sft_vs_base"]["all"]
    assert beta_pair["paired_count"] == 1
    assert beta_pair["accuracy_delta"] == -1.0
    assert beta_pair["accuracy_delta_ci95"] == [-1.0, -1.0]

    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "Contamination caveat" in markdown
    assert "fp-alpha" in markdown
    assert "fp-beta" in markdown


def test_parameter_accuracy_excludes_null_values(tmp_path: Path) -> None:
    run_dir = _prepare_run_dir(tmp_path, _alpha_records())

    summary_json_path, _ = write_reports(run_dir)

    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    sft_all = summary["contracts"][0]["variants"]["sft"]["all"]
    assert sft_all["parameter_accuracy"] == 1.0
    assert sft_all["parameter_accuracy_denominator"] == 2


def test_failures_preserve_original_records_and_output(tmp_path: Path) -> None:
    records = _alpha_records()
    run_dir = _prepare_run_dir(tmp_path, records)

    write_reports(run_dir)

    failures_path = run_dir / "failures.jsonl"
    failure_lines = failures_path.read_text(encoding="utf-8").splitlines()
    expected = [record for record in records if not record["score"]["strict_pass"]]
    assert len(failure_lines) == len(expected)
    assert [json.loads(line) for line in failure_lines] == expected
    assert json.loads(failure_lines[0])["score"]["raw_output"] == "not-json"


def test_bootstrap_edge_cases_are_exact(tmp_path: Path) -> None:
    run_dir = _prepare_run_dir(tmp_path, _beta_records())

    summary_json_path, _ = write_reports(run_dir, bootstrap_samples=7, bootstrap_seed=5)

    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    beta_pair = summary["contracts"][0]["pairwise"]["sft_vs_base"]["all"]
    assert beta_pair["paired_count"] == 1
    assert beta_pair["accuracy_delta_ci95"] == [-1.0, -1.0]
    no_pairs = summary["contracts"][0]["pairwise"]["base_vs_qwen_flash"]["all"]
    assert no_pairs["paired_count"] == 0
    assert no_pairs["left_accuracy"] is None
    assert no_pairs["accuracy_delta_ci95"] is None


def test_invalid_inputs_fail_descriptively(tmp_path: Path) -> None:
    valid_run_dir = _prepare_run_dir(tmp_path, _alpha_records())

    broken_manifest_dir = tmp_path / "broken-manifest"
    broken_manifest_dir.mkdir()
    _write_jsonl(broken_manifest_dir / "results.jsonl", _alpha_records())
    with pytest.raises(ValueError, match="manifest"):
        write_reports(broken_manifest_dir)

    empty_results_dir = tmp_path / "empty-results"
    empty_results_dir.mkdir()
    _write_json(empty_results_dir / "manifest.json", _manifest())
    (empty_results_dir / "results.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        write_reports(empty_results_dir)

    duplicate_id_dir = tmp_path / "duplicate-id"
    duplicate_id_dir.mkdir()
    _write_json(duplicate_id_dir / "manifest.json", _manifest())
    duplicate_records = _alpha_records()
    duplicate_records.append(duplicate_records[0])
    _write_jsonl(duplicate_id_dir / "results.jsonl", duplicate_records)
    with pytest.raises(ValueError, match="duplicate record_id"):
        write_reports(duplicate_id_dir)

    negative_latency_dir = tmp_path / "negative-latency"
    negative_latency_dir.mkdir()
    _write_json(negative_latency_dir / "manifest.json", _manifest())
    negative_records = _alpha_records()
    negative_records[0] = {
        **negative_records[0],
        "generation": {**negative_records[0]["generation"], "latency_ms": -1.0},
    }
    _write_jsonl(negative_latency_dir / "results.jsonl", negative_records)
    with pytest.raises(ValueError, match="latency_ms"):
        write_reports(negative_latency_dir)

    duplicate_key_dir = tmp_path / "duplicate-key"
    duplicate_key_dir.mkdir()
    _write_json(duplicate_key_dir / "manifest.json", _manifest())
    duplicate_key_records = _alpha_records()
    duplicate_key_records.append(
        {
            **duplicate_key_records[0],
            "record_id": "alpha-base-1b",
            "generation": {**duplicate_key_records[0]["generation"], "request_id": "alpha-base-1b"},
        }
    )
    _write_jsonl(duplicate_key_dir / "results.jsonl", duplicate_key_records)
    with pytest.raises(ValueError, match="duplicate case key"):
        write_reports(duplicate_key_dir)

    malformed_dir = tmp_path / "malformed"
    malformed_dir.mkdir()
    _write_json(malformed_dir / "manifest.json", _manifest())
    (malformed_dir / "results.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        write_reports(malformed_dir)

    with pytest.raises(ValueError, match="bootstrap_samples"):
        write_reports(valid_run_dir, bootstrap_samples=0)


def test_second_write_is_byte_identical(tmp_path: Path) -> None:
    run_dir = _prepare_run_dir(tmp_path, _alpha_records() + _beta_records())

    first_json_path, first_md_path = write_reports(run_dir, bootstrap_samples=25, bootstrap_seed=99)
    first_json = first_json_path.read_bytes()
    first_md = first_md_path.read_bytes()
    first_failures = (run_dir / "failures.jsonl").read_bytes()

    second_json_path, second_md_path = write_reports(run_dir, bootstrap_samples=25, bootstrap_seed=99)
    assert second_json_path.read_bytes() == first_json
    assert second_md_path.read_bytes() == first_md
    assert (run_dir / "failures.jsonl").read_bytes() == first_failures
