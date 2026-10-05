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
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MANIFEST_NAME = "manifest.json"
_RESULTS_NAME = "results.jsonl"
_SUMMARY_JSON_NAME = "summary.json"
_SUMMARY_MD_NAME = "summary.md"
_FAILURES_NAME = "failures.jsonl"
_REPORT_SCHEMA_VERSION = 1
_PAIRWISE_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("sft_vs_base", "sft", "base"),
    ("sft_vs_qwen_flash", "sft", "qwen_flash"),
    ("base_vs_qwen_flash", "base", "qwen_flash"),
)
_SLICE_ORDER: tuple[str, ...] = (
    "all",
    "exact_train",
    "non_exact_train",
    "composition_seen_atoms",
    "has_novel_atom",
)


@dataclass(frozen=True)
class _NormalizedRecord:
    payload: dict[str, Any]
    contract_name: str
    contract_fingerprint: str
    variant: str
    case_id: str
    expected_tool_names: tuple[str, ...]
    intent_count: int
    level: str
    language: str
    difficulty: str
    polarity: str
    exact_train: bool
    all_atoms_seen_train: bool
    has_novel_atom: bool
    strict_pass: bool
    valid_json: bool
    contract_valid: bool
    task_count_match: bool
    ordered_kind_name_match: bool
    parameter_match: bool | None
    forbidden_tool_violation: bool
    latency_ms: float
    prompt_tokens: int
    output_tokens: int
    reason: str


def _canonical_json(value: Any, *, compact: bool) -> str:
    separators = (",", ":") if compact else (",", ": ")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=separators, allow_nan=False)


def _read_json(path: Path, *, missing_message: str, malformed_message: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(missing_message) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(malformed_message) from exc

    if not isinstance(payload, dict):
        raise ValueError(malformed_message)
    return payload


def _require_mapping(container: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: missing or invalid {key}")
    return value


def _require_non_empty_string(container: Mapping[str, Any], key: str, *, context: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: missing or invalid {key}")
    return value


def _require_bool(container: Mapping[str, Any], key: str, *, context: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}: missing or invalid {key}")
    return value


def _require_list(container: Mapping[str, Any], key: str, *, context: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context}: missing or invalid {key}")
    return value


def _require_non_negative_number(container: Mapping[str, Any], key: str, *, context: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: missing or invalid {key}")
    number = float(value)
    if number < 0:
        raise ValueError(f"{context}: {key} must be non-negative")
    return number


def _require_non_negative_int_or_none(container: Mapping[str, Any], key: str, *, context: str) -> int:
    value = container.get(key)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: missing or invalid {key}")
    if value < 0:
        raise ValueError(f"{context}: {key} must be non-negative")
    return value


def _validate_manifest(payload: Mapping[str, Any]) -> None:
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("manifest is malformed")
    contracts = payload.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("manifest is malformed")


def _slice_membership(record: _NormalizedRecord, slice_name: str) -> bool:
    if slice_name == "all":
        return True
    if slice_name == "exact_train":
        return record.exact_train
    if slice_name == "non_exact_train":
        return not record.exact_train
    if slice_name == "composition_seen_atoms":
        return (not record.exact_train) and record.all_atoms_seen_train
    if slice_name == "has_novel_atom":
        return record.has_novel_atom
    raise KeyError(slice_name)


def _expected_tool_names(expected: Sequence[Any], *, context: str) -> tuple[str, ...]:
    tool_names: set[str] = set()
    for index, item in enumerate(expected):
        if not isinstance(item, Mapping):
            raise ValueError(f"{context}: invalid expected[{index}]")
        name = item.get("name")
        if name is None:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context}: invalid expected[{index}].name")
        tool_names.add(name)
    if not tool_names:
        tool_names.add("reply")
    return tuple(sorted(tool_names))


def _parse_record(payload: dict[str, Any], *, line_number: int) -> _NormalizedRecord:
    context = f"results line {line_number}"
    record_id = _require_non_empty_string(payload, "record_id", context=context)
    contract = _require_mapping(payload, "contract", context=context)
    case = _require_mapping(payload, "case", context=context)
    generation = _require_mapping(payload, "generation", context=context)
    score = _require_mapping(payload, "score", context=context)

    contract_name = _require_non_empty_string(contract, "name", context=context)
    contract_fingerprint = _require_non_empty_string(contract, "fingerprint", context=context)
    variant = _require_non_empty_string(payload, "variant", context=context)
    case_id = _require_non_empty_string(case, "case_id", context=context)

    overlap = _require_mapping(case, "overlap", context=context)
    expected = _require_list(case, "expected", context=context)
    expected_tool_names = _expected_tool_names(expected, context=context)

    request_id = _require_non_empty_string(generation, "request_id", context=context)
    if request_id != record_id:
        raise ValueError(f"{context}: generation.request_id must match record_id")

    parameter_match = score.get("parameter_match")
    if parameter_match is not None and not isinstance(parameter_match, bool):
        raise ValueError(f"{context}: invalid parameter_match")

    return _NormalizedRecord(
        payload=payload,
        contract_name=contract_name,
        contract_fingerprint=contract_fingerprint,
        variant=variant,
        case_id=case_id,
        expected_tool_names=expected_tool_names,
        intent_count=len(expected),
        level=_require_non_empty_string(case, "level", context=context),
        language=_require_non_empty_string(case, "language", context=context),
        difficulty=_require_non_empty_string(case, "difficulty", context=context),
        polarity=_require_non_empty_string(case, "polarity", context=context),
        exact_train=_require_bool(overlap, "exact_train", context=context),
        all_atoms_seen_train=_require_bool(overlap, "all_atoms_seen_train", context=context),
        has_novel_atom=_require_bool(overlap, "has_novel_atom", context=context),
        strict_pass=_require_bool(score, "strict_pass", context=context),
        valid_json=_require_bool(score, "valid_json", context=context),
        contract_valid=_require_bool(score, "contract_valid", context=context),
        task_count_match=_require_bool(score, "task_count_match", context=context),
        ordered_kind_name_match=_require_bool(score, "ordered_kind_name_match", context=context),
        parameter_match=parameter_match,
        forbidden_tool_violation=_require_bool(score, "forbidden_tool_violation", context=context),
        latency_ms=_require_non_negative_number(generation, "latency_ms", context=context),
        prompt_tokens=_require_non_negative_int_or_none(generation, "prompt_tokens", context=context),
        output_tokens=_require_non_negative_int_or_none(generation, "output_tokens", context=context),
        reason=_require_non_empty_string(score, "reason", context=context),
    )


def _load_records(run_dir: Path) -> list[_NormalizedRecord]:
    manifest = _read_json(
        run_dir / _MANIFEST_NAME,
        missing_message="manifest is missing",
        malformed_message="manifest is malformed",
    )
    _validate_manifest(manifest)

    results_path = run_dir / _RESULTS_NAME
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError("results.jsonl is missing") from exc

    if not lines:
        raise ValueError("results.jsonl must be non-empty")

    records: list[_NormalizedRecord] = []
    seen_record_ids: set[str] = set()
    seen_case_keys: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"results line {line_number}: malformed JSON")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"results line {line_number}: malformed JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"results line {line_number}: malformed JSON")

        record = _parse_record(payload, line_number=line_number)
        record_id = payload["record_id"]
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        case_key = (record.contract_fingerprint, record.variant, record.case_id)
        if case_key in seen_case_keys:
            raise ValueError(
                f"duplicate case key: ({record.contract_fingerprint}, {record.variant}, {record.case_id})"
            )
        seen_case_keys.add(case_key)
        records.append(record)

    return records


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * (len(ordered) + 1)) - 1))
    return ordered[index]


def _aggregate_metrics(records: Sequence[_NormalizedRecord]) -> dict[str, Any]:
    count = len(records)
    strict_passes = sum(1 for record in records if record.strict_pass)
    valid_json_passes = sum(1 for record in records if record.valid_json)
    contract_valid_passes = sum(1 for record in records if record.contract_valid)
    task_count_passes = sum(1 for record in records if record.task_count_match)
    ordered_passes = sum(1 for record in records if record.ordered_kind_name_match)
    forbidden_violations = sum(1 for record in records if record.forbidden_tool_violation)

    parameter_values = [record.parameter_match for record in records if record.parameter_match is not None]
    latency_values = [record.latency_ms for record in records]
    latency_total_ms = sum(latency_values)
    prompt_total = sum(record.prompt_tokens for record in records)
    output_total = sum(record.output_tokens for record in records)
    failure_reasons = Counter(record.reason for record in records if not record.strict_pass)

    return {
        "count": count,
        "strict_sequence_accuracy": _rate(strict_passes, count),
        "valid_json_rate": _rate(valid_json_passes, count),
        "contract_valid_rate": _rate(contract_valid_passes, count),
        "task_count_accuracy": _rate(task_count_passes, count),
        "ordered_kind_name_accuracy": _rate(ordered_passes, count),
        "parameter_accuracy": _rate(sum(1 for value in parameter_values if value), len(parameter_values)),
        "parameter_accuracy_denominator": len(parameter_values),
        "forbidden_tool_violation_rate": _rate(forbidden_violations, count),
        "latency_ms": {
            "mean": _rate(int(latency_total_ms), count)
            if float(latency_total_ms).is_integer()
            else latency_total_ms / count
            if count
            else None,
            "p50": _nearest_rank(latency_values, 0.50) if latency_values else None,
            "p95": _nearest_rank(latency_values, 0.95) if latency_values else None,
        },
        "tokens": {"prompt_total": prompt_total, "output_total": output_total},
        "output_tokens_per_second": None if latency_total_ms <= 0 else output_total / (latency_total_ms / 1000.0),
        "failure_reasons": dict(sorted(failure_reasons.items())),
    }


def _slice_summary(records: Sequence[_NormalizedRecord]) -> dict[str, Any]:
    return {
        slice_name: _aggregate_metrics([record for record in records if _slice_membership(record, slice_name)])
        for slice_name in _SLICE_ORDER
    }


def _grouped_metrics(records: Sequence[_NormalizedRecord], key_fn: Any) -> dict[str, Any]:
    grouped: dict[str, list[_NormalizedRecord]] = defaultdict(list)
    for record in records:
        for key in key_fn(record):
            grouped[str(key)].append(record)
    return {key: _aggregate_metrics(grouped[key]) for key in sorted(grouped)}


def _bootstrap_ci(differences: Sequence[int], *, samples: int, seed: int) -> list[float] | None:
    if not differences:
        return None
    observed = sum(differences) / len(differences)
    if len(differences) == 1 or all(value == differences[0] for value in differences):
        return [observed, observed]
    generator = random.Random(seed)
    means: list[float] = []
    max_index = len(differences) - 1
    for _ in range(samples):
        total = 0
        for _ in differences:
            total += differences[generator.randint(0, max_index)]
        means.append(total / len(differences))
    return [_nearest_rank(means, 0.025), _nearest_rank(means, 0.975)]


def _pairwise_summary(
    left_records: Mapping[str, _NormalizedRecord],
    right_records: Mapping[str, _NormalizedRecord],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    overlap_case_ids = sorted(set(left_records).intersection(right_records))
    if not overlap_case_ids:
        return {
            "paired_count": 0,
            "both_pass": 0,
            "left_only_pass": 0,
            "right_only_pass": 0,
            "both_fail": 0,
            "left_accuracy": None,
            "right_accuracy": None,
            "accuracy_delta": None,
            "accuracy_delta_ci95": None,
        }

    left_passes = 0
    right_passes = 0
    both_pass = 0
    left_only_pass = 0
    right_only_pass = 0
    both_fail = 0
    differences: list[int] = []
    for case_id in overlap_case_ids:
        left_pass = left_records[case_id].strict_pass
        right_pass = right_records[case_id].strict_pass
        left_passes += int(left_pass)
        right_passes += int(right_pass)
        differences.append(int(left_pass) - int(right_pass))
        if left_pass and right_pass:
            both_pass += 1
        elif left_pass:
            left_only_pass += 1
        elif right_pass:
            right_only_pass += 1
        else:
            both_fail += 1

    paired_count = len(overlap_case_ids)
    return {
        "paired_count": paired_count,
        "both_pass": both_pass,
        "left_only_pass": left_only_pass,
        "right_only_pass": right_only_pass,
        "both_fail": both_fail,
        "left_accuracy": left_passes / paired_count,
        "right_accuracy": right_passes / paired_count,
        "accuracy_delta": sum(differences) / paired_count,
        "accuracy_delta_ci95": _bootstrap_ci(differences, samples=bootstrap_samples, seed=bootstrap_seed),
    }


def _build_summary(
    records: Sequence[_NormalizedRecord],
    manifest: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped_contracts: dict[tuple[str, str], list[_NormalizedRecord]] = defaultdict(list)
    for record in records:
        grouped_contracts[(record.contract_name, record.contract_fingerprint)].append(record)

    contracts_summary: list[dict[str, Any]] = []
    for contract_name, contract_fingerprint in sorted(grouped_contracts):
        contract_records = grouped_contracts[(contract_name, contract_fingerprint)]
        variant_records: dict[str, list[_NormalizedRecord]] = defaultdict(list)
        for record in contract_records:
            variant_records[record.variant].append(record)

        variants_summary: dict[str, Any] = {}
        for variant in sorted(variant_records):
            current_records = sorted(variant_records[variant], key=lambda item: item.case_id)
            variants_summary[variant] = {
                **_slice_summary(current_records),
                "by_difficulty": _grouped_metrics(current_records, lambda record: [record.difficulty]),
                "by_expected_tool": _grouped_metrics(current_records, lambda record: list(record.expected_tool_names)),
                "by_intent_count": _grouped_metrics(current_records, lambda record: [record.intent_count]),
                "by_language": _grouped_metrics(current_records, lambda record: [record.language]),
                "by_level": _grouped_metrics(current_records, lambda record: [record.level]),
                "by_polarity": _grouped_metrics(current_records, lambda record: [record.polarity]),
            }

        pairwise_summary: dict[str, Any] = {}
        for label, left_variant, right_variant in _PAIRWISE_VARIANTS:
            left = {record.case_id: record for record in variant_records.get(left_variant, [])}
            right = {record.case_id: record for record in variant_records.get(right_variant, [])}
            pairwise_summary[label] = {}
            for slice_name in _SLICE_ORDER:
                left_slice = {
                    case_id: record for case_id, record in left.items() if _slice_membership(record, slice_name)
                }
                right_slice = {
                    case_id: record for case_id, record in right.items() if _slice_membership(record, slice_name)
                }
                pairwise_summary[label][slice_name] = _pairwise_summary(
                    left_slice,
                    right_slice,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )

        contracts_summary.append(
            {
                "contract_name": contract_name,
                "contract_fingerprint": contract_fingerprint,
                "pairwise": pairwise_summary,
                "variants": variants_summary,
            }
        )

    return {
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
        "contracts": contracts_summary,
        "schema_version": _REPORT_SCHEMA_VERSION,
        "source_manifest_identity": manifest["identity"],
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("null" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _build_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Reporting Summary",
        "",
        "Contamination caveat: overlap slices indicate train-overlap characteristics only; none should be described as fully OOD.",
        "",
    ]
    for contract in summary["contracts"]:
        lines.extend(
            [
                f"## {contract['contract_name']} ({contract['contract_fingerprint']})",
                "",
                "### Variant All Slice",
                "",
            ]
        )
        variant_rows = []
        for variant, metrics in sorted(contract["variants"].items()):
            all_metrics = metrics["all"]
            variant_rows.append(
                [
                    variant,
                    all_metrics["count"],
                    _format_number(all_metrics["strict_sequence_accuracy"]),
                    _format_number(all_metrics["valid_json_rate"]),
                    _format_number(all_metrics["contract_valid_rate"]),
                    _format_number(all_metrics["parameter_accuracy"]),
                    all_metrics["parameter_accuracy_denominator"],
                    _format_number(all_metrics["forbidden_tool_violation_rate"]),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "variant",
                    "count",
                    "strict_acc",
                    "json_rate",
                    "contract_rate",
                    "param_acc",
                    "param_denom",
                    "forbidden_rate",
                ],
                variant_rows,
            )
        )
        lines.append("")

        for label, pairwise in sorted(contract["pairwise"].items()):
            lines.extend([f"### {label}", ""])
            rows = []
            for slice_name in _SLICE_ORDER:
                metrics = pairwise[slice_name]
                rows.append(
                    [
                        slice_name,
                        metrics["paired_count"],
                        metrics["both_pass"],
                        metrics["left_only_pass"],
                        metrics["right_only_pass"],
                        metrics["both_fail"],
                        _format_number(metrics["accuracy_delta"]),
                        "null"
                        if metrics["accuracy_delta_ci95"] is None
                        else f"[{_format_number(metrics['accuracy_delta_ci95'][0])}, {_format_number(metrics['accuracy_delta_ci95'][1])}]",
                    ]
                )
            lines.append(
                _markdown_table(
                    ["slice", "pairs", "both_pass", "left_only", "right_only", "both_fail", "delta", "ci95"],
                    rows,
                )
            )
            lines.append("")

        lines.extend(["### Overlap Slices", ""])
        overlap_rows = []
        for variant, metrics in sorted(contract["variants"].items()):
            for slice_name in _SLICE_ORDER[1:]:
                slice_metrics = metrics[slice_name]
                overlap_rows.append(
                    [
                        variant,
                        slice_name,
                        slice_metrics["count"],
                        _format_number(slice_metrics["strict_sequence_accuracy"]),
                        _format_number(slice_metrics["valid_json_rate"]),
                    ]
                )
        lines.append(_markdown_table(["variant", "slice", "count", "strict_acc", "json_rate"], overlap_rows))
        lines.append("")

        lines.extend(["### Failure Reasons", ""])
        failure_rows = []
        for variant, metrics in sorted(contract["variants"].items()):
            for reason, reason_count in metrics["all"]["failure_reasons"].items():
                failure_rows.append([variant, reason, reason_count])
        lines.append(_markdown_table(["variant", "reason", "count"], failure_rows or [["none", "none", 0]]))
        lines.append("")
    return "\n".join(lines)


def _write_failures(path: Path, records: Iterable[_NormalizedRecord]) -> None:
    lines = [_canonical_json(record.payload, compact=True) for record in records if not record.strict_pass]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def write_reports(
    run_dir: Path,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260919,
) -> tuple[Path, Path]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    manifest = _read_json(
        run_dir / _MANIFEST_NAME,
        missing_message="manifest is missing",
        malformed_message="manifest is malformed",
    )
    _validate_manifest(manifest)
    records = _load_records(run_dir)

    summary = _build_summary(records, manifest, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
    summary_json_path = run_dir / _SUMMARY_JSON_NAME
    summary_md_path = run_dir / _SUMMARY_MD_NAME
    failures_path = run_dir / _FAILURES_NAME

    summary_json_path.write_text(_canonical_json(summary, compact=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_build_markdown(summary), encoding="utf-8")
    _write_failures(failures_path, records)
    return summary_json_path, summary_md_path
