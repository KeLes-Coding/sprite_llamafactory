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

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


_SLICE_FIELDS = (
    "level",
    "language",
    "polarity",
    "provenance",
    "tool_naming",
    "task_count",
    "kind",
    "tool",
    "action",
)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _quality(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    benchmark_passes = sum(bool(record.get("benchmark_compatible_pass", record["strict_pass"])) for record in records)
    strict_passes = sum(bool(record["strict_pass"]) for record in records)
    benchmark_parameter_values = [
        record.get("benchmark_parameter_match")
        for record in records
        if record.get("benchmark_parameter_match") is not None
    ]
    parameter_values = [record["parameter_match"] for record in records if record["parameter_match"] is not None]
    expected_steps = sum(int(record["expected_step_count"]) for record in records)
    predicted_steps = sum(int(record["predicted_step_count"]) for record in records)
    matched_steps = sum(int(record["aligned_kind_name_matches"]) for record in records)
    step_precision = _rate(matched_steps, predicted_steps)
    step_recall = _rate(matched_steps, expected_steps)
    step_f1 = (
        2 * step_precision * step_recall / (step_precision + step_recall)
        if step_precision is not None and step_recall is not None and step_precision + step_recall > 0
        else None
    )
    failure_reasons = Counter(str(record["reason"]) for record in records if not record["strict_pass"])
    benchmark_failure_reasons = Counter(
        str(record.get("benchmark_compatible_reason", record["reason"]))
        for record in records
        if not record.get("benchmark_compatible_pass", record["strict_pass"])
    )
    return {
        "count": count,
        "primary_metric": "benchmark_compatible_accuracy",
        "benchmark_compatible_pass": benchmark_passes,
        "benchmark_compatible_accuracy": _rate(benchmark_passes, count),
        "benchmark_parser_valid_rate": _rate(
            sum(bool(record.get("benchmark_parser_valid", record["contract_valid"])) for record in records), count
        ),
        "benchmark_tool_sequence_accuracy": _rate(
            sum(
                bool(record.get("benchmark_tool_sequence_match", record["ordered_kind_name_match"]))
                for record in records
            ),
            count,
        ),
        "benchmark_parameter_accuracy": _rate(
            sum(bool(value) for value in benchmark_parameter_values), len(benchmark_parameter_values)
        ),
        "benchmark_parameter_accuracy_denominator": len(benchmark_parameter_values),
        "benchmark_forbidden_tool_violation_rate": _rate(
            sum(bool(record.get("benchmark_forbidden_tool_violation", False)) for record in records), count
        ),
        "benchmark_failure_reasons": dict(sorted(benchmark_failure_reasons.items())),
        "strict_pass": strict_passes,
        "strict_accuracy": _rate(strict_passes, count),
        "valid_json_rate": _rate(sum(bool(record["valid_json"]) for record in records), count),
        "contract_valid_rate": _rate(sum(bool(record["contract_valid"]) for record in records), count),
        "task_count_accuracy": _rate(sum(bool(record["task_count_match"]) for record in records), count),
        "ordered_kind_name_accuracy": _rate(sum(bool(record["ordered_kind_name_match"]) for record in records), count),
        "step_kind_name": {
            "matched": matched_steps,
            "expected": expected_steps,
            "predicted": predicted_steps,
            "precision": step_precision,
            "recall": step_recall,
            "f1": step_f1,
        },
        "parameter_accuracy": _rate(sum(bool(value) for value in parameter_values), len(parameter_values)),
        "parameter_accuracy_denominator": len(parameter_values),
        "forbidden_tool_violation_rate": _rate(
            sum(bool(record["forbidden_tool_violation"]) for record in records), count
        ),
        "failure_reasons": dict(sorted(failure_reasons.items())),
    }


def _performance(records: Sequence[Mapping[str, Any]], batch_wall_time_ms: float | None) -> dict[str, Any]:
    count = len(records)
    latencies = [float(record["latency_ms"]) for record in records]
    token_records = [
        record
        for record in records
        if isinstance(record.get("prompt_tokens"), int) and isinstance(record.get("output_tokens"), int)
    ]
    input_total = sum(int(record["prompt_tokens"]) for record in token_records)
    output_total = sum(int(record["output_tokens"]) for record in token_records)
    token_total = input_total + output_total
    latency_total_ms = sum(latencies)
    succeeded = sum(record.get("error") is None for record in records)
    batch_seconds = batch_wall_time_ms / 1000.0 if batch_wall_time_ms is not None and batch_wall_time_ms > 0 else None
    service_seconds = latency_total_ms / 1000.0 if latency_total_ms > 0 else None
    return {
        "batch_wall_time_ms": batch_wall_time_ms,
        "requests": {
            "total": count,
            "succeeded": succeeded,
            "failed": count - succeeded,
            "per_second": _rate(count, batch_seconds) if batch_seconds is not None else None,
        },
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "mean": _rate(sum(latencies), count),
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "tokens": {
            "coverage": {"count": len(token_records), "rate": _rate(len(token_records), count)},
            "input": {"total": input_total, "mean": _rate(input_total, len(token_records))},
            "output": {"total": output_total, "mean": _rate(output_total, len(token_records))},
            "total": {"total": token_total, "mean": _rate(token_total, len(token_records))},
        },
        "throughput": {
            "output_tokens_per_second": _rate(output_total, batch_seconds) if batch_seconds is not None else None,
            "total_tokens_per_second": _rate(token_total, batch_seconds) if batch_seconds is not None else None,
            "service_time_output_tokens_per_second": (
                _rate(output_total, service_seconds) if service_seconds is not None else None
            ),
        },
    }


def _aggregate(records: Sequence[Mapping[str, Any]], batch_wall_time_ms: float | None = None) -> dict[str, Any]:
    return {
        "quality": _quality(records),
        "performance": _performance(records, batch_wall_time_ms),
    }


def _slice_values(record: Mapping[str, Any], field: str) -> tuple[str, ...]:
    metadata = record["metadata"]
    if field == "tool_naming":
        return (str(metadata.get(field, "unknown")),)
    if field in {"level", "language", "polarity", "provenance", "task_count"}:
        return (str(metadata[field]),)
    values = metadata[{"kind": "kinds", "tool": "tools", "action": "actions"}[field]]
    return tuple(dict.fromkeys(str(value) for value in values))


def _group_sort_key(field: str, value: str) -> tuple[int, int | str]:
    if field == "level" and value.isdigit():
        return (0, int(value))
    return (1, value)


def summarize_model(records: Sequence[Mapping[str, Any]], batch_wall_time_ms: float | None) -> dict[str, Any]:
    aggregate = _aggregate(records, batch_wall_time_ms)
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {field: defaultdict(list) for field in _SLICE_FIELDS}
    for record in records:
        for field in _SLICE_FIELDS:
            for value in _slice_values(record, field):
                grouped[field][value].append(record)

    quality = aggregate["quality"]
    return {
        "total": quality["count"],
        "primary_metric": quality["primary_metric"],
        "benchmark_compatible_pass": quality["benchmark_compatible_pass"],
        "benchmark_compatible_accuracy": quality["benchmark_compatible_accuracy"],
        "strict_pass": quality["strict_pass"],
        "strict_accuracy": quality["strict_accuracy"],
        **aggregate,
        "slices": {
            field: {
                value: _aggregate(field_records)
                for value, field_records in sorted(groups.items(), key=lambda item: _group_sort_key(field, item[0]))
            }
            for field, groups in grouped.items()
        },
    }
