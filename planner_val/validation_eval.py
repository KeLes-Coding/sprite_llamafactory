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

import argparse
import hashlib
import io
import json
import re
import shlex
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from planner_val.backends import (
    CompletionBackend,
    GenerationRequest,
    GenerationResult,
    OpenAICompatibleBackend,
    SshTunnel,
)
from planner_val.models import ContractSnapshot, EvalCase, ExpectedTask
from planner_val.production_prompt import ProductionPromptProfile, load_benchmark_20260913_profile
from planner_val.scoring import _extract_array_text, score_benchmark_compatible, score_output
from planner_val.validation_reporting import summarize_model


_VALIDATION_MEMBER = "action_sequence_sft_validation.parquet"
_NEGATIVE_ATOM_PATTERN = re.compile(r"^l0_neg_(HumanAction|RobotGesture|RobotDance|robot_status)_")


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    system: str
    user: str
    expected_raw: str
    query: str
    eval_case: EvalCase
    contract: ContractSnapshot
    level: str
    language: str
    polarity: str
    provenance: str
    kinds: tuple[str, ...]
    tools: tuple[str, ...]
    actions: tuple[str, ...]
    source_atom_ids: tuple[str, ...]
    source_item_id: str
    example_id: str
    tool_naming: str = "real"
    alias_to_name: Mapping[str, str] = field(default_factory=dict)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _string_value(container: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = container.get(key)
    return str(value) if isinstance(value, (str, int)) and str(value) else default


def _string_tuple(container: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = container.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _source_atom_ids(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    lineage = metadata.get("lineage")
    if isinstance(lineage, list):
        for item in lineage:
            if isinstance(item, Mapping) and isinstance(item.get("source_atom_id"), str):
                values.append(item["source_atom_id"])
    source = metadata.get("source")
    if isinstance(source, Mapping):
        source_ids = source.get("source_atom_ids")
        if isinstance(source_ids, list):
            values.extend(item for item in source_ids if isinstance(item, str) and item)
    return tuple(dict.fromkeys(values))


def _forbidden_functions(metadata: Mapping[str, Any]) -> frozenset[str]:
    functions = []
    for atom_id in _source_atom_ids(metadata):
        match = _NEGATIVE_ATOM_PATTERN.match(atom_id)
        if match is not None:
            functions.append(match.group(1))
    return frozenset(functions)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _message_content(messages: Any, role: str, *, case_id: str) -> str:
    if not isinstance(messages, list):
        raise ValueError(f"{case_id}: messages must be a list")
    matches = [message for message in messages if isinstance(message, Mapping) and message.get("role") == role]
    if len(matches) != 1 or not isinstance(matches[0].get("content"), str):
        raise ValueError(f"{case_id}: expected exactly one {role} message")
    return matches[0]["content"]


def _json_object(raw: str, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected JSON object")
    return value


def _expected_tasks(raw: str, *, case_id: str) -> tuple[ExpectedTask, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{case_id}: malformed assistant JSON") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{case_id}: assistant output must be a non-empty task list")

    tasks: list[ExpectedTask] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{case_id}: assistant task {index} must be an object")
        kind = item.get("kind")
        instruction = item.get("instruction")
        if kind not in {"reply", "query", "action"} or not isinstance(instruction, str) or not instruction:
            raise ValueError(f"{case_id}: invalid assistant task {index}")
        name = item.get("name")
        arguments = item.get("arguments", {})
        if kind == "reply":
            name = None
            arguments = {}
        elif not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
            raise ValueError(f"{case_id}: invalid assistant task {index}")
        tasks.append(ExpectedTask(kind=kind, instruction=instruction, name=name, arguments=arguments))
    return tuple(tasks)


def _contract(system: str, user_payload: Mapping[str, Any], *, case_id: str) -> ContractSnapshot:
    tools = user_payload.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{case_id}: user payload must contain tools")

    schemas: dict[str, Mapping[str, Any]] = {}
    kinds: dict[str, str] = {}
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"{case_id}: invalid tool {index}")
        name = tool.get("name")
        kind = tool.get("kind")
        schema = tool.get("arguments_schema")
        if not isinstance(name, str) or not name or kind not in {"query", "action"} or not isinstance(schema, Mapping):
            raise ValueError(f"{case_id}: invalid tool {index}")
        schemas[name] = schema
        kinds[name] = kind

    fingerprint_payload = json.dumps(
        {"system": system, "tools": tools}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    fingerprint = "sha256:" + hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    return ContractSnapshot(
        name="controlled",
        system_prompt=system,
        tools_context=tools,
        tool_schemas=schemas,
        tool_kinds=kinds,
        source={"member": _VALIDATION_MEMBER, "case_id": case_id},
        fingerprint=fingerprint,
    )


def _dataset_member(package_path: Path) -> str:
    return package_path.name if package_path.suffix == ".jsonl" else _VALIDATION_MEMBER


def _read_validation_rows(package_path: Path) -> list[Any]:
    if package_path.suffix == ".jsonl":
        try:
            lines = package_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise ValueError(f"dataset not found: {package_path}") from exc
        return [json.loads(line) for line in lines if line.strip()]
    try:
        with zipfile.ZipFile(package_path) as archive:
            member = archive.read(_VALIDATION_MEMBER)
    except FileNotFoundError as exc:
        raise ValueError(f"package not found: {package_path}") from exc
    except KeyError as exc:
        raise ValueError(f"missing {_VALIDATION_MEMBER}: {package_path}") from exc
    return pq.read_table(io.BytesIO(member)).to_pylist()


def _alias_to_name(metadata: Mapping[str, Any], *, case_id: str) -> dict[str, str]:
    if metadata.get("tool_naming") != "alias":
        return {}
    aliases = metadata.get("tool_aliases")
    if not isinstance(aliases, Mapping) or not all(
        isinstance(name, str) and isinstance(alias, str) for name, alias in aliases.items()
    ):
        raise ValueError(f"{case_id}: alias rows must provide tool_aliases")
    return {alias: name for name, alias in aliases.items()}


def _dealias_plan(raw: str | None, alias_to_name: Mapping[str, str]) -> str | None:
    """Rewrite aliased tool names to real names; names outside the alias map become unknown tools."""
    if raw is None or not alias_to_name:
        return raw
    array_text = _extract_array_text(raw)
    if array_text is None:
        return raw
    try:
        payload = json.loads(array_text)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, list):
        return raw
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            item["name"] = alias_to_name.get(item["name"], f"unaliased:{item['name']}")
    return json.dumps(payload, ensure_ascii=False)


def _dealias_tools(
    user_payload: Mapping[str, Any], alias_to_name: Mapping[str, str], *, case_id: str
) -> dict[str, Any]:
    tools = user_payload.get("tools")
    if not alias_to_name or not isinstance(tools, list):
        return dict(user_payload)
    renamed = []
    for tool in tools:
        if not isinstance(tool, Mapping) or tool.get("name") not in alias_to_name:
            raise ValueError(f"{case_id}: tool name is missing from tool_aliases")
        renamed.append({**tool, "name": alias_to_name[tool["name"]]})
    return {**user_payload, "tools": renamed}


def load_validation_cases(package_path: Path, limit: int | None = None) -> list[ValidationCase]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = _read_validation_rows(package_path)
    selected = rows if limit is None else rows[:limit]
    cases: list[ValidationCase] = []
    for index, row in enumerate(selected):
        case_id = f"validation-{index:04d}"
        if not isinstance(row, Mapping):
            raise ValueError(f"{case_id}: invalid row")
        messages = row.get("messages")
        system = _message_content(messages, "system", case_id=case_id)
        user = _message_content(messages, "user", case_id=case_id)
        expected_raw = _message_content(messages, "assistant", case_id=case_id)
        user_payload = _json_object(user, context=f"{case_id}: user")
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        query = user_payload.get("instruction")
        if not isinstance(query, str) or not query:
            raise ValueError(f"{case_id}: missing instruction")
        alias_to_name = _alias_to_name(metadata, case_id=case_id)
        expected = _expected_tasks(_dealias_plan(expected_raw, alias_to_name), case_id=case_id)
        if any(task.name is not None and task.name.startswith("unaliased:") for task in expected):
            raise ValueError(f"{case_id}: assistant tool name is missing from tool_aliases")
        cases.append(
            ValidationCase(
                case_id=case_id,
                system=system,
                user=user,
                expected_raw=expected_raw,
                query=query,
                eval_case=EvalCase(
                    case_id=case_id,
                    query=query,
                    expected=expected,
                    forbidden_functions=_forbidden_functions(metadata),
                    level=_string_value(metadata, "level"),
                    language=_string_value(metadata, "language"),
                    difficulty="unknown",
                    polarity=_string_value(metadata, "polarity"),
                    source_atom_ids=frozenset(_source_atom_ids(metadata)),
                ),
                contract=_contract(
                    system, _dealias_tools(user_payload, alias_to_name, case_id=case_id), case_id=case_id
                ),
                level=_string_value(metadata, "level"),
                language=_string_value(metadata, "language"),
                polarity=_string_value(metadata, "polarity"),
                provenance=_string_value(metadata, "provenance"),
                kinds=_string_tuple(metadata, "kinds"),
                tools=_string_tuple(metadata, "tools"),
                actions=_string_tuple(metadata, "actions"),
                source_atom_ids=_source_atom_ids(metadata),
                source_item_id=_string_value(metadata, "source_item_id", ""),
                example_id=_string_value(metadata, "example_id", ""),
                tool_naming=_string_value(metadata, "tool_naming", "real"),
                alias_to_name=alias_to_name,
            )
        )
    return cases


def _safe_results(backend: CompletionBackend, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
    try:
        results = backend.generate(requests)
    except Exception:  # noqa: BLE001
        results = []
    if len(results) != len(requests) or any(
        not isinstance(result, GenerationResult) or result.request_id != request.request_id
        for request, result in zip(requests, results)
    ):
        return [
            GenerationResult(request.request_id, None, 0.0, None, None, "backend_error: request failed")
            for request in requests
        ]
    return list(results)


def _scored_record(
    case: ValidationCase,
    result: GenerationResult,
    model_name: str,
    base_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scored_text = _dealias_plan(result.text, case.alias_to_name)
    score = score_output(case.eval_case, scored_text, case.contract, backend_error=result.error)
    benchmark_score = score_benchmark_compatible(
        case.eval_case, scored_text, case.contract, backend_error=result.error
    )
    aligned_kind_name_matches = sum(
        expected.kind == actual.kind and expected.name == actual.name
        for expected, actual in zip(case.eval_case.expected, score.parsed_tasks)
    )
    total_tokens = (
        result.prompt_tokens + result.output_tokens
        if result.prompt_tokens is not None and result.output_tokens is not None
        else None
    )
    record = dict(base_record or {})
    record.update(
        {
            "case_id": case.case_id,
            "model": model_name,
            "query": case.query,
            "expected": json.loads(case.expected_raw),
            "output": result.text,
            "benchmark_compatible_pass": benchmark_score.passed,
            "benchmark_compatible_reason": benchmark_score.reason,
            "benchmark_parser_valid": benchmark_score.parser_valid,
            "benchmark_tool_sequence_match": benchmark_score.tool_sequence_match,
            "benchmark_parameter_match": benchmark_score.parameter_match,
            "benchmark_forbidden_tool_violation": benchmark_score.forbidden_tool_violation,
            "benchmark_reply_slots": benchmark_score.reply_slots,
            "benchmark_expected_tool_count": benchmark_score.expected_tool_count,
            "benchmark_predicted_tool_count": benchmark_score.predicted_tool_count,
            "benchmark_errors": list(benchmark_score.errors),
            "reason": score.reason,
            "strict_pass": score.strict_pass,
            "valid_json": score.valid_json,
            "contract_valid": score.contract_valid,
            "task_count_match": score.task_count_match,
            "ordered_kind_name_match": score.ordered_kind_name_match,
            "parameter_match": score.parameter_match,
            "forbidden_tool_violation": score.forbidden_tool_violation,
            "expected_step_count": len(case.eval_case.expected),
            "predicted_step_count": len(score.parsed_tasks),
            "aligned_kind_name_matches": aligned_kind_name_matches,
            "latency_ms": result.latency_ms,
            "timing": {"total_ms": result.latency_ms},
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "tokens": {"input": result.prompt_tokens, "output": result.output_tokens, "total": total_tokens},
            "error": result.error,
            "metadata": {
                "level": case.level,
                "language": case.language,
                "polarity": case.polarity,
                "provenance": case.provenance,
                "tool_naming": case.tool_naming,
                "task_count": len(case.eval_case.expected),
                "kinds": list(case.kinds),
                "tools": list(case.tools),
                "actions": list(case.actions),
                "forbidden_functions": sorted(case.eval_case.forbidden_functions),
                "source_atom_ids": list(case.source_atom_ids),
                "source_item_id": case.source_item_id,
                "example_id": case.example_id,
            },
        }
    )
    return record


def rescore_validation(cases: Sequence[ValidationCase], output_dir: Path) -> dict[str, Any]:
    if not cases:
        raise ValueError("at least one validation case is required")
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case_by_id = {case.case_id: case for case in cases}
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        source_record = json.loads(line)
        case_id = source_record.get("case_id")
        model_name = source_record.get("model")
        if case_id not in case_by_id or not isinstance(model_name, str):
            raise ValueError(f"invalid result record at line {line_number}")
        result = GenerationResult(
            request_id=case_id,
            text=source_record.get("output"),
            latency_ms=float(source_record.get("latency_ms", 0.0)),
            prompt_tokens=source_record.get("prompt_tokens"),
            output_tokens=source_record.get("output_tokens"),
            error=source_record.get("error"),
        )
        record = _scored_record(case_by_id[case_id], result, model_name, source_record)
        records.append(record)
        records_by_model.setdefault(model_name, []).append(record)

    original_models = original_summary.get("models", {})
    summary = {**original_summary, "schema_version": 3, "cases": len(cases)}
    summary["models"] = {
        model_name: summarize_model(
            model_records,
            original_models.get(model_name, {}).get("performance", {}).get("batch_wall_time_ms"),
        )
        for model_name, model_records in records_by_model.items()
    }
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_validation(
    cases: Sequence[ValidationCase],
    backends: Mapping[str, CompletionBackend],
    output_dir: Path,
    request_builders: Mapping[str, Callable[[ValidationCase], GenerationRequest]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
    dataset_info: Mapping[str, Any] | None = None,
    prompt_profile: ProductionPromptProfile | None = None,
    run_started_ms: float | None = None,
    run_started_at: str | None = None,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("at least one validation case is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_started_at = run_started_at or _utc_now()
    effective_started_ms = run_started_ms if run_started_ms is not None else _now_ms()
    records: list[dict[str, Any]] = []
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    batch_wall_times: dict[str, float] = {}
    summary: dict[str, Any] = {
        "schema_version": 3,
        "cases": len(cases),
        "run": {
            "started_at": effective_started_at,
            "wall_time_scope": "cli_start_through_results_serialization",
            "config": dict(run_config or {}),
            "ttft": {"available": False, "reason": "non_streaming_planner_evaluation"},
        },
        "dataset": {**dict(dataset_info or {}), "cases": len(cases)},
        "models": {},
    }
    if metadata is not None:
        summary.update(metadata)
    if prompt_profile is not None:
        artifact_path = output_dir / "prompt_profile.json"
        artifact_bytes = (json.dumps(prompt_profile.artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        artifact_path.write_bytes(artifact_bytes)
        summary["prompt_profile"] = {
            **prompt_profile.metadata,
            "artifact": {"path": artifact_path.name, "sha256": hashlib.sha256(artifact_bytes).hexdigest()},
        }

    for model_name, backend in backends.items():
        request_builder = request_builders.get(model_name) if request_builders is not None else None
        requests = [
            request_builder(case)
            if request_builder is not None
            else GenerationRequest(case.case_id, case.system, case.user)
            for case in cases
        ]
        batch_started_ms = _now_ms()
        results = _safe_results(backend, requests)
        batch_wall_times[model_name] = _now_ms() - batch_started_ms
        model_records: list[dict[str, Any]] = []
        for case, result in zip(cases, results, strict=True):
            record = _scored_record(case, result, model_name)
            records.append(record)
            model_records.append(record)
        records_by_model[model_name] = model_records

    summary["models"] = {
        model_name: summarize_model(model_records, batch_wall_times[model_name])
        for model_name, model_records in records_by_model.items()
    }
    results_path = output_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary["run"]["finished_at"] = _utc_now()
    summary["run"]["wall_time_ms"] = _now_ms() - effective_started_ms
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_env_value(path: Path, name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"env file not found: {path}") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.removeprefix("export ").split("=", 1)
        if key.strip() == name:
            values = shlex.split(raw_value, comments=True)
            return values[0] if values else ""
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare vLLM SFT and qwen-flash on SFT validation cases.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vllm-model", default="gemma3-270m-sft")
    parser.add_argument("--qwen-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--qwen-model", default="qwen-flash")
    parser.add_argument("--cosa-env", type=Path)
    parser.add_argument(
        "--rescore-results",
        action="store_true",
        help="Re-score output-dir/results.jsonl without calling a model backend.",
    )
    parser.add_argument("--qwen-only", action="store_true", help="Skip the vLLM backend and SSH tunnel.")
    parser.add_argument("--vllm-only", action="store_true", help="Skip qwen; --cosa-env is not required.")
    parser.add_argument(
        "--qwen-production-profile",
        action="store_true",
        help="Use the frozen prompt configuration from benchmark run 20260913_151333_groupfiles238_range0-237.",
    )
    parser.add_argument(
        "--production-agent-repo",
        type=Path,
        help="Git repository containing the frozen ActionSequence prompt revision.",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-port", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run_started_ms = _now_ms()
    run_started_at = _utc_now()
    args = _build_parser().parse_args(argv)
    cases = load_validation_cases(args.package, limit=args.limit)
    if args.rescore_results:
        summary = rescore_validation(cases, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.qwen_only and args.vllm_only:
        raise ValueError("--qwen-only and --vllm-only are mutually exclusive")
    api_key: str | None = None
    if not args.vllm_only:
        if args.cosa_env is None:
            raise ValueError("--cosa-env is required unless --rescore-results or --vllm-only is used")
        api_key = _read_env_value(args.cosa_env, "DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(f"DASHSCOPE_API_KEY is missing from {args.cosa_env}")

    prompt_profile: ProductionPromptProfile | None = None
    if args.qwen_production_profile:
        if args.production_agent_repo is None:
            raise ValueError("--production-agent-repo is required with --qwen-production-profile")
        prompt_profile = load_benchmark_20260913_profile(args.production_agent_repo)

    backends: dict[str, OpenAICompatibleBackend] = {}
    if not args.qwen_only:
        backends[args.vllm_model] = OpenAICompatibleBackend(
            model=args.vllm_model,
            base_url=args.vllm_url,
            api_key="EMPTY",
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
    qwen_result_name = (
        f"{args.qwen_model}-production-{prompt_profile.revision[:7]}"
        if prompt_profile is not None
        else args.qwen_model
    )
    if not args.vllm_only:
        backends[qwen_result_name] = OpenAICompatibleBackend(
            model=args.qwen_model,
            base_url=args.qwen_url,
            api_key=api_key,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
    tunnel = (
        SshTunnel(host=args.ssh_host, user=args.ssh_user, ssh_port=args.ssh_port)
        if args.ssh_host is not None and not args.qwen_only
        else None
    )
    try:
        if tunnel is not None:
            tunnel.start()
        request_builders = None
        if prompt_profile is not None:
            request_builders = {
                qwen_result_name: lambda case: GenerationRequest(
                    case.case_id,
                    prompt_profile.system_prompt,
                    prompt_profile.build_user_prompt(case.query),
                )
            }
        model_config: dict[str, dict[str, str]] = {}
        if not args.qwen_only:
            model_config[args.vllm_model] = {"model": args.vllm_model, "base_url": args.vllm_url}
        if not args.vllm_only:
            model_config[qwen_result_name] = {"model": args.qwen_model, "base_url": args.qwen_url}
        summary = evaluate_validation(
            cases,
            backends,
            args.output_dir,
            request_builders=request_builders,
            run_config={
                "qwen_only": args.qwen_only,
                "vllm_only": args.vllm_only,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": False,
                "models": model_config,
            },
            dataset_info={
                "package": str(args.package),
                "package_sha256": _file_sha256(args.package),
                "member": _dataset_member(args.package),
                "requested_limit": args.limit,
            },
            prompt_profile=prompt_profile,
            run_started_ms=run_started_ms,
            run_started_at=run_started_at,
        )
    finally:
        for backend in backends.values():
            backend.close()
        if tunnel is not None:
            tunnel.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
