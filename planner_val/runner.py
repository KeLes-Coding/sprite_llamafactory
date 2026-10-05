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
import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from planner_val.backends import CompletionBackend, GenerationRequest, GenerationResult
from planner_val.contracts import build_messages
from planner_val.models import ContractSnapshot, EvalCase, ExpectedTask, OverlapInfo, ScoreResult
from planner_val.scoring import score_output


VariantName = Literal["base", "sft", "qwen_flash"]
BackendFactory = Callable[[], CompletionBackend]

_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_RESULTS_NAME = "results.jsonl"


@dataclass(frozen=True)
class _ScheduleEntry:
    record_id: str
    contract: ContractSnapshot
    variant: VariantName
    case: EvalCase
    model_id: str
    max_tokens: int


@dataclass(frozen=True)
class _ResumeExpectation:
    contract: dict[str, Any]
    case: dict[str, Any]
    request: dict[str, str]
    variant: VariantName
    model_id: str
    max_tokens: int
    seed: int
    sample_size: int | None


class RunTunnel(Protocol):
    def ensure_running(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RunConfig:
    cases: tuple[EvalCase, ...]
    contracts: tuple[ContractSnapshot, ...]
    variants: tuple[VariantName, ...]
    model_ids: Mapping[VariantName, str]
    output_dir: Path
    max_tokens: int
    seed: int
    sample_size: int | None
    dataset_fingerprint: str
    training_package_fingerprint: str
    resume: bool = False


def _now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(inner_value) for key, inner_value in value.items()}
    if isinstance(value, tuple):
        return [_json_scalar(item) for item in value]
    if isinstance(value, list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_scalar(item) for item in sorted(value)]
    raise TypeError(f"unsupported JSON value type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _build_identity(config: RunConfig) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "case_ids": [case.case_id for case in config.cases],
        "contract_fingerprints": [contract.fingerprint for contract in config.contracts],
        "variants": list(config.variants),
        "model_ids": {variant: config.model_ids[variant] for variant in config.variants},
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "sample_size": config.sample_size,
        "dataset_fingerprint": config.dataset_fingerprint,
        "training_package_fingerprint": config.training_package_fingerprint,
    }


def _serialize_expected_task(task: ExpectedTask) -> dict[str, Any]:
    return {
        "kind": task.kind,
        "instruction": task.instruction,
        "name": task.name,
        "arguments": _json_scalar(task.arguments),
    }


def _serialize_overlap(overlap: OverlapInfo) -> dict[str, Any]:
    return {
        "exact_train": overlap.exact_train,
        "exact_validation": overlap.exact_validation,
        "all_atoms_seen_train": overlap.all_atoms_seen_train,
        "has_novel_atom": overlap.has_novel_atom,
    }


def _serialize_case(case: EvalCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "query": case.query,
        "expected": [_serialize_expected_task(task) for task in case.expected],
        "forbidden_functions": sorted(case.forbidden_functions),
        "level": case.level,
        "language": case.language,
        "difficulty": case.difficulty,
        "polarity": case.polarity,
        "source_atom_ids": sorted(case.source_atom_ids),
        "overlap": _serialize_overlap(case.overlap),
    }


def _serialize_contract(contract: ContractSnapshot) -> dict[str, Any]:
    return {
        "name": contract.name,
        "fingerprint": contract.fingerprint,
        "max_steps": contract.max_steps,
        "system_prompt": contract.system_prompt,
        "tools_context": _json_scalar(contract.tools_context),
        "tool_schemas": _json_scalar(contract.tool_schemas),
        "tool_kinds": _json_scalar(contract.tool_kinds),
        "message_config": _json_scalar(contract.message_config),
        "source": _json_scalar(contract.source),
    }


def _serialize_generation(result: GenerationResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "text": result.text,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "error": result.error,
    }


def _serialize_score(score: ScoreResult) -> dict[str, Any]:
    return {
        "reason": score.reason,
        "strict_pass": score.strict_pass,
        "valid_json": score.valid_json,
        "contract_valid": score.contract_valid,
        "task_count_match": score.task_count_match,
        "ordered_kind_name_match": score.ordered_kind_name_match,
        "parameter_match": score.parameter_match,
        "forbidden_tool_violation": score.forbidden_tool_violation,
        "parsed_tasks": [
            {
                "kind": task.kind,
                "instruction": task.instruction,
                "name": task.name,
                "arguments": _json_scalar(task.arguments),
            }
            for task in score.parsed_tasks
        ],
        "parameter_errors": list(score.parameter_errors),
        "forbidden_calls": list(score.forbidden_calls),
        "raw_output": score.raw_output,
    }


def _record_id(
    *, contract: ContractSnapshot, variant: VariantName, case: EvalCase, model_id: str, max_tokens: int
) -> str:
    payload = {
        "contract_fingerprint": contract.fingerprint,
        "variant": variant,
        "case_id": case.case_id,
        "model_id": model_id,
        "max_tokens": max_tokens,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _manifest_payload(config: RunConfig) -> dict[str, Any]:
    return {
        "identity": _build_identity(config),
        "contracts": [_serialize_contract(contract) for contract in config.contracts],
        "created_at": _now_iso(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("resume manifest is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("resume manifest is malformed") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("identity"), dict):
        raise ValueError("resume manifest is malformed")
    return payload


def _validate_manifest(payload: Mapping[str, Any], config: RunConfig) -> None:
    if payload.get("identity") != _build_identity(config):
        raise ValueError("resume identity mismatch")

    contracts = payload.get("contracts")
    expected_contracts = [_serialize_contract(contract) for contract in config.contracts]
    if not isinstance(contracts, list) or contracts != expected_contracts:
        raise ValueError("resume manifest is malformed")

    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("resume manifest is malformed")

    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("resume manifest is malformed")
    if not isinstance(environment.get("platform"), str) or not environment["platform"]:
        raise ValueError("resume manifest is malformed")
    if not isinstance(environment.get("python"), str) or not environment["python"]:
        raise ValueError("resume manifest is malformed")
    pid = environment.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        raise ValueError("resume manifest is malformed")


def _require_string_field(container: Mapping[str, Any], key: str, *, line_number: int) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_int_field(container: Mapping[str, Any], key: str, *, line_number: int) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_mapping_field(
    container: Mapping[str, Any],
    key: str,
    *,
    line_number: int,
) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_bool_field(container: Mapping[str, Any], key: str, *, line_number: int) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_nullable_string_field(container: Mapping[str, Any], key: str, *, line_number: int) -> str | None:
    if key not in container:
        raise ValueError(f"resume record is malformed at line {line_number}")
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_nullable_int_field(container: Mapping[str, Any], key: str, *, line_number: int) -> int | None:
    if key not in container:
        raise ValueError(f"resume record is malformed at line {line_number}")
    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_number_field(container: Mapping[str, Any], key: str, *, line_number: int) -> int | float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_list_field(container: Mapping[str, Any], key: str, *, line_number: int) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"resume record is malformed at line {line_number}")
    return value


def _require_string_list(values: list[Any], *, line_number: int) -> None:
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"resume record is malformed at line {line_number}")


def _validate_generation(payload: Mapping[str, Any], *, record_id: str, line_number: int) -> None:
    request_id = _require_string_field(payload, "request_id", line_number=line_number)
    if request_id != record_id:
        raise ValueError(f"resume record fields do not match expected schedule at line {line_number}")
    _require_nullable_string_field(payload, "text", line_number=line_number)
    _require_number_field(payload, "latency_ms", line_number=line_number)
    _require_nullable_int_field(payload, "prompt_tokens", line_number=line_number)
    _require_nullable_int_field(payload, "output_tokens", line_number=line_number)
    _require_nullable_string_field(payload, "error", line_number=line_number)


def _validate_score(payload: Mapping[str, Any], *, line_number: int) -> None:
    _require_string_field(payload, "reason", line_number=line_number)
    _require_bool_field(payload, "strict_pass", line_number=line_number)
    _require_bool_field(payload, "valid_json", line_number=line_number)
    _require_bool_field(payload, "contract_valid", line_number=line_number)
    _require_bool_field(payload, "task_count_match", line_number=line_number)
    _require_bool_field(payload, "ordered_kind_name_match", line_number=line_number)
    parameter_match = payload.get("parameter_match")
    if parameter_match is not None and not isinstance(parameter_match, bool):
        raise ValueError(f"resume record is malformed at line {line_number}")
    _require_bool_field(payload, "forbidden_tool_violation", line_number=line_number)
    parsed_tasks = _require_list_field(payload, "parsed_tasks", line_number=line_number)
    for task in parsed_tasks:
        if not isinstance(task, Mapping):
            raise ValueError(f"resume record is malformed at line {line_number}")
        _require_string_field(task, "kind", line_number=line_number)
        instruction = task.get("instruction")
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError(f"resume record is malformed at line {line_number}")
        name = task.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"resume record is malformed at line {line_number}")
        if "arguments" not in task:
            raise ValueError(f"resume record is malformed at line {line_number}")

    parameter_errors = _require_list_field(payload, "parameter_errors", line_number=line_number)
    _require_string_list(parameter_errors, line_number=line_number)
    forbidden_calls = _require_list_field(payload, "forbidden_calls", line_number=line_number)
    _require_string_list(forbidden_calls, line_number=line_number)

    if "raw_output" not in payload:
        raise ValueError(f"resume record is malformed at line {line_number}")
    raw_output = payload.get("raw_output")
    if raw_output is not None and not isinstance(raw_output, str):
        raise ValueError(f"resume record is malformed at line {line_number}")


def _build_resume_expectations(
    expected_schedule: Mapping[str, _ScheduleEntry],
    config: RunConfig,
) -> dict[str, _ResumeExpectation]:
    expectations: dict[str, _ResumeExpectation] = {}
    for record_id, entry in expected_schedule.items():
        system_message, user_message = build_messages(entry.contract, entry.case.query)
        expectations[record_id] = _ResumeExpectation(
            contract=_serialize_contract(entry.contract),
            case=_serialize_case(entry.case),
            request={
                "system": system_message["content"],
                "user": user_message["content"],
            },
            variant=entry.variant,
            model_id=entry.model_id,
            max_tokens=entry.max_tokens,
            seed=config.seed,
            sample_size=config.sample_size,
        )
    return expectations


def _build_expected_schedule(config: RunConfig) -> dict[str, _ScheduleEntry]:
    case_ids: set[str] = set()
    for case in config.cases:
        if case.case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        case_ids.add(case.case_id)

    contract_fingerprints: set[str] = set()
    for contract in config.contracts:
        if contract.fingerprint in contract_fingerprints:
            raise ValueError(f"duplicate contract fingerprint: {contract.fingerprint}")
        contract_fingerprints.add(contract.fingerprint)

    variants: set[VariantName] = set()
    for variant in config.variants:
        if variant in variants:
            raise ValueError(f"duplicate variant: {variant}")
        variants.add(variant)

    schedule: dict[str, _ScheduleEntry] = {}
    for contract in config.contracts:
        for variant in config.variants:
            model_id = config.model_ids[variant]
            for case in config.cases:
                record_id = _record_id(
                    contract=contract,
                    variant=variant,
                    case=case,
                    model_id=model_id,
                    max_tokens=config.max_tokens,
                )
                if record_id in schedule:
                    raise ValueError(f"record_id collision: {record_id}")
                schedule[record_id] = _ScheduleEntry(
                    record_id=record_id,
                    contract=contract,
                    variant=variant,
                    case=case,
                    model_id=model_id,
                    max_tokens=config.max_tokens,
                )
    return schedule


def _load_completed_record_ids(results_path: Path, expected_schedule: Mapping[str, _ResumeExpectation]) -> set[str]:
    if not results_path.exists():
        return set()

    completed: set[str] = set()
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"resume results are malformed at line {line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"resume results are malformed at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"resume results are malformed at line {line_number}")
        record_id = payload.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"resume results are malformed at line {line_number}")
        entry = expected_schedule.get(record_id)
        if entry is None:
            raise ValueError(f"resume results reference record outside current schedule at line {line_number}")

        contract_payload = _require_mapping_field(payload, "contract", line_number=line_number)
        case_payload = _require_mapping_field(payload, "case", line_number=line_number)
        request_payload = _require_mapping_field(payload, "request", line_number=line_number)
        generation_payload = _require_mapping_field(payload, "generation", line_number=line_number)
        score_payload = _require_mapping_field(payload, "score", line_number=line_number)

        variant = _require_string_field(payload, "variant", line_number=line_number)
        model_id = _require_string_field(payload, "model_id", line_number=line_number)
        max_tokens = _require_int_field(payload, "max_tokens", line_number=line_number)
        seed = _require_int_field(payload, "seed", line_number=line_number)
        sample_size = _require_nullable_int_field(payload, "sample_size", line_number=line_number)

        if (
            variant != entry.variant
            or model_id != entry.model_id
            or max_tokens != entry.max_tokens
            or seed != entry.seed
            or sample_size != entry.sample_size
        ):
            raise ValueError(f"resume record fields do not match expected schedule at line {line_number}")
        if dict(contract_payload) != entry.contract:
            raise ValueError(f"resume record fields do not match expected schedule at line {line_number}")
        if dict(case_payload) != entry.case:
            raise ValueError(f"resume record fields do not match expected schedule at line {line_number}")
        if dict(request_payload) != entry.request:
            raise ValueError(f"resume record fields do not match expected schedule at line {line_number}")
        _validate_generation(generation_payload, record_id=record_id, line_number=line_number)
        _validate_score(score_payload, line_number=line_number)
        if record_id in completed:
            raise ValueError(f"resume results are malformed at line {line_number}")
        completed.add(record_id)
    return completed


def _sanitized_backend_error(label: str) -> str:
    return f"{label}: request failed"


def _error_results(requests: Sequence[GenerationRequest], *, label: str) -> list[GenerationResult]:
    error = _sanitized_backend_error(label)
    return [
        GenerationResult(
            request_id=request.request_id,
            text=None,
            latency_ms=0.0,
            prompt_tokens=None,
            output_tokens=None,
            error=error,
        )
        for request in requests
    ]


def _sanitize_result_error(result: GenerationResult) -> GenerationResult:
    if not result.error:
        return result
    return GenerationResult(
        request_id=result.request_id,
        text=result.text,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        error=_sanitized_backend_error("backend_error"),
    )


def _validate_result_batch(
    requests: Sequence[GenerationRequest],
    results: Any,
) -> list[GenerationResult] | None:
    if results is None or isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        return None
    if len(results) != len(requests):
        return None
    if not all(isinstance(result, GenerationResult) for result in results):
        return None
    expected_ids = [request.request_id for request in requests]
    actual_ids = [result.request_id for result in results]
    if actual_ids != expected_ids:
        return None
    return [_sanitize_result_error(result) for result in results]


def _close_resources(
    backends: Mapping[VariantName, CompletionBackend],
    tunnel: RunTunnel | None,
) -> BaseException | None:
    first_error: BaseException | None = None
    for backend in backends.values():
        try:
            backend.close()
        except BaseException as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc
    if tunnel is not None:
        try:
            tunnel.close()
        except BaseException as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc
    return first_error


def run_benchmark(
    config: RunConfig,
    backend_factories: Mapping[VariantName, BackendFactory],
    tunnel: RunTunnel | None = None,
) -> Path:
    backends: dict[VariantName, CompletionBackend] = {}
    primary_error: BaseException | None = None
    close_error: BaseException | None = None
    output_dir = config.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / _MANIFEST_NAME
        results_path = output_dir / _RESULTS_NAME
        expected_schedule = _build_expected_schedule(config)
        resume_expectations = _build_resume_expectations(expected_schedule, config)

        if not config.resume and results_path.exists() and not manifest_path.exists():
            raise ValueError("results exist without manifest")

        if config.resume:
            existing_manifest = _read_manifest(manifest_path)
            _validate_manifest(existing_manifest, config)
            completed_record_ids = _load_completed_record_ids(results_path, resume_expectations)
        else:
            if manifest_path.exists():
                raise ValueError("manifest already exists; pass resume=True to continue")
            completed_record_ids = set()
            _write_json(manifest_path, _manifest_payload(config))

        pending_by_contract_variant: list[tuple[ContractSnapshot, VariantName, list[EvalCase]]] = []
        for contract in config.contracts:
            for variant in config.variants:
                model_id = config.model_ids[variant]
                pending_cases = [
                    case
                    for case in config.cases
                    if _record_id(
                        contract=contract,
                        variant=variant,
                        case=case,
                        model_id=model_id,
                        max_tokens=config.max_tokens,
                    )
                    not in completed_record_ids
                ]
                if pending_cases:
                    pending_by_contract_variant.append((contract, variant, pending_cases))

        finished_without_pending = not pending_by_contract_variant
        if not finished_without_pending and tunnel is not None:
            tunnel.ensure_running()

        with results_path.open("a", encoding="utf-8") as handle:
            for contract, variant, pending_cases in pending_by_contract_variant:
                if variant not in backend_factories:
                    raise ValueError(f"missing backend factory for variant {variant}")
                backend = backends.get(variant)
                if backend is None:
                    backend = backend_factories[variant]()
                    backends[variant] = backend

                model_id = config.model_ids[variant]
                requests: list[GenerationRequest] = []
                record_ids: list[str] = []
                for case in pending_cases:
                    system_message, user_message = build_messages(contract, case.query)
                    record_ids.append(
                        _record_id(
                            contract=contract,
                            variant=variant,
                            case=case,
                            model_id=model_id,
                            max_tokens=config.max_tokens,
                        )
                    )
                    requests.append(
                        GenerationRequest(
                            request_id=record_ids[-1],
                            system=system_message["content"],
                            user=user_message["content"],
                        )
                    )

                try:
                    generated = backend.generate(requests)
                except Exception:  # noqa: BLE001
                    generated = _error_results(requests, label="backend_error")
                else:
                    validated = _validate_result_batch(requests, generated)
                    if validated is None:
                        generated = _error_results(requests, label="backend_error")
                    else:
                        generated = validated

                for case, request, result in zip(pending_cases, requests, generated, strict=True):
                    score = score_output(case, result.text, contract, backend_error=result.error)
                    record = {
                        "record_id": request.request_id,
                        "contract": _serialize_contract(contract),
                        "variant": variant,
                        "model_id": model_id,
                        "max_tokens": config.max_tokens,
                        "seed": config.seed,
                        "sample_size": config.sample_size,
                        "case": _serialize_case(case),
                        "request": {"system": request.system, "user": request.user},
                        "generation": _serialize_generation(result),
                        "score": _serialize_score(score),
                    }
                    handle.write(_canonical_json(record) + "\n")
                    handle.flush()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close_error = _close_resources(backends, tunnel)
        if close_error is not None and primary_error is None:
            raise close_error

    return output_dir
