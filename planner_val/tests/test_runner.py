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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from planner_val import runner as runner_module
from planner_val.backends import CompletionBackend, GenerationRequest, GenerationResult
from planner_val.models import ContractSnapshot, EvalCase, ExpectedTask, OverlapInfo
from planner_val.runner import RunConfig, run_benchmark


class _RecordingBackend(CompletionBackend):
    def __init__(
        self,
        variant: str,
        scripted_calls: Sequence[Callable[[Sequence[GenerationRequest]], list[GenerationResult]]],
    ) -> None:
        self.variant = variant
        self._scripted_calls = list(scripted_calls)
        self.generate_calls: list[tuple[GenerationRequest, ...]] = []
        self.close_calls = 0

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        call_index = len(self.generate_calls)
        self.generate_calls.append(tuple(requests))
        return self._scripted_calls[call_index](requests)

    def close(self) -> None:
        self.close_calls += 1


class _BackendFactory:
    def __init__(
        self,
        variant: str,
        scripted_calls: Sequence[Callable[[Sequence[GenerationRequest]], list[GenerationResult]]],
    ) -> None:
        self.variant = variant
        self._scripted_calls = tuple(scripted_calls)
        self.created: list[_RecordingBackend] = []

    def __call__(self) -> _RecordingBackend:
        backend = _RecordingBackend(self.variant, self._scripted_calls)
        self.created.append(backend)
        return backend


class _Tunnel:
    def __init__(self) -> None:
        self.ensure_running_calls = 0
        self.close_calls = 0

    def ensure_running(self) -> None:
        self.ensure_running_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _BrokenGenerateBackend(CompletionBackend):
    def __init__(self, generated: Any) -> None:
        self._generated = generated
        self.close_calls = 0

    def generate(self, requests: Sequence[GenerationRequest]) -> Any:
        return self._generated

    def close(self) -> None:
        self.close_calls += 1


class _BrokenFactory:
    def __init__(self, backend: CompletionBackend | None = None, exc: Exception | None = None) -> None:
        self._backend = backend
        self._exc = exc
        self.calls = 0

    def __call__(self) -> CompletionBackend:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        assert self._backend is not None
        return self._backend


class _RaisingCloseBackend(_RecordingBackend):
    def __init__(self, variant: str, close_message: str) -> None:
        super().__init__(variant, [_reply_batch, _reply_batch])
        self.close_message = close_message

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError(self.close_message)


class _RaisingTunnel(_Tunnel):
    def __init__(self, close_message: str) -> None:
        super().__init__()
        self.close_message = close_message

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError(self.close_message)


class _BaseExceptionCloseBackend(_RecordingBackend):
    def __init__(
        self,
        variant: str,
        close_error: BaseException,
        scripted_calls: Sequence[Callable[[Sequence[GenerationRequest]], list[GenerationResult]]] | None = None,
    ) -> None:
        super().__init__(variant, scripted_calls or [_reply_batch, _reply_batch])
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        raise self.close_error


class _BaseExceptionTunnel(_Tunnel):
    def __init__(self, close_error: BaseException) -> None:
        super().__init__()
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        raise self.close_error


def _reply_result(request: GenerationRequest, *, instruction: str = "done") -> GenerationResult:
    return GenerationResult(
        request_id=request.request_id,
        text=json.dumps([{"kind": "reply", "instruction": instruction}], ensure_ascii=False),
        latency_ms=12.5,
        prompt_tokens=10,
        output_tokens=4,
        error=None,
    )


def _reply_batch(requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
    return [_reply_result(request) for request in requests]


def _error_batch(requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
    return [
        GenerationResult(
            request_id=request.request_id,
            text=None,
            latency_ms=3.0,
            prompt_tokens=2,
            output_tokens=0,
            error="RuntimeError: secret endpoint failure",
        )
        for request in requests
    ]


def _make_case(case_id: str, *, exact_train: bool, has_novel_atom: bool) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        query=f"query for {case_id}",
        expected=(ExpectedTask(kind="reply", instruction="done"),),
        forbidden_functions=frozenset({"dangerous_tool"} if case_id.endswith("2") else ()),
        level="L1",
        language="zh",
        difficulty="easy",
        polarity="positive",
        source_atom_ids=frozenset({f"atom-{case_id}", f"atom-shared-{case_id[-1]}"}),
        overlap=OverlapInfo(
            exact_train=exact_train,
            exact_validation=False,
            all_atoms_seen_train=not has_novel_atom,
            has_novel_atom=has_novel_atom,
        ),
    )


def _make_contract(name: str, fingerprint: str) -> ContractSnapshot:
    if name == "controlled":
        return ContractSnapshot(
            name="controlled",
            system_prompt="controlled system",
            tools_context=({"name": "noop", "kind": "query", "arguments_schema": {"type": "object"}},),
            tool_schemas={"noop": {"type": "object", "properties": {}, "additionalProperties": False}},
            tool_kinds={"noop": "query"},
            source={"source_file": f"{name}.json"},
            fingerprint=fingerprint,
        )
    return ContractSnapshot(
        name="production",
        system_prompt="production system",
        tools_context="TOOLS CONTEXT",
        tool_schemas={},
        tool_kinds={},
        source={"prompts_path": f"{name}.py", "tool_capture_path": f"{name}.json"},
        fingerprint=fingerprint,
        max_steps=4,
        message_config={
            "user_tools_header": "TOOLS",
            "user_limit_header": "LIMITS",
            "user_instruction_header": "INSTRUCTION",
            "step_limit_template": "Use at most {max_steps} steps.",
        },
    )


def _make_config(output_dir: Path, **overrides: Any) -> RunConfig:
    defaults = {
        "cases": (
            _make_case("case-1", exact_train=False, has_novel_atom=True),
            _make_case("case-2", exact_train=True, has_novel_atom=False),
        ),
        "contracts": (
            _make_contract("controlled", "sha256:contract-controlled"),
            _make_contract("production", "sha256:contract-production"),
        ),
        "variants": ("base", "sft", "qwen_flash"),
        "model_ids": {
            "base": "base-model",
            "sft": "sft-model",
            "qwen_flash": "qwen-flash-model",
        },
        "output_dir": output_dir,
        "max_tokens": 128,
        "seed": 7,
        "sample_size": None,
        "dataset_fingerprint": "sha256:dataset-v1",
        "training_package_fingerprint": "sha256:package-v1",
        "resume": False,
    }
    defaults.update(overrides)
    return RunConfig(**defaults)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _record_pairs(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    return [(record["contract"]["name"], record["variant"], record["case"]["case_id"]) for record in records]


def _assert_no_secrets(payload: Any) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = key.lower()
            assert "api_key" not in lowered
            assert "credential" not in lowered
            assert "secret" not in lowered
            _assert_no_secrets(value)
        return
    if isinstance(payload, list):
        for item in payload:
            _assert_no_secrets(item)
        return
    if isinstance(payload, str):
        assert "secret" not in payload.lower()


def test_run_benchmark_writes_12_unique_records_in_deterministic_order(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run")
    tunnel = _Tunnel()
    factories = {variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in config.variants}

    run_dir = run_benchmark(config, factories, tunnel=tunnel)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = _load_jsonl(run_dir / "results.jsonl")
    assert len(records) == 12
    assert len({record["record_id"] for record in records}) == 12
    assert _record_pairs(records) == [
        ("controlled", "base", "case-1"),
        ("controlled", "base", "case-2"),
        ("controlled", "sft", "case-1"),
        ("controlled", "sft", "case-2"),
        ("controlled", "qwen_flash", "case-1"),
        ("controlled", "qwen_flash", "case-2"),
        ("production", "base", "case-1"),
        ("production", "base", "case-2"),
        ("production", "sft", "case-1"),
        ("production", "sft", "case-2"),
        ("production", "qwen_flash", "case-1"),
        ("production", "qwen_flash", "case-2"),
    ]
    assert manifest["identity"]["variants"] == ["base", "sft", "qwen_flash"]
    assert manifest["identity"]["case_ids"] == ["case-1", "case-2"]
    assert tunnel.ensure_running_calls == 1
    assert tunnel.close_calls == 1
    _assert_no_secrets(manifest)
    _assert_no_secrets(records)


def test_variants_receive_identical_messages_for_each_contract_and_case(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run")
    factories = {variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in config.variants}

    run_benchmark(config, factories)

    per_variant_messages: dict[str, list[tuple[str, str]]] = {}
    for variant, factory in factories.items():
        backend = factory.created[0]
        messages: list[tuple[str, str]] = []
        for batch in backend.generate_calls:
            for request in batch:
                messages.append((request.system, request.user))
        per_variant_messages[variant] = messages
    assert per_variant_messages["base"] == per_variant_messages["sft"] == per_variant_messages["qwen_flash"]


def test_resume_after_completion_keeps_jsonl_byte_identical_and_skips_backend_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    initial_config = _make_config(output_dir)
    initial_factories = {
        variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in initial_config.variants
    }

    run_benchmark(initial_config, initial_factories)
    before = (output_dir / "results.jsonl").read_bytes()

    resume_factories = {
        variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in initial_config.variants
    }
    resume_config = _make_config(output_dir, resume=True)
    run_benchmark(resume_config, resume_factories)

    after = (output_dir / "results.jsonl").read_bytes()
    assert after == before
    assert all(not factory.created for factory in resume_factories.values())


def test_resume_after_completion_skips_tunnel_start_but_still_closes_tunnel(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    initial_config = _make_config(output_dir)
    initial_factories = {
        variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in initial_config.variants
    }
    run_benchmark(initial_config, initial_factories)

    tunnel = _Tunnel()
    resume_factories = {
        variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in initial_config.variants
    }

    run_benchmark(_make_config(output_dir, resume=True), resume_factories, tunnel=tunnel)

    assert tunnel.ensure_running_calls == 0
    assert tunnel.close_calls == 1
    assert all(not factory.created for factory in resume_factories.values())


def test_resume_rejects_orphan_results_without_manifest_before_backend_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True)
    (output_dir / "results.jsonl").write_text(
        json.dumps({"record_id": "orphan-record"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    factories = {variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in ("base", "sft")}

    with pytest.raises(ValueError, match="manifest"):
        run_benchmark(_make_config(output_dir, variants=("base", "sft")), factories)

    assert all(not factory.created for factory in factories.values())


def test_resume_rejects_manifest_missing_identity_before_backend_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(json.dumps({"created_at": "now"}), encoding="utf-8")
    factories = {variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in ("base", "sft")}

    with pytest.raises(ValueError, match="manifest"):
        run_benchmark(_make_config(output_dir, resume=True, variants=("base", "sft")), factories)

    assert all(not factory.created for factory in factories.values())


@pytest.mark.parametrize(
    ("manifest_mutator", "match"),
    [
        (lambda manifest: manifest.pop("contracts"), "manifest"),
        (lambda manifest: manifest.__setitem__("created_at", 123), "manifest"),
        (lambda manifest: manifest.__setitem__("environment", []), "manifest"),
        (
            lambda manifest: manifest["contracts"][0].__setitem__("system_prompt", "tampered prompt"),
            "manifest",
        ),
    ],
)
def test_resume_rejects_identity_compatible_manifest_shape_or_snapshot_mismatches_before_backend_creation(
    tmp_path: Path,
    manifest_mutator: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",))
    factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    run_benchmark(baseline, factories)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_mutator(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")

    resume_factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    with pytest.raises(ValueError, match=match):
        run_benchmark(_make_config(output_dir, resume=True, variants=("base",)), resume_factories)

    assert all(not factory.created for factory in resume_factories.values())


def test_resume_rejects_schedule_external_record_id_before_backend_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",))
    factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    run_benchmark(baseline, factories)

    results_path = output_dir / "results.jsonl"
    results_path.write_text(
        results_path.read_text(encoding="utf-8")
        + json.dumps({"record_id": "outside-schedule"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    resume_factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}

    with pytest.raises(ValueError, match="schedule"):
        run_benchmark(_make_config(output_dir, resume=True, variants=("base",)), resume_factories)

    assert all(not factory.created for factory in resume_factories.values())


def test_resume_rejects_record_id_field_mismatch_before_backend_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",))
    factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    run_benchmark(baseline, factories)

    results_path = output_dir / "results.jsonl"
    records = _load_jsonl(results_path)
    records[0]["variant"] = "sft"
    results_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    resume_factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}

    with pytest.raises(ValueError, match="record"):
        run_benchmark(_make_config(output_dir, resume=True, variants=("base",)), resume_factories)

    assert all(not factory.created for factory in resume_factories.values())


@pytest.mark.parametrize(
    ("record_mutator", "match"),
    [
        (
            lambda record: record["request"].pop("user"),
            "record",
        ),
        (
            lambda record: record["generation"].pop("error"),
            "record",
        ),
        (
            lambda record: record["score"].pop("strict_pass"),
            "record",
        ),
        (
            lambda record: record["case"].__setitem__("query", "tampered query"),
            "record",
        ),
    ],
)
def test_resume_rejects_record_structure_or_case_evidence_mismatches_before_backend_creation(
    tmp_path: Path,
    record_mutator: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",))
    factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    run_benchmark(baseline, factories)

    results_path = output_dir / "results.jsonl"
    records = _load_jsonl(results_path)
    record_mutator(records[0])
    results_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    resume_factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    with pytest.raises(ValueError, match=match):
        run_benchmark(_make_config(output_dir, resume=True, variants=("base",)), resume_factories)

    assert all(not factory.created for factory in resume_factories.values())


def test_resume_partial_completion_only_runs_pending_schedule_entries(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",))
    factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    run_benchmark(baseline, factories)

    records = _load_jsonl(output_dir / "results.jsonl")
    kept_records = records[:1]
    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in kept_records) + "\n",
        encoding="utf-8",
    )

    resume_factory = _BackendFactory("base", [_reply_batch, _reply_batch])
    run_benchmark(_make_config(output_dir, resume=True, variants=("base",)), {"base": resume_factory})

    generate_calls = resume_factory.created[0].generate_calls
    assert len(generate_calls) == 2
    assert [request.request_id for request in generate_calls[0]] == [records[1]["record_id"]]
    assert [request.request_id for request in generate_calls[1]] == [records[2]["record_id"], records[3]["record_id"]]


def test_resume_partial_rejects_missing_seed_before_tunnel_start_or_backend_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",))
    run_benchmark(baseline, {"base": _BackendFactory("base", [_reply_batch, _reply_batch])})

    records = _load_jsonl(output_dir / "results.jsonl")
    records[0].pop("seed")
    (output_dir / "results.jsonl").write_text(
        json.dumps(records[0], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tunnel = _Tunnel()
    resume_factory = _BackendFactory("base", [_reply_batch, _reply_batch])

    with pytest.raises(ValueError, match="record"):
        run_benchmark(
            _make_config(output_dir, resume=True, variants=("base",)), {"base": resume_factory}, tunnel=tunnel
        )

    assert tunnel.ensure_running_calls == 0
    assert tunnel.close_calls == 1
    assert not resume_factory.created


def test_resume_partial_rejects_tampered_sample_size_before_tunnel_start_or_backend_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",), sample_size=11)
    run_benchmark(baseline, {"base": _BackendFactory("base", [_reply_batch, _reply_batch])})

    records = _load_jsonl(output_dir / "results.jsonl")
    records[0]["sample_size"] = 12
    (output_dir / "results.jsonl").write_text(
        json.dumps(records[0], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tunnel = _Tunnel()
    resume_factory = _BackendFactory("base", [_reply_batch, _reply_batch])

    with pytest.raises(ValueError, match="record"):
        run_benchmark(
            _make_config(output_dir, resume=True, variants=("base",), sample_size=11),
            {"base": resume_factory},
            tunnel=tunnel,
        )

    assert tunnel.ensure_running_calls == 0
    assert tunnel.close_calls == 1
    assert not resume_factory.created


@pytest.mark.parametrize(
    ("config_mutator", "match"),
    [
        (
            lambda output_dir: _make_config(
                output_dir,
                cases=(
                    _make_case("case-1", exact_train=False, has_novel_atom=True),
                    _make_case("case-1", exact_train=True, has_novel_atom=False),
                ),
            ),
            "duplicate case_id",
        ),
        (
            lambda output_dir: _make_config(
                output_dir,
                contracts=(
                    _make_contract("controlled", "sha256:dup"),
                    _make_contract("production", "sha256:dup"),
                ),
            ),
            "duplicate contract fingerprint",
        ),
        (lambda output_dir: _make_config(output_dir, variants=("base", "base")), "duplicate variant"),
    ],
)
def test_run_benchmark_rejects_duplicate_schedule_inputs_before_manifest_or_backend_creation(
    tmp_path: Path,
    config_mutator: Callable[[Path], RunConfig],
    match: str,
) -> None:
    output_dir = tmp_path / "run"
    config = config_mutator(output_dir)
    factories = {
        "base": _BackendFactory("base", [_reply_batch, _reply_batch]),
        "sft": _BackendFactory("sft", [_reply_batch, _reply_batch]),
        "qwen_flash": _BackendFactory("qwen_flash", [_reply_batch, _reply_batch]),
    }

    with pytest.raises(ValueError, match=match):
        run_benchmark(config, factories)

    assert not (output_dir / "manifest.json").exists()
    assert all(not factory.created for factory in factories.values())


def test_run_benchmark_rejects_record_id_collisions_before_manifest_or_backend_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    config = _make_config(output_dir, variants=("base",))
    factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}
    original_record_id = runner_module._record_id

    def collide_record_ids(**kwargs: Any) -> str:
        case = kwargs["case"]
        if case.case_id == "case-2":
            return original_record_id(
                contract=kwargs["contract"],
                variant=kwargs["variant"],
                case=config.cases[0],
                model_id=kwargs["model_id"],
                max_tokens=kwargs["max_tokens"],
            )
        return original_record_id(**kwargs)

    monkeypatch.setattr(runner_module, "_record_id", collide_record_ids)

    with pytest.raises(ValueError, match="record_id collision"):
        run_benchmark(config, factories)

    assert not (output_dir / "manifest.json").exists()
    assert all(not factory.created for factory in factories.values())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda config: _make_config(
                config.output_dir,
                resume=True,
                contracts=(
                    _make_contract("controlled", "sha256:changed-controlled"),
                    config.contracts[1],
                ),
            ),
            "identity",
        ),
        (
            lambda config: _make_config(config.output_dir, resume=True, dataset_fingerprint="sha256:dataset-v2"),
            "identity",
        ),
        (
            lambda config: _make_config(
                config.output_dir,
                resume=True,
                model_ids={"base": "base-model-v2", "sft": "sft-model", "qwen_flash": "qwen-flash-model"},
            ),
            "identity",
        ),
        (lambda config: _make_config(config.output_dir, resume=True, max_tokens=64), "identity"),
        (lambda config: _make_config(config.output_dir, resume=True, variants=("base", "sft")), "identity"),
        (
            lambda config: _make_config(
                config.output_dir,
                resume=True,
                cases=(config.cases[0],),
            ),
            "identity",
        ),
    ],
)
def test_resume_rejects_identity_changes_before_creating_backends(
    tmp_path: Path,
    mutator: Callable[[RunConfig], RunConfig],
    match: str,
) -> None:
    baseline = _make_config(tmp_path / "run")
    factories = {variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in baseline.variants}
    run_benchmark(baseline, factories)

    changed = mutator(baseline)
    resume_factories = {
        variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in baseline.variants
    }
    with pytest.raises(ValueError, match=match):
        run_benchmark(changed, resume_factories)
    assert all(not factory.created for factory in resume_factories.values())


def test_backend_exception_and_malformed_results_create_backend_error_records_and_continue(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run")
    tunnel = _Tunnel()

    def raise_backend(requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        raise RuntimeError("secret endpoint failure")

    def malformed_results(requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [
            GenerationResult(
                request_id="wrong-id",
                text="[]",
                latency_ms=1.0,
                prompt_tokens=1,
                output_tokens=1,
                error=None,
            )
        ]

    factories = {
        "base": _BackendFactory("base", [raise_backend, _reply_batch]),
        "sft": _BackendFactory("sft", [malformed_results, _reply_batch]),
        "qwen_flash": _BackendFactory("qwen_flash", [_reply_batch, _reply_batch]),
    }

    run_benchmark(config, factories, tunnel=tunnel)

    records = _load_jsonl(config.output_dir / "results.jsonl")
    backend_error_records = [record for record in records if record["score"]["reason"] == "backend_error"]
    assert len(records) == 12
    assert len(backend_error_records) == 4
    assert {(record["contract"]["name"], record["variant"]) for record in backend_error_records} == {
        ("controlled", "base"),
        ("controlled", "sft"),
    }
    assert all("secret" not in record["generation"]["error"].lower() for record in backend_error_records)
    assert any(record["variant"] == "qwen_flash" and record["score"]["reason"] == "passed" for record in records)
    assert tunnel.close_calls == 1
    assert all(factory.created[0].close_calls == 1 for factory in factories.values())


def test_backend_generation_errors_are_sanitized_before_scoring_and_persistence(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run", variants=("base",))
    factories = {"base": _BackendFactory("base", [_error_batch, _error_batch])}

    run_benchmark(config, factories)

    records = _load_jsonl(config.output_dir / "results.jsonl")
    assert len(records) == 4
    assert all(record["score"]["reason"] == "backend_error" for record in records)
    assert all(record["generation"]["error"] == "backend_error: request failed" for record in records)
    assert all(record["score"]["parameter_errors"] == ["backend_error: request failed"] for record in records)


@pytest.mark.parametrize(
    ("generated", "match"),
    [
        (None, None),
        ("not-a-sequence", None),
        ([{"request_id": "wrong"}], None),
    ],
)
def test_non_sequence_none_and_wrong_element_batches_become_backend_errors_and_continue(
    tmp_path: Path,
    generated: Any,
    match: str | None,
) -> None:
    config = _make_config(tmp_path / "run", variants=("base", "sft"))
    factories = {
        "base": _BrokenFactory(_BrokenGenerateBackend(generated)),
        "sft": _BackendFactory("sft", [_reply_batch, _reply_batch]),
    }

    run_benchmark(config, factories)

    records = _load_jsonl(config.output_dir / "results.jsonl")
    base_records = [record for record in records if record["variant"] == "base"]
    sft_records = [record for record in records if record["variant"] == "sft"]
    assert all(record["score"]["reason"] == "backend_error" for record in base_records)
    assert all(record["generation"]["error"] == "backend_error: request failed" for record in base_records)
    assert all(record["score"]["reason"] == "passed" for record in sft_records)


def test_wrong_length_batch_becomes_backend_errors_and_continues(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run", variants=("base", "sft"))

    def wrong_length(requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [_reply_result(requests[0])]

    factories = {
        "base": _BackendFactory("base", [wrong_length, _reply_batch]),
        "sft": _BackendFactory("sft", [_reply_batch, _reply_batch]),
    }

    run_benchmark(config, factories)

    records = _load_jsonl(config.output_dir / "results.jsonl")
    controlled_base = [
        record for record in records if record["contract"]["name"] == "controlled" and record["variant"] == "base"
    ]
    assert len(controlled_base) == 2
    assert all(record["score"]["reason"] == "backend_error" for record in controlled_base)
    assert any(record["variant"] == "sft" and record["score"]["reason"] == "passed" for record in records)


def test_same_length_wrong_ids_batch_becomes_backend_errors_and_continues(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run", variants=("base", "sft"))

    def reordered_ids(requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [
            _reply_result(
                GenerationRequest(
                    request_id=requests[1].request_id,
                    system=requests[1].system,
                    user=requests[1].user,
                )
            ),
            _reply_result(
                GenerationRequest(
                    request_id=requests[0].request_id,
                    system=requests[0].system,
                    user=requests[0].user,
                )
            ),
        ]

    factories = {
        "base": _BackendFactory("base", [reordered_ids, _reply_batch]),
        "sft": _BackendFactory("sft", [_reply_batch, _reply_batch]),
    }

    run_benchmark(config, factories)

    records = _load_jsonl(config.output_dir / "results.jsonl")
    controlled_base = [
        record for record in records if record["contract"]["name"] == "controlled" and record["variant"] == "base"
    ]
    assert len(controlled_base) == 2
    assert all(record["score"]["reason"] == "backend_error" for record in controlled_base)
    assert any(record["variant"] == "sft" and record["score"]["reason"] == "passed" for record in records)


def test_close_attempts_all_backends_and_tunnel_then_raises_first_error(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run", variants=("base", "sft"))
    base_backend = _RaisingCloseBackend("base", "base close failed")
    sft_backend = _RaisingCloseBackend("sft", "sft close failed")
    tunnel = _RaisingTunnel("tunnel close failed")
    factories = {
        "base": _BrokenFactory(base_backend),
        "sft": _BrokenFactory(sft_backend),
    }

    with pytest.raises(RuntimeError, match="base close failed"):
        run_benchmark(config, factories, tunnel=tunnel)

    assert base_backend.close_calls == 1
    assert sft_backend.close_calls == 1
    assert tunnel.close_calls == 1


def test_primary_keyboardinterrupt_still_closes_all_resources_and_reraises_primary(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run", variants=("base", "sft"))

    def _interrupt_batch(requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        raise KeyboardInterrupt("primary interrupted")

    base_backend = _BaseExceptionCloseBackend("base", SystemExit("base cleanup exited"))
    sft_backend = _BaseExceptionCloseBackend(
        "sft",
        KeyboardInterrupt("sft cleanup interrupted"),
        scripted_calls=[_reply_batch, _interrupt_batch],
    )
    tunnel = _BaseExceptionTunnel(SystemExit("tunnel cleanup exited"))
    factories = {
        "base": _BrokenFactory(base_backend),
        "sft": _BrokenFactory(sft_backend),
    }

    with pytest.raises(KeyboardInterrupt, match="primary interrupted"):
        run_benchmark(config, factories, tunnel=tunnel)

    assert base_backend.close_calls == 1
    assert sft_backend.close_calls == 1
    assert tunnel.close_calls == 1


def test_cleanup_baseexceptions_propagate_first_error_after_closing_all_resources(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run", variants=("base", "sft"))
    base_backend = _BaseExceptionCloseBackend("base", SystemExit("base cleanup exited"))
    sft_backend = _BaseExceptionCloseBackend("sft", KeyboardInterrupt("sft cleanup interrupted"))
    tunnel = _BaseExceptionTunnel(SystemExit("tunnel cleanup exited"))
    factories = {
        "base": _BrokenFactory(base_backend),
        "sft": _BrokenFactory(sft_backend),
    }

    with pytest.raises(SystemExit, match="base cleanup exited"):
        run_benchmark(config, factories, tunnel=tunnel)

    assert base_backend.close_calls == 1
    assert sft_backend.close_calls == 1
    assert tunnel.close_calls == 1


def test_factory_error_closes_created_backends_and_tunnel(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run", variants=("base", "sft"))
    base_factory = _BackendFactory("base", [_reply_batch, _reply_batch])
    factories = {
        "base": base_factory,
        "sft": _BrokenFactory(exc=RuntimeError("factory failed")),
    }
    tunnel = _Tunnel()

    with pytest.raises(RuntimeError, match="factory failed"):
        run_benchmark(config, factories, tunnel=tunnel)

    assert base_factory.created[0].close_calls == 1
    assert tunnel.close_calls == 1


def test_resume_preflight_identity_mismatch_closes_tunnel_without_starting_it(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    baseline = _make_config(output_dir, variants=("base",))
    run_benchmark(baseline, {"base": _BackendFactory("base", [_reply_batch, _reply_batch])})

    tunnel = _Tunnel()
    resume_factories = {"base": _BackendFactory("base", [_reply_batch, _reply_batch])}

    with pytest.raises(ValueError, match="identity"):
        run_benchmark(
            _make_config(output_dir, resume=True, variants=("base",), dataset_fingerprint="sha256:dataset-v2"),
            resume_factories,
            tunnel=tunnel,
        )

    assert tunnel.ensure_running_calls == 0
    assert tunnel.close_calls == 1
    assert all(not factory.created for factory in resume_factories.values())


def test_preflight_keyboardinterrupt_closes_tunnel_without_starting_and_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path / "run", variants=("base",))
    tunnel = _Tunnel()
    factory = _BackendFactory("base", [_reply_batch, _reply_batch])

    def interrupt(config: RunConfig) -> dict[str, runner_module._ScheduleEntry]:
        raise KeyboardInterrupt("preflight interrupted")

    monkeypatch.setattr(runner_module, "_build_expected_schedule", interrupt)

    with pytest.raises(KeyboardInterrupt, match="preflight interrupted"):
        run_benchmark(config, {"base": factory}, tunnel=tunnel)

    assert tunnel.ensure_running_calls == 0
    assert tunnel.close_calls == 1
    assert not factory.created


def test_manifest_and_results_are_valid_json_without_credentials(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "run")
    factories = {variant: _BackendFactory(variant, [_reply_batch, _reply_batch]) for variant in config.variants}

    run_benchmark(config, factories)

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line) for line in (config.output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert isinstance(manifest, dict)
    assert all(isinstance(record, dict) for record in results)
    _assert_no_secrets(manifest)
    _assert_no_secrets(results)
