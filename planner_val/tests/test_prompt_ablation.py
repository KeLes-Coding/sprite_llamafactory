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
from types import SimpleNamespace

from planner_val.backends import GenerationResult
from planner_val.models import ContractSnapshot, EvalCase, ExpectedTask
from planner_val.prompt_ablation import build_ablation_requests, evaluate_prompt_ablation, main
from planner_val.validation_eval import ValidationCase


def _validation_case() -> ValidationCase:
    expected_raw = '[{"kind":"reply","instruction":"acknowledged"}]'
    contract = ContractSnapshot(
        name="controlled",
        system_prompt="base system",
        tools_context=(),
        tool_schemas={},
        tool_kinds={},
        source={"case_id": "challenge-001"},
        fingerprint="sha256:test",
    )
    return ValidationCase(
        case_id="challenge-001",
        system="base system",
        user='{"tools":[],"instruction":"reply only"}',
        expected_raw=expected_raw,
        query="reply only",
        eval_case=EvalCase(
            case_id="challenge-001",
            query="reply only",
            expected=(ExpectedTask(kind="reply", instruction="acknowledged"),),
            forbidden_functions=frozenset(),
            level="1",
            language="en",
            difficulty="medium",
            polarity="positive",
            source_atom_ids=frozenset(),
        ),
        contract=contract,
        level="1",
        language="en",
        polarity="positive",
        provenance="negative_constraint",
        kinds=("reply",),
        tools=(),
        actions=(),
        source_atom_ids=(),
        source_item_id="challenge-001",
        example_id="challenge-001",
    )


def test_build_ablation_requests_changes_only_the_selected_system_rules() -> None:
    user = json.dumps(
        {
            "tools": [
                {
                    "name": "RobotDance",
                    "kind": "action",
                    "arguments_schema": {
                        "type": "object",
                        "properties": {"dance": {"enum": ["ambition_dance"]}},
                    },
                }
            ],
            "instruction": "来一段雄心壮志舞",
        },
        ensure_ascii=False,
    )
    case = SimpleNamespace(case_id="challenge-001", system="base system", user=user)

    requests = build_ablation_requests(case)

    assert list(requests) == ["A", "B", "C", "D"]
    assert all(request.user == user for request in requests.values())
    assert requests["A"].system == "base system"
    assert "ambition_dance" in requests["B"].system
    assert "被否定、引用或假设的内容不得生成任务" not in requests["B"].system
    assert "ambition_dance" not in requests["C"].system
    assert "被否定、引用或假设的内容不得生成任务" in requests["C"].system
    assert "ambition_dance" in requests["D"].system
    assert "被否定、引用或假设的内容不得生成任务" in requests["D"].system


def test_evaluate_prompt_ablation_runs_all_variants_with_the_same_case_payload(tmp_path) -> None:
    case = _validation_case()
    expected_raw = case.expected_raw

    class FakeBackend:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, requests):
            self.requests.extend(requests)
            return [GenerationResult(request.request_id, expected_raw, 1.0, 10, 5, None) for request in requests]

        def close(self) -> None:
            return None

    backends = {variant: FakeBackend() for variant in ("A", "B", "C", "D")}

    summary = evaluate_prompt_ablation([case], backends, tmp_path)

    assert list(summary["models"]) == ["A", "B", "C", "D"]
    assert all(summary["models"][variant]["benchmark_compatible_pass"] == 1 for variant in backends)
    assert all(backend.requests[0].user == case.user for backend in backends.values())
    assert backends["A"].requests[0].system == case.system
    assert "ambition_dance" in backends["B"].requests[0].system
    assert "ambition_dance" not in backends["C"].requests[0].system
    assert "ambition_dance" in backends["D"].requests[0].system


def test_cli_runs_four_variants_and_records_prompt_artifact(tmp_path, monkeypatch) -> None:
    case = _validation_case()
    created = []

    class FakeOpenAIBackend:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.requests = []
            self.closed = False
            created.append(self)

        def generate(self, requests):
            self.requests.extend(requests)
            return [GenerationResult(request.request_id, case.expected_raw, 1.0, 10, 5, None) for request in requests]

        def close(self) -> None:
            self.closed = True

    package_path = tmp_path / "dataset.zip"
    challenge_path = tmp_path / "challenge.json"
    package_path.write_bytes(b"package")
    challenge_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("planner_val.prompt_ablation.load_challenge_cases", lambda *_: [case])
    monkeypatch.setattr("planner_val.prompt_ablation.OpenAICompatibleBackend", FakeOpenAIBackend)

    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "--package",
            str(package_path),
            "--challenge",
            str(challenge_path),
            "--output-dir",
            str(output_dir),
            "--vllm-model",
            "deployed-sft",
            "--concurrency",
            "3",
        ]
    )

    assert exit_code == 0
    assert len(created) == 4
    assert all(instance.kwargs["model"] == "deployed-sft" for instance in created)
    assert all(instance.kwargs["concurrency"] == 3 for instance in created)
    assert all(instance.closed for instance in created)
    assert all(instance.requests[0].user == case.user for instance in created)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["models"]) == ["A", "B", "C", "D"]
    assert summary["prompt_ablation"]["artifact"]["path"] == "prompt_ablation.json"
    artifact = json.loads((output_dir / "prompt_ablation.json").read_text(encoding="utf-8"))
    assert list(artifact["variants"]) == ["A", "B", "C", "D"]
    assert artifact["variants"]["A"]["system_prompt"] == case.system
    assert artifact["variants"]["B"]["system_prompt"] != artifact["variants"]["C"]["system_prompt"]
