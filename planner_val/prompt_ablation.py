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
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from planner_val.backends import CompletionBackend, GenerationRequest, OpenAICompatibleBackend
from planner_val.challenge_eval import load_challenge_cases
from planner_val.validation_eval import ValidationCase, evaluate_validation


_SEMANTIC_RULES = """

补充语义与参数绑定规则：
1. tools 及其 arguments_schema 是本轮工具、enum 和参数的唯一事实来源。即使某个合法值未在训练样本中出现，也必须使用当前 schema 中与用户表达对应的值，不得替换成更熟悉的旧值。
2. 用户明确给出的城市、经纬度、速度、步数、角度和次数必须从当前 instruction 原样提取，不得召回其他案例的固定参数。英文 west longitude 和带负号的经纬度必须保留负号。
3. RobotDance 名称映射：雄心壮志舞或大展宏图=ambition_dance；埃及摇摆=egyptian_shake；Go Cortis=go_cortis_dance；idol dance number two 或女团舞二=idol_dance_2；原地舞或美美桑内=one_spot_dance；power-up dance 或助力舞=power_up_dance；单人摇摆舞或孤身摇=solo_shake；Whatever 或管他什么音乐=whatever；Gee=gee；victory dance 或胜利之舞=victory_dance。
4. RobotGesture 名称映射：击掌=high_five；挥手=wave_greet_bye；鞠躬=bow；鼓掌=clap；点头=nod。
5. get_weather 的 query_type：当前或现在=now；未来 24 小时或明天=24h；未来七天或未来一周=7d；当前加未来七天=now_7d。
6. 公网、公开网页、最新新闻、价格和近期事件使用 web_search；公司内部资料、内部知识库和产品私有文档使用 rag_query；机器人自身当前电量、日期、时间、位置、型号和当前动作使用 robot_status。不得用其他工具代替这些明确的数据源。
7. HumanAction 平移速度写入对应方向轴并保留数值：前后用 x，左右横移用 y，左为正、右为负；转向使用 yaw，左为 1、右为 -1，并原样保留 degree。
""".rstrip()

_CONTROL_RULES = """

补充约束与话语范围规则：
1. 被否定、引用或假设的内容不得生成任务；只为用户最终明确要求执行、查询或回复的内容生成任务。禁止项不是待执行项。
2. “不要”“不得”“严禁”“do not”“don't”“without”等约束的作用域必须保留，不得因其中出现工具或动作关键词而调用该工具。
3. 能力询问、解释、讨论、复述、转述和元问题使用 reply，不实际调用其提到的动作或查询工具。
4. 用户纠正前述意图时，以纠正后的最终意图为准，不为被否定的旧意图生成任务。
5. “一次”“只做一次”“不要重复”“one time only”“exactly once”只约束当前动作次数，不是独立任务；RobotGesture 和 RobotDance 的 repeat 只能省略或为 1。
6. “说一句”“告诉我”“回复”“finish by saying”是 reply 意图，不得转换成动作或查询工具。
7. 输出前逐项检查：删除所有被禁止、被引用、被假设和已被纠正的调用；不得添加 instruction 中不存在的任务。
""".rstrip()

_VARIANT_DESCRIPTIONS = {
    "A": "baseline controlled validation prompt",
    "B": "baseline plus schema semantics and dynamic value binding",
    "C": "baseline plus negation, quotation, capability, correction, and count rules",
    "D": "baseline plus both semantic and control rules",
}


class _PromptCase(Protocol):
    case_id: str
    system: str
    user: str


def build_ablation_requests(case: _PromptCase) -> Mapping[str, GenerationRequest]:
    """Build the four controlled prompt variants for one evaluation case."""
    systems = {
        "A": case.system,
        "B": case.system + _SEMANTIC_RULES,
        "C": case.system + _CONTROL_RULES,
        "D": case.system + _SEMANTIC_RULES + _CONTROL_RULES,
    }
    return {
        variant: GenerationRequest(request_id=case.case_id, system=system, user=case.user)
        for variant, system in systems.items()
    }


def evaluate_prompt_ablation(
    cases: Sequence[ValidationCase],
    backends: Mapping[str, CompletionBackend],
    output_dir: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
    dataset_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate A/B/C/D prompt variants through the existing scorer."""
    variants = ("A", "B", "C", "D")
    if tuple(backends) != variants:
        raise ValueError("prompt ablation backends must be ordered A, B, C, D")
    request_builders = {
        variant: lambda case, selected=variant: build_ablation_requests(case)[selected] for variant in variants
    }
    return evaluate_validation(
        cases,
        backends,
        output_dir,
        request_builders=request_builders,
        metadata=metadata,
        run_config=run_config,
        dataset_info=dataset_info,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_artifact(case: ValidationCase) -> dict[str, Any]:
    requests = build_ablation_requests(case)
    return {
        "schema_version": 1,
        "invariant": "Only the system prompt changes; each case keeps its original user payload and tool schema.",
        "variants": {
            variant: {
                "description": _VARIANT_DESCRIPTIONS[variant],
                "system_prompt": request.system,
                "system_prompt_sha256": hashlib.sha256(request.system.encode("utf-8")).hexdigest(),
            }
            for variant, request in requests.items()
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled SFT prompt ablations on the OOD challenge set.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vllm-model", default="gemma3-270m-sft")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cases = load_challenge_cases(args.challenge, args.package)
    if len({case.system for case in cases}) != 1:
        raise ValueError("prompt ablation requires one shared baseline system prompt")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "prompt_ablation.json"
    artifact_bytes = (json.dumps(_prompt_artifact(cases[0]), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    artifact_path.write_bytes(artifact_bytes)
    artifact_metadata = {
        "artifact": {
            "path": artifact_path.name,
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        },
        "invariant": "same checkpoint, user payload, tool schema, decoding settings, and scorer",
        "variants": dict(_VARIANT_DESCRIPTIONS),
    }

    backends = {
        variant: OpenAICompatibleBackend(
            model=args.vllm_model,
            base_url=args.vllm_url,
            api_key="EMPTY",
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
        for variant in _VARIANT_DESCRIPTIONS
    }
    try:
        summary = evaluate_prompt_ablation(
            cases,
            backends,
            args.output_dir,
            metadata={"prompt_ablation": artifact_metadata},
            run_config={
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": False,
                "model": {"name": args.vllm_model, "base_url": args.vllm_url},
                "variants": dict(_VARIANT_DESCRIPTIONS),
            },
            dataset_info={
                "challenge_path": str(args.challenge),
                "challenge_sha256": _file_sha256(args.challenge),
                "package": str(args.package),
                "package_sha256": _file_sha256(args.package),
            },
        )
    finally:
        for backend in backends.values():
            backend.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
