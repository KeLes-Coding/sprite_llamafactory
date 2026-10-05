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

"""Offline v2 tool descriptions; never mutate the frozen training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from planner_val.backends import OpenAICompatibleBackend
from planner_val.challenge_eval import load_challenge_cases
from planner_val.validation_eval import ValidationCase, _contract, _expected_tasks, evaluate_validation


# Meanings are based on Agent tool_descriptions.yaml; egyptian_shake is also
# present in the Agent's trigger-panel dance mapping. New enum values must be
# reviewed before this experimental schema accepts them.
_DANCES = {
    "abracadabr_dance": "扭胯舞",
    "all_things_grow_dance": "万物生",
    "ambition_dance": "大展宏图",
    "apt_dance": "APT 舞",
    "egyptian_shake": "埃及摇摆",
    "gee": "Gee 同名舞",
    "gentleman": "Gentleman 同名舞",
    "go_cortis_dance": "Go Cortis 舞",
    "idol_dance_1": "女团舞 1",
    "idol_dance_2": "女团舞 2",
    "karla_ok": "卡啦永远 OK",
    "lets_bounce": "来个蹦蹦",
    "luv_each_other": "相亲相爱",
    "one_and_only_dance": "热烈",
    "one_spot_dance": "美美桑内",
    "popping": "机械舞",
    "power_up_dance": "助力舞",
    "pulp_fiction_dance": "低俗小说",
    "smooth_sailing_and_prosperity": "顺风顺水顺财神",
    "solo_shake": "孤身摇",
    "swag_dance": "展臂舞",
    "sweep_kick_dance": "扫腿舞",
    "victory_dance": "胜利之舞",
    "warm_up_dance": "热场舞",
    "whatever": "管他什么音乐",
}
_GESTURES = {
    "blow_kisses_multi": "连续飞吻",
    "bow": "鞠躬",
    "clap": "鼓掌",
    "curtain_bow": "谢幕",
    "hand_heart": "上抬手比心",
    "high_five": "击掌",
    "left_hand_side_heart": "左手侧比心",
    "nod": "点头",
    "right_hand_side_heart": "右手侧比心",
    "shake_hands": "握手",
    "shake_head": "摇头",
    "this_way_please": "请这边走的指引手势",
    "wave_greet_bye": "挥手打招呼或道别",
}
_STATUS = {value: value for value in ("电量", "日期", "时间", "位置", "型号", "当前动作")}
_WEATHER_TYPES = {
    "now": "当前天气",
    "24h": "未来 24 小时的逐小时天气",
    "7d": "未来七天的天气预报",
    "now_7d": "当前天气和未来七天的天气预报",
}


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable(item) for item in value]
    return deepcopy(value)


def _describe_enum(property_schema: dict[str, Any], meanings: Mapping[str, str]) -> None:
    values = property_schema.get("enum")
    if not isinstance(values, list) or any(value not in meanings for value in values):
        missing = [value for value in values or [] if value not in meanings]
        raise ValueError(f"unreviewed enum values: {missing}")
    property_schema["description"] = (
        "可选值与含义：" + "；".join(f"{value}: {meanings[value]}" for value in values) + "。"
    )


def enrich_tool_schema(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy the controlled contract and clarify meanings without changing tool identity.

    This is an offline experimental contract. The live Agent weather handler still
    requires coordinates; publishing this schema to that runtime is out of scope.
    """
    enriched = _mutable(tools)
    for tool in enriched:
        name = tool["name"]
        schema = tool["arguments_schema"]
        if name == "RobotDance":
            props = schema["properties"]
            _describe_enum(props["dance"], _DANCES)
            props["repeat"]["description"] = "可省略；指定时只能是 1，表示执行一次，不产生额外任务。"
        elif name == "RobotGesture":
            props = schema["properties"]
            _describe_enum(props["gesture"], _GESTURES)
            props["repeat"]["description"] = "可省略；指定时只能是 1，表示执行一次，不产生额外任务。"
        elif name == "robot_status":
            _describe_enum(schema["properties"]["query"], _STATUS)
        elif name == "get_weather":
            props = schema["properties"]
            props.pop("latitude", None)
            props.pop("longitude", None)
            schema["required"] = [field for field in schema["required"] if field not in {"latitude", "longitude"}]
            props["city"]["description"] = "用户要求查询的城市名；从本次请求提取，不用其他样本的城市代替。"
            _describe_enum(props["query_type"], _WEATHER_TYPES)
        elif name in {"web_search", "rag_query"}:
            schema["properties"]["query"]["description"] = "保留当前用户子意图的检索主题，不套用其他请求的固定查询。"
        elif name == "HumanAction":
            for branch in schema["oneOf"]:
                props = branch["properties"]
                parameters = props.get("parameters")
                if not isinstance(parameters, dict):
                    continue
                fields = parameters["properties"]
                if "step" in fields:
                    fields["step"]["description"] = "移动步数；用户明确给出时保留该数值，否则为 1。"
                if "degree" in fields:
                    fields["degree"]["description"] = "用户要求的转向角度（度），不得替换成常见角度。"
    return enriched


def adapt_case_to_schema_v2(case: ValidationCase) -> ValidationCase:
    """Project a frozen evaluation row onto v2 without editing its source."""
    user_payload = json.loads(case.user)
    user_payload["tools"] = enrich_tool_schema(user_payload["tools"])
    expected = json.loads(case.expected_raw)
    weather_changed = False
    for task in expected:
        if task.get("name") == "get_weather":
            task["arguments"].pop("latitude", None)
            task["arguments"].pop("longitude", None)
            weather_changed = True

    expected_raw = (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) if weather_changed else case.expected_raw
    )
    user = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
    return replace(
        case,
        user=user,
        expected_raw=expected_raw,
        eval_case=replace(case.eval_case, expected=_expected_tasks(expected_raw, case_id=case.case_id)),
        contract=_contract(case.system, user_payload, case_id=case.case_id),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the frozen challenge under the experimental v2 contract."""
    parser = argparse.ArgumentParser(description="Run the 50-case challenge with clarified tool schemas.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="gemma3-270m-sft")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory")

    cases = [adapt_case_to_schema_v2(case) for case in load_challenge_cases(args.challenge, args.package)]
    args.output_dir.mkdir(parents=True)
    schema_artifact = {
        "schema_version": "tool-schema-description/v2",
        "system": cases[0].system,
        "tools": json.loads(cases[0].user)["tools"],
    }
    artifact_bytes = (json.dumps(schema_artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    (args.output_dir / "tool_schema.json").write_bytes(artifact_bytes)
    backend = OpenAICompatibleBackend(
        model=args.model,
        base_url=args.vllm_url,
        api_key="EMPTY",
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
    )
    try:
        summary = evaluate_validation(
            cases,
            {args.model: backend},
            args.output_dir,
            metadata={"schema_experiment": "tool-schema-description/v2 (offline only)"},
            run_config={"model": args.model, "concurrency": args.concurrency, "max_tokens": args.max_tokens},
            dataset_info={
                "package_sha256": hashlib.sha256(args.package.read_bytes()).hexdigest(),
                "challenge_sha256": hashlib.sha256(args.challenge.read_bytes()).hexdigest(),
                "contract_fingerprint": cases[0].contract.fingerprint,
                "tool_schema_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            },
        )
    finally:
        backend.close()
    quality = summary["models"][args.model]["quality"]
    print(json.dumps({"cases": summary["cases"], "quality": quality}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
