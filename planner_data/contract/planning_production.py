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

"""Configuration and contract freezing for ActionSequence production data."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from planner_data.common import is_range_spec, value_in_range
from planner_data.contract.planning_contract import (
    ActionSequenceContractBundle,
    compile_plan,
    contract_ref,
    load_contract_bundle,
)


_PACKAGE_DIR = Path(__file__).resolve().parents[1]

SOURCE_AGENT_VERSION = "local_omni-dev_9fb2bfe"
TARGET_AGENT_VERSION = "subagent_Sprite-action-sequence-reply-simplification_3911119"
SELECTED_TOOLS = (
    "HumanAction",
    "RobotGesture",
    "RobotDance",
    "get_weather",
    "web_search",
    "rag_query",
    "robot_status",
)
EXCLUDED_TOOLS = ("EndChat", "get_visual_info", "ActionSequence")
TOOL_KINDS = {
    "HumanAction": "action",
    "RobotGesture": "action",
    "RobotDance": "action",
    "robot_status": "query",
    "rag_query": "query",
    "get_weather": "query",
    "web_search": "query",
}
CURATED_CONTRACT_PATH = _PACKAGE_DIR / "contract/data/action-sequence-contract-v1.json"
CONTRACT_V2_PATH = _PACKAGE_DIR / "contract/data/action-sequence-contract-v2.json"
# Named contracts selectable by jobs and recorded on intents for compilation.
PLANNING_CONTRACTS = {"v1": CURATED_CONTRACT_PATH, "v2": CONTRACT_V2_PATH}
# Matrix-generated atoms that carry their own rule-derived params and planner
# target; kept outside L0 so the annotation dataset hash stays stable.
CURATED_ATOMS_PATH = _PACKAGE_DIR / "atoms/data/curated-atoms.jsonl"
# Frozen snapshot of the reviewed L0 projection (benchmark l0_adapter output for
# SOURCE/TARGET_AGENT_VERSION, curated atoms excluded) used for dedup/avoid lists.
LEGACY_L0_POOL_PATH = _PACKAGE_DIR / "atoms/data/legacy-l0-pool.json"
# Reviewed L0 atoms dropped from planning data (stand_up was over-represented).
_RETIRED_ATOM_IDS = frozenset(
    "l0_humanaction_stand_up_imit" + (f"_{n}" if n else "")
    for n in (0, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 3, 41, 44, 49, 52, 54, 56, 58, 7, 8, 9)
) | {"l0_web_search_funny_joke"}  # a joke request is a reply, not a search

# Enum meanings for the v2 contract descriptions (from Agent tool_descriptions.yaml).
# New enum values must be reviewed here before the v2 contract accepts them.
DANCE_MEANINGS = {
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
GESTURE_MEANINGS = {
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
STATUS_MEANINGS = {value: value for value in ("电量", "日期", "时间", "位置", "型号", "当前动作")}
WEATHER_TYPE_MEANINGS = {
    "now": "当前天气",
    "24h": "未来 24 小时的逐小时天气（今晚、明天属于此项）",
    "7d": "未来七天的天气预报",
    "now_7d": "当前天气和未来七天的天气预报",
}
# Reviewed default when a weather request names no city.
DEFAULT_WEATHER_CITY = "深圳"
# Arguments a newer contract removed; dropped from legacy atoms only when the
# target contract no longer declares them.
_LEGACY_ARGUMENTS = {"get_weather": ("latitude", "longitude")}
# Reviewed turn-degree rules, checked in order against the atom's wording:
# an explicit angle keeps its value; circle -> 360; turn around -> 180;
# vague "一点/一下" -> 15; a bare turn -> 90.
_EXPLICIT_DEGREE = re.compile(r"\d+\s*(?:度|°|degree)|[一二三四五六七八九十百]+度", re.IGNORECASE)
_TURN_DEGREE_RULES = (
    (360, re.compile(r"圈|circle|spin|360", re.IGNORECASE)),
    (180, re.compile(r"向后转|往后转|转.{0,2}身|掉头|后面|背对|around|backward|about[- ]face", re.IGNORECASE)),
    (15, re.compile(r"一点|一下|一些|一点儿|一丢|稍|微|轻轻|a little|a bit|slightly|briefly|a touch", re.IGNORECASE)),
)
_BARE_TURN_DEGREE = 90
# Reviewed forward/backward magnitude: every HumanAction move along x uses ±0.6.
_MOVE_X_MAGNITUDE = 0.6
# Weather atoms kept out of planning data until the contract covers sun times.
_EXCLUDED_WEATHER = re.compile(r"日出|日落|天黑|天亮|落山|sunrise|sunset|dusk|dawn", re.IGNORECASE)


def _atom_texts(atom: dict[str, Any]) -> str:
    target = atom.get("planner_target") or {}
    instructions = target.get("instructions") if isinstance(target, dict) else None
    parts = [str(atom.get("query") or "")]
    if isinstance(instructions, dict):
        parts.extend(str(value) for value in instructions.values())
    return "\n".join(parts)


def _excluded_atom(atom: dict[str, Any]) -> bool:
    return atom.get("expected_function") == "get_weather" and bool(
        _EXCLUDED_WEATHER.search(_atom_texts(atom))
    )


def _canonical_turn_degree(atom: dict[str, Any]) -> dict[str, Any]:
    """Apply the reviewed turn-degree rules to in-place HumanAction turns."""
    text = _atom_texts(atom)
    if _EXPLICIT_DEGREE.search(text):
        return atom
    degree = next(
        (value for value, pattern in _TURN_DEGREE_RULES if pattern.search(text)),
        _BARE_TURN_DEGREE,
    )
    atom = deepcopy(atom)
    for step in atom.get("expected_steps") or [atom]:
        params = step.get("expected_params") if isinstance(step, dict) else None
        parameters = params.get("parameters") if isinstance(params, dict) else None
        if (
            isinstance(parameters, dict)
            and "degree" in parameters
            and parameters.get("yaw") not in (0, None)
        ):
            parameters["degree"] = degree
    return atom


def _canonical_move_x(atom: dict[str, Any]) -> dict[str, Any]:
    """Apply the reviewed ±0.6 magnitude to forward/backward HumanAction moves."""
    atom = deepcopy(atom)
    for step in atom.get("expected_steps") or [atom]:
        params = step.get("expected_params") if isinstance(step, dict) else None
        parameters = params.get("parameters") if isinstance(params, dict) else None
        if not isinstance(parameters, dict) or "degree" in parameters or "x" not in parameters:
            continue
        x = parameters["x"]
        if is_range_spec(x):
            candidates = [v for v in (_MOVE_X_MAGNITUDE, -_MOVE_X_MAGNITUDE) if value_in_range(x, v)]
            if len(candidates) == 1:
                parameters["x"] = candidates[0]
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and x != 0:
            parameters["x"] = _MOVE_X_MAGNITUDE if x > 0 else -_MOVE_X_MAGNITUDE
    return atom


def _describe_enum(property_schema: dict[str, Any], meanings: dict[str, str]) -> None:
    values = property_schema.get("enum") or []
    missing = [value for value in values if value not in meanings]
    if missing:
        raise ValueError(f"unreviewed enum values: {missing}")
    property_schema["description"] = (
        "可选值与含义：" + "；".join(f"{value}: {meanings[value]}" for value in values) + "。"
    )


def build_contract_v2(bundle: ActionSequenceContractBundle) -> ActionSequenceContractBundle:
    """Derive the v2 contract: enum meanings, value-copy hints, city-only weather."""
    payload = bundle.model_dump(mode="json")
    payload["source"] = {
        "kind": "curated",
        "identity": {**payload["source"]["identity"], "description_revision": "v2"},
    }
    repeat_hint = "可省略；指定时只能是 1，表示执行一次，不产生额外任务。"
    for tool in payload["tools"]:
        name = tool["name"]
        schema = tool["arguments_schema"]
        if name == "RobotDance":
            _describe_enum(schema["properties"]["dance"], DANCE_MEANINGS)
            schema["properties"]["repeat"]["description"] = repeat_hint
        elif name == "RobotGesture":
            _describe_enum(schema["properties"]["gesture"], GESTURE_MEANINGS)
            schema["properties"]["repeat"]["description"] = repeat_hint
        elif name == "robot_status":
            _describe_enum(schema["properties"]["query"], STATUS_MEANINGS)
        elif name == "get_weather":
            props = schema["properties"]
            for key in _LEGACY_ARGUMENTS["get_weather"]:
                props.pop(key, None)
            schema["required"] = [key for key in schema["required"] if key in props]
            props["city"]["description"] = (
                f"用户要求查询的城市名；从本次请求提取，不用其他样本的城市代替；"
                f"未提及城市时用{DEFAULT_WEATHER_CITY}。"
            )
            _describe_enum(props["query_type"], WEATHER_TYPE_MEANINGS)
        elif name in {"web_search", "rag_query"}:
            schema["properties"]["query"]["description"] = "保留当前用户子意图的检索主题，不套用其他请求的固定查询。"
        elif name == "HumanAction":
            for branch in schema["oneOf"]:
                parameters = branch["properties"].get("parameters")
                if not isinstance(parameters, dict):
                    continue
                fields = parameters["properties"]
                if "step" in fields:
                    fields["step"]["description"] = "移动步数；用户明确给出时保留该数值，否则为 1。"
                if "degree" in fields:
                    fields["degree"]["description"] = "用户要求的转向角度（度），不得替换成常见角度。"
    return ActionSequenceContractBundle.model_validate(payload)


def write_contract_v2(output_path: Path = CONTRACT_V2_PATH) -> ActionSequenceContractBundle:
    """Regenerate the v2 contract file from the curated v1 contract."""
    bundle = build_contract_v2(load_contract_bundle(CURATED_CONTRACT_PATH))
    output_path.write_text(
        json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def _drop_legacy_arguments(step: dict[str, Any], bundle: ActionSequenceContractBundle) -> None:
    name = step.get("expected_function")
    params = step.get("expected_params")
    contract = next((tool for tool in bundle.tools if tool.name == name), None)
    if contract is None or not isinstance(params, dict):
        return
    declared = contract.arguments_schema.get("properties") or {}
    for key in _LEGACY_ARGUMENTS.get(str(name), ()):
        if key not in declared:
            params.pop(key, None)


def _normalize_schema_annotations(value: object) -> object:
    """Unwrap cloud descriptions accidentally serialized as nested objects."""
    if isinstance(value, list):
        return [_normalize_schema_annotations(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if (
            key == "description"
            and isinstance(item, dict)
            and set(item) == {"description"}
            and isinstance(item["description"], str)
        ):
            normalized[key] = item["description"]
        else:
            normalized[key] = _normalize_schema_annotations(item)
    return normalized


def _normalized_arguments_schema(function: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_schema_annotations(function.get("parameters") or {})
    if not isinstance(normalized, dict):
        raise ValueError("tool parameters must be an object schema")
    return normalized


def load_verified_planning_pool(
    *,
    tools: tuple[str, ...] = SELECTED_TOOLS,
    curated_path: Path = CURATED_ATOMS_PATH,
    snapshot_path: Path = LEGACY_L0_POOL_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Merge the frozen L0 projection snapshot with the curated atoms."""
    projected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    curated = load_curated_atoms(curated_path)
    merged = {
        tool: [
            row
            for row in [*projected.get(tool, []), *curated.get(tool, [])]
            if isinstance(row.get("planner_target"), dict) and row.get("id") not in _RETIRED_ATOM_IDS
        ]
        for tool in tools
    }
    return {tool: rows for tool, rows in merged.items() if rows}


def load_curated_atoms(path: Path = CURATED_ATOMS_PATH) -> dict[str, list[dict[str, Any]]]:
    """Read matrix-generated atoms grouped by tool (missing file -> empty)."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not path.is_file():
        return grouped
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            atom = json.loads(line)
            tool = atom["expected_function"] or atom["negative_tool"]
            grouped.setdefault(str(tool), []).append(atom)
    return grouped


def normalize_atom_for_planning(
    atom: dict[str, Any], bundle: ActionSequenceContractBundle
) -> dict[str, Any]:
    """Return a target-contract-valid copy of one annotated planning atom."""
    normalized = deepcopy(atom)
    target = normalized.get("planner_target")
    if not isinstance(target, dict):
        raise ValueError("atom is missing planner_target")

    if "test_input" not in normalized and "query" in normalized:
        normalized["test_input"] = normalized["query"]
    if "expected_steps" not in normalized and "expected_function" in normalized:
        normalized["expected_steps"] = [
            {
                "expected_function": normalized.get("expected_function"),
                "expected_params": deepcopy(normalized.get("expected_params")),
            }
        ]

    if target.get("kind") == "reply":
        normalized["expected_steps"] = []
    else:
        steps = normalized.get("expected_steps")
        if not isinstance(steps, list) or len(steps) != 1:
            raise ValueError("call atom must contain exactly one expected step")
        step = steps[0]
        if not isinstance(step, dict):
            raise ValueError("expected step must be an object")
        if step.get("expected_function") == "rag_query" and step.get("expected_params") is None:
            step["expected_params"] = {"query": str(normalized.get("test_input") or "").strip()}
        _drop_legacy_arguments(step, bundle)

    compiled = None
    refs = contract_ref(bundle)
    compiler_atom = {
        key: normalized[key]
        for key in ("id", "test_input", "expected_steps", "planner_target")
    }
    for language in ("zh", "en"):
        compiled = compile_plan(
            [compiler_atom],
            language=language,
            tools=bundle.tools,
            contract_ref=refs,
            provenance="generated",
        )

    assert compiled is not None
    task = compiled.tasks[0]
    if task.kind != "reply":
        normalized["expected_steps"][0]["expected_params"] = deepcopy(task.arguments)
    # ``expected_steps`` is the contract-facing shape, but the case builder reads
    # the flat ``expected_function``/``expected_params`` pair when it turns an
    # atom into an intent. Without this write-back a training run silently ships
    # the *pre-normalization* values — ``rag_query`` with ``params: null`` and
    # range specs like ``{"lt": 0}`` where the target contract demands a concrete
    # number — i.e. the normalization would apply to nothing that ends up on disk.
    steps = normalized.get("expected_steps") or []
    if steps:
        normalized["expected_function"] = steps[0].get("expected_function")
        normalized["expected_params"] = deepcopy(steps[0].get("expected_params"))
    else:
        normalized.pop("expected_function", None)
        normalized.pop("expected_params", None)
    return normalized
