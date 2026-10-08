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

"""Coverage-matrix generation of curated ActionSequence planning atoms.

Each matrix cell fixes the target arguments. The LLM only writes the user
wording and the matching planner instructions; every sentence is re-derived by
fixed wording rules and dropped when its wording does not imply the cell's
arguments, so arguments never come from the model.

Usage::

    python -m planner_data.atoms.planning_atoms generate --tools HumanAction get_weather
    python -m planner_data.atoms.planning_atoms audit
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from planner_data import llm
from planner_data.atoms import planning_slots
from planner_data.common import _norm_text, find_env_value
from planner_data.contract import planning_production as pp


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from planner_data.contract.planning_contract import ActionSequenceContractBundle


_MAX_ROUNDS = 3

# ---------------------------------------------------------------------------
# Wording rules: text -> arguments
# ---------------------------------------------------------------------------
_TURN = re.compile(r"转|旋")
_BACK = re.compile(r"后退|倒退|退|(?:往|向|朝)后(?!转|面)")
_FORWARD = re.compile(r"前进|(?:往|向|朝)前")
_LEFT = re.compile(r"左")
_RIGHT = re.compile(r"右")
_POSTURES = (
    ("sit_down", re.compile(r"坐")),
    ("lie_down", re.compile(r"躺")),
    ("stand_up", re.compile(r"站|起身|起立|直立|立正|立起|起来")),
)
_STEP = re.compile(planning_slots._ZH_NUMBER + r"\s*步")
_DEGREE = re.compile(r"(\d+|[零一二两三四五六七八九十]+)\s*(?:度|°)")

_CITY_NAMES = sorted(
    {zh for pool in planning_slots._CITIES.values() for zh, _en in pool}, key=len, reverse=True
)
_PLACE_HINT = re.compile(r"[省市县]")
_NOW = re.compile(r"现在|当前|此刻|今天|今日|这会|眼下|目前|实时")
_H24 = re.compile(r"今晚|今夜|明天|明早|明晚|明日|几点|待会|等会|一会儿|小时|夜里|晚上|凌晨|上午|中午|下午|傍晚")
_D7 = re.compile(r"这周|本周|周末|几天|一周|七天|7天|下周|星期|礼拜")


def _turn_degree(text: str) -> int | None:
    if pp._EXPLICIT_DEGREE.search(text):
        matches = _DEGREE.findall(text)
        return planning_slots._parse_cn(matches[0]) if len(matches) == 1 else None
    return next(
        (value for value, pattern in pp._TURN_DEGREE_RULES if pattern.search(text)),
        pp._BARE_TURN_DEGREE,
    )


def derive_human_action(text: str) -> dict[str, Any] | None:
    """Arguments implied by one single-intent HumanAction wording, else None."""
    postures = [action for action, pattern in _POSTURES if pattern.search(text)]
    left, right = bool(_LEFT.search(text)), bool(_RIGHT.search(text))
    forward, back = bool(_FORWARD.search(text)), bool(_BACK.search(text))
    if left and right:
        return None
    if _TURN.search(text):
        if postures or forward or back:
            return None
        degree = _turn_degree(text)
        if degree is None:
            return None
        yaw = -1 if right else 1
        return {"action": "walk", "parameters": {"x": 0, "y": 0, "yaw": yaw, "degree": degree, "step": 1}}
    directions = forward + back + left + right
    if directions > 1 or (directions and postures):
        return None
    if directions:
        steps = _STEP.findall(text)
        if len(steps) > 1:
            return None
        step = planning_slots._parse_cn(steps[0]) if steps else 1
        if step is None:
            return None
        x = pp._MOVE_X_MAGNITUDE if forward else -pp._MOVE_X_MAGNITUDE if back else 0
        y = 1 if left else -1 if right else 0
        return {"action": "walk", "parameters": {"x": x, "y": y, "yaw": 0, "step": step}}
    if len(postures) == 1:
        return {"action": postures[0]}
    return None


def derive_weather(text: str) -> dict[str, Any] | None:
    """Arguments implied by one weather wording (missing city -> default), else None."""
    cities = [city for city in _CITY_NAMES if city in text]
    if len(cities) > 1:
        return None
    if not cities and _PLACE_HINT.search(text):
        return None  # an unknown place would silently fall back to the default city
    now, h24, d7 = bool(_NOW.search(text)), bool(_H24.search(text)), bool(_D7.search(text))
    if d7 and h24:
        return None
    query_type = ("now_7d" if now else "7d") if d7 else "24h" if h24 else "now"
    return {"city": cities[0] if cities else pp.DEFAULT_WEATHER_CITY, "query_type": query_type}


_GESTURE_RULES = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("blow_kisses_multi", r"飞.{0,2}吻"),
        ("curtain_bow", r"谢幕"),
        ("bow", r"鞠.{0,2}躬"),
        ("clap", r"鼓.{0,2}掌|拍.{0,2}手|掌声"),
        ("high_five", r"击.{0,2}掌|high\s*five|give me five"),
        ("nod", r"点.{0,2}头"),
        ("shake_head", r"摇.{0,2}头"),
        ("shake_hands", r"握.{0,2}手"),
        ("this_way_please", r"这边请|请.{0,4}这边|请的手势|指引|引导|引路"),
        ("wave_greet_bye", r"挥.{0,2}手|招.{0,2}手|拜拜|再见|回头见|打.{0,2}招呼|道别|告别|bye"),
    )
)
_HEART = re.compile(r"比(?!较|如).{0,8}心|爱心|心形")


def derive_gesture(text: str) -> dict[str, Any] | None:
    """Gesture implied by one wording (a bare heart -> hand_heart), else None."""
    found = {name for name, pattern in _GESTURE_RULES if pattern.search(text)}
    if "curtain_bow" in found:
        found.discard("bow")
    if _HEART.search(text):
        left, right = "左" in text, "右" in text
        if left and right:
            return None
        found.add("left_hand_side_heart" if left else "right_hand_side_heart" if right else "hand_heart")
    return {"gesture": found.pop()} if len(found) == 1 else None


# (enum, name used in prompts, wording pattern)
_DANCES = (
    ("abracadabr_dance", "扭胯舞", r"扭胯舞|abracadabra"),
    ("all_things_grow_dance", "万物生", r"万物生"),
    ("ambition_dance", "大展宏图", r"大展宏图"),
    ("apt_dance", "APT", r"(?<![a-z])a\.?p\.?t(?![a-z])"),
    ("egyptian_shake", "埃及摇摆", r"埃及摇摆"),
    ("gee", "Gee", r"(?<![a-z])gee(?![a-z])"),
    ("gentleman", "Gentleman", r"gentleman"),
    ("go_cortis_dance", "Go Cortis", r"go\s*cortis"),
    ("idol_dance_1", "女团舞1", r"女团舞\s*(?:1|一)(?![0-9段下个支])"),
    ("idol_dance_2", "女团舞2", r"女团舞\s*(?:2|二)"),
    ("karla_ok", "卡啦永远OK", r"卡[啦拉]永远\s*ok"),
    ("lets_bounce", "来个蹦蹦", r"蹦蹦"),
    ("luv_each_other", "相亲相爱", r"相亲相爱"),
    ("one_and_only_dance", "热烈", r"热烈"),
    ("one_spot_dance", "美美桑内", r"美美桑内"),
    ("popping", "机械舞", r"机械舞|popping"),
    ("power_up_dance", "助力舞", r"助力舞"),
    ("pulp_fiction_dance", "低俗小说", r"低俗小说|pulp\s*fiction"),
    ("smooth_sailing_and_prosperity", "顺风顺水顺财神", r"顺风顺水"),
    ("solo_shake", "孤身摇", r"孤身摇"),
    ("swag_dance", "展臂舞", r"展臂舞"),
    ("sweep_kick_dance", "扫腿舞", r"扫腿舞"),
    ("victory_dance", "胜利之舞", r"胜利之?舞"),
    ("warm_up_dance", "热场舞", r"热场舞"),
    ("whatever", "管他什么音乐", r"管他什么音乐|whatever"),
)
_DANCE_RULES = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, _label, pattern in _DANCES)
_ANY_DANCE = re.compile(r"舞|跳一段|跳一个|跳一支|跳个")
_GENERIC_DANCE = "warm_up_dance"  # reviewed default for an unnamed dance


def derive_dance(text: str) -> dict[str, Any] | None:
    """Dance implied by one wording (an unnamed dance -> warm_up_dance), else None."""
    found = {name for name, pattern in _DANCE_RULES if pattern.search(text)}
    if len(found) == 1:
        return {"dance": found.pop()}
    if found or "女团舞" in text or not _ANY_DANCE.search(text):
        return None
    return {"dance": _GENERIC_DANCE}


_STATUS_RULES = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("电量", r"电量|电池|没电|充电|续航|撑多久|还能用多久|电还|多少电"),
        ("日期", r"日期|几号|几月|星期几|周几|礼拜几|哪一天|哪天"),
        ("时间", r"几点|时间|钟点|几时"),
        ("位置", r"位置|在哪|哪里|哪儿|什么地方|定位|所在的?(省份|城市)"),
        ("型号", r"型号|版本|哪款|哪一款|什么款|什么机器人|什么系列"),
        (
            "当前动作",
            r"在做什么|在干什么|在干嘛|在忙|忙吗|空闲|在做啥|在干啥|什么动作|什么姿势|当前动作|在执行|在运行|执行什么",
        ),
    )
)


def derive_robot_status(text: str) -> dict[str, Any] | None:
    """Status item implied by one wording about the robot itself, else None."""
    if any(city in text for city in _CITY_NAMES):
        return None  # a named city asks about the world, not the robot
    found = [name for name, pattern in _STATUS_RULES if pattern.search(text)]
    return {"query": found[0]} if len(found) == 1 else None


DERIVERS: dict[str, Callable[[str], dict[str, Any] | None]] = {
    "HumanAction": derive_human_action,
    "RobotGesture": derive_gesture,
    "RobotDance": derive_dance,
    "get_weather": derive_weather,
    "robot_status": derive_robot_status,
}
# Asking whether/how the robot can do something is a reply, never an action.
_CAPABILITY = re.compile(r"你会|会不会|学过|学会|多少种|哪些|什么意思|起源|为什么|怎么做|怎么跳|是什么")
_OTHER_ASSISTANT = re.compile(r"小度|小爱|天猫精灵|小艺|siri", re.IGNORECASE)
# Wording that belongs to a dedicated tool rather than public web search.
_NOT_SEARCH = re.compile(
    r"天气|气温|温度|下雨|下雪|冷不冷|热不热|紫外线|空气质量|逐际|luna|你们公司|你们的", re.IGNORECASE
)


def _search_reason(query: str, zh: str, search: str, city: str | None) -> str | None:
    if len(search) < 4 or search not in query:
        return "search_query is not a span of the query"
    if search not in zh:
        return "instruction_zh drops the search topic"
    if _NOT_SEARCH.search(query) or derive_robot_status(query) is not None:
        return "wording belongs to another tool"
    if city and city not in search:
        return "search_query misses city"
    return None


# ---------------------------------------------------------------------------
# Coverage matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One coverage cell: fixed arguments plus a writing brief for the LLM."""

    cell_id: str
    tool: str
    intent: str
    params: dict[str, Any] | None
    count: int
    slot: str | None = None  # "step" | "degree" | "city" | "foreign_city": one value per sentence
    hint: str = ""
    reply: bool = False  # negative: the wording names the tool's item but only asks about it


def _move(x: float, y: int, step: int | str = 1) -> dict[str, Any]:
    return {"action": "walk", "parameters": {"x": x, "y": y, "yaw": 0, "step": step}}


def _turn(yaw: int, degree: int | str) -> dict[str, Any]:
    return {"action": "walk", "parameters": {"x": 0, "y": 0, "yaw": yaw, "degree": degree, "step": 1}}


_PLAIN = "不说步数，或只说“一步”；不出现“一点、一小步、稍微”这类程度词"
_LITTLE = "用“一点、一小步、稍微、挪一挪”这类程度词，不出现具体步数"
_STEPS = "按给定步数写，数字可写成阿拉伯数字或中文数字"
_BARE_TURN = "只说转向，不说角度，不出现“一下、一点、一圈、转身、后面”"
_DEG = "按给定角度写，角度写成阿拉伯数字加“度”"
_BIT = "用“一下、一点、稍微”这类程度词，不说角度"
_FWD, _BK = pp._MOVE_X_MAGNITUDE, -pp._MOVE_X_MAGNITUDE
_ASPECTS = "问法分散到：整体概况、冷热温度、会不会下雨下雪、空气质量或紫外线、穿衣或出行建议、风力湿度"
_NO_CITY = "不出现任何城市或地名（包括“这里”以外的地点）"

CELLS: tuple[Cell, ...] = (
    Cell("sit_down", "HumanAction", "让机器人坐下", {"action": "sit_down"}, 9),
    Cell("lie_down", "HumanAction", "让机器人躺下", {"action": "lie_down"}, 9),
    Cell("stand_up_casual", "HumanAction", "让机器人站起来", {"action": "stand_up"}, 4,
         hint="侧重口语或带场景理由的说法，如“别坐着了”"),
    Cell("fwd_plain", "HumanAction", "让机器人往前走", _move(_FWD, 0), 4, hint=_PLAIN),
    Cell("fwd_steps", "HumanAction", "让机器人往前走指定步数", _move(_FWD, 0, "{step}"), 6, "step", _STEPS),
    Cell("fwd_little", "HumanAction", "让机器人稍微往前一点", _move(_FWD, 0), 3, hint=_LITTLE),
    Cell("back_plain", "HumanAction", "让机器人往后退", _move(_BK, 0), 4, hint=_PLAIN),
    Cell("back_steps", "HumanAction", "让机器人往后退指定步数", _move(_BK, 0, "{step}"), 6, "step", _STEPS),
    Cell("back_little", "HumanAction", "让机器人稍微往后退一点", _move(_BK, 0), 3, hint=_LITTLE),
    Cell("left_plain", "HumanAction", "让机器人向左平移（身体朝向不变）", _move(0, 1), 4, hint=_PLAIN),
    Cell("left_steps", "HumanAction", "让机器人向左平移指定步数", _move(0, 1, "{step}"), 4, "step", _STEPS),
    Cell("right_plain", "HumanAction", "让机器人向右平移（身体朝向不变）", _move(0, -1), 4, hint=_PLAIN),
    Cell("right_steps", "HumanAction", "让机器人向右平移指定步数", _move(0, -1, "{step}"), 4, "step", _STEPS),
    Cell("turn_left", "HumanAction", "让机器人向左转", _turn(1, 90), 4, hint=_BARE_TURN),
    Cell("turn_right", "HumanAction", "让机器人向右转", _turn(-1, 90), 4, hint=_BARE_TURN),
    Cell("turn_left_deg", "HumanAction", "让机器人向左转指定角度", _turn(1, "{degree}"), 5, "degree", _DEG),
    Cell("turn_right_deg", "HumanAction", "让机器人向右转指定角度", _turn(-1, "{degree}"), 5, "degree", _DEG),
    Cell("turn_left_bit", "HumanAction", "让机器人向左稍微转一点", _turn(1, 15), 3, hint=_BIT),
    Cell("turn_right_bit", "HumanAction", "让机器人向右稍微转一点", _turn(-1, 15), 4, hint=_BIT),
    Cell("turn_around", "HumanAction", "让机器人转身或向后转", _turn(1, 180), 4, hint="不说左右，不说角度"),
    Cell("spin", "HumanAction", "让机器人原地转一圈", _turn(1, 360), 3, hint="不说左右，不说角度"),
    Cell("spin_right", "HumanAction", "让机器人向右转一圈", _turn(-1, 360), 2, hint="不说角度"),
    Cell("now_city", "get_weather", "问给定城市现在或今天的天气", {"city": "{city}", "query_type": "now"}, 10,
         "city", _ASPECTS + "；不出现今晚、明天、这周等将来时段"),
    Cell("now_default", "get_weather", "问现在或今天的天气（不说城市）",
         {"city": pp.DEFAULT_WEATHER_CITY, "query_type": "now"}, 4, hint=_NO_CITY),
    Cell("h24_city", "get_weather", "问给定城市今晚或明天某时段的天气", {"city": "{city}", "query_type": "24h"}, 8,
         "city", _ASPECTS + "；时段只用今晚、明天、明早、下午、几点这类 24 小时内的说法"),
    Cell("h24_default", "get_weather", "问今晚或明天的天气（不说城市）",
         {"city": pp.DEFAULT_WEATHER_CITY, "query_type": "24h"}, 3, hint=_NO_CITY),
    Cell("d7_city", "get_weather", "问给定城市这周、周末或未来几天的天气", {"city": "{city}", "query_type": "7d"}, 8,
         "city", _ASPECTS + "；不出现现在、今天、今晚、明天"),
    Cell("d7_default", "get_weather", "问这周或未来几天的天气（不说城市）",
         {"city": pp.DEFAULT_WEATHER_CITY, "query_type": "7d"}, 3, hint=_NO_CITY),
    Cell("now7d_city", "get_weather", "同时问给定城市现在和未来一周的天气",
         {"city": "{city}", "query_type": "now_7d"}, 5, "city", "一句里同时问现在（或今天）和这周（或未来几天）"),
    Cell("now7d_default", "get_weather", "同时问现在和未来一周的天气（不说城市）",
         {"city": pp.DEFAULT_WEATHER_CITY, "query_type": "now_7d"}, 2, hint=_NO_CITY),
)

# Curated count per gesture: fill every gesture to about 9 atoms, at least 3 new.
_GESTURE_CELLS = {
    "blow_kisses_multi": (7, "说法里要有“飞吻”"),
    "bow": (4, "用“鞠躬、鞠个躬”，不说谢幕"),
    "clap": (6, "用“鼓掌、拍手、掌声”，不说击掌"),
    "curtain_bow": (6, "说法里要有“谢幕”"),
    "hand_heart": (5, "用“比心、比个心”，不说左手、右手、左边、右边"),
    "high_five": (5, "用“击掌、击个掌”，不说拍手、鼓掌"),
    "left_hand_side_heart": (6, "必须说明用左手或在左侧比心，不出现“右”"),
    "nod": (6, "说法里要有“点头”"),
    "right_hand_side_heart": (4, "必须说明用右手或在右侧比心，不出现“左”"),
    "shake_hands": (7, "说法里要有“握手”"),
    "shake_head": (8, "说法里要有“摇头”"),
    "this_way_please": (6, "用“这边请、请往这边”这类引导客人的说法，意思是做引导手势，不是让机器人自己走"),
    "wave_greet_bye": (5, "用“挥手、招手、挥挥手”"),
}
# Curated count per dance: fill every dance to about 6 atoms, at least 2 new.
_DANCE_COUNTS = {
    "abracadabr_dance": 3, "all_things_grow_dance": 2, "ambition_dance": 6, "apt_dance": 4,
    "egyptian_shake": 6, "gee": 6, "gentleman": 5, "go_cortis_dance": 6, "idol_dance_1": 2,
    "idol_dance_2": 6, "karla_ok": 4, "lets_bounce": 5, "luv_each_other": 3, "one_and_only_dance": 2,
    "one_spot_dance": 6, "popping": 3, "power_up_dance": 6, "pulp_fiction_dance": 2,
    "smooth_sailing_and_prosperity": 3, "solo_shake": 6, "swag_dance": 3, "sweep_kick_dance": 5,
    "victory_dance": 5, "warm_up_dance": 2, "whatever": 6,
}
_STATUS_CELLS = {
    "电量": (3, "可以直接问电量，也可以间接问（如“还能撑多久”“要不要充电”）"),
    "日期": (5, "问今天几号、星期几或日期"),
    "时间": (4, "问现在几点或时间"),
    "位置": (3, "问机器人自己现在在哪、所在位置"),
    "型号": (3, "问机器人自己的型号或版本"),
    "当前动作": (3, "问机器人现在在做什么、处于什么动作或姿势"),
}
_SEARCH_CELLS = (
    ("ws_news", "问最近的新闻或热搜话题", None, r"新闻|热搜|热门|讨论|头条"),
    ("ws_sports", "问最近某场比赛或赛事的结果、比分", None, r"比分|结果|谁赢|冠军|赛果"),
    ("ws_finance", "问金价、汇率或股票行情", None, r"价|行情|汇率|指数"),
    ("ws_traffic", "问给定城市的交通或地铁运营情况", "city", r"交通状况|路况|拥堵|堵|运营|运行|停运"),
    ("ws_scenic", "问给定城市某个景点今天是否开放或人多不多", "city", r"开放|开门|人多|游客|挤|门票|排队"),
    ("ws_flight", "问给定城市机场的航班或延误情况", "city", r"航班|延误|起飞|取消"),
    (
        "ws_launch",
        "问某款公开商品（手机、汽车等，不是机器人）的发布信息或售价",
        None,
        r"价|多少钱|发布|功能|配置|上市",
    ),
    ("ws_entertainment", "问最近的电影、剧集、歌曲或明星动态", None, r"新歌|剧|电影|综艺|热搜|新闻|票房|上映|推荐"),
    (
        "ws_city_events",
        "问给定城市今天或这周末有什么活动、展览或演出（不问天气）",
        "city",
        r"活动|展览|演出|音乐会|好玩",
    ),
    ("ws_world_time", "问给定海外城市现在几点（不问天气）", "foreign_city", r"几点|时间"),
)
# The search query must keep what is being looked up, not just the place or item.
_SEARCH_TOPICS = {cid: re.compile(topic) for cid, _intent, _slot, topic in _SEARCH_CELLS}
_DANCE_LABELS = {name: label for name, label, _pattern in _DANCES}
CELLS += (
    *(
        Cell(g, "RobotGesture", f"让机器人做{pp.GESTURE_MEANINGS[g]}手势", {"gesture": g}, n, hint=hint)
        for g, (n, hint) in _GESTURE_CELLS.items()
    ),
    Cell("wave_scene", "RobotGesture", "告别或打招呼，希望机器人挥手回应", {"gesture": "wave_greet_bye"}, 3,
         hint="不直接说挥手，用“拜拜、再见、回头见、打个招呼”这类告别或打招呼的话"),
    Cell("gesture_ask", "RobotGesture", "只问机器人会不会做某个具体手势，不要求它现在做", None, 6,
         hint="句中必须出现一个具体手势名（如比心、鞠躬、击掌、握手），用“你会……吗”“会不会……”问法", reply=True),
    *(
        Cell(d, "RobotDance", f"让机器人跳「{_DANCE_LABELS[d]}」", {"dance": d}, n,
             hint=f"舞名逐字写作「{_DANCE_LABELS[d]}」，说法可以是“跳X”“来一段X”或口语请求")
        for d, n in _DANCE_COUNTS.items()
    ),
    Cell("dance_unnamed", "RobotDance", "让机器人跳舞（不指定哪支）", {"dance": _GENERIC_DANCE}, 4,
         hint="不说具体舞名，只说跳个舞、来段舞之类"),
    Cell("dance_ask", "RobotDance", "只问机器人会不会跳某支具体的舞，不要求它现在跳", None, 6,
         hint="句中必须出现一个具体舞名（如机械舞、扭胯舞、万物生），用“你会……吗”“会不会……”“学过……吗”问法",
         reply=True),
    *(
        Cell(f"status_{item}", "robot_status", f"问机器人当前的{item}", {"query": item}, n,
             hint=hint + "；问的是机器人自己的状态，不出现城市名或其他地点")
        for item, (n, hint) in _STATUS_CELLS.items()
    ),
    *(Cell(cid, "web_search", intent, {"query": "{search}"}, 4, slot) for cid, intent, slot, _topic in _SEARCH_CELLS),
)

_STEP_VALUES = [n for n in range(2, 31) if planning_slots.number_split("step", n) == "train"]
_DEGREE_VALUES = [30, 45, 60, 120, 135, 150, 270]
_CITY_VALUES = [zh for zh, _en in planning_slots._CITIES["train"] if zh != pp.DEFAULT_WEATHER_CITY]
_FOREIGN_CITY_VALUES = ["东京", "首尔", "新加坡", "曼谷", "伦敦", "巴黎", "纽约", "悉尼", "柏林", "莫斯科", "迪拜"]
_SLOT_VALUES = {
    "step": _STEP_VALUES,
    "degree": _DEGREE_VALUES,
    "city": _CITY_VALUES,
    "foreign_city": _FOREIGN_CITY_VALUES,
}


def _fill(params: object, slot: str | None, value: object) -> object:
    if isinstance(params, dict):
        return {key: _fill(item, slot, value) for key, item in params.items()}
    if slot and params == "{" + slot + "}":
        return value
    return params


# ---------------------------------------------------------------------------
# Writing + validation
# ---------------------------------------------------------------------------
_SYSTEM = (
    "你为一台人形服务机器人的规划模型写训练数据：模拟真实用户当面对机器人说的一句中文话。"
    "句子要自然、口语、彼此说法不同，像不同的人在不同场合说的。"
)
_INSTRUCTION_STYLE = {
    "HumanAction": "只保留动作本身的原话：去掉客套、语气词和理由说明（如“我要拍照”），保留方向、步数、角度和程度词",
    "get_weather": "“查询<城市><时段>……”形式的简洁任务，用自己的话概括用户关心的点（如是否下雨、气温高低）；"
    "必须写出城市，用户没说城市时写" + pp.DEFAULT_WEATHER_CITY,
    "RobotGesture": "手势的简洁命令（如“鞠躬”“左手比心”“挥手道别”），去掉客套、语气词和理由",
    "RobotDance": "“跳<舞名>”形式的简洁命令，舞名照原话；用户没说舞名时写“跳个舞”",
    "robot_status": "“查询……”形式的简洁任务，写明查询项（如“查询当前电量”“查询今天是几号”）",
    "web_search": "用户原话去掉客套和语气词后的问题，必须包含 search_query 原文",
    "reply": "“回答……”形式，概括用户问的问题（如“回答是否会比心”）",
}
# Aspect labels from the writing brief that must not leak into instructions.
_BRIEF_LEAK = re.compile(r"整体概况|冷热温度|下雨下雪|或紫外线|风力湿度|天气方面")


def _prompt(cell: Cell, values: list[Any], avoid: list[str]) -> str:
    lines = [
        f"写 {len(values)} 句中文用户话语，意图：{cell.intent}。",
        "说法类型尽量分散：直接命令、随口口语、礼貌请求、带场景或理由的说法。",
        "硬性要求：",
        "- 每句只表达这一个意图，不夹带别的请求、动作或问题；",
        "- 不要求机器人先走到别处（如去椅子那边）或做它做不到的附加动作；",
        "- 句式和用词互不相同，不要只换语气词；",
    ]
    if cell.hint:
        lines.append(f"- {cell.hint}；")
    if cell.slot:
        label = {"step": "步数", "degree": "角度"}.get(cell.slot, "城市")
        lines.append(f"- 第 i 句的{label}依次为：" + "、".join(str(v) for v in values) + "；")
    if avoid:
        lines.append("- 不要与这些已有说法雷同：" + " / ".join(avoid) + "；")
    style = _INSTRUCTION_STYLE["reply" if cell.reply else cell.tool]
    lines += ["每句同时给出：", f"- instruction_zh：{style}；", "- instruction_en：instruction_zh 的自然英文。"]
    fields = '"query": "...", "instruction_zh": "...", "instruction_en": "..."'
    if cell.tool == "web_search":
        lines.append("- search_query：query 中表示检索主题的一段连续原文，逐字复制，不改写；"
            "要同时包含对象和查什么（如「巴黎今天有什么活动」，不能只写「巴黎今天」）。"
        )
        fields += ', "search_query": "..."'
    lines.append(f"只输出 JSON 数组：[{{{fields}}}]")
    return "\n".join(lines)


def validate_sentence(
    cell: Cell,
    expected: dict[str, Any] | None,
    query: str,
    zh: str,
    en: str,
    *,
    slot_value: object = None,
    search: str = "",
) -> str | None:
    """Return a rejection reason, or None when the wording implies ``expected``."""
    if not (query and zh and en):
        return "empty field"
    if _BRIEF_LEAK.search(zh):
        return "instruction_zh copies brief labels"
    if _OTHER_ASSISTANT.search(query):
        return "addresses another assistant"
    if cell.reply:
        if not _CAPABILITY.search(query):
            return "not a capability question"
        if DERIVERS[cell.tool](query) is None:
            return "names no concrete item"
        return None if zh.startswith(("回答", "说明")) else "reply instruction style"
    if cell.tool == "web_search":
        if not _SEARCH_TOPICS[cell.cell_id].search(search):
            return "search_query misses the topic"
        return _search_reason(query, zh, search, str(slot_value) if cell.slot else None)
    if cell.tool in {"RobotGesture", "RobotDance"} and _CAPABILITY.search(query):
        return "capability question"
    derive = DERIVERS[cell.tool]
    if derive(query) != expected:
        return f"query derives {derive(query)}"
    if derive(zh) != expected:
        return f"instruction_zh derives {derive(zh)}"
    if cell.tool == "get_weather" and expected and expected["city"] not in zh:
        return "instruction_zh misses city"
    return None


def _cell_params(cell: Cell, value: object, search: str) -> dict[str, Any] | None:
    if cell.reply:
        return None
    if cell.tool == "web_search":
        return {"query": search}
    return cast("dict[str, Any]", _fill(cell.params, cell.slot, value))


def _atom(cell: Cell, query: str, zh: str, en: str, params: dict[str, Any] | None) -> dict[str, Any]:
    tool = cell.tool
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    curation = {
        "cell": cell.cell_id,
        "writer": f"model:{llm.DEFAULT_MODEL}",
        "validator": "rules",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    common = {
        "id": f"cur_{tool.lower()}_{cell.cell_id}_{digest}",
        "query": query,
        "atom_name": cell.cell_id,
        "agent_version": pp.SOURCE_AGENT_VERSION,
        "cloud_raw": None,
        "universal": None,
        "negative": None,
    }
    if cell.reply:
        return {
            **common,
            "expected_function": None,
            "expected_action": None,
            "expected_params": None,
            "atom_category": tool,
            "module": tool,
            "query_key": f"{tool}:negative",
            "negative_sample": True,
            "negative_tool": tool,
            "planner_target": {"kind": "reply", "instructions": {"zh": zh, "en": en}},
            "sample_kind": f"{tool}反样本",
            "sample_polarity": "反样本",
            "curation": curation,
        }
    assert params is not None
    action = params.get("action") or params.get("gesture") or params.get("dance")
    if tool not in {"HumanAction", "RobotGesture", "RobotDance"}:
        action = None
    category = "RobotDance.dance" if tool == "RobotDance" else f"{tool}.{action}" if action else tool
    return {
        **common,
        "expected_function": tool,
        "expected_action": action,
        "expected_params": params,
        "atom_category": category,
        "module": None if action else tool,
        "query_key": None if action else f"{tool}:canonical",
        "negative_sample": False,
        "negative_tool": None,
        "planner_target": {"kind": pp.TOOL_KINDS[tool], "instructions": {"zh": zh, "en": en}},
        "sample_kind": tool,
        "sample_polarity": "正样本",
        "curation": curation,
    }


def generate_cell(
    cell: Cell,
    *,
    need: int,
    seen: set[str],
    avoid: list[str],
    rng: random.Random,
    bundle: ActionSequenceContractBundle,
    write: Callable[..., Any] = llm.chat_json,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Write + validate atoms for one cell; returns (accepted, rejected)."""
    accepted: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    pool = _SLOT_VALUES.get(cell.slot or "", [])
    for _round in range(_MAX_ROUNDS):
        missing = need - len(accepted)
        if missing <= 0:
            break
        values: list[Any] = rng.sample(pool, missing) if pool else [None] * missing
        try:
            rows = write(_prompt(cell, values, avoid), system=_SYSTEM, seed=rng.randrange(1 << 30))
        except ValueError as exc:
            rejected.append(("<reply>", str(exc)))
            continue
        for value, row in zip(values, rows if isinstance(rows, list) else [], strict=False):
            if not isinstance(row, dict):
                continue
            query, zh, en, search = (
                str(row.get(k) or "").strip() for k in ("query", "instruction_zh", "instruction_en", "search_query")
            )
            params = _cell_params(cell, value, search)
            key = _norm_text(query)
            reason = (
                "duplicate"
                if key in seen
                else validate_sentence(cell, params, query, zh, en, slot_value=value, search=search)
            )
            if reason is None:
                atom = _atom(cell, query, zh, en, params)
                try:
                    pp.normalize_atom_for_planning(atom, bundle)
                except (ValueError, TypeError) as exc:
                    reason = f"contract: {exc}"
            if reason is not None:
                rejected.append((query, reason))
                continue
            seen.add(key)
            accepted.append(atom)
    return accepted, rejected


def _existing_pool() -> dict[str, list[dict[str, Any]]]:
    return pp.load_verified_planning_pool()


def generate(
    tools: list[str], *, seed: int, path: Path = pp.CURATED_ATOMS_PATH, redo: frozenset[str] = frozenset()
) -> None:
    """Fill every under-populated cell of ``tools`` and append accepted atoms to ``path``.

    Cells named in ``redo`` lose their existing curated atoms first.
    """
    if redo and path.is_file():
        kept = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["curation"]["cell"] not in redo
        ]
        path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    bundle = pp.load_contract_bundle(pp.CONTRACT_V2_PATH)
    pool = _existing_pool()
    curated = pp.load_curated_atoms(path)
    rng = random.Random(seed)
    write = functools.partial(llm.chat_json, api_key=find_env_value("DASHSCOPE_API_KEY"))
    seen = {_norm_text(row.get("query")) for rows in pool.values() for row in rows}
    for cell in CELLS:
        if cell.tool not in tools:
            continue
        have = sum(1 for atom in curated.get(cell.tool, []) if atom["curation"]["cell"] == cell.cell_id)
        need = cell.count - have
        if need <= 0:
            continue
        avoid = [str(row["query"]) for row in pool.get(cell.tool, []) if bool(row.get("negative_sample")) == cell.reply]
        avoid = rng.sample(avoid, min(20, len(avoid)))
        accepted, rejected = generate_cell(
            cell, need=need, seen=seen, avoid=avoid, rng=rng, bundle=bundle, write=write
        )
        with path.open("a", encoding="utf-8") as handle:
            for atom in accepted:
                handle.write(json.dumps(atom, ensure_ascii=False) + "\n")
        print(f"{cell.cell_id}: +{len(accepted)}/{need}")
        for query, reason in rejected:
            print(f"   x {query} :: {reason}")


def audit() -> None:
    """Print existing positive atoms whose wording rules disagree with their arguments."""
    for tool, derive in DERIVERS.items():
        for row in _existing_pool().get(tool, []):
            if row.get("negative_sample") or str(row.get("id", "")).startswith("cur_") or pp._excluded_atom(row):
                continue
            atom = pp.normalize_atom_for_planning(
                pp._canonical_move_x(pp._canonical_turn_degree(row)),
                pp.load_contract_bundle(pp.CONTRACT_V2_PATH),
            )
            derived = derive(str(row["query"]))
            if derived != atom["expected_params"]:
                print(f"{row['id']}\t{row['query']}\t{atom['expected_params']}\tderived={derived}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--tools", nargs="+", default=sorted({cell.tool for cell in CELLS}))
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--redo", nargs="+", default=[], help="cell ids to drop and regenerate")
    sub.add_parser("audit")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.tools, seed=args.seed, redo=frozenset(args.redo))
    else:
        audit()


if __name__ == "__main__":
    main()
