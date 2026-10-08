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

"""Atom-diversity pilot (HumanAction, RobotGesture, RobotDance): scene x persona x style x noise.

Reuses the coverage-matrix pipeline (cells, wording rules, contract
normalization) and only swaps the writing prompt. Each cell is written in rounds
until new sentences stop being novel, which measures how many distinct atoms the
extra dimensions really yield, how many are rejected by the label rules, and when
generation saturates.

Run from the repository root::

    python -m planner_data.atoms.pilot --name all_dims
    python -m planner_data.atoms.pilot --name baseline --dims none
    python -m planner_data.atoms.pilot --name quick --cells sit_down fwd_steps --max-rounds 3
"""

from __future__ import annotations

import argparse
import copy
import difflib
import functools
import json
import os
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


os.environ.setdefault("BENCHMARK_LLM_MAX_ATTEMPTS", "3")

from planner_data import llm  # noqa: E402
from planner_data.atoms import dims  # noqa: E402
from planner_data.atoms import planning_atoms as pa  # noqa: E402
from planner_data.common import RUNS_DIR, _norm_text, find_env_value  # noqa: E402
from planner_data.contract import planning_production as pp  # noqa: E402


TAUS = (0.4, 0.5, 0.6, 0.7)
_SLOT_LABELS = {"step": "步数", "degree": "角度", "city": "城市", "foreign_city": "城市"}

# Label convention (see dims.HINT_OVERRIDES): "转一下" is a plain 90° turn, so drop 一下 from the
# 15° rule. Both derive paths read this module global at call time.
_SMALL_TURN = re.compile(r"一点|一些|一丢|稍|微|轻轻|a little|a bit|slightly|a touch", re.IGNORECASE)
pp._TURN_DEGREE_RULES = tuple(
    (value, _SMALL_TURN if value == 15 else pattern) for value, pattern in pp._TURN_DEGREE_RULES
)

# ---------------------------------------------------------------------------
# Similarity on sentence shape (slot values and punctuation masked)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"[0-9零一二两三四五六七八九十百]+")
_PUNCT = re.compile(r"[\s，。！？、,.!?;；:：~～…\"'“”‘’（）()]+")


def shape(text: str) -> str:
    return _NUM.sub("#", _PUNCT.sub("", text))


def grams(text: str, n: int = 2) -> frozenset[str]:
    s = shape(text)
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1)) or frozenset({s})


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def max_sim(g: frozenset[str], others: list[frozenset[str]]) -> float:
    return max((jaccard(g, o) for o in others), default=0.0)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    scene: str | None = None
    persona: str | None = None
    style: str | None = None
    noise: str | None = None
    kernel: str | None = None


def sample_spec(rng: random.Random, enabled: frozenset[str], cell_id: str = "",
                styles: tuple[str, ...] = dims.STYLES) -> Spec:
    noisy = "noise" in enabled and rng.random() < dims.NOISE_RATE
    kernels = dims.KERNELS.get(cell_id) if "kernel" in enabled else None
    return Spec(
        scene=rng.choice(dims.SCENES) if "scene" in enabled else None,
        persona=rng.choice(dims.PERSONAS) if "persona" in enabled else None,
        style=rng.choice(styles) if "style" in enabled else None,
        noise=rng.choice(dims.NOISES) if noisy else None,
        kernel=rng.choice(kernels) if kernels else None,
    )


SYSTEM = (
    "你为一台人形服务机器人的规划模型写训练数据：模拟真实用户当面对机器人说的一句中文话（语音转写成的文字）。"
    "每句都要像给定的那个人在给定场合下真会说的话，自然、口语，彼此说法不同。"
)

# Query tools. User decisions 2026-10-06: "能不能帮我查下北京天气" with a concrete subject calls the tool;
# the robot's own design specs ("你续航多久") are private knowledge (rag_query), its live state is
# robot_status; stable common knowledge ("长城有多长") is a reply; rag topics follow v330.
QUERY_TOOLS = ("get_weather", "robot_status", "web_search", "rag_query")
_SEARCH_TOOLS = {"web_search", "rag_query"}
_PRODUCTS = "Luna、Oli、TRON1、TRON2"
_INSTRUCTION_STYLE = {**pa._INSTRUCTION_STYLE, "rag_query": pa._INSTRUCTION_STYLE["web_search"]}
_PRODUCT_RE = r"Luna|Oli|TRON|逐际|LimX|你们"
# (cell, intent, hint, topic the search_query must keep, subject it must name or None)
_RAG = (
    ("rag_company", "问逐际动力这家公司本身的情况",
     "说法里要出现“逐际动力”或“你们公司”；问成立时间、总部在哪、融资、发展历程、发布过哪些产品等",
     r"逐际|LimX|公司", None),
    ("rag_product", f"问某款产品（{_PRODUCTS}）是什么、有什么功能或和别的型号有什么区别",
     "说法里要出现产品名", r"Luna|Oli|TRON", None),
    ("rag_spec_self", "问机器人自己的某一项出厂设计规格",
     "每句只问一项，各句分别问满电续航、关节或自由度数、身高体重、最快速度、负载、传感器、外壳材料中的不同项；"
     "用“你”指代机器人，不说产品名；问的是设计参数，不是此刻状态，不出现“现在、当前、还剩、还能撑”",
     r"续航|充电|电池|关节|自由度|多高|身高|多重|体重|重量|速度|多快|负载|承重|传感器|摄像头|材料|外壳|芯片|处理器", None),
    ("rag_price", f"问某款产品（{_PRODUCTS}）的价格、优惠、购买或预订渠道、交付时间中的一项",
     "说法里要出现产品名，或说“你们的机器人”", r"价|多少钱|优惠|折扣|买|购|预订|预定|订|交付|发货|到货|发布|上市|发售", _PRODUCT_RE),
    ("rag_aftersale", "问逐际动力产品售后的某一项：保修期、维修、退换货、配件、售后网点中的一项",
     "可以说产品名，也可以说“你们的机器人”", r"保修|质保|维修|修|退|换|售后|配件|网点", _PRODUCT_RE),
    ("rag_scene", f"问某款产品（{_PRODUCTS}）适合用在哪些场景，或能不能用于某个具体场合（商场讲解、婚礼、课堂、展会等）",
     "说法里要出现产品名；问的是产品适不适合，不是让眼前的你现在去做",
     r"场景|场合|适合|适用|用于|用在|当|做|能不能|可以|可不可以|行业|领域", r"Luna|Oli|TRON"),
    ("rag_tech", f"问某款产品（{_PRODUCTS}）的技术原理（怎么保持平衡、表情怎么实现、怎么紧急停止、用什么算法或系统）",
     "说法里要出现产品名，或说“你们的机器人”", r"怎|如何|咋|原理|算法|技术|靠什么|靠啥|系统|实现|做到|控制|平衡|表情|停", _PRODUCT_RE),
)
RAG_CELLS = tuple(pa.Cell(cid, "rag_query", intent, {"query": "{search}"}, 4, None, hint)
                  for cid, intent, hint, _topic, _subject in _RAG)
ALL_CELLS = (*pa.CELLS, *RAG_CELLS)
_RAG_TOPICS = {cid: (re.compile(topic, re.IGNORECASE), re.compile(subject, re.IGNORECASE) if subject else None)
               for cid, _intent, _hint, topic, subject in _RAG}
_WEATHER_WORDS = re.compile(r"天气|气温|下雨|下雪|冷不冷|热不热")
# pa's web_search topics plus everyday wordings seen in pilots (“今天能不能去”“地铁末班车”“比赛成绩”).
_WS_EXTRA = {
    "ws_sports": r"成绩|得分|获胜|赢|输|名次|排名",
    "ws_traffic": r"交通|地铁|公交|末班|首班|线路",
    "ws_scenic": r"开|参观|能去|能不能去",
    "ws_entertainment": r"歌|明星|专辑|动画|动态|火|热门",
    "ws_city_events": r"展|市集|庙会|音乐节|节",
    "ws_launch": r"买|购|预订|预定|优惠|折扣|发售",
}
_WS_TOPICS = {cid: re.compile(f"{pat.pattern}|{_WS_EXTRA[cid]}" if cid in _WS_EXTRA else pat.pattern, re.IGNORECASE)
              for cid, pat in pa._SEARCH_TOPICS.items()}
_REQUEST_HEAD = re.compile(r"(你)?(能不能|可不可以|可以|知不知道|知道|帮我|帮忙|麻烦|请问|请|告诉|想知道|想了解|查一?下|查查|看一?下|看看|找找)")


def _covered(search: str, zh: str) -> bool:
    """instruction_zh may fix typos or drop fillers, so 60% of the search characters in order is enough."""
    blocks = difflib.SequenceMatcher(None, search, zh, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) >= 0.6 * len(search)


def search_reason(cell: pa.Cell, query: str, zh: str, search: str, city: str | None) -> str | None:
    if len(search) < 4 or search not in query:
        return "search_query is not a span of the query"
    if _REQUEST_HEAD.match(search):
        return "search_query keeps the request words"
    if search.count("、") >= 2:  # the intent's option list pasted into the sentence
        return "copies the intent list"
    if not _covered(search, zh):
        return "instruction_zh drops the search topic"
    if cell.tool == "web_search":
        if not _WS_TOPICS[cell.cell_id].search(search):
            return "search_query misses the topic"
        if pa._NOT_SEARCH.search(query) or pa.derive_robot_status(query) is not None:
            return "wording belongs to another tool"
        return "search_query misses city" if city and city not in search else None
    topic, subject = _RAG_TOPICS[cell.cell_id]
    if not topic.search(search):
        return "search_query misses the topic"
    if subject is not None and not subject.search(search):
        return "search_query misses the product"
    return "wording belongs to another tool" if _WEATHER_WORDS.search(query) else None


def validate(cell: pa.Cell, params: Any, query: str, zh: str, en: str, value: Any, search: str) -> str | None:
    if cell.tool not in _SEARCH_TOOLS:
        return pa.validate_sentence(cell, params, query, zh, en, slot_value=value, search=search)
    if not (query and zh and en):
        return "empty field"
    if pa._BRIEF_LEAK.search(zh):
        return "instruction_zh copies brief labels"
    if pa._OTHER_ASSISTANT.search(query):
        return "addresses another assistant"
    return search_reason(cell, query, zh, search, str(value) if cell.slot else None)


def build_prompt(cell: pa.Cell, specs: list[Spec], values: list[Any], avoid: list[str]) -> str:
    query = cell.tool in QUERY_TOOLS
    lines = [f"写 {len(specs)} 句中文用户话语，意图都是：{cell.intent}。", "逐句按下面的设定写："]
    label = _SLOT_LABELS.get(cell.slot or "")
    for i, (spec, value) in enumerate(zip(specs, values, strict=True), 1):
        parts = [
            f"{name}：{text}"
            for name, text in (
                ("场景", spec.scene),
                ("说话人", spec.persona),
                ("说法", spec.style),
                ("口语噪声", spec.noise),
                ("动作说法", spec.kernel),
            )
            if text
        ]
        if label and value is not None:
            parts.append(f"{label}：{value}")
        lines.append(f"{i}. " + ("；".join(parts) or "自由发挥"))
    if not any(any(asdict(spec).values()) for spec in specs):
        lines.append("说法类型尽量分散：直接问、请它帮忙查、礼貌问法、带场景或理由的说法。" if query
                     else "说法类型尽量分散：直接命令、随口口语、礼貌请求、带场景或理由的说法。")
    lines += [
        "硬性要求：",
        "- 每句只表达这一个意图，不夹带别的请求、动作或问题；",
        "- 场景和说话人只影响措辞，不改变意图；不要让机器人去某个具体地点、拿东西或做别的事；",
        "- 口语噪声只能落在语气词、停顿、重复或无关的字上；"
        + ("城市、时段、要查的对象和内容必须清楚准确；" if query else "方向、步数、角度、动作词、手势名、舞名必须清楚准确；"),
        "- 句式和用词互不相同，不要只换语气词；",
        "- 可以直接问，也可以请它帮忙查（如“帮我查下……”“能不能帮我看看……”“你知道……吗”）；"
        "必须带上具体要查的内容，不要只问它会不会查；" if query else
        "- 不要写成问机器人能不能、会不会、可不可以做的问句（如“能不能……”“你能……吗”“可以……吗”），那是能力提问；",
        "- 不要称呼机器人，不给它起名字、昵称或身份称呼（如小助、小宝、小护士）；需要指代时最多用“你”，也可以不指代；"
        + (f"可以提公司名逐际动力和产品名（{_PRODUCTS}），那是在问产品，不是称呼；" if cell.tool == "rag_query"
           else "不写“机器人”；"),
    ]
    if any(spec.kernel for spec in specs):
        lines.append("- 给了“动作说法”的句子，动作必须用这个说法（可以加步数、角度、程度词或语气），不要换成别的动词；")
    if cell.hint:
        lines.append(f"- {cell.hint}；")
    if avoid:
        lines.append("- 不要与这些已有说法雷同：" + " / ".join(avoid) + "；")
    lines += [
        "每句同时给出：",
        f"- instruction_zh：{_INSTRUCTION_STYLE[cell.tool]}；",
        "- instruction_en：instruction_zh 的自然英文；",
    ]
    fields = '"query": "...", "instruction_zh": "...", "instruction_en": "..."'
    if cell.tool in _SEARCH_TOOLS:
        lines.append("- search_query：从 query 里截取的一段连续原文，逐字复制，连错别字和中间的口语字一起保留，"
                     "不删字、不补字、不调换顺序；从提到对象处截到问完为止，宁长勿短，"
                     "但不带“能不能帮我查一下”“你知不知道”这类请求开头"
                     "（如“那个Oli机器人多少钱啊”取 Oli机器人多少钱，不能写成 Oli的价格）；")
        fields += ', "search_query": "..."'
    lines.append(f"按编号顺序只输出长度为 {len(specs)} 的 JSON 数组：[{{{fields}}}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM label review: an independent reading of the sentence must give the same arguments
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = "你是标注审核员，只按字面意思判断用户这句话要机器人做什么，不推测言外之意。"


@dataclass(frozen=True)
class JudgeSpec:
    """Per-tool wording of the review prompt; the request rules are shared."""

    what: str  # what the robot is asked to do
    yes: str  # examples of requests
    ask: str  # examples of capability questions
    wish: str  # example of a wish
    omit: str  # examples without an address term
    fields: tuple[str, ...]
    output: str  # one output item


_DANCE_LIST = " / ".join(f"{name}（{label}）" for name, label, _pattern in pa._DANCES)
_DANCE_BY_LABEL = {label.lower(): name for name, label, _pattern in pa._DANCES}
_GESTURE_LIST = " / ".join(f"{name}（{meaning}）" for name, meaning in pp.GESTURE_MEANINGS.items())

JUDGES = {
    "HumanAction": JudgeSpec(
        what="做一个身体动作",
        yes="“往前走两步”“请向左转一点”“麻烦你站起来”“要不坐会儿吧”",
        ask="“你会转圈吗”“能不能往前走两步”“可以坐下吗”",
        wish="“要是你能坐下就好了”",
        omit="“找个地方坐”“咱们坐会儿吧”",
        fields=(
            "- action：sit_down / lie_down / stand_up / forward / back / left / right / turn_left / turn_right / none；"
            "left、right 指身体朝向不变的横向平移，turn_left、turn_right 指原地转向；"
            "转身或向后转记为 turn_left；转一圈没说右就记为 turn_left；request 为 false 时也照实填动作；",
            "- steps：forward/back/left/right 明说的步数，没说步数为 1；其他动作为 null；",
            "- degree：转向角度，按顺序取第一条命中的：明说角度用该值；一圈 360；转身、向后转、掉头 180；"
            "带“一点、稍微、一些、微微、轻轻”为 15（如“向左转一点”→15）；其他转向（包括“转一下”“左转”）为 90；非转向为 null。",
        ),
        output='{"i": 1, "request": true, "action": "forward", "steps": 3, "degree": null}',
    ),
    "RobotGesture": JudgeSpec(
        what="做一个手势或礼仪动作（鞠躬、点头、挥手、比心、击掌等）",
        # Label convention: a bare farewell to the robot plans a wave, not a reply.
        yes="“鞠个躬”“做一下鞠躬的动作”“请跟大家挥挥手”“比个心吧”“来，击个掌”"
            "，以及对它道别（如“拜拜”“好的，回头见”“我先走了，再见”，没提挥手也算，gesture 为 wave_greet_bye）"
            "；让它做列表外的手势也算（如“比个耶吧”“来，叉个腰”，gesture 填 none）",
        ask="“你会比心吗”“能不能鞠个躬”“可以握个手吗”",
        wish="“要是你能鼓个掌就好了”",
        omit="“给大家鞠个躬”“咱们击个掌吧”",
        fields=(
            f"- gesture：{_GESTURE_LIST} / none；"
            "用户自己对它说拜拜、再见、回头见、打个招呼也算请求 wave_greet_bye（让它去“说声再见”不算，那是让它说话）；"
            "比心没说左右为 hand_heart；不在列表里的手势（如敬礼、比耶、比OK、竖大拇指）填 none，别凑近似的；"
            "request 为 false 时也照实填手势；gesture 写成带双引号的字符串，包括 \"none\"。",
        ),
        output='{"i": 1, "request": true, "gesture": "bow"}',
    ),
    "RobotDance": JudgeSpec(
        what="跳一支舞",
        yes="“跳个机械舞”“来一段万物生”“给大家跳支舞吧”；让它跳列表外的舞也算（如“来段秧歌”，dance 填 other）",
        ask="“你会跳机械舞吗”“能不能跳个舞”“可以跳段万物生吗”",
        wish="“要是你能跳段舞就好了”",
        omit="“来段热场舞”“咱们跳个舞吧”",
        fields=(
            f"- dance：说到的舞名，从下列选：{_DANCE_LIST}；"
            "没说具体哪支舞填 unnamed；说了列表外的舞名填 other；request 为 false 时也照实填；"
            "dance 写成带双引号的字符串，包括 \"unnamed\" \"other\"。",
        ),
        output='{"i": 1, "request": true, "dance": "popping"}',
    ),
}


def judge_prompt(tool: str, queries: list[str]) -> str:
    if tool in QUERY_TOOLS:
        return query_judge_prompt(queries)
    spec = JUDGES[tool]
    lines = [
        "下面是用户当面对人形机器人说的话，逐句判断：",
        f"- request：这句话是不是在让机器人现在{spec.what}。"
        f"祈使、请求、建议算 true（如{spec.yes}）；"
        f"以下算 false：问它能不能、会不会、可不可以做（如{spec.ask}，这些是能力提问）、假设或设想（如{spec.wish}）、"
        "闲聊或描述、明确对别人说（如“小朋友们站好听讲”）、否定（别、不要）、"
        "推迟到以后或等某个条件再做（如“等客人来了再做”“一会儿再做”“明天做”“我答对了你就做”，不是现在做）、"
        "问这个动作的含义或叫法（如“这个动作英文怎么说”）、让它口头讲解或教怎么做（如“教教我怎么做”；"
        "但“示范一下”“做给我看看”算 true）、让它开口对别人说句话（如“跟大家说声再见”“替我说声谢谢”，"
        "这和用户自己对它道别不同）；"
        f"用户大多不称呼机器人，省略主语的祈使句和“咱们一起……”都算对机器人说（如{spec.omit}都是 true）；",
        "机器人没有名字，用户不会用人名、职务、亲属称呼或“你们”叫它；句中点名招呼某人"
        "（如“小李，……”“爷爷你……”“麻烦张经理……”“你们几个……”）就是在对那个人说，算 false；"
        "“给小朋友们鞠个躬”里的小朋友们只是对象，不算对别人说；",
        *spec.fields,
        "句子：",
        *(f"{i}. {q}" for i, q in enumerate(queries, 1)),
        f"按编号顺序只输出长度为 {len(queries)} 的 JSON 数组：[{spec.output}]",
    ]
    return "\n".join(lines)


_SOURCES = {"get_weather": "weather", "robot_status": "status", "web_search": "public", "rag_query": "private"}


def query_judge_prompt(queries: list[str]) -> str:
    """One review for all query tools, so a sentence that belongs to another tool is caught too."""
    items = " / ".join(pa._STATUS_CELLS)
    lines = [
        "下面是用户当面对人形服务机器人说的话，逐句判断：",
        "- ask：这句话是不是在让机器人现在告诉或查一个具体信息。直接问（如“北京明天下雨吗”“现在几点了”）、"
        "请它查（如“帮我查下金价”“能不能帮我看看航班延误没”“你知道现在几点吗”）都算 true；"
        "以下算 false：只问它会不会查、不带具体要查的内容（如“你会查天气吗”）、闲聊或陈述（如“明天要下雨了”“今天好热”）、"
        "假设或愿望、推迟到以后再查（如“等会儿再帮我查”）、明确对别人说、让它做动作；"
        "用户大多不称呼机器人，省略主语的问句都算对机器人说；句中点名招呼某人（如“小李，……”）就是在对那个人说，算 false；",
        "- source：回答需要的信息来源，按下面顺序取第一个符合的：\n"
        "  1. status：问机器人自己此刻的状态——当前电量或还能撑多久、今天几号星期几、现在几点、它现在在哪、"
        "它自己是什么型号或哪一款（如“你是哪个型号”“这台是什么型号”）、它正在做什么；\n"
        "  2. weather：只限天气本身——气温、下雨下雪、空气质量、紫外线、穿衣建议；"
        "地铁公交、航班、景点、活动即使带了城市和时间也不是 weather；\n"
        f"  3. private：句中提到逐际动力、{_PRODUCTS}、“你们公司”“你们的机器人”，或问机器人自己的出厂设计规格"
        "（满电续航、关节数、身高体重、最快速度）；包括产品功能、产品比较、价格优惠和购买、售后保修、技术原理、"
        "能不能用在某个场合；\n"
        "  4. public：需要联网查的当前公共信息——新闻热搜、比赛结果、金价汇率股票、交通航班、景点开放、活动演出、"
        "影视明星、其他商品的发布和售价、海外城市当地时间；\n"
        "  5. common：不用查就能答的稳定常识，如“长城有多长”“一年有几天”；",
        "- city：source 为 weather 时问的城市，没说城市填 null；其他填 null；",
        "- period：source 为 weather 时的时段：now（现在、只说“今天”或没说时间，如“今天天气怎么样”）/ "
        "24h（点明接下来某个时段：今晚、今天下午、明天、明早）/ "
        "7d（这周、周末、未来几天）/ now_7d（同时问现在和未来几天）；其他填 null；",
        f"- item：问的是机器人自己的状态时填问的项（{items}），问它自己是什么型号时填 型号；其他填 null。",
        "句子：",
        *(f"{i}. {q}" for i, q in enumerate(queries, 1)),
        f"按编号顺序只输出长度为 {len(queries)} 的 JSON 数组："
        '[{"i": 1, "ask": true, "source": "public", "city": null, "period": null, "item": null}]',
    ]
    return "\n".join(lines)


def _query_view(tool: str, params: Any) -> dict[str, Any]:
    """What the query judge can confirm: the source, plus city/period or the status item."""
    view = {"source": _SOURCES[tool]}
    if tool in ("get_weather", "robot_status"):
        view.update(params or {})
    return view


def judged_query(verdict: dict[str, Any]) -> dict[str, Any]:
    source = verdict.get("source")
    if source == "weather":
        city = verdict.get("city")
        city = pp.DEFAULT_WEATHER_CITY if city in (None, "", "null") else city
        return {"source": source, "city": city, "query_type": verdict.get("period")}
    if source == "status":
        return {"source": source, "query": verdict.get("item")}
    if source == "private" and verdict.get("item") == "型号":
        # The judge files “你是什么型号” under product knowledge however it is told; the contract and the
        # rule deriver both put the robot's own model under robot_status, so an explicit 型号 item settles it.
        return {"source": "status", "query": "型号"}
    return {"source": source}


_POSTURES = {"sit_down", "lie_down", "stand_up"}
_MOVES = {"forward": (pp._MOVE_X_MAGNITUDE, 0), "back": (-pp._MOVE_X_MAGNITUDE, 0), "left": (0, 1), "right": (0, -1)}


def _int(value: Any, default: int) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def judged_params(tool: str, verdict: Any) -> dict[str, Any] | None:
    """Tool arguments implied by one judge verdict, or None when it is not a request."""
    if not isinstance(verdict, dict) or verdict.get("request") is not True:
        return None
    if tool == "RobotGesture":
        gesture = verdict.get("gesture")
        return {"gesture": gesture} if gesture in pp.GESTURE_MEANINGS else None
    if tool == "RobotDance":
        dance = str(verdict.get("dance") or "")
        if dance == "unnamed":
            return {"dance": pa._GENERIC_DANCE}
        dance = _DANCE_BY_LABEL.get(dance.lower(), dance)  # the judge sometimes answers with the label
        return {"dance": dance} if dance in pa._DANCE_LABELS else None
    action = verdict.get("action")
    if action in _POSTURES:
        return {"action": action}
    if action in _MOVES:
        x, y = _MOVES[action]
        return {"action": "walk", "parameters": {"x": x, "y": y, "yaw": 0, "step": _int(verdict.get("steps"), 1)}}
    if action in ("turn_left", "turn_right"):
        yaw = 1 if action == "turn_left" else -1
        return {"action": "walk",
                "parameters": {"x": 0, "y": 0, "yaw": yaw, "degree": _int(verdict.get("degree"), 90), "step": 1}}
    return None


def _as_list(reply: Any) -> list[Any]:
    if isinstance(reply, dict):  # e.g. {"sentences": [...]}
        reply = next((v for v in reply.values() if isinstance(v, list)), [])
    return reply if isinstance(reply, list) else []


def judge_reason(tool: str, verdict: Any, expected: Any, *, error: bool = False) -> str | None:
    if error:
        return "judge_error"
    if not isinstance(verdict, dict):
        return "judge missing"
    if tool in QUERY_TOOLS:
        if verdict.get("ask") is not True:
            return "judge not request"
        judged = judged_query(verdict)
        view = _query_view(tool, expected)
        if judged.get("query_type") == "24h" and view.get("query_type") == "now":
            # The judge keeps reading a bare “今天” as 24h whatever the definition says; the rule deriver
            # already requires an explicit later-period cue (今晚/明天/…) for 24h, so trust it here.
            judged = {**judged, "query_type": "now"}
        return None if judged == view else f"judge mismatch {judged}"
    if verdict.get("request") is not True:
        return "judge not request"
    judged = judged_params(tool, verdict)
    if _about_face(judged) and _about_face(expected):  # 转身 has no side; yaw=1 is only a label convention
        return None
    return None if judged == expected else f"judge mismatch {judged}"


def _about_face(params: Any) -> bool:
    return isinstance(params, dict) and (params.get("parameters") or {}).get("degree") == 180


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------
@dataclass
class Context:
    write: Callable[..., Any]
    bundle: Any
    seen: set[str]
    existing: dict[str, list[str]]  # per tool
    existing_grams: dict[str, list[frozenset[str]]]
    judge: Callable[..., Any] | None = None
    sink: Callable[[str, dict[str, Any]], None] = lambda kind, row: None
    lock: threading.Lock = field(default_factory=threading.Lock)
    openers: Counter[str] = field(default_factory=Counter)  # accepted sentences per opener, this run + --extend


_OPENER_STRIP = re.compile(r"^[\s，,。.!！?？~～…、]+")


def opener(text: str) -> str:
    return _OPENER_STRIP.sub("", text)[:2]


def opener_full(ctx: Context, query: str, cap: float, floor: int = 50) -> bool:
    """An opener already holding `cap` of the accepted sentences takes no more ("要不/那个/麻烦" drift)."""
    total = sum(ctx.openers.values())
    return cap > 0 and total >= floor and ctx.openers[opener(query)] >= cap * total


def crowded_openers(ctx: Context, cap: float) -> list[str]:
    total = sum(ctx.openers.values())
    return [k for k, v in ctx.openers.most_common(6) if total >= 50 and v >= 0.8 * cap * total] if cap > 0 else []


@dataclass
class CellResult:
    cell_id: str
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    curve: list[dict[str, int]] = field(default_factory=list)
    calls: int = 0
    failed_calls: int = 0
    judge_calls: int = 0
    raw: int = 0
    stop: str = "max_rounds"


# Writing instructions copied into the sentence (e.g. "一口气说完没有标点").
_LEAK = re.compile(r"一口气说完|没有标点|语气词|错别字|同音字|重复的词|口语噪声|说话人|不同场景")
# Speech transcripts carry no quotes; they come from the brief's 「舞名」.
_QUOTES = re.compile(r"[「」『』《》“”\"]")
# Asking whether the robot can/may do it is a capability question (reply), not an action request.
_ABILITY_Q = re.compile(r"能不能|可不可以|能否|会不会|(能|可以)[^，。！？,.!?]{0,14}[吗嘛么]")
# The robot is referred to as 你 at most; 机器人, invented names and roles are rejected.
_NICKNAMES = r"小(助|宝|智|艾|爱|度|护士|帮手|家伙|机灵|伙子|展)|助手"
_ADDRESS = re.compile(r"机器人|" + _NICKNAMES)
_ADDRESS_RAG = re.compile(_NICKNAMES)  # “Luna机器人多少钱” asks about the product
# Exact spellings where the dance rule is looser ("顺风顺水顺才神" still hits "顺风顺水").
_DANCE_SPELLING = {
    "smooth_sailing_and_prosperity": re.compile(r"顺风顺水顺财神|顺风顺水(?!顺)"),
    "karla_ok": re.compile(r"卡啦永远\s*ok", re.IGNORECASE),
}
# 舞 after these is generic ("跳支舞" "来段舞" "跳个好看的舞"); any other "X舞" names a dance.
_GENERIC_WU = re.compile(r"舞(?:蹈|技|步|姿|台|会|曲|者|一)|(?<=[跳支段个首曲的点些起会种一两几来这那多么啥过了着])舞")


def dance_name_issue(query: str) -> str | None:
    """A misspelt or off-list dance name (user decision 2026-10-06: reply, never the nearest dance)."""
    for name, pattern in pa._DANCE_RULES:
        strict = _DANCE_SPELLING.get(name)
        if strict is not None and pattern.search(query) and not strict.search(query):
            return "misspelt dance name"
    rest = query
    for _name, pattern in pa._DANCE_RULES:
        rest = pattern.sub("", rest)
    return "unknown dance name" if "舞" in _GENERIC_WU.sub("", rest) else None


def _bucket(reason: str) -> str:
    return re.sub(r"\s*(\{.*|None)$", "", reason)


def cell_hint(cell: pa.Cell) -> str:
    hint = dims.HINT_OVERRIDES.get(cell.cell_id, cell.hint)
    if cell.tool == "RobotDance" and "「" in hint:  # the brief's 「X」 gets copied into the sentence
        hint = hint.replace("「", "").replace("」", "") + "；舞名前后不加引号或书名号"
    return hint


_SPREAD = re.compile(r"问法分散到：([^；]+)")


def _rotate_hint(hint: str, rng: random.Random) -> str:
    """A listed aspect set gets copied into instruction_zh (“整体概况”); give each round two of them."""
    m = _SPREAD.search(hint)
    if not m:
        return hint
    picked = "或".join(rng.sample(m.group(1).split("、"), 2))
    return f"{hint[: m.start()]}这批句子关心的点在{picked}之间分配，用口语问出来{hint[m.end() :]}"


def run_cell(cell: pa.Cell, args: argparse.Namespace, ctx: Context) -> CellResult:
    rng = random.Random(f"{args.seed}:{cell.cell_id}")
    res = CellResult(cell.cell_id)
    cell_grams: list[frozenset[str]] = []
    pool = pa._SLOT_VALUES.get(cell.slot or "", [])
    quiet = 0
    query_tool = cell.tool in QUERY_TOOLS
    styles = dims.QUERY_STYLES if query_tool else dims.STYLES
    address = _ADDRESS_RAG if cell.tool == "rag_query" else _ADDRESS

    def log_reject(row: dict[str, Any]) -> None:
        res.rejected.append(row)
        ctx.sink("rejected", {"cell": cell.cell_id, **row})
    for rnd in range(1, args.max_rounds + 1):
        specs = [sample_spec(rng, args.dims, cell.cell_id, styles) for _ in range(args.batch)]
        values = [rng.choice(pool) for _ in specs] if pool else [None] * len(specs)
        recent = [a["query"] for a in res.accepted[-args.avoid :]] if args.avoid else []
        existing = ctx.existing.get(cell.tool, [])
        seeds = rng.sample(existing, min(args.avoid, len(existing))) if args.avoid else []
        prompt = build_prompt(replace(cell, hint=_rotate_hint(cell.hint, rng)), specs, values, seeds + recent)
        with ctx.lock:
            crowded = crowded_openers(ctx, args.opener_cap)
        if crowded:
            prompt = prompt.replace("每句同时给出：", "- 这些开头已经用得太多，句子不要以它们开头：" + "、".join(crowded) + "；\n每句同时给出：")
        res.calls += 1
        try:
            rows = ctx.write(prompt, system=SYSTEM, seed=rng.randrange(1 << 30))
            if not _as_list(rows):  # e.g. a single object instead of the array; must not count as saturation
                raise ValueError(f"no sentence list in reply: {str(rows)[:100]}")
        except Exception as exc:  # noqa: BLE001 - one failed call skips one round
            res.failed_calls += 1
            log_reject({"query": "<call>", "reason": "call_error", "detail": str(exc)[:200]})
            continue

        def reject(query: str, reason: str, spec: Spec, **extra: Any) -> None:
            log_reject({"query": query, "reason": _bucket(reason), "detail": reason,
                        "spec": asdict(spec), "round": rnd, **extra})

        candidates: list[tuple[Spec, str, dict[str, Any], Any]] = []
        for spec, value, row in zip(specs, values, _as_list(rows), strict=False):
            if not isinstance(row, dict):
                continue
            res.raw += 1
            query, zh, en, search = (str(row.get(k) or "").strip()
                                     for k in ("query", "instruction_zh", "instruction_en", "search_query"))
            params = {"query": search} if cell.tool in _SEARCH_TOOLS else pa._cell_params(cell, value, "")
            expected = copy.deepcopy(params)
            reason = (
                "prompt leak" if _LEAK.search(query)
                else "quotes" if _QUOTES.search(query)
                else "ability question" if not query_tool and _ABILITY_Q.search(query)
                else "address term" if address.search(query)
                else (cell.tool == "RobotDance" and dance_name_issue(query))
                or validate(cell, params, query, zh, en, value, search)
            )
            atom = None
            if reason is None:
                atom = pa._atom(cell, query, zh, en, params)
                try:
                    pp.normalize_atom_for_planning(atom, ctx.bundle)
                except (ValueError, TypeError) as exc:
                    reason = f"contract: {exc}"
            if reason is None:
                key = _norm_text(query)
                with ctx.lock:
                    reason = "duplicate" if key in ctx.seen else (
                        "crowded opener" if opener_full(ctx, query, args.opener_cap) else None)
                    if reason is None:
                        ctx.seen.add(key)
                        ctx.openers[opener(query)] += 1  # counted before the judge; rejects only loosen the cap
            if reason is not None or atom is None:
                reject(query, reason or "", spec, **({"search_query": search} if search else {}))
                continue
            candidates.append((spec, query, atom, expected))

        verdicts: list[Any] | None = []
        if ctx.judge is not None and candidates:
            res.judge_calls += 1
            try:
                verdicts = _as_list(ctx.judge(judge_prompt(cell.tool, [c[1] for c in candidates]), system=JUDGE_SYSTEM,
                                              seed=rng.randrange(1 << 30)))
            except Exception:  # noqa: BLE001 - an unreviewed batch is dropped, not accepted
                verdicts = None

        valid = novel = 0
        for idx, (spec, query, atom, expected) in enumerate(candidates):
            if ctx.judge is not None:
                verdict = verdicts[idx] if verdicts and idx < len(verdicts) else None
                reason = judge_reason(cell.tool, verdict, expected, error=verdicts is None)
                if reason is not None:
                    reject(query, reason, spec, verdict=verdict)
                    continue
                atom["curation"]["validator"] = "rules+judge"
            g = grams(query)
            sim = max_sim(g, cell_grams)
            cell_grams.append(g)
            valid += 1
            novel += sim < args.tau
            atom["curation"]["writer"] = f"model:{args.model}"
            atom["pilot"] = {
                **asdict(spec),
                "round": rnd,
                "max_sim_cell": round(sim, 3),
                "max_sim_existing": round(max_sim(g, ctx.existing_grams.get(cell.tool, [])), 3),
            }
            res.accepted.append(atom)
            ctx.sink("atoms", atom)
        res.curve.append({"round": rnd, "raw": res.raw, "valid": len(res.accepted), "new_valid": valid,
                          "new_novel": novel})
        print(f"[{cell.cell_id}] r{rnd} valid+{valid} novel+{novel} total={len(res.accepted)}", flush=True)
        quiet = quiet + 1 if novel < args.min_novel else 0
        if quiet >= args.patience:
            res.stop = "saturated"
            break
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def novel_counts(accepted: list[dict[str, Any]]) -> dict[str, int]:
    """Greedy novelty in generation order, recomputed for several thresholds."""
    out: dict[str, int] = {}
    shapes = [grams(a["query"]) for a in accepted]
    for tau in TAUS:
        kept: list[frozenset[str]] = []
        for g in shapes:
            if max_sim(g, kept) < tau:
                kept.append(g)
        out[f"novel@{tau}"] = len(kept)
    return out


def summarize(results: list[CellResult], tau: float) -> dict[str, Any]:
    cells = {}
    dim_valid: dict[str, Counter[str]] = defaultdict(Counter)
    dim_novel: dict[str, Counter[str]] = defaultdict(Counter)
    dim_reject: dict[str, Counter[str]] = defaultdict(Counter)
    for res in results:
        reasons = Counter(r["reason"] for r in res.rejected)
        cells[res.cell_id] = {
            "stop": res.stop,
            "rounds": len(res.curve),
            "calls": res.calls,
            "failed_calls": res.failed_calls,
            "judge_calls": res.judge_calls,
            "raw": res.raw,
            "valid": len(res.accepted),
            "rejected": dict(reasons.most_common()),
            **novel_counts(res.accepted),
            f"novel_vs_existing@{tau}": sum(a["pilot"]["max_sim_existing"] < tau for a in res.accepted),
        }
        for atom in res.accepted:
            for dim in dims.DIMENSIONS:
                value = atom["pilot"][dim] or "-"
                dim_valid[dim][value] += 1
                dim_novel[dim][value] += atom["pilot"]["max_sim_cell"] < tau
        for rej in res.rejected:
            for dim in dims.DIMENSIONS:
                dim_reject[dim][(rej.get("spec") or {}).get(dim) or "-"] += 1
    by_dim = {
        dim: {
            value: {
                "valid": dim_valid[dim][value],
                "novel_rate": round(dim_novel[dim][value] / dim_valid[dim][value], 3) if dim_valid[dim][value] else 0,
                "reject_rate": round(
                    dim_reject[dim][value] / (dim_reject[dim][value] + dim_valid[dim][value]), 3
                ),
            }
            for value in sorted(set(dim_valid[dim]) | set(dim_reject[dim]))
        }
        for dim in dims.DIMENSIONS
    }
    totals = Counter()
    for c in cells.values():
        totals.update({k: v for k, v in c.items() if isinstance(v, int)})
    return {"totals": dict(totals), "cells": cells, "by_dimension": by_dim}


def print_summary(summary: dict[str, Any], tau: float) -> None:
    head = f"{'cell':18}{'stop':>10}{'raw':>6}{'valid':>7}" + "".join(f"{'n@' + str(t):>7}" for t in TAUS)
    print("\n" + head + f"{'vsExist':>9}  top rejects")
    for cid, c in summary["cells"].items():
        top = ", ".join(f"{k}:{v}" for k, v in list(c["rejected"].items())[:3])
        print(
            f"{cid:18}{c['stop']:>10}{c['raw']:>6}{c['valid']:>7}"
            + "".join(f"{c[f'novel@{t}']:>7}" for t in TAUS)
            + f"{c[f'novel_vs_existing@{tau}']:>9}  {top}"
        )
    t = summary["totals"]
    print(f"{'TOTAL':18}{'':>10}{t.get('raw', 0):>6}{t.get('valid', 0):>7}"
          + "".join(f"{t.get(f'novel@{x}', 0):>7}" for x in TAUS) + f"{t.get(f'novel_vs_existing@{tau}', 0):>9}")
    print(f"calls={t.get('calls', 0)} failed={t.get('failed_calls', 0)} judge_calls={t.get('judge_calls', 0)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_dims(values: list[str]) -> frozenset[str]:
    if values == ["none"]:
        return frozenset()
    unknown = set(values) - set(dims.DIMENSIONS)
    if unknown:
        raise SystemExit(f"unknown dims: {sorted(unknown)}")
    return frozenset(values)


def build_context(model: str, temperature: float, judge_model: str | None) -> Context:
    pool = pa._existing_pool()
    curated = pp.load_curated_atoms()
    seen = {_norm_text(row.get("query")) for rows in (*pool.values(), *curated.values()) for row in rows}
    existing = {
        tool: [
            str(row["query"])
            for row in (*pool.get(tool, []), *curated.get(tool, []))
            if not row.get("negative_sample")
        ]
        for tool in (*JUDGES, *QUERY_TOOLS)
    }
    api_key = find_env_value("DASHSCOPE_API_KEY")
    write = functools.partial(llm.chat_json, api_key=api_key, model=model, temperature=temperature)
    judge = functools.partial(llm.chat_json, api_key=api_key, model=judge_model, temperature=0.0) if judge_model else None
    return Context(
        write=write,
        judge=judge,
        bundle=pp.load_contract_bundle(pp.CONTRACT_V2_PATH),
        seen=seen,
        existing=existing,
        existing_grams={tool: [grams(q) for q in queries] for tool, queries in existing.items()},
    )


def run(args: argparse.Namespace, ctx: Context) -> Path:
    cells = [
        replace(c, hint=cell_hint(c))
        for c in ALL_CELLS
        if c.tool in args.tools and not c.reply and (not args.cells or c.cell_id in args.cells)
    ]
    missing = set(args.cells or []) - {c.cell_id for c in cells}
    if missing:
        raise SystemExit(f"unknown cells: {sorted(missing)}")
    out = args.out / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    config = {k: (sorted(v) if isinstance(v, frozenset) else str(v) if isinstance(v, Path)
                  else [str(x) for x in v] if k == "extend" else v)
              for k, v in vars(args).items()}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for run_dir in args.extend:  # continue earlier runs: their sentences are duplicates, avoid seeds and openers
        with (run_dir / "atoms.jsonl").open(encoding="utf-8") as f:
            for row in (json.loads(line) for line in f if line.strip()):
                ctx.seen.add(_norm_text(row["query"]))
                ctx.existing.setdefault(row["expected_function"], []).append(row["query"])
                ctx.openers[opener(row["query"])] += 1
    files = {kind: (out / f"{kind}.jsonl").open("a", encoding="utf-8") for kind in ("atoms", "rejected")}

    def sink(kind: str, row: dict[str, Any]) -> None:  # written as rounds finish, so a run can be read live
        with ctx.lock:
            files[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[kind].flush()

    ctx.sink = sink
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda c: run_cell(c, args, ctx), cells))
    finally:
        for f in files.values():
            f.close()
    summary = summarize(results, args.tau)
    summary["curves"] = {res.cell_id: res.curve for res in results}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, args.tau)
    print(f"\noutput: {out}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--dims", nargs="+", default=list(dims.DIMENSIONS), help="subset of dimensions, or 'none'")
    p.add_argument("--tools", nargs="+", choices=[*JUDGES, *QUERY_TOOLS], default=list(JUDGES))
    p.add_argument("--cells", nargs="+", default=None, help="cell ids within --tools (default: all)")
    p.add_argument("--batch", type=int, default=10, help="sentences per LLM call")
    p.add_argument("--max-rounds", type=int, default=15)
    p.add_argument("--tau", type=float, default=0.5, help="shape similarity below this counts as novel")
    p.add_argument("--min-novel", type=int, default=2, help="a round with fewer novel atoms is quiet")
    p.add_argument("--patience", type=int, default=2, help="stop a cell after this many quiet rounds")
    p.add_argument("--avoid", type=int, default=10, help="existing + recent sentences shown as 'avoid'")
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--judge-model", default=llm.DEFAULT_MODEL, help="LLM label review model")
    p.add_argument("--no-judge", action="store_true", help="accept on the rule validator alone")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=RUNS_DIR)
    p.add_argument("--extend", type=Path, nargs="*", default=[], help="earlier runs to continue (no duplicates)")
    p.add_argument("--opener-cap", type=float, default=0.06, help="max share of one 2-char opener; 0 disables")
    args = p.parse_args(argv)
    args.dims = _parse_dims(args.dims)
    return args


def main() -> None:
    args = parse_args()
    run(args, build_context(args.model, args.temperature, None if args.no_judge else args.judge_model))


if __name__ == "__main__":
    main()
