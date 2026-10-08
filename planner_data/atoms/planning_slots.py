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

"""Slot variants for ActionSequence planning atoms.

An L0 atom stores three independent literals: the user text, the planner
instructions and the target arguments. This module induces a one-slot template
from atoms whose slot value appears exactly once in every text, then renders all
three from the same value so value changes stay aligned.

Value pools are split into disjoint ``train``/``dev``/``test`` partitions so
held-out evaluation can measure unseen values.
"""

from __future__ import annotations

import hashlib
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


SlotSplit = Literal["train", "dev", "test"]
SLOT_SPLITS: tuple[str, ...] = ("train", "dev", "test")

# (zh, en). Challenge-set cities are kept out of ``train``.
_CITIES: dict[str, tuple[tuple[str, str], ...]] = {
    "train": (
        ("南京", "Nanjing"), ("厦门", "Xiamen"), ("重庆", "Chongqing"), ("青岛", "Qingdao"),
        ("天津", "Tianjin"), ("苏州", "Suzhou"), ("深圳", "Shenzhen"), ("昆明", "Kunming"),
        ("上海", "Shanghai"), ("西安", "Xi'an"), ("拉萨", "Lhasa"), ("哈尔滨", "Harbin"),
        ("武汉", "Wuhan"), ("乌鲁木齐", "Urumqi"), ("北京", "Beijing"), ("杭州", "Hangzhou"),
        ("成都", "Chengdu"), ("广州", "Guangzhou"), ("合肥", "Hefei"), ("福州", "Fuzhou"),
        ("南昌", "Nanchang"), ("济南", "Jinan"), ("郑州", "Zhengzhou"), ("太原", "Taiyuan"),
        ("石家庄", "Shijiazhuang"), ("沈阳", "Shenyang"), ("长春", "Changchun"),
        ("呼和浩特", "Hohhot"), ("兰州", "Lanzhou"), ("西宁", "Xining"), ("银川", "Yinchuan"),
        ("贵阳", "Guiyang"), ("南宁", "Nanning"), ("海口", "Haikou"), ("宁波", "Ningbo"),
        ("无锡", "Wuxi"), ("大连", "Dalian"), ("珠海", "Zhuhai"), ("桂林", "Guilin"),
        ("洛阳", "Luoyang"), ("烟台", "Yantai"), ("温州", "Wenzhou"), ("香港", "Hong Kong"),
        ("澳门", "Macau"), ("台北", "Taipei"), ("东京", "Tokyo"), ("首尔", "Seoul"),
        ("新加坡", "Singapore"), ("曼谷", "Bangkok"), ("伦敦", "London"), ("巴黎", "Paris"),
        ("纽约", "New York"), ("悉尼", "Sydney"), ("柏林", "Berlin"), ("莫斯科", "Moscow"),
        ("迪拜", "Dubai"),
    ),
    "dev": (
        ("徐州", "Xuzhou"), ("泉州", "Quanzhou"), ("绍兴", "Shaoxing"), ("常州", "Changzhou"),
        ("扬州", "Yangzhou"), ("丽江", "Lijiang"), ("开封", "Kaifeng"), ("大阪", "Osaka"),
        ("罗马", "Rome"), ("多伦多", "Toronto"), ("开罗", "Cairo"), ("里斯本", "Lisbon"),
    ),
    "test": (
        ("长沙", "Changsha"), ("大理", "Dali"), ("三亚", "Sanya"), ("内罗毕", "Nairobi"),
        ("奥斯陆", "Oslo"), ("雷克雅未克", "Reykjavik"), ("敦煌", "Dunhuang"), ("喀什", "Kashgar"),
        ("赫尔辛基", "Helsinki"), ("布宜诺斯艾利斯", "Buenos Aires"), ("惠州", "Huizhou"),
        ("遵义", "Zunyi"),
    ),
}
# Numbers are drawn at random inside the contract range; a value's split is a
# fixed function of the value, so train/dev/test never share a number.
_NUMBER_RANGES = {"degree": (1, 1080), "step": (1, 30)}
_HELD_OUT_STEPS = {"dev": frozenset({12, 17, 23, 28}), "test": frozenset({13, 19, 25, 29})}
_SYNTH_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_SYNTH_VOWELS = "aeiou"
# Real words / places a CVCVCV pseudo name can collide with.
_SYNTH_BLACKLIST = frozenset(
    {
        "banana", "bikini", "bogota", "bamako", "batumi", "bonito", "casino", "dodoma",
        "domino", "gelato", "kaduna", "kigali", "kimono", "kisumu", "lusaka", "makati",
        "malabo", "maputo", "medina", "mimosa", "moroni", "nagoya", "nakuru", "pajama",
        "potato", "safari", "salami", "samosa", "sonata", "tomato", "toledo", "tuxedo",
    }
)

# RobotDance enum -> (zh match pattern, zh name, en name). Names are nouns, so
# swapping one for another keeps the sentence grammatical.
_DANCES: dict[str, tuple[str, str, str]] = {
    "abracadabr_dance": ("扭胯舞", "扭胯舞", "twist-hip dance"),
    "all_things_grow_dance": ("万物生", "万物生", "All Things Grow dance"),
    "ambition_dance": ("大展宏图", "大展宏图", "Grand Ambition dance"),
    "apt_dance": (r"APT", "APT舞", "APT dance"),
    "egyptian_shake": ("埃及摇摆", "埃及摇摆", "Egyptian Shake dance"),
    "gee": (r"Gee", "Gee", "Gee dance"),
    "gentleman": (r"Gentleman", "Gentleman", "Gentleman dance"),
    "go_cortis_dance": (r"Go\s*Cortis", "Go Cortis", "Go Cortis dance"),
    "idol_dance_1": (r"女团舞\s*1", "女团舞1", "Idol Dance 1"),
    "idol_dance_2": (r"女团舞\s*2", "女团舞2", "Idol Dance 2"),
    "karla_ok": (r"卡啦永远\s*OK", "卡啦永远OK", "Karla OK dance"),
    "lets_bounce": ("蹦蹦", "蹦蹦", "bounce dance"),
    "luv_each_other": ("相亲相爱", "相亲相爱", "love-each-other dance"),
    "one_and_only_dance": ("热烈", "热烈", "enthusiastic dance"),
    "one_spot_dance": ("美美桑内", "美美桑内", "Meimei Sangnei dance"),
    "popping": ("机械舞", "机械舞", "popping dance"),
    "power_up_dance": ("助力舞", "助力舞", "power-up dance"),
    "pulp_fiction_dance": ("低俗小说", "低俗小说", "Pulp Fiction dance"),
    "smooth_sailing_and_prosperity": ("顺风顺水顺财神", "顺风顺水顺财神", "Smooth Sailing and Prosperity dance"),
    "solo_shake": ("孤身摇", "孤身摇", "solo shake dance"),
    "swag_dance": ("展臂舞", "展臂舞", "swag arm dance"),
    "sweep_kick_dance": ("扫腿舞", "扫腿舞", "sweep kick dance"),
    "victory_dance": ("胜利之舞", "胜利之舞", "victory dance"),
    "warm_up_dance": ("热场舞", "热场舞", "warm-up dance"),
    "whatever": ("管他什么音乐", "管他什么音乐", "Whatever dance"),
}
_DANCE_SUFFIX = r"(舞蹈|舞)?"
# A web_search city is swappable only when it modifies a generic topic
# ("北京的交通"); a landmark ("广州白云机场") would become a false entity.
_WEB_CITY_CONTEXT = r"(?=的|地铁|实时|交通)"

_CN_DIGITS = "零一二三四五六七八九"
_EN_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
)
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ZH_NUMBER = r"(\d+|[零一二两三四五六七八九十百]+)"
_EN_NUMBER = r"(\d+|" + "|".join(_EN_WORDS[1:]) + r")"


def number_split(slot: str, value: int) -> str:
    """Fixed train/dev/test partition of a numeric slot value."""
    if slot == "step":
        return next((split for split, held in _HELD_OUT_STEPS.items() if value in held), "train")
    # Everyday angles stay in train; the rest is split by a stable hash.
    if value % 15 == 0 and value <= 360:
        return "train"
    bucket = int(hashlib.blake2b(f"{slot}:{value}".encode(), digest_size=8).hexdigest(), 16) % 10
    return "train" if bucket < 8 else ("dev" if bucket == 8 else "test")


def _en_number_word(value: int) -> str | None:
    if value < len(_EN_WORDS):
        return _EN_WORDS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _EN_TENS[tens] + (f"-{_EN_WORDS[ones]}" if ones else "")
    return None


def _cn_numeral(value: int) -> str:
    if value == 2:
        return "两"
    if value < 10:
        return _CN_DIGITS[value]
    if value < 20:
        return "十" + (_CN_DIGITS[value % 10] if value % 10 else "")
    if value < 100:
        tens, ones = divmod(value, 10)
        return _CN_DIGITS[tens] + "十" + (_CN_DIGITS[ones] if ones else "")
    return str(value)


def _parse_cn(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if text == "两":
        return 2
    if text in _CN_DIGITS:
        return _CN_DIGITS.index(text)
    if "十" in text and "百" not in text:
        head, _, tail = text.partition("十")
        tens = 1 if not head else _CN_DIGITS.find(head)
        ones = 0 if not tail else _CN_DIGITS.find(tail)
        if tens < 1 or ones < 0:
            return None
        return tens * 10 + ones
    return None


def _parse_en(text: str) -> int:
    return int(text) if text.isdigit() else _EN_WORDS.index(text.lower())


@dataclass(frozen=True)
class _Span:
    """One slot occurrence inside a text: ``text[:start] + render + text[end:]``."""

    start: int
    end: int
    style: str  # "literal" | "en_literal" | "digit" | "cn" | "en_word"
    unit: str = ""


@dataclass(frozen=True)
class SlotTemplate:
    """A single-slot template induced from one atom."""

    slot: str  # "city" | "degree" | "step" | "dance"
    path: tuple[str, ...]
    original: Any
    spans: dict[str, _Span]  # keys: "query", "zh", "en"
    # Set when the slot is a substring of a string argument (web_search topic).
    param_span: _Span | None = None


def _unique(pattern: str, text: str, *, flags: int = 0) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags))
    return matches[0] if len(matches) == 1 else None


def _literal_span(text: str, value: str, style: str) -> _Span | None:
    if not value or text.count(value) != 1:
        return None
    start = text.index(value)
    return _Span(start, start + len(value), style)


def _zh_number_span(text: str, value: int, unit: str) -> _Span | None:
    match = _unique(_ZH_NUMBER + r"(?=\s*" + unit + ")", text)
    if match is None or _parse_cn(match.group(1)) != value:
        return None
    style = "digit" if match.group(1).isdigit() else "cn"
    return _Span(match.start(1), match.end(1), style)


def _en_number_span(text: str, value: int, unit: str) -> _Span | None:
    match = _unique(r"\b" + _EN_NUMBER + r"\s+(" + unit + r")s?\b", text, flags=re.IGNORECASE)
    if match is None or _parse_en(match.group(1)) != value:
        return None
    style = "digit" if match.group(1).isdigit() else "en_word"
    return _Span(match.start(1), match.end(0), style, unit)


def _value_at(params: Any, path: tuple[str, ...]) -> Any:
    current = params
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _city_en(zh: str) -> str | None:
    for pool in _CITIES.values():
        for city_zh, city_en in pool:
            if city_zh == zh:
                return city_en
    return None


def _dance_span(text: str, dance: str) -> _Span | None:
    match = _unique(_DANCES[dance][0] + r"\s*" + _DANCE_SUFFIX, text)
    if match is None:
        return None
    return _Span(match.start(), match.end(), "dance", match.group(1) or "")


def _dance_template(texts: dict[str, str], params: Any) -> SlotTemplate | None:
    dance = _value_at(params, ("dance",))
    if dance not in _DANCES:
        return None
    en = _unique(re.escape(_DANCES[dance][2]), texts["en"], flags=re.IGNORECASE)
    spans = {
        "query": _dance_span(texts["query"], dance),
        "zh": _dance_span(texts["zh"], dance),
        "en": _Span(en.start(), en.end(), "en_literal") if en else None,
    }
    if all(spans.values()):
        return SlotTemplate("dance", ("dance",), dance, spans)  # type: ignore[arg-type]
    return None


def _web_city_template(texts: dict[str, str], params: Any) -> SlotTemplate | None:
    topic = _value_at(params, ("query",))
    if not isinstance(topic, str):
        return None
    for city, city_en in (pair for pool in _CITIES.values() for pair in pool):
        query = _unique(re.escape(city) + _WEB_CITY_CONTEXT, texts["query"])
        if query is None:
            continue
        spans = {
            "query": _Span(query.start(), query.end(), "literal"),
            "zh": _literal_span(texts["zh"], city, "literal"),
            "en": _literal_span(texts["en"], city_en, "en_literal"),
        }
        param_span = _literal_span(topic, city, "literal")
        if all(spans.values()) and param_span is not None:
            return SlotTemplate("city", ("query",), city, spans, param_span)  # type: ignore[arg-type]
    return None


def induce_template(atom: dict[str, Any]) -> SlotTemplate | None:
    """Return a slot template when the slot value is unambiguous in all three texts."""
    target = atom.get("planner_target")
    if not isinstance(target, dict) or target.get("kind") == "reply":
        return None
    if atom.get("negative_sample"):
        return None
    instructions = target.get("instructions") or {}
    texts = {
        "query": str(atom.get("query") or ""),
        "zh": str(instructions.get("zh") or ""),
        "en": str(instructions.get("en") or ""),
    }
    tool = atom.get("expected_function")
    params = atom.get("expected_params")

    if tool == "get_weather":
        city = _value_at(params, ("city",))
        city_en = _city_en(city) if isinstance(city, str) else None
        if city_en is None:
            return None
        spans = {
            "query": _literal_span(texts["query"], city, "literal"),
            "zh": _literal_span(texts["zh"], city, "literal"),
            "en": _literal_span(texts["en"], city_en, "en_literal"),
        }
        if all(spans.values()):
            return SlotTemplate("city", ("city",), city, spans)  # type: ignore[arg-type]
        return None

    if tool == "HumanAction" and _value_at(params, ("action",)) == "walk":
        for slot, zh_unit, en_unit in (("degree", "度", "degree"), ("step", "步", "step")):
            value = _value_at(params, ("parameters", slot))
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            spans = {
                "query": _zh_number_span(texts["query"], value, zh_unit),
                "zh": _zh_number_span(texts["zh"], value, zh_unit),
                "en": _en_number_span(texts["en"], value, en_unit),
            }
            if all(spans.values()):
                return SlotTemplate(slot, ("parameters", slot), value, spans)  # type: ignore[arg-type]
        return None

    if tool == "RobotDance":
        return _dance_template(texts, params)
    if tool == "web_search":
        return _web_city_template(texts, params)
    return None


def _render_number(value: int, span: _Span) -> str:
    if span.style == "cn":
        return _cn_numeral(value)
    if span.style in {"digit", "en_word"} and span.unit:
        word = _en_number_word(value) if span.style == "en_word" else None
        return f"{word or value} {span.unit}{'' if value == 1 else 's'}"
    return str(value)


def _render_dance(dance: str, span: _Span, language: str) -> str:
    _pattern, zh_name, en_name = _DANCES[dance]
    if language == "en":
        return en_name
    # Keep an explicit "舞/舞蹈" suffix unless the new name already names a dance.
    return zh_name + ("" if "舞" in zh_name else span.unit)


def _render_text(text: str, span: _Span, rendered: str) -> str:
    return text[: span.start] + rendered + text[span.end :]


_REAL_CITY_NAMES = frozenset(en.lower() for pool in _CITIES.values() for _zh, en in pool)


def _synthetic_city(rng: random.Random) -> str:
    # Capitalized pronounceable CVCVCV pseudo names ("Bekaro"). Consonant-only
    # names get rendered as acronyms by translation ("LTC's pressure"); a final
    # "e" or a blacklisted name would read as an English word or a real place.
    while True:
        name = "".join(rng.choice(_SYNTH_CONSONANTS if index % 2 == 0 else _SYNTH_VOWELS) for index in range(6))
        if name[-1] != "e" and name not in _SYNTH_BLACKLIST and name not in _REAL_CITY_NAMES:
            return name.capitalize()


def _random_number(slot: str, rng: random.Random) -> int:
    low, high = _NUMBER_RANGES[slot]
    # Most turns stay within one revolution; the rest cover the full range.
    if slot == "degree" and rng.random() < 0.75:
        high = 360
    return rng.randint(low, high)


def _pick_values(template: SlotTemplate, split: str, count: int, rng: random.Random) -> list[Any]:
    if template.slot == "city":
        pool = [pair for pair in _CITIES[split] if pair[0] != template.original]
        rng.shuffle(pool)
        return pool[:count]
    if template.slot == "dance":
        # Enum values cannot be held out: every split uses the full enum.
        dances = [dance for dance in _DANCES if dance != template.original]
        rng.shuffle(dances)
        return dances[:count]
    values: list[int] = []
    for _ in range(count * 50):
        if len(values) >= count:
            break
        value = _random_number(template.slot, rng)
        if value != template.original and value not in values and number_split(template.slot, value) == split:
            values.append(value)
    return values


def _surfaces(template: SlotTemplate, value: Any) -> list[str]:
    if template.slot == "city":
        zh, en = value
        return sorted({zh, en})
    if template.slot == "dance":
        # The enum id is not copied from the text, so there is nothing to gate on.
        return []
    words = {str(value), _cn_numeral(value)}
    word = _en_number_word(value)
    if word:
        words.update({word, word.replace("-", " ")})
    return sorted(words)


def render_variant(
    atom: dict[str, Any], template: SlotTemplate, value: Any, split: str, *, synthetic: bool = False
) -> dict[str, Any]:
    """Render query, instructions and arguments of ``atom`` from one slot value."""
    variant = deepcopy(atom)
    for key in ("slot_variants", "slot_keep_original", "slot_synthetic_rate"):
        variant.pop(key, None)
    target = variant["planner_target"]
    if template.slot == "city":
        zh_value, en_value = value
        rendered = {"query": zh_value, "zh": zh_value, "en": en_value}
        argument: Any = zh_value
        if template.param_span is not None:
            topic = _value_at(variant["expected_params"], template.path)
            argument = _render_text(topic, template.param_span, zh_value)
    elif template.slot == "dance":
        rendered = {
            key: _render_dance(value, span, "en" if key == "en" else "zh")
            for key, span in template.spans.items()
        }
        argument = value
    else:
        rendered = {key: _render_number(value, span) for key, span in template.spans.items()}
        argument = value
    variant["query"] = _render_text(variant["query"], template.spans["query"], rendered["query"])
    if "test_input" in variant:
        variant["test_input"] = variant["query"]
    for language in ("zh", "en"):
        target["instructions"][language] = _render_text(
            target["instructions"][language], template.spans[language], rendered[language]
        )
    params = variant["expected_params"]
    container = params
    for key in template.path[:-1]:
        container = container[key]
    container[template.path[-1]] = argument
    if variant.get("expected_steps"):
        variant["expected_steps"][0]["expected_params"] = deepcopy(params)
    variant["slot_values"] = [
        {
            "slot": template.slot,
            "value": argument,
            "split": split,
            "surfaces": _surfaces(template, value),
            "synthetic": synthetic,
        }
    ]
    return variant


def expand_slot_variants(
    atom: dict[str, Any], *, count: int, split: str, validate: Any, synthetic: bool = False
) -> list[dict[str, Any]]:
    """Render up to ``count`` validated variants for one atom (empty when not templatable).

    ``validate`` normalizes a variant against the target contract and raises on
    an invalid one; invalid variants are skipped. With ``synthetic``, city slots
    additionally get ``count`` never-seen pseudo names marked ``synthetic``; the
    case builder decides how often those are drawn.
    """
    if count <= 0:
        return []
    if split not in SLOT_SPLITS:
        raise ValueError(f"unknown slot split: {split}")
    template = induce_template(atom)
    if template is None:
        return []
    rng = random.Random(f"{split}:{atom.get('id')}")
    values = [(value, False) for value in _pick_values(template, split, count, rng)]
    if synthetic and template.slot == "city":
        names = {_synthetic_city(rng) for _ in range(count)}
        values.extend(((name, name), True) for name in sorted(names))
    variants: list[dict[str, Any]] = []
    for value, is_synthetic in values:
        try:
            variants.append(validate(render_variant(atom, template, value, split, synthetic=is_synthetic)))
        except ValueError:
            continue
    return variants


def slot_surfaces_present(query: str, intents: list[dict[str, Any]]) -> bool:
    """True when every slot value of every intent still appears in ``query``."""
    lowered = query.lower()
    for intent in intents:
        for slot in intent.get("slot_values") or []:
            surfaces = slot.get("surfaces") or []
            # City names are copied verbatim into arguments, so case must match;
            # number words may start a sentence ("Ten steps").
            if slot.get("slot") == "city":
                found = any(str(surface) in query for surface in surfaces)
            else:
                found = any(str(surface).lower() in lowered for surface in surfaces)
            if surfaces and not found:
                return False
    return True
