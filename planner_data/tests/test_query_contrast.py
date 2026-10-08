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

"""Offline checks for query_contrast.py (no LLM calls)."""

from __future__ import annotations

import random

from planner_data.atoms import contrast, pilot
from planner_data.atoms import query_contrast as qc


def _pos(cell_id, query, value=None, search=""):
    cell = qc._CELLS[cell_id]
    return pilot.pa._atom(cell, query, query, "x", pilot.pa._cell_params(cell, value, search))


def _ctx(write, judge):
    return contrast.Context(write=write, judge=judge, seen=set(),
                            bundle=pilot.pp.load_contract_bundle(pilot.pp.CONTRACT_V2_PATH))


def test_select_gives_cross_flips_half_of_their_sources():
    atoms = [_pos("now_city", f"北京现在天气咋样{i}", "北京") for i in range(8)]
    atoms += [_pos("status_电量", f"你还剩多少电{i}") for i in range(8)]
    atoms += [_pos("ws_news", f"今天有啥新闻{i}", search=f"今天有啥新闻{i}") for i in range(8)]
    items = qc.select(atoms, 8, random.Random(0))
    flips = lambda cell: [i.flip for i in items if i.source["atom_name"] == cell]  # noqa: E731
    assert sorted(set(flips("now_city")) & {"weather_to_events", "weather_to_traffic"}) == [
        "weather_to_events", "weather_to_traffic"]
    assert flips("status_电量").count("battery_to_spec") == 4
    assert "ability" not in flips("status_电量")  # "你会查自己的电量吗" reads as a request
    assert not any(qc.FLIPS[f].target for f in flips("ws_news"))
    assert all(i.hint for i in items if qc.FLIPS[i.flip].hints)
    assert {i.value for i in items if i.flip.startswith("weather_to")} == {"北京"}


def test_requirement_and_check():
    city = _pos("now_city", "北京现在天气咋样", "北京")
    assert "回答是否能查天气" in qc.requirement(qc.Item(city, "ability"))
    assert "北京同一时间" in qc.requirement(qc.Item(city, "weather_to_events"))
    assert "search_query" in qc.requirement(qc.Item(city, "weather_to_events"))
    assert "search_query" not in qc.requirement(qc.Item(_pos("ws_world_time", "纽约几点了", "纽约",
                                                             "纽约几点了"), "world_to_time"))

    def check(item, query, zh, search=""):
        return qc.check(item, query, zh, "x", search)

    assert check(qc.Item(city, "ability"), "你会查天气吗", "回答是否能查天气") is None
    assert check(qc.Item(city, "ability"), "查下天气", "回答是否能查天气") == "no ability marker"
    assert check(qc.Item(city, "ability"), "你会查北京天气吗", "回答是否能查天气") == "ability keeps the object"
    scene = _pos("rag_scene", "Oli能用在餐厅迎宾吗", search="Oli能用在餐厅迎宾吗")
    assert check(qc.Item(scene, "ability"), "这个机器人能用在餐厅迎宾吗？你会查产品适用场景吗？",
                 "回答是否能查产品适用场景") == "ability keeps the object"
    assert check(qc.Item(city, "statement"), "北京今天好热啊", "回应用户的闲聊") is None
    assert check(qc.Item(city, "statement"), "北京今天热吗", "回应用户的闲聊") == "still a question"
    assert check(qc.Item(city, "deferred"), "等会儿再查北京天气", "回应稍后再查的安排") is None
    assert check(qc.Item(city, "others"), "小李，你查下北京天气", "查询北京天气") == "reply instruction style"
    assert check(qc.Item(city, "others"), "北京现在天气咋样", "回应") == "unchanged"
    assert check(qc.Item(city, "weather_to_events"), "北京现在有啥活动", "搜索北京现在有啥活动",
                 "北京现在有啥活动") is None
    assert check(qc.Item(city, "weather_to_events"), "上海现在有啥活动", "搜索上海现在有啥活动",
                 "上海现在有啥活动") == "search_query misses city"
    price = _pos("rag_price", "Luna多少钱", search="Luna多少钱")
    assert check(qc.Item(price, "rag_to_launch", "iPhone 17"), "iPhone 17多少钱", "搜索iPhone 17多少钱",
                 "iPhone 17多少钱") is None
    assert check(qc.Item(price, "statement"), "Luna机器人挺贵的", "回应用户的闲聊") is None  # product name is fine
    battery = _pos("status_电量", "你还剩多少电")
    assert check(qc.Item(battery, "battery_to_spec"), "你满电续航多久", "搜索你满电续航多久",
                 "你满电续航多久") is None
    spec = _pos("rag_spec_self", "你满电能跑多久", search="你满电能跑多久")
    assert check(qc.Item(spec, "spec_to_battery"), "你现在还剩多少电", "查询当前电量") is None
    assert qc.target_params(qc.Item(spec, "spec_to_battery"), "") == {"query": "电量"}


def test_judge_reason():
    city = _pos("now_city", "北京现在天气咋样", "北京")
    jr = qc.judge_reason
    assert jr(qc.Item(city, "ability"), {"ask": False, "source": "weather"}, None) is None
    assert jr(qc.Item(city, "others"), {"ask": True, "source": "weather"}, None) == "judge says ask"
    assert jr(qc.Item(city, "common"), {"ask": True, "source": "common"}, None) is None
    assert jr(qc.Item(city, "common"), {"ask": True, "source": "public"}, None) == "judge source public"
    events = qc.Item(city, "weather_to_events")
    assert jr(events, {"ask": True, "source": "public"}, {"query": "北京有啥活动"}) is None
    assert jr(events, {"ask": True, "source": "weather", "city": "北京", "period": "now"},
              {"query": "北京有啥活动"}).startswith("judge mismatch")


def test_process_batch_builds_reply_and_cross_atoms():
    city = _pos("now_city", "北京现在天气咋样", "北京")
    items = [qc.Item(city, "ability"), qc.Item(city, "weather_to_events"), qc.Item(city, "others", "小李")]
    rows = [
        {"i": 1, "query": "你会查天气吗", "instruction_zh": "回答是否能查天气", "instruction_en": "Answer"},
        {"i": 2, "query": "北京现在有啥活动", "instruction_zh": "搜索北京现在有啥活动", "instruction_en": "Search",
         "search_query": "北京现在有啥活动"},
        {"i": 3, "query": "小李，你查下北京天气", "instruction_zh": "回应用户对别人说的话", "instruction_en": "R"},
    ]
    verdicts = iter([{"ask": False, "source": "weather"}, {"ask": True, "source": "public"},
                     {"ask": True, "source": "weather", "city": "北京", "period": "now"}])
    accepted, rejected = qc.process_batch(items, _ctx(lambda p, **kw: rows, lambda p, **kw: [next(verdicts)]), 1)
    assert [a["atom_name"] for a in accepted] == ["neg_ability_now_city", "x_weather_to_events"]
    assert accepted[0]["planner_target"]["kind"] == "reply" and accepted[0]["negative_sample"] is True
    assert accepted[1]["expected_function"] == "web_search"
    assert accepted[1]["expected_params"] == {"query": "北京现在有啥活动"}
    assert [r["reason"] for r in rejected] == ["judge says ask"]
