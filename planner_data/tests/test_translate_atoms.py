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

from planner_data import assemble
from planner_data.atoms import translate_atoms as ta


def _atom(tool, query, zh, params, kind="action", **extra):
    return {"id": "a1", "query": query, "atom_name": "x", "expected_function": tool, "expected_params": params,
            "negative_sample": False, "planner_target": {"kind": kind, "instructions": {"zh": zh, "en": "junk"}},
            **extra}


def test_dance_name_must_follow_the_glossary():
    atom = _atom("RobotDance", "跳段机械舞", "跳机械舞", {"dance": "popping"})
    ok, reason = ta.build(atom, {"query": "Do some popping for us.", "instruction": "Dance popping."})
    assert reason is None and ok["planner_target"]["instructions"]["en"] == "Dance popping"
    assert ok["id"] == "a1_en" and ok["language"] == "en" and ok["source_id"] == "a1"
    assert ta.build(atom, {"query": "Do the mechanical dance.", "instruction": "Dance"})[1] == "dance name"
    generic = _atom("RobotDance", "跳个舞吧", "跳个舞", {"dance": "warm_up_dance"})
    assert ta.build(generic, {"query": "Do the Victory dance", "instruction": "Dance"})[1] == "added dance name"


def test_search_span_must_be_in_the_translation():
    atom = _atom("web_search", "帮我查下最近的新闻", "最近的新闻", {"query": "最近的新闻"}, kind="query")
    ok, _ = ta.build(atom, {"query": "Look up the latest news for me", "instruction": "Search the latest news",
                            "span_en": "the latest news"})
    assert ok["expected_params"] == {"query": "the latest news"}
    assert ta.build(atom, {"query": "Look up the news", "instruction": "x", "span_en": "latest news"})[1] == "span"


def test_city_translated_only_when_the_user_named_it():
    named = _atom("get_weather", "悉尼今天天气", "查询悉尼天气", {"city": "悉尼", "query_type": "now"}, kind="query")
    ok, _ = ta.build(named, {"query": "How's the weather in Sydney today?", "instruction": "Check Sydney weather",
                             "city_en": "Sydney"})
    assert ok["expected_params"]["city"] == "Sydney"
    default = _atom("get_weather", "今天天气咋样", "查询天气", {"city": "深圳", "query_type": "now"}, kind="query")
    ok, _ = ta.build(default, {"query": "How's the weather today?", "instruction": "Check the weather"})
    assert ok["expected_params"]["city"] == "深圳"
    assert ta.agree(ok, {"tool": "get_weather", "params": {"city": "Shenzhen", "query_type": "now"}})


def test_chinese_left_in_translation_is_rejected():
    atom = _atom("RobotGesture", "鞠个躬", "鞠躬", {"gesture": "bow"})
    assert ta.build(atom, {"query": "Please 鞠躬", "instruction": "Bow"})[1] == "chinese left"


def test_english_quote_matches_regardless_of_case():
    query = "Do the Gee. And could you also nod?"
    verdict = {"items": [{"i": 1, "kept": True, "quote": "Do the Gee."},
                         {"i": 2, "kept": True, "quote": "Could you also nod?"}]}
    assert assemble.judge_reason(verdict, query, ["Do the Gee.", "Could you nod?"]) is None


def test_english_merge_check_counts_robot_as_address():
    src = [{"query": "Bow, please."}, {"query": "Nod."}]
    assert assemble.check("Bow, please, and then nod.", src, "en") is None
    assert assemble.check("Robot, bow and nod.", src, "en") == "address term"
    assert "不写 robot" in assemble.merge_prompt(src, "en") and "机器人" not in assemble.merge_prompt(src, "en")
