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

"""Offline checks for contrast.py (no LLM calls)."""

from __future__ import annotations

import random

from planner_data.atoms import contrast, pilot


def _cell(cell_id):
    return next(c for c in pilot.pa.CELLS if c.cell_id == cell_id)


def _pos(cell_id, query, zh=None):
    cell = _cell(cell_id)
    return pilot.pa._atom(cell, query, zh or query, "x", pilot.pa._cell_params(cell, None, ""))


def _ctx(write, judge, sink=None):
    ctx = contrast.Context(write=write, judge=judge, bundle=pilot.pp.load_contract_bundle(pilot.pp.CONTRACT_V2_PATH),
                           seen=set())
    if sink is not None:
        ctx.sink = sink
    return ctx


def test_select_spreads_flips_and_unsupported_items():
    atoms = [_pos("bow", f"鞠个躬{i}") for i in range(16)] + [_pos("popping", f"跳机械舞{i}") for i in range(16)]
    atoms += [_pos("dance_unnamed", f"跳个舞{i}") for i in range(16)]
    atoms.append(_pos("sit_down", "坐下"))  # HumanAction has no negatives here yet
    items = contrast.select(atoms, 14, random.Random(0))
    assert len(items) == 42 and {i.tool for i in items} == {"RobotGesture", "RobotDance"}
    by_cell = lambda cell: {i.flip for i in items if i.source["atom_name"] == cell}  # noqa: E731
    assert by_cell("popping") == set(contrast.FLIPS) - {"say"}
    assert by_cell("bow") == set(contrast.FLIPS) - {"typo"}
    assert by_cell("dance_unnamed") == set(contrast.FLIPS) - {"typo", "say"}
    for tool in ("RobotGesture", "RobotDance"):
        unsup = [i for i in items if i.tool == tool and i.flip == "unsupported"]
        assert all((u.target, u.core) in contrast.UNSUPPORTED[tool] for u in unsup)
    assert all(i.hint for i in items if i.flip in contrast.HINTS)
    assert "回答是否会鞠躬" in contrast.requirement(contrast.Item(_pos("bow", "鞠躬"), "ability"))
    assert "说明不会跳天鹅湖" in contrast.requirement(contrast.Item(_pos("popping", "跳机械舞"), "unsupported", "天鹅湖"))
    assert "把舞名机械舞里" in contrast.requirement(contrast.Item(_pos("popping", "跳机械舞"), "typo"))


def test_check_keeps_the_item_and_flips_the_intent():
    bow = _pos("bow", "来，鞠个躬")

    def check(flip, query, zh="回应鞠躬", target="", core=""):
        return contrast.check(contrast.Item(bow, flip, target, core), query, zh, "x")

    assert check("ability", "你能鞠个躬吗") is None
    assert check("ability", "你会鞠躬吗") is None
    assert check("ability", "鞠个躬吧") == "no ability marker"
    assert check("stop", "别鞠躬了") is None
    assert check("stop", "鞠躬鞠得好") == "no stop marker"
    assert check("comment", "刚才那个鞠躬真标准") is None
    assert check("comment", "可以鞠个躬吗") == "ability question"
    assert check("others", "小朋友们，给老师鞠个躬") is None
    assert check("wish", "要是能鞠个躬就好了") is None
    assert check("wish", "要是能鼓个掌就好了").startswith("query derives")  # the item must stay
    assert check("stop", "别鞠躬了", zh="鞠躬") == "reply instruction style"
    assert check("unsupported", "来，敬个礼", zh="说明做不了敬礼", target="敬礼", core=r"敬.{0,2}礼") is None
    assert check("unsupported", "来，鞠个躬", zh="说明做不了敬礼", target="敬礼",
                 core=r"敬.{0,2}礼") == "misses unsupported item"
    assert check("unsupported", "敬个礼再鞠个躬", zh="说明做不了敬礼", target="敬礼",
                 core=r"敬.{0,2}礼").startswith("query derives")
    dance = contrast.Item(_pos("popping", "跳段机械舞"), "unsupported", "女团舞3", r"女团舞\s*(?:3|三)")
    assert contrast.check(dance, "跳段女团舞三", "说明不会跳女团舞3", "x") is None
    assert contrast.check(contrast.Item(_pos("popping", "跳"), "unsupported", "天鹅湖", "天鹅湖"),
                          "跳段天鹅湖", "说明不会跳天鹅湖", "x") is None  # generic dance words are fine


def test_check_deferred_sing_joke_typo():
    bow = _pos("bow", "来，鞠个躬")
    popping = _pos("popping", "给大家跳段机械舞吧")
    sailing = _pos("smooth_sailing_and_prosperity", "来段顺风顺水顺财神")

    def check(src, flip, query, zh):
        return contrast.check(contrast.Item(src, flip), query, zh, "x")

    assert check(bow, "deferred", "等客人来了再鞠个躬", "回应稍后再鞠躬的安排") is None
    assert check(bow, "deferred", "来，鞠个躬吧", "回应稍后再鞠躬的安排") == "no deferred marker"
    assert check(popping, "sing", "给大家唱首歌吧", "说明唱不了歌") is None
    assert check(popping, "sing", "给大家唱段机械舞吧", "说明唱不了歌").startswith("query derives")
    assert check(bow, "joke", "来，讲个笑话", "讲一个笑话") is None
    assert check(bow, "joke", "来，讲个故事", "讲一个故事") == "no joke marker"
    assert check(sailing, "typo", "来段顺风顺水顺才神", "说明不会跳顺风顺水顺才神") is None
    assert check(popping, "typo", "给大家跳段机械午吧", "说明不会跳机械午") is None
    assert check(popping, "typo", "给大家跳段机械舞吧", "说明不会跳机械舞") == "dance name spelt right"
    assert check(popping, "typo", "给大家跳段机械午吧", "说明不会跳XX") == "instruction misses the written name"
    assert check(sailing, "stop", "别跳顺风顺水顺才神了", "回应不用跳") == "misspelt dance name"


def test_check_mention_teach_say_conditional():
    bow = _pos("bow", "来，鞠个躬")
    wave = _pos("wave_greet_bye", "跟大家挥挥手")

    def check(src, flip, query, zh, hint=""):
        return contrast.check(contrast.Item(src, flip, hint=hint), query, zh, "x")

    assert check(bow, "mention", "鞠躬用英文怎么说", "回答关于鞠躬的问题") is None
    assert check(bow, "mention", "鞠个躬吧", "回答关于鞠躬的问题") == "no mention marker"
    assert check(bow, "teach", "教教我怎么鞠躬", "讲解怎么鞠躬") is None
    assert check(bow, "teach", "教我鞠躬，先示范一下", "讲解怎么鞠躬") == "demonstration"
    assert check(wave, "say", "来，跟大家说声再见", "说声再见", "说声再见") is None
    assert check(bow, "say", "来，跟大家说声谢谢", "说声谢谢", "说声谢谢") is None
    assert check(wave, "say", "跟大家挥挥手说声再见", "说声再见", "说声再见").startswith("query derives")
    assert check(bow, "say", "鞠个躬说声谢谢", "说声谢谢", "说声谢谢").startswith("query derives")
    assert check(bow, "conditional", "我答对了你就鞠个躬", "回应条件满足后再鞠躬的约定") is None
    assert check(bow, "conditional", "来，鞠个躬", "回应条件满足后再鞠躬的约定") == "no conditional marker"
    assert "说声谢谢" in contrast.requirement(contrast.Item(bow, "say", hint="说声谢谢"))


def test_unsupported_items_never_match_a_supported_rule():
    for tool, items in contrast.UNSUPPORTED.items():
        for target, _core in items:
            derived = pilot.pa.DERIVERS[tool](f"来个{target}")
            assert derived in (None, {"dance": pilot.pa._GENERIC_DANCE}), (target, derived)


def test_judge_reason():
    bow = _pos("bow", "鞠躬")
    jr = contrast.judge_reason
    assert jr(contrast.Item(bow, "stop"), {"request": False, "gesture": "bow"}) is None
    assert jr(contrast.Item(bow, "others"), {"request": True, "gesture": "bow"}) == "judge says request"
    unsup = contrast.Item(bow, "unsupported", "敬礼", "敬礼")
    assert jr(unsup, {"request": True, "gesture": "none"}) is None
    assert jr(unsup, {"request": True, "gesture": "bow"}) == "judge maps to bow"
    assert jr(unsup, {"request": False, "gesture": "none"}) == "judge not request"
    dance = contrast.Item(_pos("popping", "跳"), "unsupported", "天鹅湖", "天鹅湖")
    assert jr(dance, {"request": True, "dance": "other"}) is None
    assert jr(dance, {"request": True, "dance": "unnamed"}) == "judge maps to unnamed"
    assert jr(dance, None) == "judge missing"
    typo = contrast.Item(_pos("popping", "跳"), "typo")
    assert jr(typo, {"request": True, "dance": "popping"}) is None
    assert jr(typo, {"request": False, "dance": "popping"}) == "judge not request"
    assert jr(contrast.Item(bow, "joke"), {"request": False, "gesture": "none"}) is None


def test_process_batch_builds_reply_atoms():
    bow = _pos("bow", "来，鞠个躬")
    items = [contrast.Item(bow, "ability"), contrast.Item(bow, "others"),
             contrast.Item(bow, "unsupported", "敬礼", r"敬.{0,2}礼"), contrast.Item(bow, "stop")]
    rows = [
        {"i": 1, "query": "你能鞠个躬吗", "instruction_zh": "回答是否会鞠躬", "instruction_en": "Answer"},
        {"i": 2, "query": "王老师，您给大家鞠个躬", "instruction_zh": "回应用户对别人说的话", "instruction_en": "R"},
        {"i": 3, "query": "来，敬个礼", "instruction_zh": "说明做不了敬礼", "instruction_en": "Explain"},
        {"i": 4, "query": "鞠躬", "instruction_zh": "回应不用鞠躬", "instruction_en": "R"},
    ]
    verdicts = iter([{"request": False, "gesture": "bow"}, {"request": True, "gesture": "bow"},
                     {"request": True, "gesture": "none"}])
    prompts, sunk = [], []
    ctx = _ctx(lambda p, **kw: rows, lambda p, **kw: (prompts.append(p), [next(verdicts)])[1],
               sink=lambda kind, row: sunk.append(kind))
    accepted, rejected = contrast.process_batch(items, ctx, seed=1)
    assert [a["contrast"]["flip"] for a in accepted] == ["ability", "unsupported"]
    for a in accepted:
        assert a["planner_target"]["kind"] == "reply" and a["negative_sample"] is True
        assert a["contrast"]["source_query"] == "来，鞠个躬"
    assert accepted[1]["contrast"]["unsupported"] == "敬礼" and accepted[1]["atom_name"] == "neg_unsupported_bow"
    assert sorted(r["reason"] for r in rejected) == ["judge says request", "no stop marker"]
    assert len(prompts) == 3 and "1. 来，敬个礼" in prompts[2]  # one query per judge call
    assert sorted(sunk) == ["atoms", "atoms", "rejected", "rejected"]


def test_typo_instruction_comes_from_typo_name():
    popping = _pos("popping", "给大家跳段机械舞吧")
    items = [contrast.Item(popping, "typo"), contrast.Item(popping, "typo")]
    rows = [{"i": 1, "query": "给大家跳段机械午吧", "instruction_zh": "说明不会跳句中的错字舞名",
             "instruction_en": "E", "typo_name": "机械午"},
            {"i": 2, "query": "给大家跳段机戒舞吧", "instruction_zh": "说明不会跳机戒舞", "instruction_en": "E"}]
    ctx = _ctx(lambda p, **kw: rows, lambda p, **kw: [{"request": True, "dance": "popping"}])
    accepted, rejected = contrast.process_batch(items, ctx, seed=1)
    assert [a["planner_target"]["instructions"] for a in accepted] == [
        {"zh": "说明不会跳机械午", "en": "Explain that you cannot dance 机械午"}]
    assert rejected[0]["reason"] == "empty field"  # no typo_name, the writer's own zh is not used
    assert "typo_name" in contrast.requirement(items[0])


def test_failed_call_rejects_the_whole_batch():
    def boom(*a, **kw):
        raise RuntimeError("down")

    items = [contrast.Item(_pos("bow", "鞠躬"), "ability")]
    accepted, rejected = contrast.process_batch(items, _ctx(boom, boom), seed=1)
    assert accepted == [] and rejected[0]["reason"] == "call_error"
