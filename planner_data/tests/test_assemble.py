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

"""Offline checks for assemble.py (no LLM calls)."""

from __future__ import annotations

import json
import random

from planner_data import assemble, augment
from planner_data.atoms import contrast, pilot


BUNDLE = pilot.pp.load_contract_bundle(pilot.pp.CONTRACT_V2_PATH)


def _cell(cell_id):
    return next(c for c in pilot.pa.CELLS if c.cell_id == cell_id)


def _pos(cell_id, query, zh, en="x"):
    cell = _cell(cell_id)
    return pilot.pa._atom(cell, query, zh, en, pilot.pa._cell_params(cell, None, ""))


def _neg(source, flip, query, zh):
    return contrast.build_atom(contrast.Item(source, flip), query, zh, "y", "test")


BOW = _pos("bow", "来，给大家鞠个躬吧。", "鞠躬")
NOD = _pos("nod", "你点个头。", "点头")
GEE = _pos("gee", "跳一段Gee吧。", "跳Gee")
NEG_SING = _neg(NOD, "sing", "你给大家唱首歌吧。", "说明唱不了歌")
NEG_STOP = _neg(BOW, "stop", "别鞠躬了。", "回应不用鞠躬")


def _write_run(tmp_path, name, atoms):
    run = tmp_path / name
    run.mkdir()
    (run / "atoms.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in atoms),
                                     encoding="utf-8")
    return run


def _pool(*atoms):
    pool = assemble.Pool({}, {}, assemble.Counter())
    for a in atoms:
        a = pilot.pp.normalize_atom_for_planning(a, BUNDLE)
        if a.get("negative_sample"):
            pool.negatives.setdefault(a["contrast"]["flip"], []).append(a)
        else:
            pool.positives.setdefault(a["expected_function"], {}).setdefault(a["atom_name"], []).append(a)
    return pool


def test_load_pool_drops_duplicates_and_flagged_dances(tmp_path):
    flagged = _pos("karla_ok", "卡拉永远OK来一段", "跳卡啦永远OK")
    run_a = _write_run(tmp_path, "a", [BOW, NOD, flagged])
    run_b = _write_run(tmp_path, "b", [dict(BOW, id="other"), NEG_SING])
    pool = assemble.load_pool([run_a, run_b], BUNDLE)
    assert pool.dropped == {"duplicate": 1, "dance name": 1}
    assert sorted(pool.positives["RobotGesture"]) == ["bow", "nod"]
    assert [a["query"] for a in pool.negatives["sing"]] == [NEG_SING["query"]]


def test_sample_case_never_repeats_a_cell_or_flip():
    pool = _pool(BOW, NOD, GEE, NEG_SING, NEG_STOP)
    rng = random.Random(0)
    for _ in range(200):
        case = assemble.sample_case(pool, 3, 0.5, rng)
        if case is None:
            continue
        cells = [assemble.source_cell(a) for a in case]
        assert len(cells) == len(set(cells))  # NEG_STOP (bow) never joins BOW
        flips = [a["contrast"]["flip"] for a in case if a.get("negative_sample")]
        assert len(flips) == len(set(flips))
    assert assemble.sample_case(pool, 4, 0.0, rng) is None  # only 3 positive cells


def test_render_row_matches_v330_format():
    pool = _pool(BOW, NEG_SING)
    atoms = [pool.positives["RobotGesture"]["bow"][0], pool.negatives["sing"][0]]
    row = assemble.render_row(atoms, "先鞠个躬，再唱首歌。", "zh", BUNDLE)
    system, tools, instruction, tasks = augment.parse_row(row)
    assert system == BUNDLE.system_prompt.content
    assert [t["name"] for t in tools] == list(BUNDLE.display_order)
    assert instruction == "先鞠个躬，再唱首歌。"
    assert tasks == [
        {"kind": "action", "instruction": "鞠躬", "name": "RobotGesture", "arguments": {"gesture": "bow"}},
        {"kind": "reply", "instruction": "说明唱不了歌"},
    ]
    augment.validate(tools, tasks)
    meta = row["metadata"]
    assert (meta["level"], meta["polarity"], meta["flips"], meta["actions"]) == (2, "negative", ["sing"], ["bow"])


def _ctx(write, judge, rows):
    ctx = assemble.Context(write=write, judge=judge, bundle=BUNDLE)
    ctx.sink = lambda kind, row: rows.append((kind, row))
    return ctx


def test_level_one_uses_the_atom_sentence_without_calls():
    rows = []
    atom = _pool(BOW).positives["RobotGesture"]["bow"][0]
    ctx = _ctx(lambda *a, **k: 1 / 0, lambda *a, **k: 1 / 0, rows)
    row = assemble.process_case([atom], ctx, 0)
    assert json.loads(row["messages"][1]["content"])["instruction"] == BOW["query"]
    assert [k for k, _ in rows] == ["rows"]


def test_run_uses_every_atom_once_for_level_one_and_sizes_the_rest_by_share(tmp_path):
    atoms = _write_run(tmp_path, "a", [BOW, NOD, GEE])
    args = assemble.parse_args(["--name", "t", "--atoms", str(atoms), "--levels", "1", "2", "--l1-share", "0.5",
                                "--max-rounds", "1", "--out", str(tmp_path / "runs")])
    ctx = assemble.Context(write=lambda *a, **k: 1 / 0, judge=lambda *a, **k: 1 / 0, bundle=BUNDLE)
    out = assemble.run(args, ctx)
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(json.loads(r["messages"][1]["content"])["instruction"] for r in rows) == \
        sorted(a["query"] for a in (BOW, NOD, GEE))
    summary = (out / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert ["L2", "4", "0"] in [line.split() for line in summary]  # 3 L1 rows at 50% → 3 L2 rows, asked ×1.3 + 1


def test_merge_kept_only_when_judge_finds_every_intent_in_order():
    pool = _pool(BOW, NEG_SING)
    atoms = [pool.positives["RobotGesture"]["bow"][0], pool.negatives["sing"][0]]
    prompts = []

    def write(prompt, **_):
        prompts.append(prompt)
        return {"query": "先给大家鞠个躬，然后唱首歌吧。"}

    verdicts = iter([
        {"items": [{"i": 1, "kept": True, "quote": "鞠个躬"}, {"i": 2, "kept": True, "quote": "唱首歌"}], "extra": []},
        {"items": [{"i": 1, "kept": True, "quote": "鞠个躬"}], "extra": []},
        {"items": [{"i": 1, "kept": True, "quote": "鞠个躬"}, {"i": 2, "kept": True, "quote": "唱首歌"}],
         "extra": ["跳舞"]},
        {"items": [{"i": 1, "kept": True, "quote": "唱首歌"}, {"i": 2, "kept": True, "quote": "鞠个躬"}], "extra": []},
        {"items": [{"i": 1, "kept": True, "quote": "鞠躬"}, {"i": 2, "kept": True, "quote": "唱首歌"}], "extra": []},
    ])
    rows = []
    ctx = _ctx(write, lambda *a, **k: next(verdicts), rows)
    assert assemble.process_case(atoms, ctx, 0) is not None
    assert "1. 来，给大家鞠个躬吧。" in prompts[0] and "2. 你给大家唱首歌吧。" in prompts[0]
    for _ in range(4):
        assert assemble.process_case(atoms, ctx, 0) is None
    reasons = [row["reason"] for kind, row in rows if kind == "rejected"]
    assert reasons == ["judge misses [2]", "judge extra", "quote mismatch [1, 2]", "quote not found"]


def test_merge_that_moves_a_sentence_is_labelled_in_the_new_order():
    pool = _pool(BOW, NEG_SING)
    atoms = [pool.positives["RobotGesture"]["bow"][0], pool.negatives["sing"][0]]
    verdict = {"items": [{"i": 1, "kept": True, "quote": "给大家鞠个躬"}, {"i": 2, "kept": True, "quote": "唱首歌"}],
               "extra": []}
    rows = []
    ctx = _ctx(lambda *a, **k: {"query": "你先唱首歌吧，然后给大家鞠个躬。"}, lambda *a, **k: verdict, rows)
    row = assemble.process_case(atoms, ctx, 0)
    assert [t.get("name") for t in json.loads(row["messages"][-1]["content"])] == [None, "RobotGesture"]


def test_merge_rejects_quotes_the_judge_attached_to_the_wrong_sentence():
    """Seen in smoke4: the writer swapped 鼓掌/谢幕 and the judge quoted them by position, so the order check
    passed and the label order came out reversed."""
    pool = _pool(BOW, NEG_SING)
    atoms = [pool.positives["RobotGesture"]["bow"][0], pool.negatives["sing"][0]]
    verdict = {"items": [{"i": 1, "kept": True, "quote": "先唱首歌吧"}, {"i": 2, "kept": True, "quote": "给大家鞠个躬"}],
               "extra": []}
    rows = []
    ctx = _ctx(lambda *a, **k: {"query": "先唱首歌吧，然后给大家鞠个躬。"}, lambda *a, **k: verdict, rows)
    assert assemble.process_case(atoms, ctx, 0) is None
    assert rows[0][1]["reason"] == "quote mismatch [1, 2]"


def test_merge_rule_rejects_before_judging():
    pool = _pool(BOW, NOD)
    atoms = [pool.positives["RobotGesture"]["bow"][0], pool.positives["RobotGesture"]["nod"][0]]
    rows = []
    ctx = _ctx(lambda *a, **k: {"query": "机器人先鞠躬再点头。"}, lambda *a, **k: 1 / 0, rows)
    assert assemble.process_case(atoms, ctx, 0) is None
    assert rows[0][1]["reason"] == "address term"


def test_judge_prompt_compares_with_source_sentences_not_labels():
    pool = _pool(BOW, NEG_SING)
    atoms = [pool.positives["RobotGesture"]["bow"][0], pool.negatives["sing"][0]]
    text = assemble.judge_prompt("先鞠躬再唱歌", atoms)
    assert "1. 来，给大家鞠个躬吧。" in text and "2. 你给大家唱首歌吧。" in text
    assert "说明唱不了歌" not in text


def test_cells_sharing_a_gesture_count_as_one():
    wave = _pos("wave_greet_bye", "拜拜。", "挥手告别")
    scene = _pos("wave_scene", "跟小朋友们挥挥手。", "挥手")
    pool = _pool(wave, scene)
    assert assemble.sample_case(pool, 2, 0.0, random.Random(0)) is None


def test_instruction_end_punctuation_is_stripped():
    atom = _pool(_pos("bow", "高兴地鞠个躬。", "鞠躬。")).positives["RobotGesture"]["bow"][0]
    row = assemble.render_row([atom], atom["query"], "zh", BUNDLE)
    assert json.loads(row["messages"][2]["content"])[0]["instruction"] == "鞠躬"


def _unit(op, query, *atoms):
    return {"id": f"rel_{op}", "op": op, "query": query, "unit_atoms": list(atoms),
            "cells": sorted({assemble.source_cell(a) for a in atoms})}


def test_relation_unit_fills_its_task_count_and_blocks_its_cells():
    pool = _pool(BOW, NOD, GEE)
    bow = pool.positives["RobotGesture"]["bow"][0]
    pool.units = [_unit("count", "鞠三个躬", bow, bow, bow)]
    rng = random.Random(0)
    seen_unit = False
    for _ in range(100):
        case = assemble.sample_case(pool, 4, 0.0, rng, rel_rate=0.5)
        if case is None:  # three atom cells alone cannot fill four tasks
            continue
        assert sum(len(assemble.tasks_of(p)) for p in case) == 4
        if any("unit_atoms" in p for p in case):
            seen_unit = True
            assert all(assemble.source_cell(p) != "gesture:bow" for p in case if "unit_atoms" not in p)
    assert seen_unit


def test_single_relation_unit_needs_no_merge_and_labels_every_task():
    bow = _pool(BOW).positives["RobotGesture"]["bow"][0]
    unit = _unit("count", "鞠三个躬", bow, bow, bow)
    rows = []
    row = assemble.process_case([unit], _ctx(lambda *a, **k: 1 / 0, lambda *a, **k: 1 / 0, rows), 0)
    assert len(json.loads(row["messages"][2]["content"])) == 3
    assert row["metadata"]["ops"] == ["count"] and row["metadata"]["source_queries"] == ["鞠三个躬"]


def test_address_term_already_in_a_source_is_allowed():
    rag = dict(BOW, query="Luna机器人能干啥？")
    assert assemble.check("Luna机器人能干啥？再鞠个躬。", [rag, BOW]) is None
    assert assemble.check("机器人，Luna机器人能干啥？", [rag]) == "address term"
