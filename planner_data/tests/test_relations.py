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

"""Offline checks for relations.py (no LLM calls)."""

from __future__ import annotations

import json
import random

from planner_data.atoms import pilot, relations


BUNDLE = pilot.pp.load_contract_bundle(pilot.pp.CONTRACT_V2_PATH)


def _pos(cell_id, query):
    cell = next(c for c in pilot.pa.CELLS if c.cell_id == cell_id)
    atom = pilot.pa._atom(cell, query, cell.intent, "x", pilot.pa._cell_params(cell, None, ""))
    return pilot.pp.normalize_atom_for_planning(atom, BUNDLE)


BOW, WAVE, POP = _pos("bow", "鞠个躬吧"), _pos("wave_greet_bye", "挥挥手"), _pos("popping", "跳段机械舞")
FWD, BLOW = _pos("fwd_plain", "往前走"), _pos("blow_kisses_multi", "来个连续飞吻")


def test_expected_labels():
    assert relations.Unit("only", BOW, WAVE).expected == ["A"]
    assert relations.Unit("before", BOW, WAVE).expected == ["A", "B"]
    assert relations.Unit("count", BOW, n=3).expected == ["A", "A", "A"]
    assert relations.Unit("count", BOW, n=3).label_atoms == [BOW] * 3
    assert relations.Unit("while", FWD, WAVE).label_atoms == [FWD, WAVE]


def test_sample_respects_operator_pools():
    rng = random.Random(0)
    positives = [BOW, WAVE, POP, FWD, BLOW]
    for _ in range(30):
        count = relations.sample(positives, "count", rng)
        assert count.a is not BLOW and count.a is not FWD and 2 <= count.n <= 5
        assert relations.sample(positives, "speed", rng).a is FWD
        w = relations.sample(positives, "while", rng)
        assert w.a is FWD and w.b["expected_function"] == "RobotGesture"
        pair = relations.sample(positives, "only", rng)
        assert relations._value(pair.a) != relations._value(pair.b)


def test_check():
    c = relations.check
    assert c(relations.Unit("only", BOW, WAVE), "别挥手了，鞠个躬就行") is None
    assert c(relations.Unit("only", BOW, WAVE), "挥挥手，鞠个躬") == "no only marker"
    assert c(relations.Unit("count", BOW, n=3), "鞠三个躬吧") is None
    assert c(relations.Unit("count", BOW, n=3), "鞠两个躬吧") == "wrong count"
    assert c(relations.Unit("count", BOW, n=2), "鞠二个躬吧") is None
    assert c(relations.Unit("speed", FWD, hint="快点"), "快点往前走") is None
    assert c(relations.Unit("speed", FWD, hint="快点"), "往前走，慢一点") is None
    assert c(relations.Unit("before", BOW, WAVE), "挥手之前先鞠个躬") is None
    assert c(relations.Unit("while", FWD, WAVE), "往右边让一让，挥挥手") == "no while marker"
    assert c(relations.Unit("while", FWD, WAVE), "边往前走边挥挥手") is None
    assert c(relations.Unit("before", BOW, WAVE), "A：鞠躬之前先挥手") == "prompt leak"
    assert c(relations.Unit("either", BOW, WAVE), "你能鞠躬或者挥手吗") == "ability question"
    assert c(relations.Unit("only", POP, WAVE), "别挥手，跳段机械午就行") == "lost name popping"
    assert c(relations.Unit("only", BOW, WAVE), "别挥手了，弯个腰就行") == "lost name bow"


def test_judge_reason():
    jr = relations.judge_reason
    unit = relations.Unit("before", BOW, WAVE)
    assert jr(unit, {"do": ["A", "B"], "other": False}) is None
    assert jr(unit, {"do": ["B", "A"], "other": False}) == "judge BA"
    assert jr(unit, {"do": ["A", "B"], "other": True}) == "judge extra"
    assert jr(unit, {"do": [], "other": False}) == "judge -"
    assert jr(unit, None) == "judge missing"
    assert jr(relations.Unit("count", BOW, n=2), {"do": ["a", "A"]}) is None


def test_process_renders_n_tasks():
    rows = []
    ctx = relations.Context(write=lambda *a, **k: {"query": "给大家鞠三个躬"},
                            judge=lambda *a, **k: {"do": ["A", "A", "A"], "other": False}, bundle=BUNDLE,
                            sink=lambda kind, row: rows.append((kind, row)))
    row = relations.process(relations.Unit("count", BOW, n=3), ctx, 1)
    tasks = json.loads(row["messages"][-1]["content"])
    assert [t["arguments"] for t in tasks] == [{"gesture": "bow"}] * 3
    assert row["metadata"]["op"] == "count" and row["metadata"]["level"] == 3
    assert rows[0] == ("rows", row)
    kind, unit = rows[1]
    assert kind == "units" and unit["op"] == "count" and unit["query"] == "给大家鞠三个躬"
    assert unit["unit_atoms"] == [BOW] * 3 and unit["cells"] == [relations.assemble.source_cell(BOW)]


def test_process_rejects_judge_disagreement():
    rows = []
    ctx = relations.Context(write=lambda *a, **k: {"query": "挥手之前先鞠个躬"},
                            judge=lambda *a, **k: {"do": ["B", "A"], "other": False}, bundle=BUNDLE,
                            sink=lambda kind, row: rows.append((kind, row)))
    assert relations.process(relations.Unit("before", BOW, WAVE), ctx, 1) is None
    assert rows[0][0] == "rejected" and rows[0][1]["reason"] == "judge BA"
