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

"""Offline checks for pilot.py (no LLM calls)."""

from __future__ import annotations

import functools
import json
import random

from planner_data.atoms import pilot


def _ctx(write, judge=None):
    return pilot.Context(
        write=write,
        judge=judge,
        bundle=pilot.pp.load_contract_bundle(pilot.pp.CONTRACT_V2_PATH),
        seen=set(),
        existing={"HumanAction": ["请坐下"]},
        existing_grams={"HumanAction": [pilot.grams("请坐下")]},
    )


def _cell(cell_id):
    return next(c for c in pilot.pa.CELLS if c.cell_id == cell_id)


def test_shape_masks_numbers_and_punctuation():
    assert pilot.shape("往前走 3 步！") == pilot.shape("往前走五步")
    assert pilot.jaccard(pilot.grams("往前走三步"), pilot.grams("往前走十步")) == 1.0


def test_prompt_lists_specs_and_slot_values():
    spec = pilot.Spec(scene="商场导览", persona="外地来的游客", style="礼貌请求", noise=None)
    text = pilot.build_prompt(_cell("fwd_steps"), [spec], [7], [])
    assert "场景：商场导览" in text and "说话人：外地来的游客" in text and "步数：7" in text
    assert "说法类型尽量分散" not in text
    assert "说法类型尽量分散" in pilot.build_prompt(_cell("sit_down"), [pilot.Spec()], [None], [])


def test_sample_spec_respects_enabled_dims():
    rng = random.Random(0)
    assert pilot.sample_spec(rng, frozenset()) == pilot.Spec()
    spec = pilot.sample_spec(rng, frozenset({"scene"}), "sit_down")
    assert spec.scene and not (spec.persona or spec.style or spec.noise or spec.kernel)
    spec = pilot.sample_spec(rng, frozenset({"kernel"}), "fwd_steps")
    assert spec.kernel in pilot.dims.KERNELS["fwd_steps"]
    text = pilot.build_prompt(_cell("fwd_steps"), [spec], [3], [])
    assert f"动作说法：{spec.kernel}" in text and "不要换成别的动词" in text


def test_kernels_pass_the_rule_validator():
    for cell_id, kernels in pilot.dims.KERNELS.items():
        cell = _cell(cell_id)
        value = {"step": 3, "degree": 45}.get(cell.slot or "")
        params = pilot.pa._cell_params(cell, value, "")
        for kernel in kernels:
            query = kernel.replace("{n}", "")
            if cell.slot == "step":
                query += "三步"
            elif cell.slot == "degree":
                query += "45度"
            assert pilot.pa.derive_human_action(query) == params, (cell_id, query)


def test_turn_a_bit_convention():
    assert pilot.pa.derive_human_action("向左转一下")["parameters"]["degree"] == 90
    assert pilot.pa.derive_human_action("向左稍微转一点")["parameters"]["degree"] == 15


def test_judged_params_mapping():
    jp = functools.partial(pilot.judged_params, "HumanAction")
    assert jp({"request": True, "action": "forward", "steps": "3"}) == pilot.pa._cell_params(
        _cell("fwd_steps"), 3, "")
    assert jp({"request": True, "action": "turn_right", "degree": 15}) == pilot.pa._cell_params(
        _cell("turn_right_bit"), None, "")
    assert jp({"request": True, "action": "sit_down"}) == {"action": "sit_down"}
    assert jp({"request": False, "action": "sit_down"}) is None
    about_face = pilot.pa._cell_params(_cell("turn_around"), None, "")
    reason = functools.partial(pilot.judge_reason, "HumanAction")
    assert reason({"request": True, "action": "turn_right", "degree": 180}, about_face) is None
    assert reason({"request": True, "action": "turn_right", "degree": 90}, about_face) == (
        "judge mismatch " + str(jp({"request": True, "action": "turn_right", "degree": 90})))


def test_judged_params_gesture_and_dance():
    gesture = functools.partial(pilot.judged_params, "RobotGesture")
    assert gesture({"request": True, "gesture": "bow"}) == pilot.pa._cell_params(_cell("bow"), None, "")
    assert gesture({"request": True, "gesture": "salute"}) is None
    dance = functools.partial(pilot.judged_params, "RobotDance")
    assert dance({"request": True, "dance": "popping"}) == pilot.pa._cell_params(_cell("popping"), None, "")
    assert dance({"request": True, "dance": "unnamed"}) == pilot.pa._cell_params(_cell("dance_unnamed"), None, "")
    assert dance({"request": True, "dance": "other"}) is None
    assert dance({"request": True, "dance": "APT"}) == {"dance": "apt_dance"}
    assert "「" not in pilot.cell_hint(_cell("apt_dance")) and "APT" in pilot.cell_hint(_cell("apt_dance"))
    for tool in pilot.JUDGES:
        text = pilot.judge_prompt(tool, ["句子一"])
        assert "1. 句子一" in text and "能力提问" in text
    assert "“拜拜”" in pilot.judge_prompt("RobotGesture", ["拜拜"]).split("以下算 false")[0]
    assert "等客人来了再做" in pilot.judge_prompt("RobotGesture", ["x"]).split("以下算 false")[1]
    assert "popping（机械舞）" in pilot.judge_prompt("RobotDance", ["x"])
    assert "bow（鞠躬）" in pilot.judge_prompt("RobotGesture", ["x"])


def test_gesture_cell_runs_with_tool_specific_judge():
    replies = [
        {"query": "给大家鞠个躬吧", "instruction_zh": "鞠躬", "instruction_en": "Bow"},
        {"query": "来，鼓个掌", "instruction_zh": "鼓掌", "instruction_en": "Clap"},  # rule mismatch
    ]
    prompts = []

    def judge(prompt, **_k):
        prompts.append(prompt)
        return [{"i": 1, "request": True, "gesture": "bow"}]

    args = pilot.parse_args(["--name", "t", "--batch", "2", "--max-rounds", "1", "--avoid", "0"])
    res = pilot.run_cell(_cell("bow"), args, _ctx(lambda *a, **k: replies, judge))
    assert [a["query"] for a in res.accepted] == ["给大家鞠个躬吧"]
    assert res.accepted[0]["expected_params"]["gesture"] == "bow"
    assert "gesture：" in prompts[0] and res.accepted[0]["pilot"]["max_sim_existing"] == 0.0


def test_dance_name_issue():
    ok = ["来段顺风顺水顺财神", "来段顺风顺水", "跳个机械舞", "给大家跳支舞吧", "来段舞蹈", "跳个好看的舞",
          "听说你能跳很多舞，来段展臂舞", "这舞挺好看，跳个相亲相爱", "卡啦永远OK来一段", "跳个女团舞1"]
    assert [q for q in ok if pilot.dance_name_issue(q)] == []
    assert pilot.dance_name_issue("来段顺风顺水顺才神") == "misspelt dance name"
    assert pilot.dance_name_issue("卡拉永远OK来一段") == "misspelt dance name"
    for q in ("跳个热肠舞", "来段广场舞", "跳个女团舞"):
        assert pilot.dance_name_issue(q) == "unknown dance name", q


def test_dance_cell_rejects_off_list_names(tmp_path):
    replies = [{"query": "来段热肠舞", "instruction_zh": "跳热场舞", "instruction_en": "Dance"},
               {"query": "来段热场舞", "instruction_zh": "跳热场舞", "instruction_en": "Dance"}]
    judge = lambda *a, **k: [{"i": 1, "request": True, "dance": "warm_up_dance"}]  # noqa: E731
    args = pilot.parse_args(["--name", "t", "--batch", "2", "--max-rounds", "1", "--avoid", "0",
                             "--tools", "RobotDance", "--cells", "warm_up_dance", "--out", str(tmp_path)])
    out = pilot.run(args, _ctx(lambda *a, **k: replies, judge))
    assert [json.loads(x)["query"] for x in (out / "atoms.jsonl").read_text().splitlines()] == ["来段热场舞"]
    (rej,) = [json.loads(x) for x in (out / "rejected.jsonl").read_text().splitlines()]
    assert rej["cell"] == "warm_up_dance" and rej["reason"] == "unknown dance name"


def test_run_selects_cells_by_tool(tmp_path):
    args = pilot.parse_args(["--name", "t", "--tools", "RobotDance", "--cells", "sit_down", "--out", str(tmp_path)])
    try:
        pilot.run(args, _ctx(lambda *a, **k: []))
    except SystemExit as exc:
        assert "sit_down" in str(exc)
    else:
        raise AssertionError("sit_down is not a RobotDance cell")


def test_judge_filters_after_rules():
    replies = [
        {"query": "请坐下", "instruction_zh": "坐下", "instruction_en": "Sit down"},
        {"query": "要是你坐下就好了", "instruction_zh": "坐下", "instruction_en": "Sit down"},
        {"query": "调个坐姿看看", "instruction_zh": "坐下", "instruction_en": "Sit down"},
    ]
    verdicts = [
        {"i": 1, "request": True, "action": "sit_down"},
        {"i": 2, "request": False, "action": "sit_down"},
        {"i": 3, "request": True, "action": "none"},
    ]
    prompts = []

    def judge(prompt, **_k):
        prompts.append(prompt)
        return {"items": verdicts}

    args = pilot.parse_args(["--name", "t", "--batch", "3", "--max-rounds", "1", "--avoid", "0"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(lambda *a, **k: replies, judge))
    assert [a["query"] for a in res.accepted] == ["请坐下"]
    assert res.accepted[0]["curation"]["validator"] == "rules+judge"
    assert [r["reason"] for r in res.rejected] == ["judge not request", "judge mismatch"]
    assert res.judge_calls == 1 and "3. 调个坐姿看看" in prompts[0]


def test_ability_questions_are_rejected_before_judge():
    queries = ["你能不能坐下", "可以请你先坐好吗？", "你能坐一会儿吗", "你会不会坐下呀", "要不你坐会儿吧"]
    replies = [{"query": q, "instruction_zh": "坐下", "instruction_en": "Sit down"} for q in queries]
    args = pilot.parse_args(["--name", "t", "--batch", "5", "--max-rounds", "1", "--avoid", "0", "--no-judge"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(lambda *a, **k: replies))
    assert [a["query"] for a in res.accepted] == ["要不你坐会儿吧"]
    assert [r["reason"] for r in res.rejected] == ["ability question"] * 4


def test_address_terms_are_rejected():
    queries = ["嘿，小护士，你坐下吧", "小助，坐一会儿吧", "机器人同学，请坐下", "机器人，你先坐下吧", "你先坐下吧",
               "来个「坐下」吧"]
    replies = [{"query": q, "instruction_zh": "坐下", "instruction_en": "Sit down"} for q in queries]
    args = pilot.parse_args(["--name", "t", "--batch", "6", "--max-rounds", "1", "--avoid", "0", "--no-judge"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(lambda *a, **k: replies))
    assert [a["query"] for a in res.accepted] == ["你先坐下吧"]
    assert [r["reason"] for r in res.rejected] == ["address term"] * 4 + ["quotes"]


def test_judge_error_drops_the_batch():
    def judge(*_a, **_k):
        raise ValueError("bad json")

    replies = [{"query": "请坐下", "instruction_zh": "坐下", "instruction_en": "Sit down"}]
    args = pilot.parse_args(["--name", "t", "--batch", "1", "--max-rounds", "1", "--avoid", "0"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(lambda *a, **k: replies, judge))
    assert not res.accepted and [r["reason"] for r in res.rejected] == ["judge_error"]


def test_run_cell_validates_dedups_and_saturates():
    replies = [
        {"query": "请坐下", "instruction_zh": "坐下", "instruction_en": "Sit down"},
        {"query": "你先坐一会儿吧", "instruction_zh": "坐下", "instruction_en": "Sit down"},
        {"query": "往前走", "instruction_zh": "往前走", "instruction_en": "Walk forward"},  # wrong label
    ]
    args = pilot.parse_args(["--name", "t", "--batch", "3", "--max-rounds", "6", "--avoid", "0"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(lambda *a, **k: replies))
    assert [a["query"] for a in res.accepted] == ["请坐下", "你先坐一会儿吧"]
    reasons = {r["reason"] for r in res.rejected}
    assert {"query derives", "duplicate"} <= reasons
    assert res.stop == "saturated" and len(res.curve) == 1 + args.patience
    assert res.accepted[0]["pilot"]["max_sim_existing"] == 1.0
    summary = pilot.summarize([res], args.tau)
    assert summary["cells"]["sit_down"]["valid"] == 2


def test_wrapped_reply_and_prompt_leak():
    replies = {"sentences": [
        {"query": "请坐下", "instruction_zh": "坐下", "instruction_en": "Sit down"},
        {"query": "你坐下吧一口气说完没有标点", "instruction_zh": "坐下", "instruction_en": "Sit down"},
    ]}
    args = pilot.parse_args(["--name", "t", "--batch", "2", "--max-rounds", "1", "--avoid", "0"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(lambda *a, **k: replies))
    assert [a["query"] for a in res.accepted] == ["请坐下"]
    assert [r["reason"] for r in res.rejected] == ["prompt leak"]


def test_failed_call_is_counted_not_raised():
    def boom(*_a, **_k):
        raise ValueError("bad json")

    args = pilot.parse_args(["--name", "t", "--max-rounds", "2", "--avoid", "0"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(boom))
    assert res.failed_calls == 2 and not res.accepted


def test_single_object_reply_is_a_failed_call_not_saturation():
    single = {"query": "你坐下吧", "instruction_zh": "坐下", "instruction_en": "Sit down"}
    args = pilot.parse_args(["--name", "t", "--max-rounds", "4", "--avoid", "0", "--no-judge"])
    res = pilot.run_cell(_cell("sit_down"), args, _ctx(lambda *a, **k: single))
    assert res.failed_calls == 4 and res.stop != "saturated"


def _qcell(cell_id):
    return next(c for c in pilot.ALL_CELLS if c.cell_id == cell_id)


def test_search_reason_span_topic_and_product():
    price = _qcell("rag_price")
    ok = ("那个Oli机器人多少钱啊", "Oli机器人多少钱", "Oli多少钱", "Oli机器人多少钱")
    assert pilot.validate(price, None, ok[0], ok[1], "How much", None, ok[3]) is None
    assert pilot.validate(price, None, ok[0], ok[1], "x", None, "Oli的价格") == "search_query is not a span of the query"
    assert pilot.validate(price, None, "那个多少钱啊", "那个多少钱", "x", None, "那个多少钱") == "search_query misses the product"
    assert pilot.validate(price, None, ok[0], "问价格", "x", None, ok[3]) == "instruction_zh drops the search topic"
    # zh may fix typos and drop 的
    aftersale = _qcell("rag_aftersale")
    q = "麻烦看看逐际动里的产品的售后网点"
    assert pilot.validate(aftersale, None, q, "逐际动力产品的售后网点", "x", None, "逐际动里的产品的售后网点") is None
    scenic = _qcell("ws_scenic")
    q = "香港迪士尼今天能不能去啊"
    assert pilot.validate(scenic, None, q, "香港迪士尼今天能不能去", "x", "香港", "香港迪士尼今天能不能去") is None
    assert pilot.validate(scenic, None, q, "香港迪士尼今天能不能去", "x", "澳门",
                          "香港迪士尼今天能不能去") == "search_query misses city"
    ent = _qcell("ws_entertainment")
    q = "你能不能告诉我现在什么歌最火"
    assert pilot.validate(ent, None, q, "现在什么歌最火", "x", None, "现在什么歌最火") is None
    assert pilot.validate(ent, None, q, "现在什么歌最火", "x", None, "告诉我现在什么歌最火") == (
        "search_query keeps the request words")
    tech = _qcell("rag_tech")
    q = "你们的Luna机器人咋保持平衡的"
    assert pilot.validate(tech, None, q, "Luna机器人怎么保持平衡", "x", None, "你们的Luna机器人咋保持平衡的") is None


def test_query_judge_reason():
    reason = pilot.judge_reason
    weather = {"city": "深圳", "query_type": "now"}
    assert reason("get_weather", {"ask": True, "source": "weather", "city": None, "period": "now"}, weather) is None
    assert reason("get_weather", {"ask": True, "source": "weather", "city": None, "period": "24h"}, weather) is None
    assert reason("get_weather", {"ask": True, "source": "weather", "city": None, "period": "7d"}, weather)
    assert reason("get_weather", {"ask": True, "source": "weather", "city": None, "period": "now"},
                  {"city": "深圳", "query_type": "24h"})
    assert reason("get_weather", {"ask": False, "source": "weather"}, weather) == "judge not request"
    assert reason("robot_status", {"ask": True, "source": "status", "item": "型号"}, {"query": "型号"}) is None
    assert reason("robot_status", {"ask": True, "source": "private", "item": "型号"}, {"query": "型号"}) is None
    assert reason("robot_status", {"ask": True, "source": "private", "item": None}, {"query": "型号"})
    assert reason("rag_query", {"ask": True, "source": "private"}, {"query": "Oli多少钱"}) is None
    assert reason("rag_query", {"ask": True, "source": "public"}, {"query": "Oli多少钱"})
    text = pilot.judge_prompt("rag_query", ["Oli多少钱"])
    assert "1. Oli多少钱" in text and "private" in text and "型号" in text


def test_query_cell_allows_polite_questions_and_product_names():
    hint = pilot._rotate_hint("问法分散到：甲、乙、丙；不出现明天", random.Random(0))
    assert hint.endswith("用口语问出来；不出现明天") and hint.count("或") == 1 and "问法分散到" not in hint
    prompt = pilot.build_prompt(_qcell("rag_price"), [pilot.Spec()], [None], [])
    assert "search_query" in prompt and "帮我查下" in prompt
    replies = [{"query": q, "instruction_zh": "Luna机器人多少钱", "instruction_en": "How much is Luna",
                "search_query": "Luna机器人多少钱"} for q in ("你能不能告诉我Luna机器人多少钱", "小助，Luna机器人多少钱")]
    args = pilot.parse_args(["--name", "t", "--batch", "2", "--max-rounds", "1", "--avoid", "0", "--no-judge",
                             "--tools", "rag_query"])
    res = pilot.run_cell(_qcell("rag_price"), args, _ctx(lambda *a, **k: replies))
    assert [a["query"] for a in res.accepted] == ["你能不能告诉我Luna机器人多少钱"]
    assert res.accepted[0]["expected_params"] == {"query": "Luna机器人多少钱"}
    assert [r["reason"] for r in res.rejected] == ["address term"]
