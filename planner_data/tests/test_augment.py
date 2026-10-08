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

import json
import random

from planner_data import augment
from planner_data.contract import planning_production as pp
from planner_data.contract.planning_contract import render_planner_messages


BUNDLE = pp.load_contract_bundle(pp.CONTRACT_V2_PATH)
WALK = {"kind": "action", "instruction": "往前走三步", "name": "HumanAction",
        "arguments": {"action": "walk", "parameters": {"x": 0.6, "y": 0, "yaw": 0, "step": 3}}}
WEATHER = {"kind": "query", "instruction": "查北京天气", "name": "get_weather",
           "arguments": {"city": "北京", "query_type": "now"}}
ASK = {"kind": "reply", "instruction": "回答你会不会跳舞"}


def _row(tasks, instruction="往前走三步，查北京天气", language="zh"):
    messages = render_planner_messages(system_prompt=BUNDLE.system_prompt.content, tools=BUNDLE.tools,
                                       instruction=instruction)
    messages.append({"role": "assistant", "content": json.dumps(tasks, ensure_ascii=False)})
    return {"messages": messages, "metadata": {"example_id": "e1", "language": language, "tool_naming": "real"}}


def _args(**kw):
    return augment.parse_args(["in.jsonl", "-o", "out.jsonl", *sum(([f"--{k}", str(v)] for k, v in kw.items()), [])])


def test_fixture_is_valid():
    _, tools, _, tasks = augment.parse_row(_row([WALK, WEATHER]))
    augment.validate(tools, tasks)


def test_masked_renames_tools_and_arguments():
    row, reason = augment.augment_row(_row([WALK, WEATHER, ASK]), "masked", _args())
    assert reason == "ok"
    _, tools, _, tasks = augment.parse_row(row)
    text = json.dumps(tools, ensure_ascii=False)
    for word in ("HumanAction", "get_weather", '"city"', '"step"', '"yaw"', '"parameters"', "query 必须"):
        assert word not in text
    walk, weather, ask = tasks
    mapping = row["metadata"]["augment"]["mapping"]
    assert walk["name"] == mapping["tools"]["HumanAction"]
    params = mapping["arguments"]["HumanAction"]
    assert walk["arguments"] == {params["action"]: "walk",
                                 params["parameters"]: {params["x"]: 0.6, params["y"]: 0, params["yaw"]: 0,
                                                        params["step"]: 3}}
    assert weather["arguments"][mapping["arguments"]["get_weather"]["city"]] == "北京"
    assert ask == ASK
    rag = next(t for t in tools if t["name"] == mapping["tools"]["rag_query"])
    assert rag["description"].count(mapping["arguments"]["rag_query"]["query"]) == 1  # mention follows the rename
    assert row["metadata"]["tool_naming"] == "masked"


def test_drop_used_turns_only_that_intent_into_a_reply():
    rng = random.Random(0)
    _, tools, _, tasks = augment.parse_row(_row([WALK, WEATHER]))
    new_tools, new_tasks, dropped = augment.drop_used(tools, tasks, "zh", rng)
    assert len(new_tools) == len(tools) - 1 and dropped[0] not in {t["name"] for t in new_tools}
    kept = WEATHER if dropped == ["HumanAction"] else WALK
    gone = WALK if kept is WEATHER else WEATHER
    assert kept in new_tasks
    assert {"kind": "reply", "instruction": f"说明当前无法处理：{gone['instruction']}"} in new_tasks
    assert [t.get("name") for t in new_tasks].index(kept["name"]) == [WALK, WEATHER].index(kept)  # order kept


def test_drop_used_skips_reply_only_rows():
    _, tools, _, tasks = augment.parse_row(_row([ASK], instruction="你会跳舞吗"))
    assert augment.drop_used(tools, tasks, "zh", random.Random(0)) is None


def test_drop_unused_keeps_answer_and_called_tools():
    for seed in range(20):
        _, tools, _, tasks = augment.parse_row(_row([WALK]))
        new_tools, new_tasks, dropped = augment.drop_unused(tools, tasks, random.Random(seed))
        names = {t["name"] for t in new_tools}
        assert new_tasks == tasks and "HumanAction" in names and dropped and not names & set(dropped)


def test_mask_skipped_when_text_names_a_real_tool():
    row, reason = augment.augment_row(_row([WALK], instruction="用 HumanAction 往前走三步"), "masked", _args())
    assert row is None and reason == "names a real tool"


def test_run_only_augments_real_rows_and_marks_negatives():
    alias = _row([WALK])
    alias["metadata"] = {**alias["metadata"], "example_id": "e2", "tool_naming": "alias"}
    out, stats = augment.run([_row([WALK, WEATHER]), alias], _args())
    assert stats["skip non-real input"] == 1
    variants = {r["metadata"]["augment"]["variant"]: r for r in out}
    assert set(variants) == set(augment.VARIANTS)
    assert variants["drop_used"]["metadata"]["polarity"] == "negative"
    for r in out:
        _, tools, _, tasks = augment.parse_row(r)
        augment.validate(tools, tasks)
