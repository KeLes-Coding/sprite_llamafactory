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

"""Sealed challenge v2: new sentences written from label specs by another model family, double-judged.

Labels come from the spec, never from the writer: each case is a list of pool atoms (their labels only) or a
relation unit; deepseek writes one utterance for it in scenes/personas the training writer never saw, and two
other judges (qwen-max, glm-4.6) relabel the utterance blind. A case is kept only when both relabels match the
spec position by position; the rest go to disputed.jsonl for a human. Sentences close to any pool query drop.

    python -m planner_data.challenge_v2 --pool runs/pool-XXXX --out-json planner_val/challenges/challenge_v2.json
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from planner_data import assemble
from planner_data.atoms import audit, pilot
from planner_data.atoms import translate_atoms as ta
from planner_data.atoms.pilot import find_env_value, llm, pp


SCENES = ("机场候机厅", "图书馆少儿区", "汽车4S店展厅", "社区养老活动室", "游乐园排队区", "婚礼现场",
          "公司年会后台", "高校开放日", "健身房前台", "小区物业大厅", "书店签售会", "火车站问询处")
PERSONAS = ("退休的中学老师", "来出差的外国工程师", "刚放学的初中生", "带双胞胎的爸爸", "跑新闻的记者",
            "公司HR", "做直播的博主", "餐馆老板", "健身教练", "第一次来中国的留学生", "坐轮椅的老人", "新来的保安")
FLIP_HINT = {
    "ability": "只问你会不会、能不能做（能力提问），不是让你现在做",
    "stop": "叫你别做、不用做",
    "comment": "评价你刚才做的，不提新要求",
    "others": "对在场的别人说、让那个人去做或问那个人（句中点明对方），不是对你提要求",
    "wish": "说愿望或假设（“要是……就好了”之类），不直接提要求",
    "deferred": "让你待会儿或以后再做，不是现在",
    "unsupported": "让你现在做一个你做不了的动作或舞",
    "sing": "让你现在唱歌",
    "joke": "让你现在讲笑话",
    "mention": "问这个动作或舞本身（含义、来历），不是让你做",
    "teach": "让你口头讲讲怎么做（讲解，不是示范）",
    "say": "让你说一句话",
    "statement": "随口说一句与话题有关的陈述或感叹，不提问也不请你查",
    "common": "问一个不用查就能回答的常识问题",
}
REL_HOW = {
    "only": "说别{B}了，只要你{A}", "correct": "先说让你{B}，马上改口换成{A}（最后只要{A}）",
    "retract": "先让你{A}和{B}，接着把{B}收回", "either": "说{A}或者{B}都行（{A}先提，两个都要说到）",
    "before": "让你先{A}再{B}，但句子里先提到{B}（“{B}之前先{A}”这类说法）", "while": "让你边{A}边{B}",
    "count": "让你连着{A}{n}次", "speed": "让你{hint}地{A}",
}
_CN_NUM = "零一两三四五"
_SEARCH = ("web_search", "rag_query")
_CITY_EN = {"深圳": "Shenzhen", "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "杭州": "Hangzhou",
            "成都": "Chengdu", "悉尼": "Sydney", "东京": "Tokyo", "伦敦": "London", "纽约": "New York",
            "巴黎": "Paris", "首尔": "Seoul", "新加坡": "Singapore", "西安": "Xi'an", "南京": "Nanjing"}
SYSTEM = "你为一台人形服务机器人的规划模型写测试题：每题是一位用户当面对机器人一口气说的一段话。"


def _zh(atom: dict[str, Any]) -> str:
    return str(atom["planner_target"]["instructions"]["zh"]).rstrip("。")


def describe(atom: dict[str, Any]) -> str:
    if atom.get("negative_sample"):
        return f"{FLIP_HINT[atom['contrast']['flip']]}（你的回应会是“{_zh(atom)}”，只用来定这一项说的对象，不写进句子）"
    if atom["planner_target"]["kind"] == "query":
        return f"让你查：{_zh(atom)}" + ("［检索］" if atom["expected_function"] in _SEARCH else "")
    return f"让你现在{_zh(atom)}"


def describe_unit(op: str, a: dict[str, Any], b: dict[str, Any] | None, n: int, hint: str) -> str:
    return REL_HOW[op].format(A=_zh(a), B=_zh(b) if b else "", n=_CN_NUM[n], hint=hint)


def write_prompt(items: list[str], language: str, scene: str, persona: str) -> str:
    lines = [f"写一段用户当面对你一口气说的自然口语，按顺序包含下面每一项，一项不多一项不少，不调换顺序："]
    lines += [f"{i}. {item}" for i, item in enumerate(items, 1)]
    lines += [f"说话场景：{scene}；说话人：{persona}。可以顺带一句跟场景或身份有关的背景、理由或感叹，"
              "但它不能是请求、问题或嘱咐；除此之外不加任何别的事，每项二三十字以内。"
              + ("这段话只说这一件事。" if len(items) == 1 else ""),
              "上面每项是内容说明，“让你”“查询”“当前”这类说明字眼不要照抄，换成这位说话人自己的说法。",
              "不称呼你、不写“机器人”，不加引号；动作名、舞名、手势名、数字、方向、城市照上面写。"]
    out = '{"query": "...", "spans": {"项号": "..."}}'
    if language == "en":
        lines.append(f"用英文写，像英语母语者说话；舞名和手势名用这张对照表里的英文：{ta._GLOSSARY}。")
        lines.append("另在 instructions 里给每一项一条简短英文祈使短语（回应类写成 Reply/Explain/Say……）。")
        out = '{"query": "...", "spans": {"项号": "..."}, "instructions": {"1": "...", ...}}'
    lines.append("标了［检索］的项，在 spans 里给出句中对应检索主题的一段连续原文（一字不差）。")
    lines.append(f"只输出 JSON：{out}")
    return "\n".join(lines)


def review_prompt(queries: list[str]) -> str:
    lines = [audit.REVIEW_RULES, "- 同一个动作要求做几次就列几次；只要其中一个、改口、收回的，只列最后真正要做的。",
             "", "下面每段话是用户一口气对机器人说的，可能包含几件事。逐段按先后顺序列出机器人该做的全部任务（不要参考任何已有答案）："]
    lines += [f"{i}. {q}" for i, q in enumerate(queries, 1)]
    lines.append('只输出 JSON 数组：[{"i": 1, "tasks": [{"tool": "...", "params": {...}}]}]'
                 "（tool 与 params 的写法同上：HumanAction|RobotGesture|RobotDance|get_weather|robot_status|"
                 "web_search|rag_query|reply；walk 的 params 写 {\"action\": \"walk\", \"x\": 1, \"y\": 0, \"yaw\": 0, "
                 "\"step\": 2, \"degree\": null}（前 x=1、后 x=-1、左 y=1、右 y=-1、左转 yaw=1、右转 yaw=-1，"
                 "转身时 degree 填度数；坐躺站写 {\"action\": \"sit_down\"}；手势写 {\"gesture\": \"bow\"}；舞写 {\"dance\": \"popping\"}；"
                 "天气写 {\"city\": \"深圳\", \"query_type\": \"now\"}；状态写 {\"query\": \"电量\"}；检索与 reply 的 params 填 null）")
    return "\n".join(lines)


def _merge_replies(xs: list[Any], is_reply: Any) -> list[Any]:
    """Judges often fold neighbouring spoken items into one reply; the order of real tool calls still has to match."""
    out: list[Any] = []
    for x in xs:
        if not (out and is_reply(x) and is_reply(out[-1])):
            out.append(x)
    return out


def matches(labels: list[dict[str, Any]], tasks: Any) -> bool:
    if not isinstance(tasks, list):
        return False
    labels = _merge_replies(labels, lambda a: a["planner_target"]["kind"] == "reply")
    tasks = _merge_replies(tasks, lambda t: isinstance(t, dict) and t.get("tool") == "reply")
    if len(tasks) != len(labels):
        return False
    def city(a: dict[str, Any], t: dict[str, Any]) -> dict[str, Any]:  # judges may name an English city in Chinese
        params = t.get("params")
        if t.get("tool") == "get_weather" and isinstance(params, dict) and params.get("city") in _CITY_EN \
                and a.get("expected_function") == "get_weather" and a["expected_params"]["city"] not in _CITY_EN:
            return {**t, "params": {**params, "city": _CITY_EN[params["city"]]}}
        return t

    return all(isinstance(t, dict) and ta.agree(a, city(a, t)) for a, t in zip(labels, tasks))


def _set_params(atom: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    atom = json.loads(json.dumps(atom))
    atom["expected_params"] = params
    atom["expected_steps"][0]["expected_params"] = params
    return atom


def finalize(spec: dict[str, Any], reply: Any) -> tuple[list[dict[str, Any]] | None, str | None, str]:
    """Label atoms with this sentence's spans/cities/instructions, or why the writing is unusable."""
    if not isinstance(reply, dict):
        return None, "no json", ""
    query = str(reply.get("query") or "").strip()
    if not query or pilot._QUOTES.search(query) or pilot._LEAK.search(query) or "［" in query or "让你" in query:
        return None, "rule", query
    size = len(query.split()) if spec["language"] == "en" else len(query)
    if size > 15 + 30 * len(spec["items"]):
        return None, "too long", query
    if spec["language"] == "zh" and pilot._ADDRESS.search(query) and not any(
            pilot._ADDRESS.search(a["query"]) for a in spec["labels"]):
        return None, "address term", query
    if spec["language"] == "en" and ta._CJK.search(query):
        return None, "chinese left", query
    spans = {str(k): str(v).strip() for k, v in (reply.get("spans") or {}).items()}
    instructions = {str(k): str(v).strip().rstrip(".") for k, v in (reply.get("instructions") or {}).items()}
    labels = []
    for item_no, atom in spec["slots"]:
        key = str(item_no)
        if atom.get("expected_function") in _SEARCH:
            span = spans.get(key, "")
            if not span or span not in query:
                return None, "span", query
            atom = _set_params(atom, {"query": span})
        if atom.get("expected_function") == "get_weather" and spec["language"] == "en":
            city = atom["expected_params"]["city"]
            named = _CITY_EN.get(city, "")
            if named and named.lower() in query.lower():
                atom = _set_params(atom, {**atom["expected_params"], "city": named})
            elif city != "深圳":
                return None, "city", query
        if spec["language"] == "en":
            text = instructions.get(key, "")
            if not text or ta._CJK.search(text):
                return None, "instruction", query
            atom = json.loads(json.dumps(atom))
            atom["planner_target"]["instructions"]["en"] = text
        labels.append(atom)
    return labels, None, query


def make_specs(pool: assemble.Pool, rng: random.Random, plan: dict[str, int]) -> list[dict[str, Any]]:
    specs = []
    pos = [a for cells in pool.positives.values() for rows in cells.values() for a in rows]
    by_tool = {t: [a for rows in cells.values() for a in rows] for t, cells in pool.positives.items()}
    negs = {f: rows for f, rows in pool.negatives.items() if f in FLIP_HINT}
    action_pos = [a for a in pos if a["expected_function"] in ("HumanAction", "RobotGesture", "RobotDance")]

    def add(category: str, language: str, slots: list[tuple[int, dict[str, Any]]], items: list[str],
            tags: list[str]) -> None:
        specs.append({"category": category, "language": language, "slots": slots, "items": items, "tags": tags,
                      "labels": [a for _, a in slots]})

    def ok(atoms: list[dict[str, Any]], language: str) -> bool:  # English weather needs a city we can name
        return language == "zh" or all(
            (a.get("expected_function") != "get_weather" or a["expected_params"]["city"] in _CITY_EN)
            and (a.get("contrast") or {}).get("flip") not in assemble.EN_REQUEST_FLIPS for a in atoms)

    for language in ("zh", "en"):
        tools = sorted(by_tool)
        for k in range(plan[f"{language}_pos"]):
            atom = rng.choice(by_tool[tools[k % len(tools)]])
            while not ok([atom], language):
                atom = rng.choice(by_tool[tools[k % len(tools)]])
            add("single_positive", language, [(1, atom)], [describe(atom)], [atom["expected_function"]])
        flips = sorted(f for f in negs if language == "zh" or f not in assemble.EN_REQUEST_FLIPS)
        for k in range(plan[f"{language}_neg"]):
            atom = rng.choice(negs[flips[k % len(flips)]])
            add("single_negative", language, [(1, atom)], [describe(atom)], [atom["contrast"]["flip"]])
        for k in range(plan[f"{language}_multi"]):
            level = 2 + k % plan[f"{language}_max_level"]
            case = None
            while case is None or not ok(case, language):
                case = assemble.sample_case(pool, level, 0.3, rng)
            add("multi_intent", language, list(enumerate(case, 1)), [describe(a) for a in case],
                [f"L{level}"])
    ops = sorted(REL_HOW)
    for k in range(plan["zh_rel"]):
        op = ops[k % len(ops)]
        a = rng.choice([x for x in action_pos if op != "speed" or x["atom_name"].startswith(("fwd", "back"))])
        b = rng.choice([x for x in action_pos if assemble.source_cell(x) != assemble.source_cell(a)])
        n = rng.choice((2, 3))
        hint = rng.choice(("快点", "慢慢", "小心点", "轻轻")) if op == "speed" else ""
        expect = {"only": "A", "correct": "A", "retract": "A", "either": "A", "before": "AB", "while": "AB",
                  "count": "A" * n, "speed": "A"}[op]
        slots = [(1, a if x == "A" else b) for x in expect]
        add("relation", "zh", slots, [describe_unit(op, a, b, n, hint)], [op])
    return specs


def _grams(text: str) -> set[str]:
    chars = [c for c in text.lower() if c.isalnum()]
    return {x + y for x, y in zip(chars, chars[1:])}


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    bundle = pp.load_contract_bundle(pp.CONTRACT_V2_PATH)
    pool = assemble.load_pool([args.pool], bundle)
    for flip in ("typo",):  # a misspelt Chinese name cannot be specified to a writer
        pool.negatives.pop(flip, None)
    plan = {"zh_pos": 60, "zh_neg": 42, "zh_multi": 80, "zh_rel": 40, "en_pos": 30, "en_neg": 21, "en_multi": 30}
    plan = {k: max(1, round(v * args.scale)) for k, v in plan.items()} | {"zh_max_level": 5, "en_max_level": 3}
    specs = make_specs(pool, rng, plan)
    train = [_grams(a["query"]) for a in pool.atoms]
    key = find_env_value("DASHSCOPE_API_KEY")
    write = functools.partial(llm.chat_json, api_key=key, model=args.writer, temperature=0.9, system=SYSTEM)
    judges = [functools.partial(llm.chat_json, api_key=key, model=m, temperature=0.0) for m in args.judges]
    out_dir = args.runs / f"challenge_v2-{datetime.now():%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    lock = threading.Lock()
    kept: list[dict[str, Any]] = []
    disputed = (out_dir / "disputed.jsonl").open("a", encoding="utf-8")

    def one(arg: tuple[int, dict[str, Any]]) -> None:
        idx, spec = arg
        seed = args.seed * 1000 + idx
        record = {"idx": idx, "category": spec["category"], "language": spec["language"], "items": spec["items"]}
        try:
            reply = assemble._call(write, write_prompt(spec["items"], spec["language"], rng.choice(SCENES),
                                                       rng.choice(PERSONAS)), SYSTEM, seed)
            labels, reason, query = finalize(spec, reply)
            if labels is not None:
                grams = _grams(query)
                if max(len(grams & g) / max(1, len(grams | g)) for g in train) > 0.6:
                    labels, reason = None, "close to pool"
            verdicts = []
            if labels is not None:
                for judge in judges:
                    got = pilot._as_list(judge(review_prompt([query]), seed=seed))
                    verdicts.append(got[0].get("tasks") if got and isinstance(got[0], dict) else None)
                if not all(matches(labels, v) for v in verdicts):
                    reason = "judges disagree"
        except Exception as exc:  # noqa: BLE001
            labels, reason, query, verdicts = None, f"error {str(exc)[:120]}", "", []
        with lock:
            if reason is None:
                kept.append({**record, "query": query, "labels": labels, "tags": spec["tags"]})
            else:
                disputed.write(json.dumps({**record, "query": query, "reason": reason,
                                           "spec": [audit.label(a) for a in spec["labels"]],
                                           "verdicts": verdicts if labels else None}, ensure_ascii=False) + "\n")
                disputed.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(one, enumerate(specs)))
    disputed.close()

    cases = []
    for n, case in enumerate(sorted(kept, key=lambda c: c["idx"]), 1):
        row = assemble.render_row(case["labels"], case["query"], case["language"], bundle)
        expected = json.loads(row["messages"][-1]["content"])
        level = len(expected)
        neg_tools = sorted({a["negative_tool"] for a in case["labels"] if a.get("negative_sample")
                            and a.get("negative_tool")} - {t.get("name") for t in expected})
        cases.append({
            "case_id": f"v2_{n:03d}", "category": case["category"], "language": case["language"],
            "difficulty": "easy" if level == 1 else ("medium" if level <= 3 else "hard"),
            "polarity": "negative" if any(t["kind"] == "reply" for t in expected) else "positive",
            "query": case["query"], "expected": expected, "forbidden_functions": neg_tools,
            "novelty_tags": [f"writer:{args.writer}", *case["tags"]],
        })
    if args.append and args.out_json.exists():
        old = json.loads(args.out_json.read_text(encoding="utf-8"))["cases"]
        seen = {c["query"] for c in old}
        cases = old + [c for c in cases if c["query"] not in seen]
        for n, case in enumerate(cases, 1):
            case["case_id"] = f"v2_{n:03d}"
    payload = {"schema_version": 1, "name": "challenge_v2", "authored_after_training": False, "sealed": True,
               "writer": args.writer, "judges": args.judges, "pool": str(args.pool), "cases": cases}
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"specs": len(specs), "kept": len(cases)}, indent=1))
    print(f"specs {len(specs)} kept {len(cases)} -> {args.out_json}; disputed: {out_dir / 'disputed.jsonl'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--writer", default="glm-4.6")
    p.add_argument("--judges", nargs="+", default=["qwen-max", "deepseek-v3.2"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--scale", type=float, default=1.0, help="multiply the case plan (smoke runs)")
    p.add_argument("--runs", type=Path, default=pilot.RUNS_DIR)
    p.add_argument("--append", action="store_true", help="keep the cases already in --out-json and add new ones")
    run(p.parse_args())


if __name__ == "__main__":
    main()
