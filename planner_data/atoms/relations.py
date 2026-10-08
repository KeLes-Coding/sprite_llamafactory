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

"""Relation units: phrasings that change which actions run, how often, or in what order.

One or two accepted positive atoms (A, B) are said in one utterance through an operator, and the label
is computed from the operator (user decisions 2026-10-06):

    only      别做B，A就行                  -> A
    correct   B……哦不对，A                  -> A
    retract   A和B，B就算了                 -> A
    either    A或者B都行                    -> A (the first one)
    before    B之前先A                      -> A, B (execution order, not word order)
    while     边A边B                        -> A, B (word order)
    count     鞠三个躬                      -> A x N (gesture/dance; repeat must stay 1)
    speed     快点往前走                    -> A (manner is ignored)

A judge reads the utterance with candidate labels and lists what to do now; the row is kept only when
its list equals the operator's label. Rows are written in the assemble.py format.

    python -m planner_data.atoms.relations --name rel_smoke \
        --atoms runs/no_ability-1006_155239 runs/gesture_dance-1006_170755 --per-op 10
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from planner_data import assemble
from planner_data.atoms import contrast, pilot
from planner_data.atoms.pilot import find_env_value, llm, pa, pp


_CN = "零一两三四五六七八九"
_ACTION_TOOLS = {"HumanAction", "RobotGesture", "RobotDance"}
_MOVES = {"fwd_plain", "fwd_steps", "fwd_little", "back_plain", "back_steps", "back_little",
          "left_plain", "left_steps", "right_plain", "right_steps"}
# Gestures that read naturally with a count; "连续飞吻" is already plural, "请这边走" is not repeated.
_COUNTABLE = {"bow", "clap", "nod", "high_five", "shake_head", "wave_greet_bye", "hand_heart",
              "left_hand_side_heart", "right_hand_side_heart", "shake_hands", "curtain_bow"}
# Every gesture/dance must still be named the way the derive rules read it ("机械午" is not 机械舞).
_NAME_RULES = {**dict(pa._GESTURE_RULES), **dict(pa._DANCE_RULES)}


@dataclass(frozen=True)
class Op:
    how: str  # {A}/{B} are the labels, {n}/{hint} the count/phrasing
    expect: tuple[str, ...]  # letters, "A*" repeats A n times
    marker: re.Pattern[str]
    pair: bool = True
    hints: tuple[str, ...] = ()  # rotated so the writer does not settle on the first example


OPS: dict[str, Op] = {
    "only": Op("意思是不要做B，只做A，句式类似“{hint}”（只借句式，内容用A、B的）",
               ("A",), re.compile(r"别|不要|不用|用不着|只|就行|就好|甭|不必|免了"),
               hints=("别B了，A就行", "只要A，不用B", "A就好，B就免了", "不要B，换成A", "先不管B，光A就行",
                      "用不着B，A吧")),
    "correct": Op("先说让你做B，话没说完马上改口换成A，改口用“{hint}”这类说法，最后只要A",
                  ("A",), re.compile(r"不对|不是|算了|还是|哦不|换|改|说错|等等|不要|废了"),
                  hints=("哦不对", "算了，还是", "说错了，是", "等等，换成", "不不不，改成", "哎不要那个了，")),
    "retract": Op("先说让你做A和B两件事，接着又用“{hint}”这类说法把B收回", ("A",),
                  re.compile(r"算了|不用|别|免了|不要|甭|不做|不跳|先不|取消|不必"),
                  hints=("B就算了", "B就免了", "B先不做了", "B不用了", "哦B取消吧", "B就别做了")),
    "either": Op("一句话里同时提到A和B，说两个里做哪个都行（用“或者、要么……要么、……也行”之类的说法），A先提；"
                  "A和B都必须出现", ("A",), re.compile(r"或|还是|都行|随便|都可以|都成|也行|要么")),
    "before": Op("让你先做A再做B，但句子里先提到B，用“B之前先A”“B以前先A一下”这类说法", ("A", "B"),
                 re.compile(r"之前|以前|前先|前头先")),
    "while": Op("让你边做A边做B（如“边……边……”“一边……一边……”），A先提", ("A", "B"),
                 re.compile(r"(?<![左右上下旁身那这外里前后])边")),
    "count": Op("把句子改成让你连做{n}次（数字用中文，如“鞠三个躬”“挥三下手”“连跳两遍”“点两下头”），其余尽量不变",
                ("A*",), re.compile(r"[两二三四五]"), pair=False),
    "speed": Op("在句子里加上速度或方式的修饰（如“{hint}”），其余尽量不变", ("A",),
                 re.compile(r"快|慢|小心|轻|赶紧|稳|麻利|利索|悠着"), pair=False,
                 hints=("快点", "慢慢地", "小心点", "轻轻地", "赶紧", "稳一点", "麻利点", "别太快")),
}


@dataclass(frozen=True)
class Unit:
    op: str
    a: dict[str, Any]
    b: dict[str, Any] | None = None
    n: int = 1
    hint: str = ""

    @property
    def expected(self) -> list[str]:
        return [x for e in OPS[self.op].expect for x in (["A"] * self.n if e == "A*" else [e])]

    @property
    def label_atoms(self) -> list[dict[str, Any]]:
        return [self.a if x == "A" else self.b for x in self.expected]  # type: ignore[misc]


def label(atom: dict[str, Any]) -> str:
    return str(atom["planner_target"]["instructions"]["zh"]).rstrip("。")


def _value(atom: dict[str, Any]) -> str:
    params = atom.get("expected_params") or {}
    return str(params.get("gesture") or params.get("dance") or atom["atom_name"])


def sample(positives: list[dict[str, Any]], op: str, rng: random.Random) -> Unit | None:
    hint = rng.choice(OPS[op].hints) if OPS[op].hints else ""
    if op == "count":
        pool = [a for a in positives if a["expected_function"] == "RobotDance"
                or (a["expected_function"] == "RobotGesture" and _value(a) in _COUNTABLE)]
        return Unit(op, rng.choice(pool), n=rng.choice((2, 2, 3, 3, 4, 5))) if pool else None
    if op == "speed":
        pool = [a for a in positives if a["atom_name"] in _MOVES]
        return Unit(op, rng.choice(pool), hint=hint) if pool else None
    if op == "while":
        a_pool = [a for a in positives if a["atom_name"] in _MOVES and "steps" not in a["atom_name"]]
        b_pool = [a for a in positives if a["expected_function"] == "RobotGesture"]
        return Unit(op, rng.choice(a_pool), rng.choice(b_pool)) if a_pool and b_pool else None
    a = rng.choice(positives)
    same_tool = [p for p in positives if p["expected_function"] == a["expected_function"]]
    for _ in range(20):  # half the pairs share a tool: "别跳机械舞，跳万物生就行"
        b = rng.choice(same_tool if rng.random() < 0.5 else positives)
        if assemble.source_cell(b) != assemble.source_cell(a) and _value(b) != _value(a):
            return Unit(op, a, b, hint=hint)
    return None


SYSTEM = "你为一台人形服务机器人的规划模型合成训练数据：按要求把用户的意思说成一句自然口语。"


def write_prompt(unit: Unit) -> str:
    op = OPS[unit.op]
    how = op.how.format(n=_CN[unit.n] if unit.n < len(_CN) else unit.n, hint=unit.hint)
    lines = ["下面是同一位用户当面对你说过的话："]
    if unit.b is None:
        lines.append(f"原句：{unit.a['query']}")
    else:
        lines += [f"A：{unit.a['query']}（A 指：{label(unit.a)}）", f"B：{unit.b['query']}（B 指：{label(unit.b)}）"]
    lines += [
        f"要求：{how}",
        "- 沿用原句的说法、场景和口吻，可以删掉多余的背景，背景冲突时统一成一个场景；",
        "- 动作名、舞名、手势名、数字、方向、角度、步数一个字不改，不新增别的请求；",
        "- 不写“机器人”，不给你起名字或称呼，不加引号或书名号，不出现字母 A、B。",
        '只输出 JSON：{"query": "..."}',
    ]
    return "\n".join(lines)


def judge_prompt(query: str, unit: Unit) -> str:
    cands = [f"A. {label(unit.a)}"] + ([f"B. {label(unit.b)}"] if unit.b is not None else [])
    return "\n".join([
        f"用户当面对机器人说：{query}",
        "候选动作：" + "；".join(cands),
        "按下面的约定，列出机器人现在应该依次做的动作（只用候选字母，可重复；一个都不做就输出空列表）：",
        "- 被否掉、改口换掉、说完又撤回的不做；",
        "- “X或者Y都行”只做先说的那个；",
        "- “边X边Y”两个都做，按说的先后；“X之前先Y”“先Y再X”按实际先后做；",
        "- 说做几次就重复几次（如“鞠三个躬”列三次）；快慢、轻重等方式修饰不影响做什么；",
        "- 推迟到以后、假设、问能不能做、对别人说的，都不算现在做；",
        "- 动作名、舞名、数字、方向和候选不一致的，不算这个候选。",
        "另外，句中如果还有候选之外、让机器人现在做的事，other 填 true。",
        '只输出 JSON：{"do": ["A"], "other": false}',
    ])


def check(unit: Unit, query: str) -> str | None:
    if not query:
        return "empty"
    if pilot._LEAK.search(query) or re.search(r"[AB][：:.、]|候选", query):
        return "prompt leak"
    if pilot._QUOTES.search(query):
        return "quotes"
    if pilot._ADDRESS.search(query):
        return "address term"
    if contrast.ABILITY.search(query):
        return "ability question"
    if not OPS[unit.op].marker.search(query):
        return f"no {unit.op} marker"
    if unit.op == "count" and _CN[unit.n] not in query and not (unit.n == 2 and "二" in query):
        return "wrong count"
    for atom in (unit.a, unit.b):  # an unnamed dance ("跳个舞") has no name to keep
        rule = _NAME_RULES.get(_value(atom)) if atom and atom["atom_name"] != "dance_unnamed" else None
        if rule is not None and not rule.search(query):
            return f"lost name {_value(atom)}"
    return pilot.dance_name_issue(query)


def judge_reason(unit: Unit, verdict: Any) -> str | None:
    if not isinstance(verdict, dict) or not isinstance(verdict.get("do"), list):
        return "judge missing"
    if verdict.get("other") is True:
        return "judge extra"
    got = [str(x).strip().upper() for x in verdict["do"]]
    return None if got == unit.expected else f"judge {''.join(got) or '-'}"


@dataclass
class Context:
    write: Callable[..., Any]
    judge: Callable[..., Any]
    bundle: Any
    sink: Callable[[str, dict[str, Any]], None] = lambda kind, row: None
    lock: threading.Lock = field(default_factory=threading.Lock)


def process(unit: Unit, ctx: Context, seed: int) -> dict[str, Any] | None:
    sources = [unit.a["query"]] + ([unit.b["query"]] if unit.b is not None else [])

    def reject(query: str, reason: str, **extra: Any) -> None:
        ctx.sink("rejected", {"op": unit.op, "sources": sources, "query": query, "reason": reason, **extra})

    try:
        reply = assemble._call(ctx.write, write_prompt(unit), SYSTEM, seed)
    except RuntimeError as exc:
        reject("<call>", "call_error", error=str(exc))
        return None
    query = str(reply.get("query") or "").strip() if isinstance(reply, dict) else ""
    reason = check(unit, query)
    if reason is not None:
        reject(query, reason)
        return None
    try:
        verdict = assemble._call(ctx.judge, judge_prompt(query, unit), pilot.JUDGE_SYSTEM, seed)
    except RuntimeError as exc:
        reject(query, "judge_error", error=str(exc))
        return None
    reason = judge_reason(unit, verdict)
    if reason is not None:
        reject(query, reason, verdict=verdict)
        return None
    row = assemble.render_row(unit.label_atoms, query, "zh", ctx.bundle)
    row["metadata"].update(op=unit.op, source_queries=sources)
    ctx.sink("rows", row)
    ctx.sink("units", {  # one slot for assemble.py: said as one sentence, labelled by the operator
        "id": "rel_" + hashlib.sha1(query.encode()).hexdigest()[:12], "op": unit.op, "query": query,
        "sources": sources, "cells": sorted({assemble.source_cell(a) for a in (unit.a, unit.b) if a is not None}),
        "unit_atoms": unit.label_atoms,
    })
    return row


def load_positives(run_dirs: list[Path], bundle: Any) -> list[dict[str, Any]]:
    pool = assemble.load_pool(run_dirs, bundle)
    return [a for tool, cells in pool.positives.items() if tool in _ACTION_TOOLS
            for rows in cells.values() for a in rows if not a.get("contrast")]


def run(args: argparse.Namespace, ctx: Context) -> Path:
    rng = random.Random(args.seed)
    positives = load_positives(args.atoms, ctx.bundle)
    units = [u for op in args.ops for u in (sample(positives, op, rng) for _ in range(args.per_op)) if u]
    out = args.out / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False,
                                                indent=2), encoding="utf-8")
    files = {kind: (out / f"{kind}.jsonl").open("a", encoding="utf-8") for kind in ("rows", "rejected", "units")}

    def sink(kind: str, row: dict[str, Any]) -> None:
        with ctx.lock:
            files[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[kind].flush()

    ctx.sink = sink
    seeds = [rng.randrange(1 << 30) for _ in units]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(lambda u: process(u[0], ctx, u[1]), zip(units, seeds)))
    finally:
        for f in files.values():
            f.close()
    asked, ok = Counter(u.op for u in units), Counter(u.op for u, r in zip(units, results) if r is not None)
    lines = [f"{'op':<8} {'asked':>5} {'rows':>5}"] + [f"{op:<8} {asked[op]:>5} {ok[op]:>5}" for op in args.ops]
    report = "\n".join(lines)
    (out / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\noutput: {out}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--atoms", type=Path, nargs="+", required=True, help="run directories with positive atoms")
    p.add_argument("--ops", nargs="+", choices=list(OPS), default=list(OPS))
    p.add_argument("--per-op", type=int, default=10)
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--judge-model", default=llm.DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=pilot.RUNS_DIR)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    api_key = find_env_value("DASHSCOPE_API_KEY")
    ctx = Context(
        write=functools.partial(llm.chat_json, api_key=api_key, model=args.model, temperature=args.temperature),
        judge=functools.partial(llm.chat_json, api_key=api_key, model=args.judge_model, temperature=0.0),
        bundle=pp.load_contract_bundle(pp.CONTRACT_V2_PATH),
    )
    run(args, ctx)


if __name__ == "__main__":
    main()
