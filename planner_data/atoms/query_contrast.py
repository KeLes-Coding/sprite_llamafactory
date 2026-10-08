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

"""Query-tool contrast: minimal edits that move a query positive to reply or to a neighbouring tool.

Reply flips keep the topic but drop the request: an ability question without an object ("你会查天气吗"),
a statement, words for someone else, a deferred request, or a swap to stable common knowledge.
Cross flips keep the sentence but move it to the tool next door (weather -> city events, Luna price ->
phone price, battery left -> rated endurance, local time -> foreign time), so each boundary is seen
from both sides with the same wording.

    python -m planner_data.atoms.query_contrast --name q_neg \
        --source runs/query_smoke5-1006_212421 runs/query_smoke6-1006_212812
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from planner_data.atoms import contrast, pilot
from planner_data.atoms.pilot import _norm_text, find_env_value, llm, pa, pp


_CELLS = {c.cell_id: c for c in pilot.ALL_CELLS}
# One category word per source cell for "你会查X吗"; the writer copies whatever it is given, so no lists.
# Status items and the robot's own specs have none: "你会查自己的电量吗" reads as a request.
_CATEGORY = {
    **{c.cell_id: "天气" for c in pa.CELLS if c.tool == "get_weather"},
    "ws_news": "新闻", "ws_finance": "行情", "ws_flight": "航班", "ws_sports": "比赛结果", "ws_traffic": "交通",
    "ws_scenic": "景点开放时间", "ws_city_events": "本地活动", "ws_entertainment": "娱乐新闻",
    "ws_launch": "新品价格", "ws_world_time": "国外时间",
    "rag_company": "公司信息", "rag_product": "产品介绍", "rag_price": "产品价格", "rag_aftersale": "售后政策",
    "rag_scene": "产品适用场景", "rag_tech": "技术原理",
}
_WEATHER_CITY = ("now_city", "h24_city", "d7_city", "now7d_city")


@dataclass(frozen=True)
class QFlip:
    how: str  # {topic}/{hint}/{value} are filled in
    reply: str = ""  # instruction_zh example; empty for cross flips
    target: str = ""  # cell the rewrite belongs to; empty for reply flips
    sources: tuple[str, ...] = ()  # source cells; empty means every query cell
    marker: re.Pattern[str] | None = None
    hints: tuple[str, ...] = ()


FLIPS: dict[str, QFlip] = {
    "ability": QFlip("改成只问你会不会、能不能查{topic}，把具体要查的城市、时段、产品名、事件都删掉",
                     "回答是否能查{topic}", sources=tuple(_CATEGORY), marker=contrast.ABILITY),
    "statement": QFlip("改成{hint}，不提问，也不请你查", "回应用户的闲聊",
                       hints=("随口感叹一句", "陈述自己已经知道的情况", "跟旁人聊起这件事", "抱怨一句")),
    "others": QFlip("改成对在场的{hint}说、请对方去查或问对方，不再是问你（句中点明对方，别只在句首加称呼）",
                    "回应用户对别人说的话", hints=contrast.HINTS["others"]),
    "deferred": QFlip("改成{hint}再查，不是让你现在查", "回应稍后再查的安排", marker=contrast.FLIPS["deferred"].marker,
                      hints=("等会儿", "待会儿", "明天出门前", "等我到了酒店", "下次来的时候", "吃完饭")),
    "common": QFlip("把整个问题换成一个不用查就能答的{hint}问题：原句要问的对象全部删掉，不要在原问题后面追加；"
                     "开头的口头禅和口吻保留", "回答……（写出具体问的常识）",
                    hints=("历史常识", "地理常识", "单位换算", "生活常识", "自然科学常识", "语文常识")),
    "weather_to_events": QFlip("把问天气改成问{value}同一时间的{hint}，城市和时间说法不变",
                               target="ws_city_events", sources=_WEATHER_CITY,
                               hints=("活动", "展览", "演出", "音乐节", "庙会", "市集")),
    "weather_to_traffic": QFlip("把问天气改成问{value}的{hint}，城市不变", target="ws_traffic", sources=_WEATHER_CITY,
                                hints=("地铁末班车", "地铁几点开", "公交线路", "路况", "堵不堵车")),
    "launch_to_rag": QFlip("把其中的商品换成逐际动力的{hint}机器人，仍问原句问的价格或购买的事，其余尽量不变",
                           target="rag_price", sources=("ws_launch",), hints=("Luna", "Oli", "TRON1", "TRON2")),
    "rag_to_launch": QFlip("把其中的产品换成{hint}（一款公开商品），其余尽量不变", target="ws_launch",
                           sources=("rag_price",),
                           hints=("华为Mate 70", "iPhone 17", "小米SU7", "大疆Mini 5", "Switch 2", "特斯拉Model Y")),
    "battery_to_spec": QFlip("改成问你出厂设计的{hint}（参数），不是眼下的电量；不出现“现在、当前、还剩、还能撑”",
                             target="rag_spec_self", sources=("status_电量",),
                             hints=("满电续航", "电池容量", "充满电要多久")),
    "spec_to_battery": QFlip("改成问你眼下的电量（意思是{hint}），不再问设计参数", target="status_电量",
                             sources=("rag_spec_self",), hints=("还剩多少电", "电够不够用", "还能撑多久", "要不要充电")),
    "time_to_world": QFlip("改成问{hint}现在几点，其余尽量不变", target="ws_world_time", sources=("status_时间",),
                           hints=tuple(pa._SLOT_VALUES["foreign_city"])),
    "world_to_time": QFlip("去掉城市，改成问这里现在几点，其余尽量不变", target="status_时间",
                           sources=("ws_world_time",)),
}
_STATEMENT_Q = re.compile(r"[？?吗]")
_SEARCH_RULE = ("另给 search_query：改写后句子里的一段连续原文，逐字复制，从提到对象处截到问完为止，"
                "不带“帮我查”“你知道”这类请求开头")


@dataclass(frozen=True)
class Item:
    source: dict[str, Any]
    flip: str
    hint: str = ""

    @property
    def tool(self) -> str:
        return str(self.source["expected_function"])

    @property
    def target(self) -> pa.Cell | None:
        name = FLIPS[self.flip].target
        return _CELLS[name] if name else None

    @property
    def value(self) -> str | None:
        """Slot value the target cell checks: the kept weather city, or the new foreign city."""
        target = self.target
        if target is None or not target.slot:
            return None
        if target.slot == "foreign_city":
            return self.hint
        return str(self.source["expected_params"].get("city") or "")


def select(atoms: list[dict[str, Any]], per_cell: int, rng: random.Random,
           only: list[str] | None = None) -> list[Item]:
    """Sample positives per cell and spread the flips over them; cross flips get half of their source cells."""
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        if atom.get("expected_function") in pilot.QUERY_TOOLS and not atom.get("negative_sample"):
            by_cell[atom["atom_name"]].append(atom)
    hints = {name: rng.sample(f.hints, len(f.hints)) for name, f in FLIPS.items() if f.hints}
    used: Counter[str] = Counter()
    out: list[Item] = []
    for cell_id in sorted(by_cell):
        rows = rng.sample(by_cell[cell_id], min(per_cell, len(by_cell[cell_id])))
        usable = [n for n, f in FLIPS.items() if (only is None or n in only) and (not f.sources or cell_id in f.sources)]
        cross = rng.sample([n for n in usable if FLIPS[n].target], sum(bool(FLIPS[n].target) for n in usable))
        reply = rng.sample([n for n in usable if not FLIPS[n].target], len(usable) - len(cross))
        if not usable:
            continue
        for i, src in enumerate(rows):
            # Cross flips take every other row of their source cells; the reply flips share the rest.
            if cross and (i % 2 == 0 or not reply):
                name = cross[(i // 2) % len(cross)]
            else:
                name = reply[(i // 2 if cross else i) % len(reply)]
            hint = ""
            if name in hints:
                hint = hints[name][used[name] % len(hints[name])]
                used[name] += 1
            out.append(Item(src, name, hint))
    return out


def requirement(item: Item) -> str:
    flip = FLIPS[item.flip]
    topic = _CATEGORY.get(item.source["atom_name"], "")
    how = flip.how.format(topic=topic, hint=item.hint, value=item.value or "")
    target = item.target
    if target is None:
        return f"{how}（instruction_zh 示例：{flip.reply.format(topic=topic)}）"
    if target.tool in pilot._SEARCH_TOOLS:
        return (f"{how}（改写后仍是现在问你；instruction_zh 是改写后的句子去掉客套和语气词后剩下的问句，"
                f"必须包含 search_query 原文；{_SEARCH_RULE}）")
    return f"{how}（改写后仍是现在问你；instruction_zh 是“查询……”形式的简洁任务，写明查询项）"


SYSTEM = contrast.SYSTEM


def build_prompt(items: list[Item]) -> str:
    lines = [
        "下面每句是用户当面对机器人说的话。按每句的要求改写成一句新话：只改必要的部分，"
        "保留原句的场景、口吻、口语噪声和大致长度。",
        "硬性要求：",
        "- 不给机器人起名字或称呼，需要指代时最多用“你”；",
        "- 不加引号或书名号；",
        "- 要求里没写“仍是现在问你”的，改写后不能再是让你现在查或告诉一个具体信息；",
        "- 每句给出 instruction_zh（按要求后的示例或格式写简洁任务）和 instruction_en（instruction_zh 的自然英文）。",
        "句子：",
    ]
    for i, item in enumerate(items, 1):
        lines += [f"{i}. 原句：{item.source['query']}", f"   要求：{requirement(item)}"]
    lines.append('按编号顺序只输出 JSON 数组：[{"i": 1, "query": "...", "instruction_zh": "...", "instruction_en": "..."}]'
                 '（要求里让另给 search_query 的，再加 "search_query" 字段）')
    return "\n".join(lines)


def target_params(item: Item, search: str) -> dict[str, Any] | None:
    target = item.target
    if target is None:
        return None
    if target.tool in pilot._SEARCH_TOOLS:
        return {"query": search}
    return pa._cell_params(target, None, "")


def check(item: Item, query: str, zh: str, en: str, search: str) -> str | None:
    """Rule rejection reason for one rewrite, or None."""
    if not (query and zh and en):
        return "empty field"
    if _norm_text(query) == _norm_text(item.source["query"]):
        return "unchanged"
    if pilot._LEAK.search(query):
        return "prompt leak"
    if pilot._QUOTES.search(query):
        return "quotes"
    target = item.target
    rag = "rag_query" in (item.tool, target.tool if target else "")
    if (pilot._ADDRESS_RAG if rag else pilot._ADDRESS).search(query) or pa._OTHER_ASSISTANT.search(query):
        return "address term"
    if target is not None:
        return pilot.validate(target, target_params(item, search), query, zh, en, item.value, search)
    if not zh.startswith(("回答", "回应", "说明")):
        return "reply instruction style"
    flip = FLIPS[item.flip]
    if flip.marker is not None and not flip.marker.search(query):
        return f"no {item.flip} marker"
    if item.flip == "statement" and _STATEMENT_Q.search(query):
        return "still a question"
    if item.flip == "ability":  # "……能不能用在餐厅迎宾啊？你会查产品适用场景吗" still asks the real question
        city = (item.source.get("expected_params") or {}).get("city")
        if (city and city in query) or re.search(pilot._PRODUCT_RE, query, re.IGNORECASE) \
                or len(re.findall(r"[？?]|吗", query)) > 1:
            return "ability keeps the object"
    return None


def judge_reason(item: Item, verdict: Any, params: dict[str, Any] | None) -> str | None:
    """Reply flips: no ask, or (common) an ask the robot answers itself. Cross flips: the target tool."""
    if item.target is not None:
        return pilot.judge_reason(item.target.tool, verdict, params)
    if not isinstance(verdict, dict):
        return "judge missing"
    if item.flip == "common":
        return None if verdict.get("source") == "common" else f"judge source {verdict.get('source')}"
    return None if verdict.get("ask") is False else "judge says ask"


def build_atom(item: Item, query: str, zh: str, en: str, params: dict[str, Any] | None,
               writer: str) -> dict[str, Any]:
    src = _CELLS[item.source["atom_name"]]
    if item.target is None:
        cell = replace(src, cell_id=f"neg_{item.flip}_{src.cell_id}", params=None, slot=None, reply=True)
    else:
        cell = replace(item.target, cell_id=f"x_{item.flip}")
    atom = pa._atom(cell, query, zh, en, params)
    atom["curation"].update(writer=f"model:{writer}", validator="rules+judge")
    atom["contrast"] = {"flip": item.flip, "source_id": item.source["id"], "source_query": item.source["query"],
                        "hint": item.hint or None}
    return atom


def process_batch(items: list[Item], ctx: contrast.Context, seed: int
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def reject(item: Item, query: str, reason: str, **extra: Any) -> None:
        row = {"flip": item.flip, "cell": item.source["atom_name"], "source": item.source["query"],
               "query": query, "reason": re.sub(r"\s*(\{.*|None)$", "", reason), "detail": reason, **extra}
        rejected.append(row)
        ctx.sink("rejected", row)

    rows: list[Any] = []
    try:
        for attempt in range(2):
            try:
                rows = pilot._as_list(ctx.write(build_prompt(items), system=SYSTEM, seed=seed + attempt * 1000))
            except Exception:  # noqa: BLE001
                if attempt:
                    raise
            if rows:
                break
        if not rows:
            raise ValueError("no sentence list in reply")
    except Exception as exc:  # noqa: BLE001 - a failed call drops one batch
        for item in items:
            reject(item, "<call>", "call_error", error=str(exc)[:200])
        return accepted, rejected

    by_i = {r.get("i"): r for r in rows if isinstance(r, dict)}
    candidates: list[tuple[Item, str, dict[str, Any] | None, dict[str, Any]]] = []
    for idx, item in enumerate(items, 1):
        row = by_i.get(idx) or (rows[idx - 1] if idx - 1 < len(rows) and isinstance(rows[idx - 1], dict) else {})
        query, zh, en, search = (str(row.get(k) or "").strip()
                                 for k in ("query", "instruction_zh", "instruction_en", "search_query"))
        reason = check(item, query, zh, en, search)
        params = target_params(item, search)
        atom = None
        if reason is None:
            atom = build_atom(item, query, zh, en, params, ctx.writer)
            try:
                pp.normalize_atom_for_planning(atom, ctx.bundle)
            except (ValueError, TypeError) as exc:
                reason = f"contract: {exc}"
        if reason is None:
            with ctx.lock:
                key = _norm_text(query)
                reason = "duplicate" if key in ctx.seen else None
                ctx.seen.add(key)
        if reason is not None or atom is None:
            reject(item, query, reason or "", instruction_zh=zh, search_query=search)
            continue
        candidates.append((item, query, params, atom))

    for item, query, params, atom in candidates:  # one query per judge call, as in contrast.py
        verdicts, error = None, ""
        for attempt in range(2):
            try:
                verdicts = pilot._as_list(ctx.judge(pilot.query_judge_prompt([query]), system=pilot.JUDGE_SYSTEM,
                                                    seed=seed + attempt * 1000))
                break
            except Exception as exc:  # noqa: BLE001 - an unreviewed rewrite is dropped
                error = str(exc)[:200]
        verdict = verdicts[0] if verdicts else None
        reason = "judge_error" if verdicts is None else judge_reason(item, verdict, params)
        if reason is not None:
            reject(item, query, reason, verdict=verdict, **({"error": error} if verdicts is None else {}))
            continue
        accepted.append(atom)
        ctx.sink("atoms", atom)
    return accepted, rejected


def summarize(items: list[Item], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> str:
    asked = Counter(i.flip for i in items)
    ok = Counter(a["contrast"]["flip"] for a in accepted)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rejected:
        reasons[r["flip"]][r["reason"]] += 1
    lines = [f"{'flip':<20} {'asked':>5} {'valid':>5}  top rejects"]
    for flip in FLIPS:
        top = ", ".join(f"{k}:{v}" for k, v in reasons[flip].most_common(3))
        lines.append(f"{flip:<20} {asked[flip]:>5} {ok[flip]:>5}  {top}")
    lines.append(f"{'TOTAL':<20} {sum(asked.values()):>5} {len(accepted):>5}")
    return "\n".join(lines)


def run(args: argparse.Namespace, ctx: contrast.Context) -> Path:
    rng = random.Random(args.seed)
    atoms = [a for src in args.source for a in contrast.load_atoms(src)]
    if args.cells:
        atoms = [a for a in atoms if a["atom_name"] in args.cells]
    items = select(atoms, args.per_cell, rng, args.flips)
    ctx.seen |= {_norm_text(a["query"]) for a in atoms}
    out = args.out / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False,
                                                indent=2), encoding="utf-8")
    files = {kind: (out / f"{kind}.jsonl").open("a", encoding="utf-8") for kind in ("atoms", "rejected")}

    def sink(kind: str, row: dict[str, Any]) -> None:
        with ctx.lock:
            files[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[kind].flush()

    ctx.sink = sink
    rng.shuffle(items)
    batches = [items[i : i + args.batch] for i in range(0, len(items), args.batch)]
    seeds = [rng.randrange(1 << 30) for _ in batches]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda b: process_batch(b[0], ctx, b[1]), zip(batches, seeds)))
    finally:
        for f in files.values():
            f.close()
    accepted = [a for acc, _ in results for a in acc]
    rejected = [r for _, rej in results for r in rej]
    report = summarize(items, accepted, rejected)
    (out / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\noutput: {out}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--source", type=Path, nargs="+", required=True, help="pilot run directories with atoms.jsonl")
    p.add_argument("--cells", nargs="+", default=None, help="only these source cells")
    p.add_argument("--flips", nargs="+", choices=list(FLIPS), default=None, help="only these rewrites")
    p.add_argument("--per-cell", type=int, default=8, help="positives rewritten per source cell")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--judge-model", default=llm.DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=pilot.RUNS_DIR)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    api_key = find_env_value("DASHSCOPE_API_KEY")
    ctx = contrast.Context(
        write=functools.partial(llm.chat_json, api_key=api_key, model=args.model, temperature=args.temperature),
        judge=functools.partial(llm.chat_json, api_key=api_key, model=args.judge_model, temperature=0.0),
        bundle=pp.load_contract_bundle(pp.CONTRACT_V2_PATH),
        seen=set(),
        writer=args.model,
    )
    run(args, ctx)


if __name__ == "__main__":
    main()
