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

"""Assemble accepted atoms into L1-L10 planner training rows (sharegpt jsonl, same format as v330).

L1 rows use every atom sentence once, as is (about --l1-share of all rows). An Ln row (n >= 2) samples n
atoms (positives, with reply negatives mixed in), has the writer merge their sentences into one utterance,
and keeps it only when a judge finds every intent, in order, with nothing extra. Rows are written as they
finish; run augment.py on rows.jsonl.

    python -m planner_data.assemble --name v340_zh --atoms runs/no_ability-1006_155239 runs/gesture_dance-1006_170755 \
        runs/neg_contrast6-1006_185017 --count 20
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from planner_data.atoms import pilot
from planner_data.atoms.pilot import _norm_text, find_env_value, llm, pa, pp
from planner_data.contract import planning_contract as pc


_TOOL_KEYS = {"HumanAction": "action", "RobotGesture": "gesture", "RobotDance": "dance"}


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------
@dataclass
class Pool:
    positives: dict[str, dict[str, list[dict[str, Any]]]]  # tool -> cell -> atoms
    negatives: dict[str, list[dict[str, Any]]]  # flip -> atoms
    dropped: Counter[str]
    units: list[dict[str, Any]] = field(default_factory=list)  # relations.py: one sentence, several tasks

    @property
    def atoms(self) -> list[dict[str, Any]]:
        pos = [a for cells in self.positives.values() for rows in cells.values() for a in rows]
        return pos + [a for rows in self.negatives.values() for a in rows]


def load_units(run_dirs: list[Path]) -> list[dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        with (run_dir / "units.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    unit = json.loads(line)
                    units.setdefault(_norm_text(unit["query"]), unit)
    return list(units.values())


def tasks_of(part: dict[str, Any]) -> list[dict[str, Any]]:
    """The label atoms of one slot: an atom labels itself, a relation unit carries its own."""
    return part.get("unit_atoms") or [part]


def cells_of(part: dict[str, Any]) -> set[str]:
    return set(part["cells"]) if "unit_atoms" in part else {source_cell(part)}


def load_pool(run_dirs: list[Path], bundle: Any) -> Pool:
    positives: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    negatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    for run_dir in run_dirs:
        with (run_dir / "atoms.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                atom = json.loads(line)
                key = _norm_text(atom["query"])
                if key in seen:
                    dropped["duplicate"] += 1
                    continue
                if atom.get("expected_function") == "RobotDance" and atom.get("language", "zh") == "zh" \
                        and pilot.dance_name_issue(atom["query"]):
                    dropped["dance name"] += 1  # accepted before the spelling rule
                    continue
                seen.add(key)
                atom = pp.normalize_atom_for_planning(atom, bundle)
                if atom.get("negative_sample"):
                    negatives[atom["contrast"]["flip"]].append(atom)
                else:
                    positives[atom["expected_function"]][atom["atom_name"]].append(atom)
    return Pool({t: dict(c) for t, c in positives.items()}, dict(negatives), dropped)


_CELLS = {c.cell_id: c for c in pilot.ALL_CELLS}


def source_cell(atom: dict[str, Any]) -> str:
    """What an atom is about: the gesture/dance value, else the cell (a negative's is its source positive's).

    Several cells share one value (every wave cell is wave_greet_bye), and "打个招呼……挥挥手" merged
    from two of them reads as one intent. Cross-tool contrasts (x_*) are their own topic.
    """
    name = str(atom["atom_name"])
    cell_id = name.split("_", 2)[2] if atom.get("negative_sample") else name
    cell = _CELLS.get(cell_id)
    params = (cell.params if cell else None) or {}
    for key in ("gesture", "dance"):
        if key in params:
            return f"{key}:{params[key]}"
    return cell_id


def sample_case(pool: Pool, level: int, neg_rate: float, rng: random.Random,
                rel_rate: float = 0.0) -> list[dict[str, Any]] | None:
    """Slots adding up to `level` tasks: no cell or flip twice, so no "鞠个躬……别鞠躬了" and no two "唱首歌"."""
    tools = sorted(pool.positives)
    flips = sorted(pool.negatives)
    case: list[dict[str, Any]] = []
    cells: set[str] = set()
    used_flips: set[str] = set()
    filled = 0
    while filled < level:
        for _try in range(20):
            roll = rng.random()
            if pool.units and roll < rel_rate:
                atom = rng.choice(pool.units)
                if len(tasks_of(atom)) > level - filled:
                    continue
            elif flips and roll < rel_rate + neg_rate:
                free = [f for f in flips if f not in used_flips]
                if not free:
                    continue
                flip = rng.choice(free)
                atom = rng.choice(pool.negatives[flip])
            else:
                tool = rng.choice(tools)
                cell = rng.choice(sorted(pool.positives[tool]))
                atom = rng.choice(pool.positives[tool][cell])
            if cells_of(atom) & cells:
                continue
            cells |= cells_of(atom)
            if atom.get("negative_sample"):
                used_flips.add(atom["contrast"]["flip"])
            case.append(atom)
            filled += len(tasks_of(atom))
            break
        else:
            return None
    return case


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def _strip_end(target: dict[str, Any]) -> dict[str, Any]:
    """Some atom instructions end with “。” (“左移二十六步。”); labels should not vary on that."""
    instructions = {k: str(v).strip().rstrip("。.！!") for k, v in target["instructions"].items()}
    return {**target, "instructions": instructions}


def render_row(parts: list[dict[str, Any]], query: str, language: str, bundle: Any) -> dict[str, Any]:
    atoms = [a for part in parts for a in tasks_of(part)]
    compiler_atoms = [{"id": a["id"], "test_input": a["test_input"], "expected_steps": a["expected_steps"],
                       "planner_target": _strip_end(a["planner_target"])} for a in atoms]
    plan = pc.compile_plan(compiler_atoms, language=language, tools=bundle.tools,
                           contract_ref=pc.contract_ref(bundle), provenance="generated")
    tasks = plan.tasks
    for task in tasks:
        if isinstance(task, pc.CallTask):
            pc.validate_task_against_contract(task, bundle.tools)
    messages = pc.render_planner_messages(system_prompt=bundle.system_prompt.content, tools=bundle.tools,
                                          instruction=query)
    messages.append({"role": "assistant", "content": pc.render_assistant_target(tasks)})
    actions = sorted({t.arguments[_TOOL_KEYS[t.name]] for t in tasks
                      if isinstance(t, pc.CallTask) and t.name in _TOOL_KEYS})
    return {
        "messages": messages,
        "metadata": {
            "language": language,
            "level": len(tasks),
            "polarity": "negative" if any(a.get("negative_sample") for a in atoms) else "positive",
            "kinds": sorted({t.kind for t in tasks}),
            "tools": sorted({t.name for t in tasks if isinstance(t, pc.CallTask)}),
            "actions": actions,
            "flips": [a["contrast"]["flip"] for a in atoms if a.get("negative_sample")],
            "ops": [p["op"] for p in parts if "unit_atoms" in p],
            "semantic_id": plan.semantic_id,
            "lineage": plan.lineage,
            "source_queries": [p["query"] for p in parts],
            "contract_display_order": list(bundle.display_order),
            "provenance": "generated",
        },
    }


# ---------------------------------------------------------------------------
# Merge (L2+)
# ---------------------------------------------------------------------------
SYSTEM = "你为一台人形服务机器人的规划模型合成训练数据：把同一位用户的几句话合成一段自然口语。"


def merge_prompt(atoms: list[dict[str, Any]], language: str = "zh") -> str:
    lines = [
        f"下面是同一位用户当面依次对你说的 {len(atoms)} 句话。把它们合成这位用户一口气说完的一段自然口语：",
        "- 每句里的请求或问题都要保留，严格按编号先后出现（哪怕调换后更顺也不能调换），不新增请求，也不把两件事并成一件；",
        "- 每句的性质不变：让你做的仍是让你现在做，问你会不会的仍是问，对别人说的仍是对那个人说，"
        "叫你别做、以后再做、评价你、许愿的照旧；",
        "- 动作名、舞名、手势名、数字、方向、角度、步数一个字不改；",
        "- 背景互相矛盾时统一成一个场景，可以删掉多余的背景和寒暄；句与句之间按这位用户自己的说话习惯衔接；",
        "- 对别人说的话单独成句，点名的人只出现在那一句里，前后对你的请求不能读起来像在对那个人说；",
        "- 不写“机器人”，不给你起名字或称呼，不加引号或书名号。",
    ]
    if language == "en":
        lines[-1] = "- 句子是英文，合成的也是英语母语者的自然口语；不写 robot，不给你起名字或称呼，不加引号。"
    lines.append("句子：")
    lines += [f"{i}. {a['query']}" for i, a in enumerate(atoms, 1)]
    lines.append('只输出 JSON：{"query": "..."}')
    return "\n".join(lines)


def judge_prompt(query: str, atoms: list[dict[str, Any]]) -> str:
    """Compare with the source sentences, not the labels: a reply label (“说明不会跳最炫民族风”) is the answer,
    not what the user said, and the judge marked it missing."""
    lines = [f"用户原本分开对机器人说了下面 {len(atoms)} 句话："]
    lines += [f"{i}. {a['query']}" for i, a in enumerate(atoms, 1)]
    lines += [
        f"合并成了一段话：{query}",
        "逐句核对合并后的话是否保留了这句里的请求或问题：意思和性质都不能变——让机器人现在做的仍是现在做，"
        "问会不会的仍是问，对别人说的仍是对那个人说，叫它别做、以后再做、评价、许愿的照旧；"
        "动作名、舞名、手势名、数字、方向一字不差（原句里的错别字或不存在的名字也要照抄）；",
        "保留了的，在 quote 里从合并后的话中原样摘出对应这一句的最短片段（一个字都不改，不同句的片段不重叠）；",
        "再看合并后有没有原句里都没有的新请求或问题（寒暄、背景、连接词不算）。",
        '只输出 JSON：{"items": [{"i": 1, "kept": true, "quote": "..."}], "extra": []}',
    ]
    return "\n".join(lines)


_ADDRESS_EN = re.compile(r"(?i)\brobot\b|" + pilot._NICKNAMES)


def check(query: str, atoms: list[dict[str, Any]], language: str = "zh") -> str | None:
    if not query:
        return "empty"
    if pilot._LEAK.search(query):
        return "prompt leak"
    if pilot._QUOTES.search(query):
        return "quotes"
    # “Luna机器人多少钱” names a product; only an address term the sources did not have is new
    address = _ADDRESS_EN if language == "en" else pilot._ADDRESS
    if len(address.findall(query)) > sum(len(address.findall(a["query"])) for a in atoms):
        return "address term"
    if language == "en":
        return None
    off_name = any(t.get("contrast", {}).get("flip") in ("typo", "unsupported") for a in atoms for t in tasks_of(a))
    if not off_name and pilot.dance_name_issue(query):
        return pilot.dance_name_issue(query)
    return None


def _grams(text: str) -> set[str]:
    chars = [c for c in text if c.isalnum()]
    grams = {a + b for a, b in zip(chars, chars[1:])}
    return grams or set(chars)


def misattributed(quotes: list[str], sources: list[str]) -> list[int]:
    """Quotes sharing fewer bigrams with their own source than with another one (1-based)."""
    bad = []
    for i, quote in enumerate(quotes):
        scores = [len(_grams(quote) & _grams(src)) for src in sources]
        if scores[i] == 0 or max(scores) > scores[i]:
            bad.append(i + 1)
    return bad


def judge_reason(verdict: Any, query: str, sources: list[str]) -> str | None:
    """Every source kept, quoted verbatim, quotes in source order (the judge's own order_ok passed reorders),
    and each quote really from its own source (the judge also quoted a swapped pair by position)."""
    n = len(sources)
    if not isinstance(verdict, dict):
        return "judge missing"
    items = {r.get("i"): r for r in verdict.get("items") or [] if isinstance(r, dict)}
    missing = [i for i in range(1, n + 1) if items.get(i, {}).get("kept") is not True]
    if missing:
        return f"judge misses {missing}"
    position = -1
    query = query.casefold()  # English judges capitalise a quote taken from mid-sentence
    for i in range(1, n + 1):
        quote = str(items[i].get("quote") or "").strip().casefold()
        found = query.find(quote, position + 1) if quote else -1
        if found < 0:
            return "quote not found" if not quote or quote not in query else "order"
        position = found
    bad = misattributed([str(items[i]["quote"]).strip() for i in range(1, n + 1)], sources)
    if bad:
        return f"quote mismatch {bad}"
    if verdict.get("extra"):
        return "judge extra"
    return None


def reorder(atoms: list[dict[str, Any]], verdict: dict[str, Any], query: str
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Put the atoms (and the judge's items) in the order their quotes appear in the merged query."""
    items = {r.get("i"): r for r in verdict.get("items") or [] if isinstance(r, dict)}
    order = sorted(range(1, len(atoms) + 1),
                   key=lambda i: query.casefold().find(str(items[i]["quote"]).strip().casefold()))
    return [atoms[i - 1] for i in order], {**verdict, "items": [{**items[i], "i": n} for n, i in enumerate(order, 1)]}


@dataclass
class Context:
    write: Callable[..., Any]
    judge: Callable[..., Any]
    bundle: Any
    sink: Callable[[str, dict[str, Any]], None] = lambda kind, row: None
    lock: threading.Lock = field(default_factory=threading.Lock)
    language: str = "zh"


def _call(fn: Callable[..., Any], prompt: str, system: str, seed: int) -> Any:
    error: Exception | None = None
    for attempt in range(2):  # malformed JSON is occasional; one retry recovers most
        try:
            return fn(prompt, system=system, seed=seed + attempt * 1000)
        except Exception as exc:  # noqa: BLE001
            error = exc
    raise RuntimeError(str(error)[:200])


def process_case(atoms: list[dict[str, Any]], ctx: Context, seed: int) -> dict[str, Any] | None:
    level = sum(len(tasks_of(a)) for a in atoms)

    def reject(query: str, reason: str, **extra: Any) -> None:
        ctx.sink("rejected", {"level": level, "sources": [a["query"] for a in atoms], "query": query,
                              "reason": reason, **extra})

    if len(atoms) == 1:
        query = atoms[0]["query"]
    else:
        try:
            reply = _call(ctx.write, merge_prompt(atoms, ctx.language), SYSTEM, seed)
        except RuntimeError as exc:
            reject("<call>", "call_error", error=str(exc))
            return None
        query = str((reply or {}).get("query") or "").strip() if isinstance(reply, dict) else ""
        reason = check(query, atoms, ctx.language)
        if reason is not None:
            reject(query, reason)
            return None
        try:
            verdict = _call(ctx.judge, judge_prompt(query, atoms), pilot.JUDGE_SYSTEM, seed)
        except RuntimeError as exc:
            reject(query, "judge_error", error=str(exc))
            return None
        reason = judge_reason(verdict, query, [a["query"] for a in atoms])
        if reason == "order":  # the writer moved a sentence (8% of tput32): label in the query's order instead
            atoms, verdict = reorder(atoms, verdict, query)
            reason = judge_reason(verdict, query, [a["query"] for a in atoms])
        if reason is not None:
            reject(query, reason, verdict=verdict)
            return None
    row = render_row(atoms, query, ctx.language, ctx.bundle)
    ctx.sink("rows", row)
    return row


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def plan_level(pool: Pool, level: int, count: int, neg_rate: float, rng: random.Random,
               taken: set[tuple[str, ...]], rel_rate: float = 0.0) -> list[list[dict[str, Any]]]:
    cases: list[list[dict[str, Any]]] = []
    for _ in range(count * 20):
        if len(cases) == count:
            break
        case = sample_case(pool, level, neg_rate, rng, rel_rate)
        key = tuple(a["id"] for a in case or [])
        if case is None or key in taken:
            continue
        taken.add(key)
        cases.append(case)
    return cases


def run(args: argparse.Namespace, ctx: Context) -> Path:
    rng = random.Random(args.seed)
    pool = load_pool(args.atoms, ctx.bundle)
    pool.units = load_units(args.relations)
    out = args.out / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False,
                                                indent=2), encoding="utf-8")
    files = {kind: (out / f"{kind}.jsonl").open("a", encoding="utf-8") for kind in ("rows", "rejected")}

    def sink(kind: str, row: dict[str, Any]) -> None:
        with ctx.lock:
            files[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[kind].flush()

    ctx.sink = sink
    # L1 is every atom (and every one-task relation unit) once; the other levels share the rest evenly.
    l1 = [[a] for a in pool.atoms + [u for u in pool.units if len(tasks_of(u)) == 1]] if 1 in args.levels else []
    others = [lv for lv in args.levels if lv != 1]
    count = args.count or (round(len(l1) * (1 - args.l1_share) / args.l1_share / len(others)) if others else 0)
    taken: set[tuple[str, ...]] = set()
    accepted: Counter[int] = Counter()
    asked: Counter[int] = Counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _round in range(args.max_rounds):
                cases = l1 if _round == 0 else []
                for level in others:
                    need = count - accepted[level]
                    if need > 0:
                        cases += plan_level(pool, level, int(need * 1.3) + 1, args.neg_rate, rng, taken,
                                            args.rel_rate)
                if not cases:
                    break
                seeds = [rng.randrange(1 << 30) for _ in cases]
                for case, row in zip(cases, executor.map(lambda c: process_case(c[0], ctx, c[1]),
                                                         zip(cases, seeds))):
                    level = sum(len(tasks_of(a)) for a in case)
                    asked[level] += 1
                    if row is not None:
                        accepted[level] += 1
    finally:
        for f in files.values():
            f.close()
    lines = [f"pool: {sum(len(r) for c in pool.positives.values() for r in c.values())} positives, "
             f"{sum(len(r) for r in pool.negatives.values())} negatives, dropped {dict(pool.dropped)}",
             f"{'level':<6} {'asked':>6} {'rows':>6}"]
    lines += [f"L{lv:<5} {asked[lv]:>6} {accepted[lv]:>6}" for lv in args.levels]
    report = "\n".join(lines)
    (out / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\noutput: {out}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--atoms", type=Path, nargs="+", required=True, help="run directories with atoms.jsonl")
    p.add_argument("--levels", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--count", type=int, help="rows per level above L1 (default: sized by --l1-share)")
    p.add_argument("--l1-share", type=float, default=0.22, help="L1 (every atom once) as a share of all rows")
    p.add_argument("--neg-rate", type=float, default=0.2, help="chance that a slot is a reply negative")
    p.add_argument("--relations", type=Path, nargs="*", default=[], help="relations.py runs with units.jsonl")
    p.add_argument("--rel-rate", type=float, default=0.15, help="chance that a slot is a relation unit")
    p.add_argument("--language", choices=("zh", "en"), default="zh", help="en: translate_atoms.py runs")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.7)
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
        language=args.language,
    )
    run(args, ctx)


if __name__ == "__main__":
    main()
