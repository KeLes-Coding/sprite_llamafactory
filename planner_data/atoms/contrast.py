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

"""Contrast negatives: rewrite accepted positive atoms into minimal-pair reply atoms.

Each positive sentence is edited as little as possible so that its label flips to
``reply``: an ability question, a stop/negation, a comment on a past performance,
words meant for someone else, a wish, or the same request for a gesture/dance the
robot does not have. Scene, tone and noise are inherited from the positive, so the
negatives are as varied as the positives they come from.

    python -m planner_data.atoms.contrast --name gd_neg --source runs/gesture_dance-1006_170755 --per-cell 12
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
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from planner_data.atoms import pilot
from planner_data.atoms.pilot import _norm_text, find_env_value, llm, pa, pp


# Items the robot cannot perform, with the pattern the rewritten sentence must match;
# none of them matches a supported gesture/dance rule (user decision 2026-10-06: reply, no nearest match).
UNSUPPORTED: dict[str, tuple[tuple[str, str], ...]] = {
    "RobotGesture": (
        ("敬礼", r"敬.{0,2}礼"), ("竖大拇指", r"大拇指"), ("抱拳", r"抱.{0,2}拳"), ("作揖", r"作.{0,2}揖"),
        ("拥抱", r"拥抱|抱抱|抱一下"), ("比耶", r"比.{0,2}耶|剪刀手"), ("比个OK", r"ok"), ("举手", r"举.{0,2}手"),
        ("叉腰", r"叉.{0,2}腰"), ("握拳加油", r"握.{0,2}拳"), ("摊手", r"摊.{0,2}手"), ("耸肩", r"耸.{0,2}肩"),
        ("捂脸", r"捂.{0,2}脸"), ("嘘的手势", r"嘘"),
    ),
    "RobotDance": (
        ("天鹅湖", r"天鹅湖"), ("广场舞", r"广场舞"), ("华尔兹", r"华尔兹"), ("探戈", r"探戈"), ("芭蕾", r"芭蕾"),
        ("恰恰", r"恰恰"), ("小苹果", r"小苹果"), ("科目三", r"科目三"), ("极乐净土", r"极乐净土"),
        ("最炫民族风", r"最炫民族风"), ("江南Style", r"江南\s*style"), ("秧歌", r"秧歌"), ("肚皮舞", r"肚皮舞"),
        ("踢踏舞", r"踢踏舞"), ("女团舞3", r"女团舞\s*(?:3|三)"),
    ),
}
TOOLS = tuple(UNSUPPORTED)

# "你会鞠躬吗" is an ability question too, although pilot's filter only needs the 能/可以 forms.
ABILITY = re.compile(pilot._ABILITY_Q.pattern + r"|会[^，。！？,.!?]{0,14}[吗嘛么]")
_DEMO = re.compile(r"示范|演示|做给|做一遍|做一下|来一个|来一下|看")
_HAND = re.compile(r"挥|招|摆|摇|手")


@dataclass(frozen=True)
class Flip:
    how: str  # rewrite instruction; {kind}/{item}/{hint}/{name} are filled in
    reply: str  # instruction_zh example
    marker: re.Pattern[str] | None = None  # the rewrite must contain it
    keeps_item: bool = True  # the original gesture/dance stays in the sentence
    tools: tuple[str, ...] = ("RobotGesture", "RobotDance")
    named_dance: bool = False  # only for sources that name a dance


FLIPS: dict[str, Flip] = {
    "ability": Flip("改成问你能不能、会不会、可不可以做这件事（能力提问），不是让你现在做", "回答是否会{item}", ABILITY),
    "stop": Flip("改成叫你别做、不用做或停下（用“别、不要、不用、停、算了”之类的词）", "回应不用{item}",
                 re.compile(r"别|不要|不用|甭|停|算了|不必")),
    "comment": Flip("改成{hint}，不提新的要求", "回应对{item}的评价"),
    "others": Flip("改成对在场的{hint}说、让对方去做，不再是让你做（句中点明对方，别只在句首加称呼）",
                   "回应用户对别人说的话"),
    "wish": Flip("改成假设或愿望（如“要是……就好了”“如果……”），不直接提要求", "回应想看{item}的愿望"),
    # User decision 2026-10-06: a request for later ("等客人来了再鞠躬") is a reply, not an action now.
    "deferred": Flip("改成{hint}再做，不是让你现在做", "回应稍后再{item}的安排",
                     re.compile(r"等|待会|一会|过会|稍后|晚点|晚些|明天|下次|下回|之后|以后|回头(?!见)|了.{0,8}再|后再")),
    "unsupported": Flip("把其中的{kind}换成{item}，其余尽量不变，仍是让你现在做", "说明{cannot}{item}",
                        keeps_item=False),
    # User decision 2026-10-06: singing is a reply that says it cannot sing; a joke is a reply that tells one.
    "sing": Flip("把其中的{kind}换成唱歌（{hint}，别用原来的舞名），其余尽量不变，仍是让你现在唱",
                 "说明唱不了歌", re.compile(r"唱|歌"), keeps_item=False),
    "joke": Flip("把其中的{kind}换成{hint}，其余尽量不变，仍是让你现在讲", "讲一个笑话",
                 re.compile(r"笑话|段子"), keeps_item=False),
    # User decision 2026-10-06: a misspelt dance name is a reply (not the nearest dance).
    "typo": Flip("把舞名{name}里的一个字改成同音或形近的错字，句子其他字一个都不改，仍是让你现在跳",
                 "说明不会跳{typo_name}", keeps_item=False, tools=("RobotDance",), named_dance=True),
    "mention": Flip("改成{hint}（问的是这个{kind}本身），不是让你做；原句里让你做、让人看的话都删掉", "回答关于{item}的问题",
                    re.compile(r"什么|怎么|哪|为啥|多少|吗|[？?]")),
    # User decision 2026-10-06: "示范一下" is an action; "教我/讲讲" is a reply.
    "teach": Flip("改成让你口头教用户怎么做，或讲讲做这个{kind}的要领、讲究（是讲解，不是让你示范、做给他看；"
                  "别写成“能不能/可以……吗”的问句）",
                  "讲解怎么{item}", re.compile(r"教|讲|说说|怎么|要领|要点|讲究|诀窍|技巧|注意")),
    # User decision 2026-10-06: asking it to say something ("跟大家说声再见") is a reply only, no gesture.
    "say": Flip("改成让你{hint}，不再提原来的动作", "{hint}", re.compile(r"说|道|讲"),
                keeps_item=False, tools=("RobotGesture",)),
    # User decision 2026-10-06: a condition not met yet ("我答对了你就跳") is a reply, like deferred.
    "conditional": Flip("改成带条件的约定（{hint}……就做），条件现在还没发生", "回应条件满足后再{item}的约定",
                        re.compile(r"就|的话|只要|如果|要是|假如|万一")),
}

# Rotated per item so the rewrites do not collapse into one template ("小朋友们，……" / "你刚才……真好").
HINTS: dict[str, tuple[str, ...]] = {
    "comment": ("夸你刚才做的这个动作", "吐槽你刚才做的这个动作不太标准", "回忆你以前做过这个动作",
                "跟旁人聊起你做这个动作的样子", "问你刚才做这个动作时的情况",
                "讲自己昨天在别处看到有人做这个动作", "说起自己以前学这个动作的事"),
    # No "宝贝"/"新来的同事": the judge reads them as names for the robot itself.
    "others": ("小朋友们", "王老师", "各位同学", "张经理", "旁边那位师傅", "你们几个", "小李", "爷爷", "隔壁桌的阿姨"),
    "deferred": ("等客人到了", "等活动开始", "等音乐响起来", "待会儿", "一会儿", "明天", "下次", "等用户把手头的事忙完"),
    "sing": ("不说歌名", "不说歌名", "生日快乐歌", "儿歌", "月亮代表我的心", "两只老虎", "甜蜜蜜", "茉莉花", "一首英文歌"),
    "joke": ("讲个笑话", "讲个冷笑话", "说个段子", "讲个适合小朋友的笑话", "来个笑话"),
    "mention": ("问它是什么意思", "问它用英文怎么说", "问它的来历或出处", "问它一般在什么场合做"),
    # No "说声你好": "打个招呼" plans a wave.
    "say": ("说声再见", "说声谢谢", "说句欢迎光临", "说句辛苦了", "道声晚安", "说声对不起"),
    "conditional": ("我答对了你就", "只要有人进门你就", "要是小朋友表现好你就", "如果音乐停了你就",
                    "你要是赢了就"),
}


@dataclass(frozen=True)
class Item:
    source: dict[str, Any]
    flip: str
    target: str = ""  # unsupported item shown to the writer
    core: str = ""  # pattern the rewrite must match
    hint: str = ""  # rotated variant for comment/others

    @property
    def tool(self) -> str:
        return str(self.source["expected_function"])

    @property
    def params(self) -> dict[str, Any]:
        return self.source["expected_params"]


def item_label(source: dict[str, Any]) -> str:
    zh = str(source["planner_target"]["instructions"]["zh"])
    return zh[1:] if zh.startswith("做") else zh


def select(atoms: list[dict[str, Any]], per_cell: int, rng: random.Random,
           only: list[str] | None = None) -> list[Item]:
    """Sample positives per cell and spread the flips (and unsupported items) evenly over them."""
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        if atom.get("expected_function") in TOOLS and not atom.get("negative_sample"):
            if atom["expected_function"] == "RobotDance" and pilot.dance_name_issue(atom["query"]):
                continue  # accepted before the spelling rule; not a positive any more
            by_cell[atom["atom_name"]].append(atom)
    targets = {tool: rng.sample(items, len(items)) for tool, items in UNSUPPORTED.items()}
    hints = {flip: rng.sample(pool, len(pool)) for flip, pool in HINTS.items()}
    used: Counter[str] = Counter()
    out: list[Item] = []
    for cell_id in sorted(by_cell):
        rows = rng.sample(by_cell[cell_id], min(per_cell, len(by_cell[cell_id])))
        tool = rows[0]["expected_function"]
        usable = [name for name, f in FLIPS.items() if (only is None or name in only)
                  and tool in f.tools and not (f.named_dance and cell_id == "dance_unnamed")]
        if not usable:
            continue
        flips = rng.sample(usable, len(usable))
        for i, src in enumerate(rows):
            flip = flips[i % len(flips)]
            if flip == "unsupported":
                tool = src["expected_function"]
                target, core = targets[tool][used[tool] % len(targets[tool])]
                used[tool] += 1
                out.append(Item(src, flip, target, core))
            elif flip in hints:
                out.append(Item(src, flip, hint=hints[flip][used[flip] % len(hints[flip])]))
                used[flip] += 1
            else:
                out.append(Item(src, flip))
    return out


def requirement(item: Item) -> str:
    flip = FLIPS[item.flip]
    dance = item.tool == "RobotDance"
    how = flip.how.format(kind="舞" if dance else "手势", item=item.target, hint=item.hint,
                          name=pa._DANCE_LABELS.get(item.params.get("dance", ""), ""))
    if item.flip == "typo":  # the writer copies any placeholder literally; the name comes back in its own field
        return f"{how}（另输出 typo_name：句中写错后的舞名，原样照抄；instruction_zh 和 instruction_en 都留空）"
    reply = flip.reply.format(item=item.target if item.flip == "unsupported" else item_label(item.source),
                              cannot="不会跳" if dance else "做不了", hint=item.hint)
    return f"{how}（instruction_zh 示例：{reply}）"


SYSTEM = "你为一台人形服务机器人的规划模型改写训练数据：把用户的一句话按要求做最小改动，改出意思不同的新话。"


def build_prompt(items: list[Item]) -> str:
    lines = [
        "下面每句是用户当面对机器人说的话。按每句的要求改写成一句新话：只改必要的部分，"
        "保留原句的场景、口吻、口语噪声和大致长度。",
        "硬性要求：",
        "- 不写“机器人”，不给它起名字或称呼，需要指代时最多用“你”；",
        "- 不加引号或书名号；",
        "- 除要求里写明“仍是让你现在……”的以外，改写后不能再是让你现在做这个动作；",
        "- 每句同时给出 instruction_zh（“回答……”“回应……”“说明……”“讲……”形式的简洁任务，参照要求后的示例）"
        "和 instruction_en（instruction_zh 的自然英文）。",
        "句子：",
    ]
    for i, item in enumerate(items, 1):
        lines += [f"{i}. 原句：{item.source['query']}", f"   要求：{requirement(item)}"]
    lines.append('按编号顺序只输出 JSON 数组：[{"i": 1, "query": "...", "instruction_zh": "...", "instruction_en": "..."}]'
                 '（要求里让另输出 typo_name 的，再加 "typo_name" 字段）')
    return "\n".join(lines)


def check(item: Item, query: str, zh: str, en: str) -> str | None:
    """Rule rejection reason for one rewrite, or None."""
    if not (query and zh and en):
        return "empty field"
    if pilot._LEAK.search(query):
        return "prompt leak"
    if pilot._QUOTES.search(query):
        return "quotes"
    if pilot._ADDRESS.search(query) or pa._OTHER_ASSISTANT.search(query):
        return "address term"
    if not zh.startswith(("回答", "回应", "说", "讲", "道")):
        return "reply instruction style"
    if item.flip == "ability":
        if not ABILITY.search(query):
            return "no ability marker"
    elif ABILITY.search(query):
        return "ability question"
    marker = FLIPS[item.flip].marker
    if marker is not None and not marker.search(query):
        return f"no {item.flip} marker"
    if item.flip == "teach" and _DEMO.search(query):
        return "demonstration"
    derived = pa.DERIVERS[item.tool](query)
    if item.flip == "say":  # "说声再见" hits the wave rule; only a waved hand makes it a gesture
        if derived is None or (str(derived.get("gesture")).startswith("wave") and not _HAND.search(query)):
            return None
        return f"query derives {derived}"
    if FLIPS[item.flip].keeps_item:
        if derived != item.params:
            return f"query derives {derived}"
        return pilot.dance_name_issue(query) if item.tool == "RobotDance" else None
    if item.flip == "typo":  # "顺才神" still hits the loose rule, "机械午" hits none
        name = re.search(r"跳(.+?)[。！!.]?$", zh)
        if not name or name.group(1) not in query:
            return "instruction misses the written name"
        if derived == item.params:
            return None if pilot.dance_name_issue(query) else "dance name spelt right"
        return None if derived in (None, {"dance": pa._GENERIC_DANCE}) else f"query derives {derived}"
    if item.flip == "unsupported":
        if not re.search(item.core, query, re.IGNORECASE):
            return "misses unsupported item"
        return None if derived in (None, {"dance": pa._GENERIC_DANCE}) else f"query derives {derived}"
    return None if derived is None else f"query derives {derived}"  # sing / joke: no dance or gesture left


def judge_reason(item: Item, verdict: Any) -> str | None:
    """The judge must see no request at all, or (unsupported) a request for nothing on the list."""
    if not isinstance(verdict, dict):
        return "judge missing"
    request = verdict.get("request")
    if item.flip == "typo":  # the judge may read through the typo; it only has to stay a request
        return None if request is True else "judge not request"
    if item.flip != "unsupported":
        return None if request is False else "judge says request"
    if request is not True:
        return "judge not request"
    if item.tool == "RobotGesture":
        value = verdict.get("gesture")
        return None if value in (None, "", "none") else f"judge maps to {value}"
    value = verdict.get("dance")
    return None if value == "other" else f"judge maps to {value}"


_CELLS = {c.cell_id: c for c in pa.CELLS}


def build_atom(item: Item, query: str, zh: str, en: str, writer: str) -> dict[str, Any]:
    src_cell = _CELLS[item.source["atom_name"]]
    cell = replace(src_cell, cell_id=f"neg_{item.flip}_{src_cell.cell_id}", params=None, slot=None, reply=True)
    atom = pa._atom(cell, query, zh, en, None)
    atom["curation"].update(writer=f"model:{writer}", validator="rules+judge")
    atom["contrast"] = {"flip": item.flip, "source_id": item.source["id"], "source_query": item.source["query"],
                        "unsupported": item.target or None}
    return atom


@dataclass
class Context:
    write: Callable[..., Any]
    judge: Callable[..., Any]
    bundle: Any
    seen: set[str]
    writer: str = llm.DEFAULT_MODEL
    sink: Callable[[str, dict[str, Any]], None] = lambda kind, row: None
    lock: threading.Lock = field(default_factory=threading.Lock)


def process_batch(items: list[Item], ctx: Context, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def reject(item: Item, query: str, reason: str, **extra: Any) -> None:
        row = {"flip": item.flip, "cell": item.source["atom_name"], "source": item.source["query"],
               "query": query, "reason": re.sub(r"\s*(\{.*|None)$", "", reason), "detail": reason, **extra}
        rejected.append(row)
        ctx.sink("rejected", row)

    rows: list[Any] = []
    try:
        for attempt in range(2):  # malformed JSON from the writer is occasional; one retry recovers most batches
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
    candidates: list[tuple[Item, str, dict[str, Any]]] = []
    for idx, item in enumerate(items, 1):
        row = by_i.get(idx) or (rows[idx - 1] if idx - 1 < len(rows) and isinstance(rows[idx - 1], dict) else {})
        query, zh, en = (str(row.get(k) or "").strip() for k in ("query", "instruction_zh", "instruction_en"))
        if item.flip == "typo":
            name = str(row.get("typo_name") or "").strip()
            zh = FLIPS["typo"].reply.format(typo_name=name) if name else ""
            en = f"Explain that you cannot dance {name}" if name else ""
        reason = check(item, query, zh, en)
        atom = None
        if reason is None:
            atom = build_atom(item, query, zh, en, ctx.writer)
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
            reject(item, query, reason or "", instruction_zh=zh)
            continue
        candidates.append((item, query, atom))

    # One query per judge call: in a mixed batch of mostly non-requests the judge drifts to request=false
    # (13 of 14 rejected unsupported dances were requests when judged alone).
    for item, query, atom in candidates:
        verdicts, error = None, ""
        for attempt in range(2):  # single-query calls occasionally fail; one retry
            try:
                verdicts = pilot._as_list(ctx.judge(pilot.judge_prompt(item.tool, [query]),
                                                    system=pilot.JUDGE_SYSTEM, seed=seed + attempt * 1000))
                break
            except Exception as exc:  # noqa: BLE001 - an unreviewed rewrite is dropped, not accepted
                error = str(exc)[:200]
        verdict = verdicts[0] if verdicts else None
        reason = "judge_error" if verdicts is None else judge_reason(item, verdict)
        if reason is not None:
            reject(item, query, reason, verdict=verdict, **({"error": error} if verdicts is None else {}))
            continue
        accepted.append(atom)
        ctx.sink("atoms", atom)
    return accepted, rejected


def load_atoms(source: Path) -> list[dict[str, Any]]:
    with (source / "atoms.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(items: list[Item], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> str:
    asked = Counter(i.flip for i in items)
    ok = Counter(a["contrast"]["flip"] for a in accepted)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rejected:
        reasons[r["flip"]][r["reason"]] += 1
    lines = [f"{'flip':<12} {'asked':>5} {'valid':>5}  top rejects"]
    for flip in FLIPS:
        top = ", ".join(f"{k}:{v}" for k, v in reasons[flip].most_common(3))
        lines.append(f"{flip:<12} {asked[flip]:>5} {ok[flip]:>5}  {top}")
    lines.append(f"{'TOTAL':<12} {sum(asked.values()):>5} {len(accepted):>5}")
    return "\n".join(lines)


def run(args: argparse.Namespace, ctx: Context) -> Path:
    rng = random.Random(args.seed)
    atoms = load_atoms(args.source)
    if args.cells:
        atoms = [a for a in atoms if a["atom_name"] in args.cells]
    items = select(atoms, args.per_cell, rng, args.flips)
    ctx.seen |= {_norm_text(a["query"]) for a in atoms}
    out = args.out / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False,
                                                indent=2), encoding="utf-8")
    files = {kind: (out / f"{kind}.jsonl").open("a", encoding="utf-8") for kind in ("atoms", "rejected")}

    def sink(kind: str, row: dict[str, Any]) -> None:  # written as batches finish, so a run can be read live
        with ctx.lock:
            files[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[kind].flush()

    ctx.sink = sink
    rng.shuffle(items)
    by_tool: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_tool[item.tool].append(item)
    batches = [rows[i : i + args.batch] for rows in by_tool.values() for i in range(0, len(rows), args.batch)]
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
    p.add_argument("--source", type=Path, required=True, help="pilot run directory with atoms.jsonl")
    p.add_argument("--cells", nargs="+", default=None, help="only these source cells")
    p.add_argument("--flips", nargs="+", choices=list(FLIPS), default=None, help="only these rewrites")
    p.add_argument("--per-cell", type=int, default=12, help="positives rewritten per source cell")
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
    ctx = Context(
        write=functools.partial(llm.chat_json, api_key=api_key, model=args.model, temperature=args.temperature),
        judge=functools.partial(llm.chat_json, api_key=api_key, model=args.judge_model, temperature=0.0),
        bundle=pp.load_contract_bundle(pp.CONTRACT_V2_PATH),
        seen=set(),
        writer=args.model,
    )
    run(args, ctx)


if __name__ == "__main__":
    main()
