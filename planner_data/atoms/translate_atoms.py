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

"""English atoms from the clean zh pool, for ``assemble.py --language en``.

The upstream ``instructions.en`` are not usable (pinyin dance names, "Respond that there's no need to nod" for
摇头, whole translated sentences for queries), so the writer translates the user sentence *and* the zh
instruction together, with one English name per dance and gesture. Rules check names, search spans and cities;
another model family then relabels the English sentence blind (audit.review_prompt) and disagreements drop.
Typo flips are skipped: a misspelt Chinese dance name has no English equivalent.

    python -m planner_data.atoms.translate_atoms --name en_atoms --pool runs/pool-XXXX --per-cell 10
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from planner_data.atoms import audit, pilot
from planner_data.atoms.pilot import find_env_value, llm, pa


DANCE_EN = {
    "abracadabr_dance": "Abracadabra dance", "all_things_grow_dance": "All Things Grow dance",
    "ambition_dance": "Ambition dance", "apt_dance": "APT dance", "egyptian_shake": "Egyptian Shake",
    "gee": "Gee", "gentleman": "Gentleman", "go_cortis_dance": "Go Cortis dance", "idol_dance_1": "Idol Dance 1",
    "idol_dance_2": "Idol Dance 2", "karla_ok": "Karla OK", "lets_bounce": "Let's Bounce",
    "luv_each_other": "Luv Each Other", "one_and_only_dance": "One and Only dance",
    "one_spot_dance": "One Spot dance", "popping": "popping", "power_up_dance": "Power Up dance",
    "pulp_fiction_dance": "Pulp Fiction dance", "smooth_sailing_and_prosperity": "Smooth Sailing dance",
    "solo_shake": "Solo Shake", "swag_dance": "Swag dance", "sweep_kick_dance": "Sweep Kick dance",
    "victory_dance": "Victory dance", "warm_up_dance": "warm-up dance", "whatever": "Whatever",
}
GESTURE_EN = {
    "blow_kisses_multi": "blow kisses", "bow": "bow", "clap": "clap", "curtain_bow": "curtain call bow",
    "hand_heart": "raised-hands heart", "high_five": "high five", "left_hand_side_heart": "left-hand heart",
    "nod": "nod", "right_hand_side_heart": "right-hand heart", "shake_hands": "shake hands",
    "shake_head": "shake head", "this_way_please": "this-way-please gesture", "wave_greet_bye": "wave",
}
_DANCE_ZH = {name: label for name, label, _ in pa._DANCES}
_GLOSSARY = "；".join(
    [f"{_DANCE_ZH[k]} → {v}" for k, v in DANCE_EN.items()]
    + [f"{pilot.pp.GESTURE_MEANINGS[k]} → {v}" for k, v in GESTURE_EN.items()]
    + ["逐际动力 → LimX Dynamics", "Luna / Oli / TRON1 / TRON2 原样保留", "击掌 → high five（不是 clap）"]
)
_CJK = re.compile(r"[\u4e00-\u9fff]")
_SEARCH = ("web_search", "rag_query")
SYSTEM = "你把人形服务机器人规划模型的中文训练样本翻译成自然的英文。"


def _named_dance(zh_query: str) -> str | None:
    return next((name for name, pattern in pa._DANCE_RULES if pattern.search(zh_query)), None)


def prompt(atoms: list[dict[str, Any]]) -> str:
    lines = [
        "下面每条是用户当面对机器人说的一句中文，和标注里对应的一条简短指令。逐条翻译成英文：",
        "- query 译成英语母语者当面对机器人会说的自然口语，仍是用户对机器人说话；请求还是提问、现在做还是以后做、"
        "对机器人说还是对旁边的人说、背景和语气都保持不变，不改成第三人称旁白；",
        "- 数字、方向、角度、步数、城市的意思不变；舞名和手势名只用对照表里的英文，原句没说舞名的不要加；",
        "- instruction 译成简短的英文祈使短语，和译文用同一套名字；",
        "- 给了 span 的，在 span_en 里填译文中对应这段的英文，必须是你的 query 译文里一字不差的连续片段；",
        "- 给了 city 的，在 city_en 里填城市的英文名。",
        f"对照表：{_GLOSSARY}",
        "",
    ]
    for i, atom in enumerate(atoms, 1):
        item = {"i": i, "query": atom["query"], "instruction": atom["planner_target"]["instructions"]["zh"]}
        params = atom.get("expected_params") or {}
        if atom.get("expected_function") in _SEARCH:
            item["span"] = params["query"]
        if atom.get("expected_function") == "get_weather" and params.get("city") in atom["query"]:
            item["city"] = params["city"]
        lines.append(json.dumps(item, ensure_ascii=False))
    lines.append('只输出 JSON：{"items": [{"i": 1, "query": "...", "instruction": "...", "span_en": null, '
                 '"city_en": null}]}')
    return "\n".join(lines)


def build(atom: dict[str, Any], item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The English atom, or why the translation is unusable."""
    query = str(item.get("query") or "").strip()
    instruction = str(item.get("instruction") or "").strip().rstrip(".!")
    if not query or not instruction:
        return None, "empty"
    if _CJK.search(query) or _CJK.search(instruction):
        return None, "chinese left"
    tool = atom.get("expected_function")
    params = copy.deepcopy(atom.get("expected_params"))
    if tool == "RobotDance":
        named = _named_dance(atom["query"])
        if named and DANCE_EN[named].lower() not in query.lower():
            return None, "dance name"
        if not named and any(v.lower() in query.lower() for v in DANCE_EN.values() if v != "popping"):
            return None, "added dance name"
    if tool in _SEARCH:
        span = str(item.get("span_en") or "").strip()
        if not span or span not in query:
            return None, "span"
        params = {"query": span}
    if tool == "get_weather" and params["city"] in atom["query"]:
        city = str(item.get("city_en") or "").strip()
        if not city or _CJK.search(city) or city.lower() not in query.lower():
            return None, "city"
        params["city"] = city
    out = copy.deepcopy(atom)
    out.update(id=f"{atom['id']}_en", query=query, language="en", source_id=atom["id"], expected_params=params)
    out["planner_target"]["instructions"]["en"] = instruction
    return out, None


def agree(atom: dict[str, Any], verdict: dict[str, Any]) -> bool:
    if atom.get("expected_function") == "get_weather" and verdict.get("tool") == "get_weather":
        got, want = verdict.get("params") or {}, atom["expected_params"]
        norm = lambda c: {"深圳": "shenzhen"}.get(str(c), str(c)).strip().lower()  # noqa: E731
        return got.get("query_type") == want["query_type"] and norm(got.get("city")) == norm(want["city"])
    return audit.agree(atom, verdict)


def sample(atoms: list[dict[str, Any]], per_cell: int, rng: random.Random,
           done: set[str] = frozenset()) -> list[dict[str, Any]]:
    """Up to `per_cell` atoms per cell (0: all of them), leaving out ids in `done`."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        flip = (atom.get("contrast") or {}).get("flip")
        if flip == "typo" or atom["id"] in done:
            continue
        key = f"{flip}:{atom.get('negative_tool')}" if atom.get("negative_sample") else atom["atom_name"]
        groups[key].append(atom)
    return [a for key in sorted(groups)
            for a in rng.sample(groups[key], min(per_cell or len(groups[key]), len(groups[key])))]


def run(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    atoms = [json.loads(line) for line in (args.pool / "atoms.jsonl").open(encoding="utf-8") if line.strip()]
    done = {json.loads(line)["source_id"] for run_dir in args.skip
            for line in (run_dir / "atoms.jsonl").open(encoding="utf-8") if line.strip()}
    todo = sample(atoms, args.per_cell, rng, done)
    out = args.out / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    key = find_env_value("DASHSCOPE_API_KEY")
    write = functools.partial(llm.chat_json, api_key=key, model=args.model, temperature=0.3, system=SYSTEM)
    review = functools.partial(llm.chat_json, api_key=key, model=args.review_model, temperature=0.0)
    lock = threading.Lock()
    files = {k: (out / f"{k}.jsonl").open("a", encoding="utf-8") for k in ("atoms", "rejected")}
    reasons: Counter[str] = Counter()

    def sink(kind: str, row: dict[str, Any], reason: str | None = None) -> None:
        with lock:
            if reason:
                reasons[reason] += 1
            files[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[kind].flush()

    def batch(chunk: list[dict[str, Any]]) -> None:
        try:
            items = {it.get("i"): it for it in pilot._as_list(write(prompt(chunk), seed=args.seed))
                     if isinstance(it, dict)}
        except Exception as exc:  # noqa: BLE001
            for atom in chunk:
                sink("rejected", {"id": atom["id"], "reason": "call_error", "error": str(exc)[:200]}, "call_error")
            return
        built = []
        for i, atom in enumerate(chunk, 1):
            en, reason = build(atom, items.get(i) or {})
            if en is None:
                sink("rejected", {"id": atom["id"], "zh": atom["query"], "item": items.get(i), "reason": reason},
                     reason)
            else:
                built.append(en)
        if not built:
            return
        try:  # shuffled so neighbours do not give the label away
            order = rng.sample(built, len(built))
            verdicts = {v.get("i"): v for v in pilot._as_list(review(audit.review_prompt(order), seed=args.seed))
                        if isinstance(v, dict)}
        except Exception as exc:  # noqa: BLE001
            for en in built:
                sink("rejected", {"id": en["id"], "reason": "review_error", "error": str(exc)[:200]}, "review_error")
            return
        for i, en in enumerate(order, 1):
            verdict = verdicts.get(i) or {}
            if agree(en, verdict):
                sink("atoms", en, "kept")
            else:
                sink("rejected", {"id": en["id"], "zh": atom_zh[en["source_id"]], "en": en["query"],
                                  "label": audit.label(en), "review": verdict, "reason": "review"}, "review")

    atom_zh = {a["id"]: a["query"] for a in todo}
    chunks = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    rng.shuffle(chunks)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(batch, chunks))
    finally:
        for f in files.values():
            f.close()
    summary = {"pool": str(args.pool), "sampled": len(todo), "reasons": dict(reasons)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--pool", type=Path, required=True, help="compile_pool.py run with atoms.jsonl")
    p.add_argument("--per-cell", type=int, default=10,
                   help="atoms per cell (per flip and tool for negatives); 0: the whole pool")
    p.add_argument("--skip", type=Path, nargs="*", default=[], help="earlier runs whose kept atoms are not redone")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--review-model", default="deepseek-v3.2")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=pilot.RUNS_DIR)
    run(p.parse_args())


if __name__ == "__main__":
    main()
