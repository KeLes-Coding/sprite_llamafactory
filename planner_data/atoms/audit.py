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

"""Quality audit of the atom pools before scaling up.

Three passes over the current canonical runs:
1. rules: every atom re-checked with today's rules (they changed while the early runs were made);
2. stats: label balance, duplicates, one sentence with two labels, opener/template concentration;
3. review: a stratified sample relabelled blind by a different model family (deepseek), so the
   generator's own judge (qwen-max) does not grade its own work.

    python -m planner_data.atoms.audit --name audit1 --review-per-group 60
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from planner_data.atoms import contrast, pilot, query_contrast
from planner_data.atoms.pilot import _norm_text, find_env_value, llm, pa, pp


RUNS = pilot.RUNS_DIR
POOLS: dict[str, tuple[str, ...]] = {
    "action_pos": ("no_ability-1006_155239",),
    "gesture_dance_pos": ("gesture_dance-1006_170755", "wave_bare-1006_172113"),
    "action_neg": ("neg_contrast6-1006_185017", "neg_newops2-1006_202946"),
    "query_pos": ("query_smoke5-1006_212421", "query_smoke6-1006_212812", "query_full-1007_150246"),
    "query_neg": ("q_neg3-1007_142128", "q_neg_full-1007_151801"),
}
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CELLS = {c.cell_id: c for c in pilot.ALL_CELLS}


def load(pools: dict[str, tuple[str, ...]]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for group, dirs in pools.items():
        for d in dirs:
            rows += [(group, a) for a in contrast.load_atoms(RUNS / d)]
    return rows


def label(atom: dict[str, Any]) -> str:
    target = atom["planner_target"]
    if target["kind"] == "reply":
        return "reply"
    return f"{atom['expected_function']} {json.dumps(atom['expected_params'], ensure_ascii=False, sort_keys=True)}"


def _search_cell(atom: dict[str, Any]) -> pa.Cell | None:
    flip = (atom.get("contrast") or {}).get("flip")
    if flip in query_contrast.FLIPS and query_contrast.FLIPS[flip].target:
        return _CELLS[query_contrast.FLIPS[flip].target]
    return _CELLS.get(atom["atom_name"])


def rule_issues(atom: dict[str, Any]) -> list[str]:
    """Today's rules applied to one stored atom."""
    query = atom["query"]
    zh, en = (atom["planner_target"]["instructions"].get(k) or "" for k in ("zh", "en"))
    tool = atom["expected_function"]
    issues = []
    if not (query and zh and en):
        issues.append("empty field")
    if pilot._LEAK.search(query):
        issues.append("prompt leak")
    if pilot._QUOTES.search(query):
        issues.append("quotes")
    rag = tool == "rag_query" or (atom.get("negative_tool") == "rag_query")
    if (pilot._ADDRESS_RAG if rag else pilot._ADDRESS).search(query) or pa._OTHER_ASSISTANT.search(query):
        issues.append("address term")
    if _CJK.search(en):
        issues.append("chinese in instruction_en")
    if atom["planner_target"]["kind"] == "reply":
        return issues
    params = atom["expected_params"]
    if tool in pa.DERIVERS:
        derived = pa.DERIVERS[tool](query)
        if derived != params:
            issues.append(f"rule derives {derived}")
        if tool in contrast.TOOLS or tool == "HumanAction":
            if pilot._ABILITY_Q.search(query):
                issues.append("ability question")
        if tool == "RobotDance" and pilot.dance_name_issue(query):
            issues.append(str(pilot.dance_name_issue(query)))
    elif tool in pilot._SEARCH_TOOLS:
        cell = _search_cell(atom)
        reason = pilot.search_reason(cell, query, zh, str(params.get("query", "")), None) if cell else "unknown cell"
        if reason:
            issues.append(reason)
    return issues


def _openers(texts: list[str]) -> Counter[str]:
    return Counter(re.sub(r"^[\s，,。.!！?？~～…]+", "", t)[:2] for t in texts)


def stats(rows: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group, atom in rows:
        by_group[group].append(atom)
    for group, atoms in by_group.items():
        queries = [a["query"] for a in atoms]
        cells = Counter(a["atom_name"] for a in atoms)
        openers = _openers(queries)
        grams = Counter(g for q in queries for g in {pilot.shape(q)[i : i + 4] for i in range(len(pilot.shape(q)) - 3)})
        out[group] = {
            "atoms": len(atoms),
            "cells": len(cells),
            "per_cell_min_median_max": (min(cells.values()), statistics.median(cells.values()), max(cells.values())),
            "unique_norm": len({_norm_text(q) for q in queries}),
            "len_median": statistics.median(len(q) for q in queries),
            "len_p90": sorted(len(q) for q in queries)[int(0.9 * len(queries))],
            "top_openers": [(k, round(v / len(atoms), 3)) for k, v in openers.most_common(6)],
            "top_4grams": [(k, round(v / len(atoms), 3)) for k, v in grams.most_common(8)],
            "labels": Counter(a["planner_target"]["kind"] if a["planner_target"]["kind"] == "reply"
                              else a["expected_function"] for a in atoms),
        }
    seen: dict[str, set[str]] = defaultdict(set)
    where: dict[str, list[str]] = defaultdict(list)
    for group, atom in rows:
        key = _norm_text(atom["query"])
        seen[key].add(label(atom))
        where[key].append(group)
    out["duplicates_across_pools"] = sum(len(v) - 1 for v in where.values() if len(v) > 1)
    out["conflicting_labels"] = [{"query": k, "labels": sorted(v)} for k, v in seen.items() if len(v) > 1]
    return out


# ---- blind relabel by another model family -------------------------------------------------------------------

_GESTURES = "、".join(f"{k}（{v}）" for k, v in pp.GESTURE_MEANINGS.items())
_DANCES = "、".join(f"{n}（{lab}）" for n, lab, _ in pa._DANCES)
REVIEW_RULES = f"""你在审核一台人形服务机器人的规划标注。机器人只有这些能力：
- HumanAction：坐下 sit_down / 躺下 lie_down / 站起来 stand_up / 走路 walk（前后左右走 N 步，或左右转 N 度；只说“转一下”是 90 度，“稍微/一点”是 15 度，“转身/向后转”是 180 度）；
- RobotGesture 只限：{_GESTURES}；
- RobotDance 只限：{_DANCES}；没说舞名就是 warm_up_dance；
- get_weather{{city, query_type: now|24h|7d|now_7d}}（没说城市默认深圳；只说“今天”算 now；今晚/明天/某时段算 24h；这周/未来几天算 7d；同时问现在和未来几天算 now_7d）；
- robot_status{{query: 电量|日期|时间|位置|型号|当前动作}}：机器人自己此刻的状态；
- web_search{{query}}：需要联网查的时效信息（新闻、行情、航班、比赛、交通、景点开放、活动、娱乐、其他商品、海外当地时间）；
- rag_query{{query}}：逐际动力公司及其产品 Luna/Oli/TRON1/TRON2 的资料，以及机器人自己的出厂设计参数（续航、关节、身高体重、速度）；
其余一律是 reply（机器人直接用话回答）。已确定的口径：
- 对动作说“能不能/可以…吗/会不会”是能力提问 → reply；但查询类“能不能帮我查下北京天气”有具体对象 → 调对应工具；“你会查天气吗”没对象 → reply；
- 推迟或带条件的（等会儿再…、我答对了你就…）、愿望假设、对在场别人说的、评论过去的动作、让它教/讲解、问某动作的含义、让它说句话（说声谢谢）→ reply；
- 列表外的手势或舞（敬礼、天鹅湖）、舞名写错字、唱歌 → reply；讲笑话 → reply；稳定常识（长城多长）→ reply；
- 对机器人道别（拜拜/再见）→ RobotGesture wave_greet_bye；“示范一下”算让它做；
- 用户一般不称呼机器人，省略主语的话都算对它说。"""


def review_prompt(atoms: list[dict[str, Any]]) -> str:
    lines = [REVIEW_RULES, "", "逐句给出你认为正确的标注（不要参考任何已有答案），并判断这句话是否像真人会说的自然口语。句子："]
    lines += [f"{i}. {a['query']}" for i, a in enumerate(atoms, 1)]
    lines.append(
        f"按编号只输出长度为 {len(atoms)} 的 JSON 数组："
        '[{"i": 1, "tool": "HumanAction|RobotGesture|RobotDance|get_weather|robot_status|web_search|rag_query|reply", '
        '"params": {"action": "walk", "x": 1, "y": 0, "yaw": 0, "step": 2, "degree": null}'
        '（前 x=1、后 x=-1、左 y=1、右 y=-1、左转 yaw=1、右转 yaw=-1，转身时 degree 填度数）'
        ' 或 {"action": "sit_down"} 或 {"gesture": "bow"} 或 {"dance": "popping"} 或 '
        '{"city": "深圳", "query_type": "now"} 或 {"query": "电量"}（web_search/rag_query 与 reply 填 null）, '
        '"natural": true, "note": "有疑问时一句话说明，否则空"}]'
    )
    return "\n".join(lines)


def agree(atom: dict[str, Any], verdict: dict[str, Any]) -> bool:
    tool = verdict.get("tool")
    if atom["planner_target"]["kind"] == "reply":
        return tool == "reply"
    if tool != atom["expected_function"]:
        return False
    params, got = atom["expected_params"], verdict.get("params") or {}
    if tool in ("RobotGesture", "RobotDance", "robot_status"):
        return got == params
    if tool == "get_weather":
        return got.get("city") == params.get("city") and got.get("query_type") == params.get("query_type")
    if tool == "HumanAction":
        if params["action"] != "walk":
            return got.get("action") == params["action"]
        p = params["parameters"]
        sign = lambda v: (v > 0) - (v < 0)  # noqa: E731 - direction is what matters; magnitudes are conventions
        keys = ("x", "y", "yaw")
        return got.get("action") == "walk" and all(sign(float(got.get(k) or 0)) == sign(p.get(k, 0)) for k in keys) \
            and int(got.get("step") or 1) == int(p.get("step", 1)) \
            and (not p.get("yaw") or int(got.get("degree") or 90) == int(p.get("degree", 90)))
    return True  # web_search / rag_query: the tool is the label


def review(rows: list[tuple[str, dict[str, Any]]], per_group: int, model: str, seed: int, workers: int
           ) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group, atom in rows:
        by_group[group].append(atom)
    sample = [(g, a) for g, atoms in by_group.items() for a in rng.sample(atoms, min(per_group, len(atoms)))]
    rng.shuffle(sample)  # mixed batches: labels must not be guessable from the neighbours
    batches = [sample[i : i + 8] for i in range(0, len(sample), 8)]
    call = functools.partial(llm.chat_json, api_key=find_env_value("DASHSCOPE_API_KEY"), model=model, temperature=0.0)

    def run(batch: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        try:
            verdicts = pilot._as_list(call(review_prompt([a for _, a in batch]), seed=seed))
        except Exception as exc:  # noqa: BLE001
            verdicts = [{"error": str(exc)[:200]}] * len(batch)
        by_i = {v.get("i"): v for v in verdicts if isinstance(v, dict)}
        out = []
        for i, (group, atom) in enumerate(batch, 1):
            v = by_i.get(i) or {}
            out.append({"group": group, "id": atom["id"], "cell": atom["atom_name"], "query": atom["query"],
                        "label": label(atom), "flip": (atom.get("contrast") or {}).get("flip"),
                        "review": v, "agree": bool(v) and "error" not in v and agree(atom, v),
                        "natural": v.get("natural")})
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for rs in pool.map(run, batches) for r in rs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--review-per-group", type=int, default=60)
    p.add_argument("--review-model", default="deepseek-v3.2")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    out = RUNS / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    rows = load(POOLS)

    issues = [{"group": g, "id": a["id"], "cell": a["atom_name"], "query": a["query"], "label": label(a),
               "issues": rule_issues(a)} for g, a in rows]
    issues = [r for r in issues if r["issues"]]
    with (out / "rule_issues.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in issues)
    summary = stats(rows)
    summary["rule_issues"] = {g: Counter(i.split(" {")[0] for r in issues if r["group"] == g for i in r["issues"])
                              for g in POOLS}

    if args.review_per_group:
        reviewed = review(rows, args.review_per_group, args.review_model, args.seed, args.workers)
        with (out / "review.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in reviewed)
        summary["review"] = {g: {"n": sum(r["group"] == g for r in reviewed),
                                 "agree": sum(r["agree"] for r in reviewed if r["group"] == g),
                                 "unnatural": sum(r["natural"] is False for r in reviewed if r["group"] == g)}
                             for g in POOLS}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=list),
                                      encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "conflicting_labels"}, ensure_ascii=False,
                     indent=1, default=list))
    print("conflicting_labels:", len(summary["conflicting_labels"]))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
