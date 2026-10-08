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

"""Distance of the challenge/probe sets from v3.4.0 training data, and strict parameter scores.

Compares the SFT models with the qwen-flash anchor once step/degree/city/query_type are checked exactly.
Clause nearest-neighbour values (min_nn/mean_nn) can vary slightly between runs because of hash-order
tie-breaking; the bench/strict scores are deterministic.

usage: PYTHONPATH=. .venv/bin/python planner_val/analysis/v340_generalization.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from planner_val.challenge_eval import load_challenge_cases
from planner_val.scoring import _extract_array_text, parse_plan, score_benchmark_compatible


RUNS = Path("/home/limx/Workspace/data_gen/261005/runs")
EXPORT = RUNS / "export_v340/v3-4-0-74a63ca3"
REF = EXPORT / "action_sequence_sft_validation_1k.jsonl"
RES = Path("planner_val/results")
SETS = {
    "challenge_50": (Path("planner_val/challenges/challenge_50.json"), {
        "270M": "gemma3-270m-lora-v340-challenge_50-20261008-100649",
        "0.6B": "qwen3-0.6b-lora-v340-challenge_50-20261008-101101",
        "qwen": "qwen-flash-challenge_50-20261008-150408",
    }),
    "challenge_v2": (Path("planner_val/challenges/challenge_v2.json"), {
        "270M": "gemma3-270m-lora-v340-challenge_v2-20261008-100649",
        "0.6B": "qwen3-0.6b-lora-v340-challenge_v2-20261008-101101",
        "qwen": "qwen-flash-challenge_v2-20261008-101414",
    }),
    "challenge_v3": (Path("planner_val/challenges/challenge_v3.json"), {
        "270M": "gemma3-270m-lora-v340-challenge_v3-20261008-103843",
        "0.6B": "qwen3-0.6b-lora-v340-challenge_v3-20261008-104104",
        "qwen": "qwen-flash-challenge_v3-20261008-103632",
    }),
    "probe_v1": (Path("planner_val/challenges/probe_v1.json"), {
        "270M": "gemma3-270m-lora-v340-probe_v1-20261008-144750",
        "0.6B": "qwen3-0.6b-lora-v340-probe_v1-20261008-144523",
        "qwen": "qwen-flash-probe_v1-20261008-145956",
    }),
}
CONFIGS = ["270M", "0.6B", "qf-v340", "qf-prod", "qf-prod-adj"]

# ---------------------------------------------------------------- training data
SPLIT = re.compile(r"[，。！？；,.!?;、\n:：]+|然后|接着|再|and then|then|also")
NORM = re.compile(r"[\s\W_]+", re.UNICODE)


def bigrams(text):
    t = NORM.sub("", text.lower())
    return {t[i : i + 2] for i in range(len(t) - 1)} or ({t} if t else set())


def clauses(text):
    return [c for c in (s.strip() for s in SPLIT.split(text)) if len(NORM.sub("", c)) >= 2]


def sig(tasks):
    out = []
    for t in tasks:
        if t.get("kind") == "reply":
            out.append(("reply",))
            continue
        a = t.get("arguments") or {}
        name = t.get("name")
        if name == "HumanAction":
            p = a.get("parameters") or {}
            d = tuple((k, (v > 0) - (v < 0)) for k, v in sorted(p.items()) if k in ("x", "y", "yaw"))
            out.append((name, a.get("action"), d))
        elif name in ("RobotGesture", "RobotDance"):
            out.append((name, a.get("gesture") or a.get("dance")))
        elif name == "robot_status":
            out.append((name, a.get("query")))
        elif name == "get_weather":
            out.append((name, a.get("query_type")))
        else:
            out.append((name,))
    return tuple(out)


def merged(s):  # adjacent reply slots collapse, matching the benchmark's reply handling
    out = []
    for x in s:
        if not (out and x == ("reply",) and out[-1] == ("reply",)):
            out.append(x)
    return tuple(out)


train_sigs, train_steps, train_degrees, train_cities = Counter(), Counter(), Counter(), Counter()
train_text = []
for line in (EXPORT / "action_sequence_sft_train.jsonl").open(encoding="utf-8"):
    msgs = json.loads(line)["messages"]
    user = json.loads(next(m["content"] for m in msgs if m["role"] == "user"))
    tasks = json.loads(next(m["content"] for m in msgs if m["role"] == "assistant"))
    train_text.append(user["instruction"])
    train_sigs[merged(sig(tasks))] += 1
    for t in tasks:
        a = t.get("arguments") or {}
        p = a.get("parameters") or {}
        if "step" in p:
            train_steps[p["step"]] += 1
        if "degree" in p:
            train_degrees[p["degree"]] += 1
        if "city" in a:
            train_cities[a["city"]] += 1

atom_text = [json.loads(l)["query"] for l in (RUNS / "pool-1007_152342/atoms.jsonl").open(encoding="utf-8")]
for d in ("en_atoms-1007_152431", "en_full-1007_173915", "en_full2-1007_174856"):
    atom_text += [json.loads(l)["query"] for l in (RUNS / d / "atoms.jsonl").open(encoding="utf-8")]
atom_bg = [bigrams(c) for t in atom_text for c in (clauses(t) or [t])]
index = defaultdict(list)
for i, bg in enumerate(atom_bg):
    for g in bg:
        index[g].append(i)
train_set = set(train_text)


def clause_nn(clause):
    q = bigrams(clause)
    hits = Counter()
    for g in q:
        for i in index.get(g, ()):
            hits[i] += 1
    best = 0.0
    for i, inter in hits.most_common(200):
        best = max(best, inter / len(q | atom_bg[i]))
    return best


def norm_city(c):
    return str(c or "").lower().replace("市", "").strip()


# ---------------------------------------------------------------- scoring helpers
def adjust_prod(raw):
    """Map production-schema differences onto the v3.4.0 contract: drop weather coordinates, expand repeat=N."""
    text = _extract_array_text(raw or "")
    try:
        tasks = json.loads(text)
    except Exception:
        return raw
    out = []
    for t in tasks if isinstance(tasks, list) else []:
        a = t.get("arguments") if isinstance(t, dict) else None
        if isinstance(a, dict):
            a.pop("latitude", None)
            a.pop("longitude", None)
            n = a.get("repeat")
            if t.get("name") in ("RobotGesture", "RobotDance") and isinstance(n, int) and n > 1:
                a["repeat"] = 1
                out += [json.loads(json.dumps(t)) for _ in range(n)]
                continue
        out.append(t)
    return json.dumps(out, ensure_ascii=False)


def strict_errors(case, raw):
    """Exact-value checks the benchmark skips: step, degree, city, query_type."""
    parsed = parse_plan(raw, case.contract)
    exp = [t for t in case.eval_case.expected if t.kind != "reply"]
    act = [t for t in parsed.tasks if t.kind != "reply"]
    errs = []
    for e, a in zip(exp, act):
        ea, aa = dict(e.arguments or {}), dict(a.arguments or {})
        if e.name == "HumanAction":
            ep, ap = ea.get("parameters") or {}, aa.get("parameters") or {}
            for k in ("step", "degree"):
                if k in ep and ep.get(k) != ap.get(k):
                    errs.append(f"{k} {ap.get(k)}!={ep.get(k)}")
        if e.name == "get_weather":
            ec, ac = norm_city(ea.get("city")), norm_city(aa.get("city"))
            if ec and not (ec in ac or ac in ec) or (ec and not ac):
                errs.append(f"city {aa.get('city')}!={ea.get('city')}")
            if "query_type" in ea and ea["query_type"] != aa.get("query_type"):
                errs.append(f"qt {aa.get('query_type')}!={ea['query_type']}")
    return errs


def load_rows(run):
    rows = defaultdict(dict)
    for line in (RES / run / "results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        m = r["model"]
        key = "qf-v340" if m == "qwen-flash-v340-prompt" else "qf-prod" if m.startswith("qwen-flash-prod") else None
        rows[key or m][r["case_id"]] = r
    return rows


# ---------------------------------------------------------------- main
report = {}
for set_name, (path, runs) in SETS.items():
    cases = {c.case_id: c for c in load_challenge_cases(path, REF)}
    raw_cases = {c["case_id"]: c for c in json.loads(path.read_text(encoding="utf-8"))["cases"]}
    by_cfg = {}
    for short, run in runs.items():
        rows = load_rows(run)
        for model, rs in rows.items():
            cfg = short if short != "qwen" else model
            by_cfg[cfg] = rs
    by_cfg["qf-prod-adj"] = {}
    for cid, r in by_cfg["qf-prod"].items():
        raw = adjust_prod(r["output"])
        s = score_benchmark_compatible(cases[cid].eval_case, raw, cases[cid].contract)
        by_cfg["qf-prod-adj"][cid] = {"benchmark_compatible_pass": s.passed, "output": raw,
                                      "benchmark_compatible_reason": s.reason}
    items = []
    for cid, case in cases.items():
        q = case.query
        cl = clauses(q) or [q]
        nns = [clause_nn(c) for c in cl]
        exp = raw_cases[cid]["expected"]
        s = merged(sig(exp))
        vals_unseen = []
        for t in exp:
            a = t.get("arguments") or {}
            p = a.get("parameters") or {}
            if "step" in p and p["step"] not in train_steps:
                vals_unseen.append(f"step={p['step']}")
            if "degree" in p and p["degree"] not in train_degrees:
                vals_unseen.append(f"degree={p['degree']}")
            if "city" in a and a["city"] not in train_cities:
                vals_unseen.append(f"city={a['city']}")
        item = {
            "id": cid, "query": q, "cat": raw_cases[cid].get("category"),
            "min_nn": min(nns), "mean_nn": sum(nns) / len(nns), "exact_in_train": q in train_set,
            "sig_seen": train_sigs.get(s, 0), "vals_unseen": vals_unseen, "n_tools": sum(x != ("reply",) for x in s),
        }
        for cfg in CONFIGS:
            r = by_cfg[cfg][cid]
            bench = bool(r["benchmark_compatible_pass"])
            se = strict_errors(case, r["output"]) if bench else []
            item[cfg] = {"bench": bench, "strict": bench and not se, "strict_err": se,
                         "reason": r["benchmark_compatible_reason"]}
        items.append(item)
    report[set_name] = items

json.dump(report, open(RES / "v340_generalization.json", "w"), ensure_ascii=False, indent=1)


def pct(n, d):
    return f"{n}/{d}" if d else "-"


val_q = []
for line in REF.open(encoding="utf-8"):
    msgs = json.loads(line)["messages"]
    val_q.append(json.loads(next(m["content"] for m in msgs if m["role"] == "user"))["instruction"])
vn = sorted(min(clause_nn(c) for c in (clauses(q) or [q])) for q in val_q[:300])
print(f"val_1k[:300] min-clause NN: p25 {vn[75]:.2f} median {vn[150]:.2f} p75 {vn[225]:.2f}")


def mcnemar(a, b):
    from math import comb

    n = a + b
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


for set_name, items in report.items():
    n = len(items)
    print(f"\n######## {set_name}  n={n}")
    mn = sorted(i["min_nn"] for i in items)
    print(f"min-clause NN to atoms: p25 {mn[n // 4]:.2f} median {mn[n // 2]:.2f} p75 {mn[3 * n // 4]:.2f};"
          f" exact-in-train {sum(i['exact_in_train'] for i in items)}; plan-signature seen in train"
          f" {sum(i['sig_seen'] > 0 for i in items)}/{n}; unseen values {sum(bool(i['vals_unseen']) for i in items)}")
    print("            " + "  ".join(f"{c:>12}" for c in CONFIGS))
    print("bench       " + "  ".join(f"{pct(sum(i[c]['bench'] for i in items), n):>12}" for c in CONFIGS))
    print("strict-par  " + "  ".join(f"{pct(sum(i[c]['strict'] for i in items), n):>12}" for c in CONFIGS))
    buckets = {
        "NN>=0.8": lambda i: i["min_nn"] >= 0.8,
        "0.5<=NN<0.8": lambda i: 0.5 <= i["min_nn"] < 0.8,
        "NN<0.5": lambda i: i["min_nn"] < 0.5,
        "sig seen": lambda i: i["sig_seen"] > 0,
        "sig unseen": lambda i: i["sig_seen"] == 0,
        "vals unseen": lambda i: bool(i["vals_unseen"]),
    }
    for label, f in buckets.items():
        sub = [i for i in items if f(i)]
        print(f"{label:<12}" + "  ".join(f"{pct(sum(i[c]['strict'] for i in sub), len(sub)):>12}" for c in CONFIGS)
              + "   (strict)")
    for x, y in (("0.6B", "270M"), ("0.6B", "qf-prod-adj"), ("270M", "qf-prod-adj"), ("0.6B", "qf-v340")):
        a = sum(i[x]["strict"] and not i[y]["strict"] for i in items)
        b = sum(i[y]["strict"] and not i[x]["strict"] for i in items)
        print(f"  McNemar strict {x} vs {y}: {a} vs {b}  p={mcnemar(a, b):.3g}")
    cats = sorted({i["cat"] for i in items})
    if len(cats) <= 12:
        for cat in cats:
            sub = [i for i in items if i["cat"] == cat]
            print(f"  {cat:<22}" + "  ".join(
                f"{sum(i[c]['bench'] for i in sub)}/{sum(i[c]['strict'] for i in sub)}".rjust(10) for c in CONFIGS)
                + f"   n={len(sub)} (bench/strict)")
