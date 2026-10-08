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

"""Compare pilot runs on diversity that matters, not just bigram novelty.

    python -m planner_data.atoms.analyze runs/all_dims-* runs/baseline-*

Per run:
- valid / novel@0.5 (whole sentence, inflated by scene padding)
- core: distinct shapes of the clause that carries the action (wrapping removed)
- kernel: distinct action phrasings (words right after the direction / posture)
- leak: prompt instructions copied into the sentence
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from planner_data.atoms import pilot


LEAK = re.compile(r"一口气说完|没有标点|语气词|错别字|同音|重复的词|口语噪声|说话人|不同场景")
_CLAUSE = re.compile(r"[，。！？、,.!?；;\s…]+")
_KERNEL = re.compile(r"[前后左右][^，。！？、,.!?\s]{0,3}|坐[^，。！？\s]{0,2}|躺[^，。！？\s]{0,2}|站[^，。！？\s]{0,2}|转[^，。！？\s]{0,3}")


def core(atom: dict) -> str | None:
    hits = [c for c in _CLAUSE.split(atom["query"]) if c and pilot.pa.derive_human_action(c) == atom["expected_params"]]
    return pilot.shape(hits[-1]) if hits else None


def kernel(atom: dict) -> str:
    return "|".join(pilot.shape(m) for m in _KERNEL.findall(atom["query"]))


def analyze(run: Path) -> dict:
    atoms = [json.loads(line) for line in (run / "atoms.jsonl").open(encoding="utf-8")]
    by_cell = defaultdict(list)
    for atom in atoms:
        by_cell[atom["atom_name"]].append(atom)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return {
        "run": run.name,
        "raw": summary["totals"].get("raw", 0),
        "valid": len(atoms),
        "novel@0.5": summary["totals"].get("novel@0.5", 0),
        "core": sum(len({core(a) for a in cell} - {None}) for cell in by_cell.values()),
        "kernel": sum(len({kernel(a) for a in cell}) for cell in by_cell.values()),
        "leak": sum(bool(LEAK.search(a["query"])) for a in atoms),
        "len_median": statistics.median(len(a["query"]) for a in atoms) if atoms else 0,
        "top_kernels": {
            cid: Counter(kernel(a) for a in by_cell[cid]).most_common(4) for cid in ("fwd_steps", "left_plain")
        },
    }


def main() -> None:
    rows = [analyze(Path(p)) for p in sys.argv[1:]]
    cols = ("raw", "valid", "novel@0.5", "core", "kernel", "leak", "len_median")
    print(f"{'run':28}" + "".join(f"{c:>11}" for c in cols))
    for r in rows:
        print(f"{r['run']:28}" + "".join(f"{r[c]:>11}" for c in cols))
    for r in rows:
        print(f"\n{r['run']} top kernels:")
        for cid, top in r["top_kernels"].items():
            print(f"  {cid}: {top}")


if __name__ == "__main__":
    main()
