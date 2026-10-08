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

"""Compile the canonical runs (audit.POOLS) into one clean atom pool for relations.py / assemble.py.

Early runs predate later label rules (no address terms, dance spellings, no intent lists), so every atom is
re-checked with today's rules and dropped when it fails; each sentence is kept once. instruction_en is not
checked here: English rows get their instructions from translate.py.

    python -m planner_data.atoms.compile_pool --name pool
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from typing import Any

from planner_data.atoms import audit
from planner_data.atoms.pilot import RUNS_DIR, _norm_text


_IGNORED = {"chinese in instruction_en"}


def kind(atom: dict[str, Any]) -> str:
    flip = (atom.get("contrast") or {}).get("flip")
    if atom["planner_target"]["kind"] == "reply":
        return f"reply/{flip}"
    return f"{atom['expected_function']}" + (f"/{flip}" if flip else "")


def compile_pool(rows: list[tuple[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group, atom in rows:
        issues = [i for i in audit.rule_issues(atom) if i not in _IGNORED]
        key = _norm_text(atom["query"])
        if key in seen:
            issues.append("duplicate")
        if issues:
            dropped.append({"group": group, "id": atom["id"], "query": atom["query"], "issues": issues})
            continue
        seen.add(key)
        kept.append({**atom, "pool_group": group})
    return kept, dropped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    kept, dropped = compile_pool(audit.load(audit.POOLS))
    out = RUNS_DIR / f"{args.name}-{datetime.now():%m%d_%H%M%S}"
    out.mkdir(parents=True)
    for name, rows in (("atoms", kept), ("dropped", dropped)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    summary = {
        "pools": audit.POOLS,
        "kept": len(kept),
        "dropped": Counter(i.split(" {")[0] for r in dropped for i in r["issues"]),
        "dropped_by_group": Counter(r["group"] for r in dropped),
        "kinds": dict(sorted(Counter(kind(a) for a in kept).items())),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
