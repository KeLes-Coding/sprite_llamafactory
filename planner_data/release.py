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

r"""Export assembled runs as a LlamaFactory sharegpt release (train + validation_1k).

Rows from every source run are pooled in order, shuffled with ``--seed``, and ``--val-size`` rows of
level >= ``--min-val-level`` are held out for validation. The release directory is named after the
train file's SHA256 and carries dataset_info.json, manifest.json and SHA256SUMS.

    PLANNER_DATA_RUNS=/path/to/runs \
    python -m planner_data.release --build-id 3.4.0 --seed 340 \
        --sources v340_zh-1007_180008 v340_en-1007_180524 --out /tmp/release_check
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from planner_data.common import RUNS_DIR


TRAIN_NAME = "action_sequence_sft_train"
VAL_NAME = "action_sequence_sft_validation_1k"
_SPEC = {
    "columns": {"messages": "messages"},
    "formatting": "sharegpt",
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}


def _dump(messages: list[list[dict[str, Any]]]) -> str:
    return "".join(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in messages)


def export(
    sources: list[str],
    *,
    build_id: str,
    seed: int,
    out_root: Path,
    runs_dir: Path = RUNS_DIR,
    val_size: int = 1000,
    min_val_level: int = 2,
) -> Path:
    """Write one release directory under ``out_root`` and return its path."""
    rows: list[tuple[int, list[dict[str, Any]]]] = []
    for source in sources:
        with open(runs_dir / source / "rows.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append((row["metadata"]["level"], row["messages"]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    eligible = [i for i, (level, _) in enumerate(rows) if level >= min_val_level]
    val_idx = set(rng.sample(eligible, val_size))
    train = [m for i, (_, m) in enumerate(rows) if i not in val_idx]
    val = [m for i, (_, m) in enumerate(rows) if i in val_idx]
    roles = ["system", "user", "assistant"]
    if not all(len(m) == 3 and [x["role"] for x in m] == roles for m in train + val):
        raise ValueError("every row must be exactly system/user/assistant")

    train_text, val_text = _dump(train), _dump(val)
    digest = hashlib.sha256(train_text.encode()).hexdigest()
    out = out_root / f"v{build_id.replace('.', '-')}-{digest[:8]}"
    out.mkdir(parents=True)
    files = {
        f"{TRAIN_NAME}.jsonl": train_text,
        f"{VAL_NAME}.jsonl": val_text,
        "dataset_info.json": json.dumps(
            {name: {"file_name": f"{name}.jsonl", **_SPEC} for name in (TRAIN_NAME, VAL_NAME)}, indent=2
        ),
        "manifest.json": json.dumps(
            {
                "build_id": build_id,
                "sources": [f"runs/{source}" for source in sources],
                "augmentation": "none",
                "splits": {"train": len(train), "validation_1k": len(val)},
                "levels": dict(sorted(collections.Counter(level for level, _ in rows).items())),
                "seed": seed,
            },
            indent=2,
        ),
    }
    sums = []
    for name, text in files.items():
        data = text.encode("utf-8")
        (out / name).write_bytes(data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
    (out / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    print(out, len(train), len(val))
    return out


def main() -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, help="run names under the runs directory")
    p.add_argument("--build-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=RUNS_DIR / "releases")
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--min-val-level", type=int, default=2)
    args = p.parse_args()
    export(
        args.sources,
        build_id=args.build_id,
        seed=args.seed,
        out_root=args.out,
        val_size=args.val_size,
        min_val_level=args.min_val_level,
    )


if __name__ == "__main__":
    main()
