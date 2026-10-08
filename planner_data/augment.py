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

"""Hammer-style training variants for compiled planner SFT rows (LlamaFactory sharegpt jsonl).

Each real-named input row can yield:

- ``masked``: tool names *and* argument names become random strings, so routing and argument
  filling must come from the descriptions. Mentions of an argument name inside descriptions
  follow the rename; argument values (enum members etc.) are kept, as in Hammer.
- ``drop_used``: one tool the answer calls is removed from the tool list. Its tasks become
  replies saying the request can't be handled *right now* (the robot normally can; only this
  tool list lacks it, see system prompt rule 6). Other tasks are unchanged.
- ``drop_unused``: some tools the answer does not call are removed; the answer is unchanged.

Every variant shuffles the tool order and is re-validated against its own tool list. The two
drop variants are masked too with ``--negative-mask-rate``, so masked names never imply "a
tool will be called".

    python -m planner_data.augment samples/v330_train_mid2000.jsonl -o runs/augment/mid2000.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from planner_data.contract.planning_contract import (
    CallTask,
    ReplyTask,
    SubagentToolContract,
    validate_task_against_contract,
)


VARIANTS = ("masked", "drop_used", "drop_unused")
_UNAVAILABLE = {
    "zh": "说明当前无法处理：{}",
    "en": "Explain that this can't be handled right now: {}",
}


# ---------------------------------------------------------------------------
# Row I/O
# ---------------------------------------------------------------------------
def parse_row(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str, list[dict[str, Any]]]:
    system, user, assistant = row["messages"]
    payload = json.loads(user["content"])
    return system["content"], payload["tools"], payload["instruction"], json.loads(assistant["content"])


def render_row(
    row: dict[str, Any], tools: list[dict[str, Any]], instruction: str, tasks: list[dict[str, Any]]
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    system, user, assistant = out["messages"]
    user["content"] = json.dumps({"tools": tools, "instruction": instruction}, ensure_ascii=False,
                                 separators=(",", ":"))
    assistant["content"] = json.dumps(tasks, ensure_ascii=False, separators=(",", ":"))
    out["messages"] = [system, user, assistant]
    return out


def validate(tools: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
    contracts = [SubagentToolContract.model_validate(tool) for tool in tools]
    for task in tasks:
        if task["kind"] == "reply":
            ReplyTask.model_validate(task)
        else:
            validate_task_against_contract(CallTask.model_validate(task), contracts)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def random_name(rng: random.Random, taken: set[str]) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        name = rng.choice(string.ascii_letters) + "".join(rng.choices(alphabet, k=rng.randint(5, 11)))
        if name not in taken:
            taken.add(name)
            return name


def property_names(schema: Any) -> list[str]:
    names: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                names.extend(value)
            names.extend(property_names(value))
    elif isinstance(schema, list):
        for value in schema:
            names.extend(property_names(value))
    return list(dict.fromkeys(names))


def _rename_text(text: str, mapping: dict[str, str]) -> str:
    for old, new in mapping.items():
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, text)
    return text


def rename_schema(schema: Any, mapping: dict[str, str]) -> Any:
    if isinstance(schema, list):
        return [rename_schema(value, mapping) for value in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            out[key] = {mapping[name]: rename_schema(sub, mapping) for name, sub in value.items()}
        elif key == "required" and isinstance(value, list):
            out[key] = [mapping.get(name, name) for name in value]
        elif key == "description" and isinstance(value, str):
            out[key] = _rename_text(value, mapping)
        else:
            out[key] = rename_schema(value, mapping)
    return out


def rename_arguments(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {mapping.get(key, key): rename_arguments(sub, mapping) for key, sub in value.items()}
    if isinstance(value, list):
        return [rename_arguments(sub, mapping) for sub in value]
    return value


def mask(
    tools: list[dict[str, Any]], tasks: list[dict[str, Any]], rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    taken: set[str] = set()
    tool_names: dict[str, str] = {}
    arg_names: dict[str, dict[str, str]] = {}
    masked_tools = []
    for tool in tools:
        params = {name: random_name(rng, taken) for name in property_names(tool["arguments_schema"])}
        tool_names[tool["name"]] = random_name(rng, taken)
        arg_names[tool["name"]] = params
        masked_tools.append({
            **tool,
            "name": tool_names[tool["name"]],
            "description": _rename_text(tool["description"], params),
            "arguments_schema": rename_schema(tool["arguments_schema"], params),
        })
    masked_tasks = [
        task if task["kind"] == "reply" else {
            **task,
            "name": tool_names[task["name"]],
            "arguments": rename_arguments(task["arguments"], arg_names[task["name"]]),
        }
        for task in tasks
    ]
    return masked_tools, masked_tasks, {"tools": tool_names, "arguments": arg_names}


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def _called(tasks: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(task["name"] for task in tasks if task["kind"] != "reply"))


def drop_used(
    tools: list[dict[str, Any]], tasks: list[dict[str, Any]], language: str, rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]] | None:
    called = _called(tasks)
    if not called:
        return None
    dropped = rng.choice(called)
    template = _UNAVAILABLE.get(language, _UNAVAILABLE["en"])
    new_tasks = [
        {"kind": "reply", "instruction": template.format(task["instruction"])}
        if task["kind"] != "reply" and task["name"] == dropped else task
        for task in tasks
    ]
    return [tool for tool in tools if tool["name"] != dropped], new_tasks, [dropped]


def drop_unused(
    tools: list[dict[str, Any]], tasks: list[dict[str, Any]], rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]] | None:
    called = set(_called(tasks))
    unused = [tool["name"] for tool in tools if tool["name"] not in called]
    most = min(len(unused), len(tools) - 1)  # keep at least one tool on show
    if most < 1:
        return None
    dropped = rng.sample(unused, rng.randint(1, most))
    return [tool for tool in tools if tool["name"] not in dropped], tasks, dropped


def _facets(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kinds": sorted({task["kind"] for task in tasks}), "tools": sorted(_called(tasks))}


def augment_row(row: dict[str, Any], variant: str, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    """One augmented copy of ``row`` (or None) and a reason code for the summary."""
    meta = row.get("metadata") or {}
    rng = random.Random(f"{args.seed}:{meta.get('example_id')}:{variant}")
    system, tools, instruction, tasks = parse_row(row)
    real_names = [tool["name"] for tool in tools]
    language = str(meta.get("language") or "zh")

    dropped: list[str] = []
    if variant == "drop_used":
        result = drop_used(tools, tasks, language, rng)
    elif variant == "drop_unused":
        result = drop_unused(tools, tasks, rng)
    else:
        result = (tools, tasks, [])
    if result is None:
        return None, "not applicable"
    tools, tasks, dropped = result

    masked = variant == "masked" or rng.random() < args.negative_mask_rate
    mapping = None
    if masked:
        # As in compiler._alias_tools: renaming breaks the link if any text names a real tool.
        texts = [instruction, system, *(tool["description"] for tool in tools), *(t["instruction"] for t in tasks)]
        if any(name in text for name in real_names for text in texts):
            if variant == "masked":
                return None, "names a real tool"
            masked = False
        else:
            tools, tasks, mapping = mask(tools, tasks, rng)
    tools = rng.sample(tools, len(tools))
    try:
        validate(tools, tasks)
    except ValueError as exc:  # pydantic ValidationError and PlanningCompileError are ValueErrors
        return None, f"invalid: {str(exc)[:80]}"

    out = render_row(row, tools, instruction, tasks)
    new_meta = out.setdefault("metadata", {})
    new_meta.update(_facets(tasks))
    new_meta["example_id"] = f"{meta.get('example_id')}_{variant}"
    new_meta["tool_naming"] = "masked" if masked else "real"
    new_meta.pop("tool_aliases", None)
    if variant == "drop_used":
        new_meta["polarity"] = "negative"
    new_meta["augment"] = {"variant": variant, "dropped_tools": dropped, "masked": masked,
                           **({"mapping": mapping} if mapping else {})}
    return out, "ok"


def run(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    out = []
    for row in rows:
        if (row.get("metadata") or {}).get("tool_naming", "real") != "real":
            stats["skip non-real input"] += 1
            continue
        stats["real input"] += 1
        for variant in VARIANTS:
            if random.Random(f"{args.seed}:{row['metadata'].get('example_id')}:{variant}:rate").random() >= getattr(
                args, f"{variant}_rate"
            ):
                continue
            new, reason = augment_row(row, variant, args)
            stats[f"{variant}: {reason}"] += 1
            if new is not None:
                out.append(new)
    return out, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="LlamaFactory sharegpt jsonl (compiled planner SFT rows)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="augmented rows only (jsonl)")
    parser.add_argument("--masked-rate", type=float, default=1.0, help="share of real rows given a masked copy")
    parser.add_argument("--drop-used-rate", type=float, default=1.0, help="share given a missing-tool copy")
    parser.add_argument("--drop-unused-rate", type=float, default=1.0, help="share given a tool-subset copy")
    parser.add_argument("--negative-mask-rate", type=float, default=0.5,
                        help="share of drop_* copies that are also masked")
    parser.add_argument("--seed", default="0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    out, stats = run(rows, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for key, value in sorted(stats.items()):
        print(f"{key:60s} {value}")
    print(f"wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
