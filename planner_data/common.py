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

"""Small helpers vendored from the Omni benchmark package (behavior unchanged)."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("PLANNER_DATA_RUNS") or Path(__file__).resolve().parent / "runs")
_ENV_FILE_RELPATHS = (".env.local", ".env")

_RANGE_LOWER_KEYS = ("min", "gte", "gt")
_RANGE_UPPER_KEYS = ("max", "lte", "lt")
_RANGE_KEYS = _RANGE_LOWER_KEYS + _RANGE_UPPER_KEYS


def _norm_text(text: Any) -> str:
    """Whitespace/punctuation-insensitive normalization for dedup comparison."""
    s = str(text or "").strip()
    s = re.sub(r"\s+", "", s)
    return s.strip("。.!！？?，,、；;：: \t\n")


def _is_number(value: Any) -> bool:
    """True for a real int/float (``bool`` is excluded — it is not a magnitude)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def is_range_spec(spec: Any) -> bool:
    """True when ``spec`` is a numeric-range dict (min/max/gt/gte/lt/lte)."""
    if not isinstance(spec, dict):
        return False
    if "value" in spec or "default" in spec:
        return False
    present = [key for key in _RANGE_KEYS if key in spec]
    if not present:
        return False
    return all(_is_number(spec[key]) and math.isfinite(spec[key]) for key in present)


def value_in_range(spec: dict[str, Any], value: Any) -> bool:
    """Whether numeric ``value`` satisfies every bound present in ``spec``."""
    if not _is_number(value):
        return False
    if "min" in spec and value < spec["min"]:
        return False
    if "gte" in spec and value < spec["gte"]:
        return False
    if "gt" in spec and value <= spec["gt"]:
        return False
    if "max" in spec and value > spec["max"]:
        return False
    if "lte" in spec and value > spec["lte"]:
        return False
    if "lt" in spec and value >= spec["lt"]:
        return False
    return True


def find_env_value(var: str) -> str | None:
    """Resolve ``var`` from the process env, else ``PLANNER_DATA_ENV_FILE``, else repo-root env files."""
    from_env = os.environ.get(var)
    if from_env:
        return from_env
    candidates = [Path(p) for p in os.environ.get("PLANNER_DATA_ENV_FILE", "").split(os.pathsep) if p]
    candidates += [REPO_ROOT / rel for rel in _ENV_FILE_RELPATHS]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped.startswith(f"{var}="):
                    continue
                value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
        except OSError:
            continue
    return None
