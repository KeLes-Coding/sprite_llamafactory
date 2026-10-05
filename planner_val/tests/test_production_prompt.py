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

from pathlib import Path

from planner_val.production_prompt import BENCHMARK_20260913_REVISION, load_benchmark_20260913_profile


def test_load_benchmark_profile_uses_frozen_git_objects(monkeypatch) -> None:
    prompt_source = """
MAX_TURN_DEGREE = 1080
SYSTEM_PROMPT = f"system {MAX_TURN_DEGREE}"
USER_TOOLS_HEADER = "Tools"
USER_LIMIT_HEADER = "Limit"
USER_INSTRUCTION_HEADER = "Instruction"
STEP_LIMIT_TEMPLATE = "At most {max_steps} tasks"
"""
    planner_source = '_CONTEXT_TOOL_NAMES = ("second", "first")\n'
    descriptions = """
tools:
  first:
    description: first tool
  ignored:
    description: ignored tool
  second:
    description: second tool
"""

    def fake_git_text(repo_path: Path, revision: str, object_path: str | None = None) -> str:
        assert repo_path == Path("/agent")
        assert revision == BENCHMARK_20260913_REVISION
        if object_path is None:
            return BENCHMARK_20260913_REVISION + "\n"
        return {
            "src/omni/subagents/action_sequence/prompts.py": prompt_source,
            "src/omni/subagents/action_sequence/planner.py": planner_source,
            "src/omni/tools/tool_descriptions.yaml": descriptions,
        }[object_path]

    monkeypatch.setattr("planner_val.production_prompt._git_text", fake_git_text)

    profile = load_benchmark_20260913_profile(Path("/agent"))

    assert profile.system_prompt == "system 1080"
    assert profile.revision == BENCHMARK_20260913_REVISION
    assert profile.max_steps == 12
    assert (
        profile.build_user_prompt("do it") == 'Tools\n{\n  "second": {\n    "description": "second tool"\n  },\n'
        '  "first": {\n    "description": "first tool"\n  }\n}\n\n'
        "Limit\nAt most 12 tasks\n\nInstruction\ndo it"
    )
    assert profile.metadata["name"] == "benchmark-20260913-production"
    assert profile.metadata["agent_revision"] == BENCHMARK_20260913_REVISION
    assert len(profile.metadata["system_prompt_sha256"]) == 64
    assert len(profile.metadata["tools_context_sha256"]) == 64
