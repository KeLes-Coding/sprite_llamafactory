# Planner Validation

This evaluator compares the deployed SFT model with `qwen-flash` on the ground-truth rows from
`action_sequence_sft_validation.parquet`.

It sends each row's original `system` and `user` messages through the OpenAI-compatible chat-completions API, then
scores the response against the row's `assistant` task sequence. It does not use V3.1.2 because that dataset does not
contain subagent ground truth.

## Run 10 Cases

Let the script create and close the SSH tunnel:

```bash
.venv/bin/python -m planner_val.validation_eval \
  --package data/v3-2-0-1.0.0-134feaba-parquet.zip \
  --limit 10 \
  --vllm-url http://127.0.0.1:8001/v1 \
  --vllm-model gemma3-270m-sft \
  --cosa-env /home/limx/Workspace/omni_dev/local.multiVersions/cosa_personification/config/env.local \
  --ssh-host 8.130.18.173 \
  --output-dir planner_val/results/validation-smoke-10
```

If the tunnel already exists, omit `--ssh-host`. The equivalent manual tunnel is:

```bash
ssh -N -p 1024 \
  -L 8001:127.0.0.1:8001 \
  root@8.130.18.173
```

Increase `--limit` to evaluate more of the 1,839 validation rows.

## Run Qwen Only

When the vLLM server is unavailable, skip both the vLLM backend and SSH tunnel:

```bash
.venv/bin/python -m planner_val.validation_eval \
  --package data/v3-2-0-1.0.0-134feaba-parquet.zip \
  --limit 1839 \
  --qwen-only \
  --cosa-env /home/limx/Workspace/omni_dev/local.multiVersions/cosa_personification/config/env.local \
  --concurrency 2 \
  --output-dir "planner_val/results/qwen-flash-full-$(date +%Y%m%d-%H%M%S)"
```

## Outputs

- `results.jsonl`: one row per case and model, including source metadata, all score dimensions, end-to-end case time,
  input/output/total tokens, expected tasks, and raw output.
- `summary.json`: run and model-batch wall time, non-secret runtime configuration, dataset identity, quality metrics,
  latency percentiles, token totals, request/token throughput, failure reasons, and slices by level, language, polarity,
  provenance, task count, kind, tool, and action.
- `prompt_profile.json`: the complete frozen system prompt, exact serialized tools context, user prompt template, source
  revision, and hashes when `--qwen-production-profile` is enabled. `summary.json` references this artifact by path and
  SHA-256.

`benchmark_compatible_accuracy` is the primary quality metric. It reproduces the planner-sequence rules used by the
2026-09-13 benchmark and keeps the production parser and contract validation as a gate. `strict_accuracy` remains in
every aggregate and slice as a diagnostic for exact SFT target reproduction.

The compatible scorer applies these historical rules:

- `query` and `action` tasks are compared as one ordered tool sequence; tool count, order, and function must match.
- Reply instruction text is not compared. Negative/reply lineage still constrains forbidden tools, and a negative
  action constraint covers `HumanAction`, `RobotGesture`, and `RobotDance` as one action family.
- Parameters are ignored for `web_search`, `get_weather`, and `rag_query`. Historical field exemptions are applied to
  `HumanAction.parameters.step`, `HumanAction.parameters.distance`, `HumanAction.parameters.degree`,
  `RobotDance.repeat`, and `RobotGesture.repeat`.
- `HumanAction.walk` compares the direction sign of `x`, `y`, and `yaw` within `[-1, 1]`, rather than exact magnitude.
  Explicit range constraints remain scored.

Strict scoring still requires the complete ordered `reply`/`query`/`action` array and exact tool-specific parameters.

Planner evaluation uses non-streaming completions, so TTFT is intentionally unavailable and recorded as such. The
reported per-case latency is request-to-complete-response time. Overall token throughput uses model batch wall time;
slice-level service throughput uses the sum of case latencies because slices do not have an independent concurrent
wall-clock interval. Both throughput numerators include only provider-reported tokens, while elapsed-time denominators
include all requests, including failures. Treat throughput as effective delivered-token throughput and always read it
with `tokens.coverage` when a provider omits usage data.

Step-level kind/name matching reports both precision (`aligned matches / predicted steps`) and recall
(`aligned matches / expected steps`), plus F1 and the raw counts. This keeps over-generation and under-generation
visible instead of combining them into an ambiguous step accuracy.

The cosa environment file is read locally for `DASHSCOPE_API_KEY`; the key is never written to result files. A 10-case
run is a connectivity smoke test. Use a larger sample for comparisons because provider output may vary between runs.

## Re-score Existing Outputs Offline

When scoring rules or report dimensions change, rebuild `results.jsonl` and `summary.json` without calling a model:

```bash
.venv/bin/python -m planner_val.validation_eval \
  --package data/v3-2-0-1.0.0-134feaba-parquet.zip \
  --limit 1839 \
  --output-dir planner_val/results/EXISTING_RUN \
  --rescore-results
```

This mode does not read `--cosa-env` or create a backend. It preserves existing outputs, latency/token values, run
metadata, dataset metadata, and model batch wall time while replacing derived score fields and aggregates.

## Run Qwen With the 2026-09-13 Production Prompt

This profile reconstructs the exact ActionSequence request configuration used by benchmark run
`20260913_151333_groupfiles238_range0-237`: agent revision
`3911119e607e1f7cd7baba92d67941c36871dead`, its seven selected tool descriptions, and `max_steps=12`.
All prompt files are read from immutable Git objects, so uncommitted changes in the agent worktree do not affect the
request.

```bash
.venv/bin/python -m planner_val.validation_eval \
  --package data/v3-2-0-1.0.0-134feaba-parquet.zip \
  --limit 10 \
  --qwen-only \
  --qwen-production-profile \
  --production-agent-repo /home/limx/Workspace/omni_dev/local.multiVersions/subagent \
  --cosa-env /home/limx/Workspace/omni_dev/local.multiVersions/cosa_personification/config/env.local \
  --concurrency 2 \
  --output-dir "planner_val/results/qwen-flash-production-smoke-$(date +%Y%m%d-%H%M%S)"
```

The validation ground truth remains unchanged. Only Qwen's request messages are replaced, and the model is reported as
`qwen-flash-production-3911119`. The summary records the full agent revision, `max_steps`, and SHA-256 hashes for the
rendered system/tool prompts and all source Git objects.

## Try the Clarified Tool Schema (Offline Experiment)

This experiment keeps the frozen package and 50-case challenge unchanged on disk. It adds reviewed enum meanings and
parameter descriptions to the per-request tool schema, and changes weather arguments to `city` plus `query_type`.
Weather targets are projected in memory to match. It does **not** update the live Agent weather tool, train a model, or
mask function names. Run it against a serving SFT checkpoint:

```bash
.venv/bin/python -m planner_val.tool_schema_v2 \
  --package data/v3-2-0-1.0.0-134feaba-parquet.zip \
  --challenge planner_val/challenges/challenge_50.json \
  --vllm-url http://127.0.0.1:8001/v1 \
  --model gemma3-270m-sft \
  --output-dir planner_val/results/schema-v2-challenge-50
```

Use a fresh output directory per run. Compare individual cases, tool choices, parser validity, and non-weather slices
with the frozen baseline. Since weather expectations have changed, do **not** treat the full 50-case strict delta as
an apples-to-apples regression or claim that this offline schema is ready for live tool execution.
The run records the actual system prompt and tool schema in `tool_schema.json`, with its SHA-256 in `summary.json`.

## Run the SFT Prompt Ablation

The controlled prompt ablation runs the same SFT checkpoint and challenge cases with four system prompts. The user
payload and tool schema remain unchanged: A is the baseline, B adds schema semantics and dynamic value binding, C adds
negation and discourse-control rules, and D combines B and C.

```bash
.venv/bin/python -m planner_val.prompt_ablation \
  --package data/v3-2-0-1.0.0-134feaba-parquet.zip \
  --challenge planner_val/challenges/challenge_50.json \
  --vllm-url http://127.0.0.1:8001/v1 \
  --vllm-model gemma3-270m-sft \
  --concurrency 8 \
  --output-dir planner_val/results/prompt-ablation-50-sft
```

The output directory contains `results.jsonl`, `summary.json`, and `prompt_ablation.json`. The prompt artifact records
the complete system prompt and SHA-256 hash for every variant. Run the experiment more than once before interpreting
small case-count changes because the deployed serving path can produce minor output drift even at temperature zero.