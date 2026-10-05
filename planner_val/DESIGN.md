# Planner Validation Design

> **Current scope (2026-09-19):** V3.1.2 does not contain subagent ground truth. The active evaluator therefore uses
> `action_sequence_sft_validation.parquet` directly and compares the deployed SFT model with `qwen-flash` through one
> OpenAI-compatible API path. See `planner_val/README.md` and `planner_val/validation_eval.py`. The broader V3.1.2,
> dual-contract, contamination, and bootstrap design below is retained as historical context and is not the current
> execution path.

## Goal

Build a reproducible offline benchmark that compares three planner variants:

1. a vLLM deployment of the original `models/gemma-3-270m-it` base model;
2. a vLLM deployment of the full-SFT weights produced by
	`llamafactory_runs/gemma3-270m-full-sft_260919_v1/sft.yaml`;
3. the current `qwen-flash` planner baseline through DashScope's OpenAI-compatible endpoint.

The benchmark measures ordered planning only. It does not start an Omni realtime session, run ASR, call tools, connect to a robot, or score physical execution.

## Observed Inputs

- The training run uses full fine-tuning and `template: gemma2`. The SFT model must therefore be loaded directly as a complete model, without `adapter_name_or_path`.
- The v3.2.0 package contains 16,993 training rows and 1,839 validation rows.
- The local V3.1.2 snapshot contains 2,380 runnable cases and 2,203 unique query strings.
- Of those V3.1.2 cases, 319 have query text exactly present in the v3.2.0 training split and 30 have query text exactly present in its validation split.
- V3.1.2 has 449 distinct `source_atom_id` values; 444 also occur in the v3.2.0 training split.

V3.1.2 is consequently a useful same-domain regression and recombination benchmark, but it is not a clean out-of-distribution test set.

## Experiment Matrix

Every selected case runs against all three variants under both contracts:

| Contract | Input construction | Question answered |
| --- | --- | --- |
| `controlled` | Frozen system prompt and `{"tools": ..., "instruction": ...}` payload extracted from the v3.2.0 training package | How do the models compare under the exact SFT contract? |
| `production` | Current ActionSequence system prompt and rendered tool-description context from an explicitly selected Agent checkout | Can the SFT model directly replace the production planner? |

The benchmark records a SHA-256 fingerprint for each contract. Results from different contract fingerprints must not be merged.

A contract snapshot stores production rendering constants (`USER_*` headers and `STEP_LIMIT_TEMPLATE`) in an immutable `message_config` semantic field. This field participates in the fingerprint; filesystem paths and file hashes remain provenance-only metadata and do not participate.

All variants use the same contract-specific messages, case order, OpenAI-compatible chat-completions request shape,
`max_tokens=1024`, and `temperature=0`.

## Contract Sources

### Controlled

The controlled contract is extracted from `action_sequence_sft_train.parquet` inside the training-package zip:

- system content comes from the row's `system` message;
- tools come from the JSON object in the row's `user` message;
- the benchmark verifies that all rows share the same recorded `system_prompt_sha256` and `tool_contracts_sha256` values;
- the user message for a benchmark case is serialized as compact JSON with the frozen tools and the case query as `instruction`.

This avoids copying a prompt by hand and silently drifting away from what Gemma saw during training.

### Production

Production inputs are assembled from explicit, immutable sources:

- `<agent-root>/src/omni/subagents/action_sequence/prompts.py` supplies `SYSTEM_PROMPT`, headers, and `STEP_LIMIT_TEMPLATE`;
- `<agent-root>/src/omni/tools/tool_descriptions.yaml` supplies the rendered planner tool context;
- an explicit benchmark tool-capture JSON supplies function parameter schemas for contract validation.

The snapshot loader reads these files without importing the external Agent package. The resolved prompt, rendered context, schemas, source paths, and hashes are written into the run manifest.

## Dataset Projection

Each V3.1.2 item becomes one immutable `EvalCase`:

- `query` is the model input;
- each positive intent becomes an ordered `query` or `action` task based on its tool name;
- an intent with no tool becomes a `reply` task;
- `params` and `action` are projected into the tool's native argument shape;
- `forbidden_functions` remains a separate negative constraint;
- level, language, difficulty, polarity, and `source_atom_id` values remain available for breakdowns.

Repeated tools remain repeated tasks. Order is significant.

## Contamination Labels

Every case receives independent overlap flags:

- `exact_train`: query text appears exactly in the v3.2.0 train split;
- `exact_validation`: query text appears exactly in the v3.2.0 validation split;
- `all_atoms_seen_train`: every source atom appears in the train split;
- `has_novel_atom`: at least one source atom is absent from the train split.

The report exposes at least these slices:

- `all`;
- `exact_train`;
- `non_exact_train`;
- `composition_seen_atoms` (`non_exact_train` and `all_atoms_seen_train`);
- `has_novel_atom`.

No V3.1.2 slice is described as fully OOD unless its lineage proves that claim.

## Parsing And Scoring

The parser extracts the outermost JSON array but otherwise requires strict task objects:

- reply: exactly `kind` and `instruction`;
- query/action: exactly `kind`, `instruction`, `name`, and `arguments`;
- `instruction` and `name` must be non-empty strings;
- `arguments` must be an object;
- tool name and kind must agree with the selected contract.

The primary metric is `strict_sequence_accuracy`. A case passes only when all of the following pass:

1. output contains a parseable task array;
2. task count matches;
3. ordered kind/tool sequence matches;
4. all benchmark-checked parameters match semantically;
5. no forbidden function appears.

Parameter policy follows the existing benchmark's practical semantics:

- `HumanAction`: exact action, directional sign for `x/y/yaw`, and explicit/range checks for `step` and `degree`;
- `RobotGesture` and `RobotDance`: exact selected action; omit/default `repeat=1` is equivalent;
- `robot_status`: exact canonical query enum;
- `get_weather`, `web_search`, and `rag_query`: tool selection is scored, while volatile/free-text parameters are reported but excluded from the primary strict decision, matching the benchmark's existing default leniency;
- nested numeric range specs use inclusive/exclusive bounds as declared by the dataset.

The scorer emits one stable failure reason per result, in precedence order:

1. `backend_error`
2. `invalid_json`
3. `invalid_task_schema`
4. `contract_violation`
5. `forbidden_tool`
6. `task_count_mismatch`
7. `task_kind_mismatch`
8. `task_name_mismatch`
9. `task_parameter_mismatch`
10. `passed`

Secondary metrics are `valid_json_rate`, `contract_valid_rate`, `task_count_accuracy`, `ordered_kind_name_accuracy`, `parameter_accuracy`, `forbidden_tool_violation_rate`, latency, prompt tokens, output tokens, and throughput.

## Backends

All three variants use one `OpenAICompatibleBackend`. Base and SFT point at vLLM OpenAI-compatible endpoints;
qwen-flash points at DashScope's compatible endpoint. Model loading, chat-template selection, batching, and GPU lifecycle
belong to the remote vLLM deployments and are outside this harness.

For a remote training host, the harness can own an SSH local-forward process for the duration of a run. The default
forward is equivalent to:

```bash
ssh -N -p 1024 -L 8001:127.0.0.1:8001 \
	-o ExitOnForwardFailure=yes \
	-o ServerAliveInterval=30 \
	-o ServerAliveCountMax=3 \
	root@8.130.18.173
```

The tunnel only transports HTTP. It does not start, stop, or inspect vLLM on the server. Direct endpoints remain
supported when the harness runs on the server or the operator manages networking separately.

Requests may run concurrently, but results preserve input order. The OpenAI SDK handles bounded transport retries.
Invalid model output is never retried by harness code. No API key, provider response header, or raw exception payload is
written to disk.

## Run Artifacts

Each run writes to `planner_val/results/<UTC timestamp>/`:

- `manifest.json`: arguments, source fingerprints, contract snapshots, model identifiers, seed, and environment metadata;
- `results.jsonl`: one append-only record per `(case, contract, variant)`;
- `summary.json`: aggregate and paired statistics;
- `summary.md`: human-readable report;
- `failures.jsonl`: non-passing records with raw model output.

Resume is permitted only when the existing manifest identity matches the requested dataset, contracts, variants, generation settings, and seed.

## Sampling And Statistics

The default smoke run selects 250 cases deterministically, balanced across L1-L5 as evenly as availability permits. A full run uses all 2,380 local cases.

Paired comparisons are reported for:

- SFT versus base;
- SFT versus qwen-flash;
- base versus qwen-flash.

Each comparison includes both-pass, left-only-pass, right-only-pass, both-fail, accuracy delta, and a deterministic 95% paired bootstrap confidence interval. Summaries are also broken down by contract, level, intent count, polarity, tool, language, difficulty, and contamination slice.

## Non-Goals

- speech recognition or TTS quality;
- Main-Agent routing into ActionSequence;
- real tool execution or robot motion;
- judging the natural-language content of reply tasks;
- claiming V3.1.2 is an unseen benchmark;
- changing LlamaFactory training code or the external benchmark/Agent repositories.

## Acceptance Criteria

1. Unit tests run without loading a real model or making network calls.
2. A fake-backend integration test completes both contracts and all three variants, resumes without duplicates, and writes deterministic summaries.
3. A 250-case real smoke run can be launched with one documented command against configured OpenAI-compatible endpoints.
4. Base, SFT, and qwen-flash receive byte-identical contract messages for each paired case.
5. Results distinguish exact-train overlap from non-exact composition cases.
6. Every failure record includes enough raw evidence to reproduce the score.
