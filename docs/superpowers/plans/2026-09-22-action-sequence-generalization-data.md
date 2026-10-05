# ActionSequence Generalization Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest reproducible Hammer-style dataset transformation in LLaMA Factory, use it to isolate naming/order shortcuts, then make only the source-generation changes proven necessary for value and language generalization.

**Architecture:** Phase 1 is entirely local to this LLaMA Factory checkout: it reads the existing immutable Parquet ZIP, creates original and function-aliased training rows, shuffles tool order, validates every transformed target against its transformed JSON Schema, and emits a new LLaMA Factory package with an audit manifest. Phase 2 is a separate, gated change in `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data`: it retains the existing L0, planner-target, lineage, and deterministic compiler design while adding value coverage and multiple surface realizations before export. The two phases meet only through versioned Parquet packages.

**Tech Stack:** Python 3.11+, `pyarrow`, `jsonschema`, `pytest`, LLaMA Factory SFT, existing `planner_val` evaluation tools.

**Spec:** Approved design from the 2026-09-22 conversation, grounded in `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/docs/superpowers/specs/2026-09-16-action-sequence-training-data-design.md`.

## Global Constraints

- Do not change LLaMA Factory dataset loading or training internals.
- Do not modify the source package `data/v3-2-0-1.0.0-134feaba-parquet.zip` in place.
- Do not copy Benchmark Webmon, catalog, audio, simulator, Agent runtime, or review-lifecycle code into this repository.
- Preserve each row's system prompt, instruction, task order, arguments, lineage, source ids, and split unless a named transform explicitly owns that field.
- Phase 1 may rename function names and reorder tools only; it must not rename parameter keys, rewrite instructions, randomize values, or synthesize irrelevance labels.
- Every alias must be unique within a row and must be updated in both the user tool list and every assistant call task.
- Keep one original copy and two transformed copies per training row by default. Do not augment validation rows.
- Use a deterministic per-row seed derived from the global seed, source split, stable row identity, and copy index.
- Preserve total optimizer exposure in the first experiment: train the 3x dataset for one epoch instead of the current dataset for three epochs.
- Keep existing ID validation and the 50-case OOD challenge unchanged. Add separate alias-contract evaluation; do not tune against a new final holdout.
- Report strict parameter grounding in addition to `benchmark_compatible_accuracy`; the compatible scorer skips parameters for `get_weather`, `web_search`, and `rag_query`.
- Phase 2 starts only after Phase 1 completes and its alias diagnostic confirms a measurable naming/order shortcut.
- Do not perform Git commits, staging, branch creation, or other Git writes. The user controls repository state.

## File Structure

### Phase 1: LLaMA Factory repository

- Create `planner_val/training_data.py`: Parquet package I/O, deterministic function aliasing, tool shuffling, validation, manifest generation, and CLI.
- Create `planner_val/tests/test_training_data.py`: unit and package-level tests for transformation correctness and reproducibility.
- Modify `planner_val/README.md`: build, inspect, train, and evaluate commands.
- Create `llamafactory_runs/gemma3-270m-full-sft-hammer-function-v1/sft.yaml`: one-epoch full-SFT experiment config using the generated package.
- Generated, not committed: `data/action-sequence-hammer-function-v1/` and `data/action-sequence-hammer-function-v1-parquet.zip`.
- Generated, not committed: `planner_val/results/alias-contract-baseline-*`, `planner_val/results/alias-contract-hammer-*`, and comparison reports.

### Phase 2: Benchmark worktree

- Modify `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/case_builder.py`: support explicit realization variants without changing semantic programs.
- Modify `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/planning_contract.py`: expose deterministic schema-derived value candidates without changing task compilation semantics.
- Modify `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/test_planning_datagen.py`: realization and semantic-identity tests.
- Modify `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/test_planning_contract.py`: enum and numeric candidate coverage tests.
- Create `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/training_coverage.py`: build-time coverage report and release gates.
- Create `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/test_training_coverage.py`: coverage aggregation and failure tests.

---

## Phase 1: Minimal Local Hammer Experiment

### Task 1: Package Reader and Stable Row Identity

**Files:**
- Create: `planner_val/training_data.py`
- Create: `planner_val/tests/test_training_data.py`

**Interfaces:**
- Produces: `read_package(path: Path) -> dict[str, list[dict[str, Any]]]`.
- Produces: `stable_row_id(row: Mapping[str, Any], split: str) -> str`.
- Produces: `PackageMembers = dict[str, list[dict[str, Any]]]`, keyed by `train` and `validation`.

- [ ] **Step 1: Write tests for ZIP loading and stable identity**

  Build a temporary ZIP using the same `pyarrow.Table.from_pylist()` pattern as `planner_val/tests/test_dataset.py::_make_training_zip`. Include one train row and one validation row with `messages` and `metadata.example_id`.

  ```python
  def test_read_package_loads_both_splits(tmp_path: Path) -> None:
      package = _make_package(tmp_path / "source.zip", train_rows=[_row("train-1")], validation_rows=[_row("val-1")])
      members = read_package(package)
      assert [row["metadata"]["example_id"] for row in members["train"]] == ["train-1"]
      assert [row["metadata"]["example_id"] for row in members["validation"]] == ["val-1"]


  def test_stable_row_id_prefers_example_id_and_falls_back_to_content() -> None:
      row = _row("example-1")
      assert stable_row_id(row, "train") == "example-1"
      row["metadata"].pop("example_id")
      assert stable_row_id(row, "train") == stable_row_id(row, "train")
      assert stable_row_id(row, "train") != stable_row_id(row, "validation")
  ```

- [ ] **Step 2: Run the focused tests and confirm RED**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q \
    planner_val/tests/test_training_data.py::test_read_package_loads_both_splits \
    planner_val/tests/test_training_data.py::test_stable_row_id_prefers_example_id_and_falls_back_to_content
  ```

  Expected: collection or import failure because `planner_val.training_data` does not exist.

- [ ] **Step 3: Implement strict package reading**

  Use `zipfile.ZipFile`, `io.BytesIO`, and `pyarrow.parquet.read_table`. Require exactly these members:

  ```python
  _SPLIT_MEMBERS = {
      "train": "action_sequence_sft_train.parquet",
      "validation": "action_sequence_sft_validation.parquet",
  }
  ```

  Reject missing members, non-object rows, malformed `messages`, and duplicate stable row ids within one split. For fallback identity, hash canonical JSON of `{"split": split, "row": row}` with SHA-256.

- [ ] **Step 4: Re-run the focused tests and the existing dataset tests**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q \
    planner_val/tests/test_training_data.py \
    planner_val/tests/test_dataset.py
  ```

  Expected: all tests pass.

### Task 2: Function Alias and Tool-Order Transform

**Files:**
- Modify: `planner_val/training_data.py`
- Modify: `planner_val/tests/test_training_data.py`

**Interfaces:**
- Consumes: `stable_row_id()` from Task 1.
- Produces: `TransformConfig(seed: int, copies: int, function_mask_ratio: float, shuffle_tools: bool)`.
- Produces: `transform_row(row: Mapping[str, Any], *, split: str, copy_index: int, config: TransformConfig) -> dict[str, Any]`.
- Produces: `alias_map_for_row(tool_names: Sequence[str], *, row_id: str, copy_index: int, seed: int) -> dict[str, str]`.

- [ ] **Step 1: Write tests for synchronized aliasing**

  Use a fixture containing `RobotDance`, `get_weather`, one reply task, and two call tasks.

  ```python
  def test_transform_row_renames_tools_and_targets_together() -> None:
      transformed = transform_row(
          _row("example-1"),
          split="train",
          copy_index=1,
          config=TransformConfig(seed=42, copies=3, function_mask_ratio=1.0, shuffle_tools=False),
      )
      payload = _user_payload(transformed)
      tasks = _assistant_tasks(transformed)
      aliases = transformed["metadata"]["augmentation"]["function_aliases"]
      assert [tool["name"] for tool in payload["tools"]] == [aliases["RobotDance"], aliases["get_weather"]]
      assert [task.get("name") for task in tasks] == [aliases["RobotDance"], None, aliases["get_weather"]]
      assert tasks[0]["arguments"] == {"dance": "warm_up_dance"}
      assert payload["instruction"] == _user_payload(_row("example-1"))["instruction"]
  ```

  Also test:

  - copy index `0` is byte-semantically unchanged and records `variant="original"`;
  - aliases match `^[A-Za-z][A-Za-z0-9_]{9,15}$` and are unique within the row;
  - alias output is deterministic for the same seed and changes for another seed;
  - `function_mask_ratio=0.0` never renames;
  - `shuffle_tools=True` changes order for at least one deterministic fixture but preserves the tool set;
  - validation rows are not transformed by package construction.

- [ ] **Step 2: Run the focused transform tests and confirm RED**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_training_data.py -k 'transform or alias or shuffle'
  ```

  Expected: failures for missing transform APIs.

- [ ] **Step 3: Implement deterministic transformation**

  Deep-copy each row. Parse only the `user` and `assistant` message contents as JSON. Generate the RNG seed from:

  ```python
  material = f"{config.seed}:{split}:{row_id}:{copy_index}".encode("utf-8")
  rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
  ```

  For transformed copies, decide masking once per row using `rng.random() < function_mask_ratio`. When selected, alias every tool in that row and update each non-reply task's `name`. Do not touch `kind`, `instruction`, `arguments_schema`, argument keys, or values.

  Add metadata without destroying original metadata:

  ```json
  {
    "augmentation": {
      "schema_version": "action-sequence-augmentation/v1",
      "variant": "function_alias",
      "source_example_id": "example-1",
      "copy_index": 1,
      "seed": 42,
      "function_aliases": {"RobotDance": "...", "get_weather": "..."},
      "tools_shuffled": true
    }
  }
  ```

- [ ] **Step 4: Re-run transform tests**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_training_data.py -k 'transform or alias or shuffle'
  ```

  Expected: all selected tests pass.

### Task 3: Transformed Contract Validation

**Files:**
- Modify: `planner_val/training_data.py`
- Modify: `planner_val/tests/test_training_data.py`

**Interfaces:**
- Consumes: transformed rows from Task 2.
- Produces: `validate_training_row(row: Mapping[str, Any]) -> None`.
- Produces: `ValidationStats(rows: int, tasks: int, aliases: int)`.

- [ ] **Step 1: Write validation failure tests**

  Test these exact corruptions independently:

  - assistant references the original name after schema aliasing;
  - assistant references a tool absent from the user payload;
  - task `kind` differs from tool `kind`;
  - required argument is removed;
  - unknown argument is added;
  - assistant JSON is malformed;
  - reply task contains `name` or `arguments`.

  Each test must assert a stable error prefix containing the source example id and task index.

- [ ] **Step 2: Run validation tests and confirm RED**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_training_data.py -k validation
  ```

  Expected: failures for missing validation implementation.

- [ ] **Step 3: Implement per-row contract validation**

  Build `{name: tool}` from the current row's user payload. For every call task:

  - require exact task keys `kind`, `instruction`, `name`, and `arguments`;
  - require tool existence and matching kind;
  - validate arguments with `Draft202012Validator(tool["arguments_schema"])`;
  - require non-empty instruction and name.

  For every reply task, require exact keys `kind` and `instruction`. Reject duplicate tool names and malformed JSON before writing output.

- [ ] **Step 4: Validate both transformed fixtures and original real-package samples**

  Add a parametrized test that reads the first and last 10 rows of each split from `data/v3-2-0-1.0.0-134feaba-parquet.zip`, validates originals, transforms train rows, and validates the results.

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_training_data.py -k validation
  ```

  Expected: all selected tests pass.

### Task 4: Deterministic Parquet Package and Manifest

**Files:**
- Modify: `planner_val/training_data.py`
- Modify: `planner_val/tests/test_training_data.py`

**Interfaces:**
- Consumes: reader, transform, and validator from Tasks 1-3.
- Produces: `build_package(source: Path, output: Path, config: TransformConfig) -> dict[str, Any]`.
- Produces ZIP members: train Parquet, unchanged validation Parquet, `dataset_info.json`, and `augmentation_manifest.json`.

- [ ] **Step 1: Write package-level tests**

  For two train rows and one validation row with `copies=3`, assert:

  ```python
  assert manifest["source"]["train_rows"] == 2
  assert manifest["output"]["train_rows"] == 6
  assert manifest["output"]["validation_rows"] == 1
  assert manifest["variants"]["original"] == 2
  assert manifest["variants"]["function_alias"] == 4
  ```

  Also assert:

  - validation canonical JSON equals its source rows;
  - two builds with the same config produce identical uncompressed member hashes;
  - source and output paths must differ;
  - existing output is rejected unless `overwrite=True` is passed;
  - manifest records source package SHA-256, config, row counts, task counts, variant counts, and member SHA-256 values;
  - `dataset_info.json` names exactly `action_sequence_hammer_function_v1_sft_train` and `action_sequence_hammer_function_v1_sft_validation`.

- [ ] **Step 2: Run package tests and confirm RED**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_training_data.py -k package
  ```

  Expected: failures for missing `build_package()`.

- [ ] **Step 3: Implement atomic package writing**

  Write all members into a temporary sibling path, validate every output row, close the ZIP, then rename it to the requested output. Use `pyarrow.parquet.write_table(..., compression="zstd")`. Do not claim whole-ZIP byte reproducibility because ZIP metadata can differ; the manifest's uncompressed member hashes are the reproducibility contract.

  The manifest must use:

  ```json
  {
    "schema_version": "action-sequence-augmentation-package/v1",
    "source": {"path": "...", "sha256": "...", "train_rows": 16993, "validation_rows": 1839},
    "config": {"seed": 42, "copies": 3, "function_mask_ratio": 1.0, "shuffle_tools": true},
    "output": {"train_rows": 50979, "validation_rows": 1839},
    "variants": {"original": 16993, "function_alias": 33986},
    "members": {"action_sequence_sft_train.parquet": {"sha256": "...", "rows": 50979}}
  }
  ```

- [ ] **Step 4: Re-run all training-data tests**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_training_data.py
  ```

  Expected: all tests pass.

### Task 5: CLI and Real-Package Smoke Build

**Files:**
- Modify: `planner_val/training_data.py`
- Modify: `planner_val/tests/test_training_data.py`
- Modify: `planner_val/README.md`

**Interfaces:**
- Produces: `python -m planner_val.training_data build ...`.
- Produces: `python -m planner_val.training_data inspect ...`.

- [ ] **Step 1: Write CLI tests**

  Test `build --help`, invalid ratios outside `[0, 1]`, `copies < 1`, identical input/output, and a successful tiny build. Capture stdout and require one JSON summary object with output path, member hashes, and row counts.

- [ ] **Step 2: Run CLI tests and confirm RED**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_training_data.py -k cli
  ```

  Expected: failures for missing CLI.

- [ ] **Step 3: Implement the CLI**

  Provide these arguments:

  ```text
  build --input PATH --output PATH --copies INT --function-mask-ratio FLOAT
        --seed INT [--shuffle-tools|--no-shuffle-tools] [--overwrite]
  inspect --input PATH
  ```

  Defaults: `copies=3`, `function_mask_ratio=1.0`, `seed=42`, and shuffle enabled. A ratio of `1.0` is chosen for the first diagnostic because copies 1 and 2 must both remove the naming shortcut; ratio ablation is deferred until the basic effect is measured.

- [ ] **Step 4: Document exact commands**

  Add to `planner_val/README.md`:

  ```bash
  .venv/bin/python -m planner_val.training_data build \
    --input data/v3-2-0-1.0.0-134feaba-parquet.zip \
    --output data/action-sequence-hammer-function-v1-parquet.zip \
    --copies 3 \
    --function-mask-ratio 1.0 \
    --shuffle-tools \
    --seed 42
  ```

- [ ] **Step 5: Build and inspect the full package**

  Run the documented build command, then:

  ```bash
  .venv/bin/python -m planner_val.training_data inspect \
    --input data/action-sequence-hammer-function-v1-parquet.zip
  ```

  Expected:

  - train rows: `50,979`;
  - validation rows: `1,839`;
  - original variants: `16,993`;
  - function-alias variants: `33,986`;
  - zero validation failures.

### Task 6: LLaMA Factory Registration and One-Epoch Training Config

**Files:**
- Create generated directory: `data/action-sequence-hammer-function-v1/`
- Create: `llamafactory_runs/gemma3-270m-full-sft-hammer-function-v1/sft.yaml`
- Modify: `planner_val/README.md`

**Interfaces:**
- Consumes: package from Task 5.
- Produces dataset names: `action_sequence_hammer_function_v1_sft_train` and `action_sequence_hammer_function_v1_sft_validation`.
- Produces output model directory: `saves/gemma3-270m/full/sft-hammer-function-v1`.

- [ ] **Step 1: Extract the package without editing global `data/dataset_info.json`**

  Extract to `data/action-sequence-hammer-function-v1/`. Keep the generated `dataset_info.json` beside the Parquet files so `dataset_dir` remains self-contained.

- [ ] **Step 2: Create the experiment config from the existing baseline**

  Copy `llamafactory_runs/gemma3-270m-full-sft_260919_v1/sft.yaml` and change only:

  ```yaml
  dataset_dir: data/action-sequence-hammer-function-v1
  dataset: action_sequence_hammer_function_v1_sft_train
  eval_dataset: action_sequence_hammer_function_v1_sft_validation
  output_dir: saves/gemma3-270m/full/sft-hammer-function-v1
  num_train_epochs: 1.0
  swanlab_run_name: gemma3-270m-full-sft-hammer-function-v1
  ```

  Remove the literal `swanlab_api_key` field from the copied config. Credentials must come from the environment.

- [ ] **Step 3: Run a dataset preprocessing smoke test**

  Add `test_generated_package_loads_with_llamafactory_dataset_parser` to `planner_val/tests/test_training_data.py`. Use the same LLaMA Factory dataset parser selected by the config, limit both splits to two rows, and assert that the train and validation datasets load and that the first aliased row retains its assistant target after template preprocessing.

  Then run:

  ```bash
  .venv/bin/python -m py_compile planner_val/training_data.py
  WANDB_DISABLED=true .venv/bin/pytest -q \
    planner_val/tests/test_training_data.py::test_generated_package_loads_with_llamafactory_dataset_parser
  ```

- [ ] **Step 4: Run training on the configured GPU environment**

  Run:

  ```bash
  llamafactory-cli train \
    llamafactory_runs/gemma3-270m-full-sft-hammer-function-v1/sft.yaml
  ```

  Expected: one epoch over 50,979 rows, matching the baseline's approximate `16,993 × 3` sample exposure.

### Task 7: Alias-Contract Diagnostic Evaluator

**Files:**
- Modify: `planner_val/training_data.py`
- Create: `planner_val/tests/test_alias_eval.py`
- Create: `planner_val/alias_eval.py`
- Modify: `planner_val/README.md`

**Interfaces:**
- Consumes: validation rows from the original package.
- Produces: `build_alias_validation_cases(package: Path, seed: int) -> list[ValidationCase]`, where each case owns its transformed `ContractSnapshot`, transformed expected tasks, and aliased user payload.
- Produces: `python -m planner_val.alias_eval` with `results.jsonl`, `summary.json`, and `alias_manifest.json`.

- [ ] **Step 1: Write request-construction tests**

  Assert that every validation request gets a fresh deterministic alias map and shuffled tool order, while expected tasks are renamed with the same map. Assert no source validation row is mutated.

- [ ] **Step 2: Run tests and confirm RED**

  Run:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q planner_val/tests/test_alias_eval.py
  ```

  Expected: import failure because `planner_val.alias_eval` does not exist.

- [ ] **Step 3: Implement alias evaluation by reusing the existing backend and scorer**

  Reuse `planner_val.validation_eval.load_validation_cases`, `ValidationCase`, `evaluate_validation`, `planner_val.backends`, and strict `score_output` scoring. Build a new per-case `ContractSnapshot` after aliasing because `planner_val.runner` assumes a shared contract cross-product and is not the owning abstraction for this suite. Score against the transformed per-case contract, not the original controlled contract. Record both raw aliased predictions and de-aliased names for human inspection.

- [ ] **Step 4: Run the same alias suite against baseline and Hammer checkpoints**

  Run twice with identical seed, concurrency, temperature, and max tokens. Store outputs under distinct directories. Also run the unchanged original validation and 50-case challenge for the Hammer checkpoint.

- [ ] **Step 5: Apply the Phase 1 decision gate**

  Continue to Phase 2 only after recording all of:

  - baseline original ID accuracy;
  - baseline alias-contract accuracy;
  - Hammer original ID accuracy;
  - Hammer alias-contract accuracy;
  - Hammer 50-case OOD overall and five slices;
  - parser-valid rate, extra/forbidden-call rate, and strict parameter grounding.

  Phase 1 succeeds when:

  - Hammer alias-contract accuracy improves by at least 15 percentage points over baseline;
  - Hammer original ID accuracy falls by no more than 2 percentage points;
  - parser-valid rate does not fall by more than 2 percentage points.

  Do not require a large 50-case OOD gain from function aliasing alone. If alias accuracy does not improve materially, stop before source-chain work and run a 0.5B-1.5B capacity control with the same package.

---

## Phase 2: Minimal Source-Generation Improvements

Phase 2 changes the Benchmark worktree only after Task 7's decision gate. It does not migrate Benchmark into LLaMA Factory.

### Task 8: Multiple Surface Realizations per Semantic Program

**Files:**
- Modify: `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/case_builder.py`
- Modify: `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/test_planning_datagen.py`

**Interfaces:**
- Produces: `realization_modes: Sequence[Literal["concat", "naturalize"]]` in the training-generation request.
- Preserves: identical ordered intents and semantic id across realization variants.

- [ ] **Step 1: Write failing generation tests**

  For one fixed two-intent sequence and one language, request `("concat", "naturalize")` and assert two rows:

  - distinct `query` values;
  - identical intent payloads and source atom ids;
  - identical compiled plan tasks and semantic id;
  - metadata `realization_variant` values `concat` and `naturalize`.

- [ ] **Step 2: Run focused Benchmark tests and confirm RED**

  Run from the Benchmark worktree:

  ```bash
  .venv/bin/pytest -q \
    omni_benchmark/datagen/test_planning_datagen.py \
    -k realization
  ```

- [ ] **Step 3: Implement realization fan-out after sequence selection**

  Draw the intent sequence once. Render each requested realization independently. Keep the existing `rewrite.naturalize()` meaning-preservation and question-drift checks. Do not ask the LLM to generate or repair plan tasks.

- [ ] **Step 4: Re-run focused generator tests**

  Run:

  ```bash
  .venv/bin/pytest -q \
    omni_benchmark/datagen/test_planning_datagen.py \
    omni_benchmark/datagen/test_training_cleanup.py
  ```

  Expected: all tests pass.

### Task 9: Schema-Derived Value Coverage

**Files:**
- Modify: `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/planning_contract.py`
- Modify: `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/test_planning_contract.py`
- Modify: the canonical L0 source or reviewed planner-annotation input used to produce the next dataset version.

**Interfaces:**
- Produces: `schema_value_candidates(schema: Mapping[str, Any]) -> tuple[Any, ...]` for enum, const, bounded integer, and bounded number leaves.
- Preserves: current `compile_plan()` behavior for existing atoms.

- [ ] **Step 1: Write candidate-generation tests**

  Assert:

  - every enum member is returned exactly once;
  - const returns only its const value;
  - integer ranges include both boundaries, zero when valid, and at least one interior value;
  - numeric ranges respect exclusive bounds and `multipleOf`;
  - strings without enum/const produce no invented value.

- [ ] **Step 2: Run focused tests and confirm RED**

  Run:

  ```bash
  .venv/bin/pytest -q omni_benchmark/datagen/test_planning_contract.py -k value_candidates
  ```

- [ ] **Step 3: Implement candidate extraction without changing current instantiation**

  Reuse the existing schema traversal and numeric-validation helpers. Do not synthesize cities, search queries, or RAG queries from JSON Schema; those require reviewed source templates.

- [ ] **Step 4: Extend reviewed source coverage**

  Add source atoms or parameterized source templates so the next dataset includes:

  - all 25 `RobotDance.dance` enum values;
  - all 13 `RobotGesture.gesture` enum values;
  - all 6 `robot_status.query` enum values;
  - all 4 `get_weather.query_type` enum values;
  - `HumanAction` movement and turn values spanning schema boundaries and interior values;
  - reviewed city/coordinate pairs beyond the current 18-city set.

  Each generated value must update both the natural-language request and assistant arguments from one structured source. Do not use global string replacement.

- [ ] **Step 5: Re-run contract and generation tests**

  Run:

  ```bash
  .venv/bin/pytest -q \
    omni_benchmark/datagen/test_planning_contract.py \
    omni_benchmark/datagen/test_planning_datagen.py
  ```

### Task 10: Coverage Audit and Release Gates

**Files:**
- Create: `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/training_coverage.py`
- Create: `/home/limx/Workspace/benchmark.worktree/benchmark.action-sequence-data/omni_benchmark/datagen/test_training_coverage.py`

**Interfaces:**
- Produces: `audit_training_cases(cases: Iterable[Mapping[str, Any]], contract: ActionSequenceContractBundle) -> dict[str, Any]`.
- Produces: nonzero CLI exit when a configured release gate fails.

- [ ] **Step 1: Write coverage-report tests**

  Use small fixtures and assert counts for:

  - rows and semantic programs;
  - language and task-count distribution;
  - tool and tool-sequence distribution;
  - enum values present and missing from targets;
  - unique values per leaf argument path;
  - realization variants per semantic program;
  - reply-only, tool-only, and mixed rows;
  - candidate tool-set and display-order variants.

- [ ] **Step 2: Run tests and confirm RED**

  Run:

  ```bash
  .venv/bin/pytest -q omni_benchmark/datagen/test_training_coverage.py
  ```

- [ ] **Step 3: Implement recursive schema-aware coverage**

  Traverse `properties`, `oneOf`, `anyOf`, and `allOf`. Compare observed assistant argument leaves against enum/const values from the contract. Keep the report descriptive by default; release gates are explicit CLI options.

- [ ] **Step 4: Add these initial release gates**

  Require:

  - zero missing planner targets;
  - zero contract-invalid compiled tasks;
  - 100% enum target coverage for the four enum families listed in Task 9;
  - at least two same-language surface realizations for the source-chain generalization slice;
  - nonzero counts for every task length L1-L10;
  - no exact query overlap across train, validation, and future holdout;
  - no semantic id split across train and validation.

- [ ] **Step 5: Generate and archive the V3.3 coverage report before export**

  The report must be written beside the generated dataset manifest and copied into the training package manifest. A failed gate prevents package publication but does not delete generated cases.

### Task 11: New Package, Source-vs-Transform Ablation, and Final Decision

**Files:**
- Generated in Benchmark: next versioned case dataset and coverage report.
- Generated in LLaMA Factory: source-only, transform-only, and combined packages/configs.
- Modify: `planner_val/README.md` with final reproducibility commands.

**Interfaces:**
- Produces four comparable training conditions:
  - `baseline`: existing source, no local transform;
  - `name`: existing source plus function alias/tool shuffle;
  - `source`: improved source generation, no local transform;
  - `combined`: improved source plus function alias/tool shuffle.

- [ ] **Step 1: Export the improved source package and verify its manifest**

  Require package hashes, split counts, coverage report hash, contract hashes, source dataset ids, recipe version, and realization distribution.

- [ ] **Step 2: Build `source` and `combined` LLaMA Factory packages**

  Use the Phase 1 builder unchanged for `combined`. Do not add source-generation logic to `planner_val.training_data`.

- [ ] **Step 3: Normalize optimizer exposure across all four conditions**

  Record `train_rows × epochs`, global batch size, optimizer steps, learning rate, warmup ratio, and checkpoint selection. Prefer equal optimizer steps; when row counts differ, use `max_steps` rather than rounded epochs.

- [ ] **Step 4: Evaluate all four conditions**

  Run:

  - original 1,839-row ID validation;
  - alias-contract validation;
  - current 50-case OOD diagnostic;
  - a newly authored, sealed OOD holdout not used for source or transform tuning.

  Report per-slice strict grounding for unseen values, paraphrases, multi-tool composition, negative/control language, and ambiguity/interference.

- [ ] **Step 5: Select the production data recipe**

  Adopt `combined` only if it improves sealed OOD and alias-contract accuracy without exceeding the 2-point ID regression budget or increasing extra/forbidden tool calls. Otherwise retain the smallest condition that passes those gates and document the rejected transform with its evidence.

---

## Final Verification

- [ ] Run focused local tests:

  ```bash
  WANDB_DISABLED=true .venv/bin/pytest -q \
    planner_val/tests/test_training_data.py \
    planner_val/tests/test_alias_eval.py
  ```

- [ ] Run local diagnostics:

  ```bash
  .venv/bin/python -m py_compile \
    planner_val/training_data.py \
    planner_val/alias_eval.py
  ```

- [ ] Run focused Benchmark tests from its worktree:

  ```bash
  .venv/bin/pytest -q \
    omni_benchmark/datagen/test_planning_contract.py \
    omni_benchmark/datagen/test_planning_datagen.py \
    omni_benchmark/datagen/test_training_cleanup.py \
    omni_benchmark/datagen/test_training_coverage.py
  ```

- [ ] Run Ruff only on changed files in each repository.
- [ ] Verify every generated package with its manifest and LLaMA Factory loader before training.
- [ ] Record commands, package hashes, model/checkpoint identity, runtime parameters, and result directories in the final experiment report.
- [ ] Confirm no credential values are present in new configs, manifests, logs, or documentation.

## Stop Conditions

- Stop Phase 1 if transformed rows cannot be validated against their per-row transformed contracts.
- Stop Phase 1 if aliasing requires parameter-name masking; parameter descriptions are not yet sufficient for that experiment.
- Stop before Phase 2 if baseline alias-contract accuracy is not materially below its original ID accuracy, because a naming shortcut has then not been demonstrated locally.
- Stop source publication if enum coverage, split isolation, planner-target completeness, or contract validity gates fail.
- Do not interpret the current 50-case challenge as final acceptance; it has already influenced the design and is now a development regression set.