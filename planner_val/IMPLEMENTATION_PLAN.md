# Planner Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic offline harness in `planner_val/` that compares Gemma base, Gemma full-SFT, and qwen-flash planning under controlled and production contracts.

**Architecture:** Load V3.1.2 into immutable cases, derive contamination flags from the v3.2.0 package, construct two fingerprinted prompt contracts, and run interchangeable batch backends through one parser/scorer. Persist append-only records and derive paired aggregate reports without invoking Omni runtime or robot execution.

**Tech Stack:** Python 3.11, pytest, OpenAI Python SDK, OpenSSH, PyArrow, PyYAML, jsonschema, NumPy.

**Spec:** `planner_val/DESIGN.md`

## Global Constraints

- Keep all new implementation, tests, docs, and generated-result ignore rules under `planner_val/`.
- Do not modify `src/llamafactory`, the training yaml, the training package, or either external repository.
- All variants use OpenAI-compatible chat completions with `temperature=0` and `max_tokens=1024` by default.
- Base and SFT model loading and chat-template configuration belong to their remote vLLM deployments.
- Provider credentials must never be persisted in manifests, results, or errors.
- Unit and integration tests must use fake backends and must not load model weights, require a GPU, or call the network.
- Every Python source file must carry the repository's Apache 2.0 license header.
- Results from different contract fingerprints must never be aggregated together.
- Git staging and commits are intentionally excluded; the user retains control of all Git state changes.

---

### Task 1: Package Skeleton And Dataset Projection

**Files:**
- Create: `planner_val/__init__.py`
- Create: `planner_val/models.py`
- Create: `planner_val/dataset.py`
- Create: `planner_val/tests/__init__.py`
- Create: `planner_val/tests/test_dataset.py`

**Interfaces:**
- Produces: `ExpectedTask`, `EvalCase`, `OverlapInfo`, `TrainingCorpusIndex` dataclasses.
- Produces: `load_v312_cases(dataset_root: Path) -> list[EvalCase]`.
- Produces: `load_training_corpus_index(package_path: Path) -> TrainingCorpusIndex`.
- Produces: `annotate_overlap(cases: Sequence[EvalCase], index: TrainingCorpusIndex) -> list[EvalCase]`.
- Produces: `select_cases(cases: Sequence[EvalCase], sample_size: int | None, seed: int) -> list[EvalCase]`.

- [ ] **Step 1: Write dataset projection tests**

Create fixtures that cover a query task, an action task, a reply/negative task, repeated tools, forbidden functions, a numeric range parameter, and source lineage:

```python
def test_load_v312_cases_preserves_order_repetition_and_negative_constraints(tmp_path: Path):
    write_case_file(
        tmp_path,
        level="L3",
        data=[{
            "case_id": "case-1",
            "query": "查天气，再查天气，然后说句话",
            "language_code": "zh",
            "difficulty": "plain",
            "intents": [
                {"tool": "get_weather", "params": {"city": "北京"}, "source_atom_id": "a1"},
                {"tool": "get_weather", "params": {"city": "上海"}, "source_atom_id": "a2"},
                {"tool": None, "negative": True, "negative_tool": "RobotDance", "source_atom_id": "a3"},
            ],
            "forbidden_functions": ["RobotDance"],
        }],
    )

    cases = load_v312_cases(tmp_path)

    assert [task.kind for task in cases[0].expected] == ["query", "query", "reply"]
    assert [task.name for task in cases[0].expected] == ["get_weather", "get_weather", None]
    assert cases[0].forbidden_functions == frozenset({"RobotDance"})
    assert cases[0].source_atom_ids == frozenset({"a1", "a2", "a3"})
```

Add tests proving that `HumanAction` combines `action` and nested `params`, that missing `case_id` fails with the source path in the error, and that sampling is stable and level-balanced.

- [ ] **Step 2: Run the dataset tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_dataset.py
```

Expected: collection fails because `planner_val.dataset` and `planner_val.models` do not exist.

- [ ] **Step 3: Implement immutable data models**

Define these exact public shapes in `planner_val/models.py`:

```python
@dataclass(frozen=True)
class ExpectedTask:
    kind: Literal["reply", "query", "action"]
    instruction: str
    name: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class OverlapInfo:
    exact_train: bool = False
    exact_validation: bool = False
    all_atoms_seen_train: bool = False
    has_novel_atom: bool = False

@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    expected: tuple[ExpectedTask, ...]
    forbidden_functions: frozenset[str]
    level: str
    language: str
    difficulty: str
    polarity: str
    source_atom_ids: frozenset[str]
    overlap: OverlapInfo = OverlapInfo()

@dataclass(frozen=True)
class TrainingCorpusIndex:
    train_instructions: frozenset[str]
    validation_instructions: frozenset[str]
    train_source_atom_ids: frozenset[str]
    validation_source_atom_ids: frozenset[str]
```

Freeze nested argument mappings by copying them through a JSON round trip at construction time, so later scorer code cannot mutate gold values.

- [ ] **Step 4: Implement V3.1.2 loading and intent projection**

Read `case/L*/*.json` in sorted path order, validate each file's `data` list, and map tools with these constants:

```python
ACTION_TOOLS = frozenset({"HumanAction", "RobotGesture", "RobotDance"})
QUERY_TOOLS = frozenset({"get_weather", "web_search", "rag_query", "robot_status", "get_visual_info"})
```

For `HumanAction`, produce `{"action": action, "parameters": params}` when parameters exist and `{"action": action}` otherwise. For flat tools use `params`; when a dataset intent supplies `action` for a flat selector, insert it under `gesture` or `dance`. A tool-less intent produces a reply task with its source sub-query when available, falling back to the case query.

- [ ] **Step 5: Implement package indexing, overlap labels, and sampling**

Open the package with `zipfile.ZipFile`, read both parquet members through `pyarrow.parquet.read_table(io.BytesIO(...))`, and extract the user JSON's `instruction` plus `metadata.lineage[*].source_atom_id`.

Sampling must sort by `(level, case_id)`, shuffle each level with an independent `random.Random(f"{seed}:{level}")`, allocate `sample_size // level_count` per level, distribute the remainder in level order, then return the selected cases sorted by `case_id`.

- [ ] **Step 6: Run the focused tests and format the task files**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_dataset.py
.venv/bin/ruff check planner_val/models.py planner_val/dataset.py planner_val/tests/test_dataset.py
.venv/bin/ruff format --check planner_val/models.py planner_val/dataset.py planner_val/tests/test_dataset.py
```

Expected: all tests pass and Ruff reports no violations.

---

### Task 2: Fingerprinted Controlled And Production Contracts

**Files:**
- Create: `planner_val/contracts.py`
- Create: `planner_val/tests/test_contracts.py`

**Interfaces:**
- Consumes: training package path and explicit Agent/tool-capture paths.
- Produces: `ContractSnapshot` with `name`, `system_prompt`, `tools_context`, `tool_schemas`, `source`, and `fingerprint`.
- Produces: `load_controlled_contract(package_path: Path) -> ContractSnapshot`.
- Produces: `load_production_contract(agent_root: Path, tool_capture_path: Path, max_steps: int) -> ContractSnapshot`.
- Produces: `build_messages(contract: ContractSnapshot, query: str) -> tuple[dict[str, str], dict[str, str]]`.

- [ ] **Step 1: Write contract extraction and fingerprint tests**

Use a temporary zip with a tiny parquet table and assert exact controlled serialization:

```python
def test_controlled_contract_rebuilds_training_user_json(training_package: Path):
    contract = load_controlled_contract(training_package)
    system, user = build_messages(contract, "向前走一步")

    assert system == {"role": "system", "content": "controlled system"}
    assert json.loads(user["content"]) == {
        "tools": CONTROLLED_TOOLS,
        "instruction": "向前走一步",
    }
    assert contract.fingerprint.startswith("sha256:")
```

Add tests that reject mixed system/tool hashes across parquet rows, that the production prompt includes all three headers exactly once, that changing one tool description changes the fingerprint, and that only schemas from the explicit capture are admitted.

- [ ] **Step 2: Run contract tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_contracts.py
```

Expected: import failure for `planner_val.contracts`.

- [ ] **Step 3: Implement `ContractSnapshot` and canonical hashing**

Add this public model to `planner_val/models.py`:

```python
@dataclass(frozen=True)
class ContractSnapshot:
    name: Literal["controlled", "production"]
    system_prompt: str
    tools_context: JSONValue
    tool_schemas: Mapping[str, Mapping[str, Any]]
    tool_kinds: Mapping[str, Literal["query", "action"]]
    message_config: Mapping[str, JSONValue]
    source: Mapping[str, str]
    fingerprint: str
    max_steps: int = 12
```

Define `JSONScalar = str | int | float | bool | None` and recursive `JSONValue` aliases before the dataclass. Include `tool_kinds` and `message_config` in canonical hashing; the controlled loader reads kinds from training tools, while production assigns query/action from the same explicit tool-name sets used by the production parser. For production, `message_config` contains the three user headers and `STEP_LIMIT_TEMPLATE`; for controlled it is empty. `source` remains provenance-only and is excluded from the fingerprint.

Hash canonical JSON encoded with `ensure_ascii=False`, `sort_keys=True`, and `separators=(",", ":")`. Include contract name, system prompt, context, schemas, and max steps in the hash.

- [ ] **Step 4: Implement controlled extraction**

Read the first train row to get the system message and `tools`. Validate all unique `metadata.system_prompt_sha256` and `metadata.tool_contracts_sha256` values have cardinality one. Convert each tool into `{name: arguments_schema}` and `{name: kind}` maps without executing package content.

- [ ] **Step 5: Implement production extraction without importing the Agent package**

Use `runpy.run_path()` only for the standalone `prompts.py`, load YAML with `yaml.safe_load`, and read schemas from the explicit tool-capture JSON. Render the same context subset and separators as production:

```python
context_names = (
    "HumanAction",
    "RobotGesture",
    "RobotDance",
    "get_weather",
    "web_search",
    "rag_query",
    "robot_status",
    "get_visual_info",
)
tools_context = json.dumps(
    {name: descriptions[name] for name in context_names if name in descriptions},
    ensure_ascii=False,
    indent=2,
)
```

Build the user message with `USER_TOOLS_HEADER`, `USER_LIMIT_HEADER`, `STEP_LIMIT_TEMPLATE`, and `USER_INSTRUCTION_HEADER`. Fail with a path-specific error if any required source or header is missing.

- [ ] **Step 6: Run contract tests and focused quality checks**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_contracts.py
.venv/bin/ruff check planner_val/contracts.py planner_val/models.py planner_val/tests/test_contracts.py
.venv/bin/ruff format --check planner_val/contracts.py planner_val/models.py planner_val/tests/test_contracts.py
```

Expected: all tests and checks pass.

---

### Task 3: Strict Parser And Semantic Scorer

**Files:**
- Create: `planner_val/scoring.py`
- Create: `planner_val/tests/test_scoring.py`
- Create: `planner_val/requirements.txt`

**Interfaces:**
- Consumes: `EvalCase`, raw output, and `ContractSnapshot`.
- Produces: `ParsedTask` and `ScoreResult` dataclasses.
- Produces: `parse_plan(raw: str, contract: ContractSnapshot) -> ParseResult`.
- Produces: `score_output(case: EvalCase, raw: str | None, contract: ContractSnapshot, backend_error: str | None = None) -> ScoreResult`.

- [ ] **Step 1: Declare isolated runtime extras**

Write exact direct dependencies not already guaranteed by the root project:

```text
openai>=1.0.0,<2.0.0
jsonschema>=4.23.0,<5.0.0
```

Do not edit the root `pyproject.toml`.

- [ ] **Step 2: Write parser and scorer tests**

Cover code-fenced arrays, malformed JSON, extra task keys, kind/tool mismatch, forbidden tools, repeated ordered tools, count mismatch, wrong gesture, walk direction sign, range bounds, `repeat` omission versus `1`, and skipped free-text query parameters.

The main success test must assert every component metric:

```python
def test_score_output_requires_complete_ordered_semantics(case: EvalCase, contract: ContractSnapshot):
    raw = json.dumps([
        {"kind": "action", "instruction": "前进一步", "name": "HumanAction", "arguments": {
            "action": "walk", "parameters": {"x": 0.6, "y": 0, "yaw": 0, "step": 1}
        }},
        {"kind": "reply", "instruction": "说明是否会跳舞"},
    ], ensure_ascii=False)

    result = score_output(case, raw, contract)

    assert result.reason == "passed"
    assert result.valid_json is True
    assert result.contract_valid is True
    assert result.task_count_match is True
    assert result.ordered_kind_name_match is True
    assert result.parameter_match is True
    assert result.strict_pass is True
```

- [ ] **Step 3: Run scorer tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_scoring.py
```

Expected: import failure for `planner_val.scoring`.

- [ ] **Step 4: Implement strict structural parsing and contract validation**

Extract from the first `[` through the last `]`, decode JSON, enforce exact key sets, map tool names to contract kinds, and validate arguments with `jsonschema.Draft202012Validator` against the selected contract schema. Return errors as data; malformed model output must not raise out of `score_output`.

Use these exact public result shapes:

```python
@dataclass(frozen=True)
class ParsedTask:
    kind: Literal["reply", "query", "action"]
    instruction: str
    name: str | None
    arguments: Mapping[str, Any]

@dataclass(frozen=True)
class ParseResult:
    tasks: tuple[ParsedTask, ...] = ()
    valid_json: bool = False
    contract_valid: bool = False
    error: str | None = None

@dataclass(frozen=True)
class ScoreResult:
    reason: str
    strict_pass: bool
    valid_json: bool
    contract_valid: bool
    task_count_match: bool
    ordered_kind_name_match: bool
    parameter_match: bool | None
    forbidden_tool_violation: bool
    parsed_tasks: tuple[ParsedTask, ...]
    parameter_errors: tuple[str, ...]
    forbidden_calls: tuple[str, ...]
    raw_output: str | None
```

- [ ] **Step 5: Implement semantic parameter matching**

Use recursive expected-subset comparison with these explicit special cases:

```python
PARAMETER_SKIPPED_TOOLS = frozenset({"get_weather", "web_search", "rag_query"})
DIRECTION_PATHS = frozenset({
    "parameters.x",
    "parameters.y",
    "parameters.yaw",
})
DEFAULT_EQUIVALENTS = {"repeat": (None, 1)}
```

Support `min`, `max`, `gt`, `gte`, `lt`, and `lte` numeric range dictionaries. For direction paths compare signs; for all other checked values compare exact values after normalizing flat gesture/dance selectors.

- [ ] **Step 6: Implement failure precedence and evidence**

`ScoreResult` must contain booleans for each component, the stable reason, parsed tasks, expected tasks, parameter errors, forbidden calls, and raw output. Select the first failing reason in the precedence order specified by `DESIGN.md`.

- [ ] **Step 7: Run all focused tests and quality checks**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_scoring.py
.venv/bin/ruff check planner_val/scoring.py planner_val/tests/test_scoring.py
.venv/bin/ruff format --check planner_val/scoring.py planner_val/tests/test_scoring.py
```

Expected: all tests and checks pass.

---

### Task 4: Unified OpenAI-Compatible Backend And SSH Tunnel

**Files:**
- Create: `planner_val/backends.py`
- Create: `planner_val/tests/test_openai_backend.py`

**Interfaces:**
- Produces: `GenerationRequest(request_id: str, system: str, user: str)`.
- Produces: `GenerationResult(request_id: str, text: str | None, latency_ms: float, prompt_tokens: int | None, output_tokens: int | None, error: str | None)`.
- Produces: `CompletionBackend.generate(requests: Sequence[GenerationRequest]) -> list[GenerationResult]` protocol.
- Produces: `OpenAICompatibleBackend` and `SshTunnel`.

- [ ] **Step 1: Write backend and tunnel contract tests using fakes**

Patch an OpenAI-compatible client and `subprocess.Popen`, then assert:

- requests use `chat.completions.create` with the configured model, `temperature=0`, and `max_tokens`;
- results preserve request order and provider usage counts;
- every request produces one result, including provider failures and malformed completions;
- error records never contain raw provider messages or credentials;
- the SSH process uses the configured `ssh -N -L` forward plus keepalive options and closes cleanly.

- [ ] **Step 2: Run backend tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_openai_backend.py
```

Expected: imports fail because `OpenAICompatibleBackend` and `SshTunnel` do not exist.

- [ ] **Step 3: Implement the backend protocol and OpenAI-compatible client**

Create one OpenAI client from `model`, `base_url`, and `api_key`. Call `chat.completions.create` with system/user
messages, `temperature=0`, and the configured `max_tokens`. Use `ThreadPoolExecutor(max_workers=concurrency)` while
preserving input order. Preserve provider usage token counts when available. Return a sanitized error result for every
failed request and leave bounded network retries to the OpenAI SDK.

Define the exact protocol, including lifecycle cleanup used by the runner:

```python
class CompletionBackend(Protocol):
    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...
    def close(self) -> None: ...
```

- [ ] **Step 4: Implement the managed SSH tunnel**

Launch an `ssh -N -L` process with `ExitOnForwardFailure`, `ServerAliveInterval`, and `ServerAliveCountMax`. Keep its
process handle, expose `ensure_running()` for a pre-run liveness check, and terminate it on context exit. The tunnel
must not own the remote vLLM process.

- [ ] **Step 5: Add explicit client and tunnel teardown**

Make both `close()` methods idempotent. Close only the OpenAI HTTP client and local SSH process; there is no local model
or GPU lifecycle.

- [ ] **Step 6: Run focused tests and quality checks**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_openai_backend.py
.venv/bin/ruff check planner_val/backends.py planner_val/tests/test_openai_backend.py
.venv/bin/ruff format --check planner_val/backends.py planner_val/tests/test_openai_backend.py
```

Expected: all tests and checks pass.

---

### Task 5: Resumable Paired Runner

**Files:**
- Create: `planner_val/runner.py`
- Create: `planner_val/tests/test_runner.py`

**Interfaces:**
- Consumes: selected cases, contracts, backend factories, generation settings, and output directory.
- Produces: `RunConfig`, `RunManifest`, and append-only `ResultRecord` values.
- Produces: `run_benchmark(config: RunConfig, backend_factories: Mapping[str, BackendFactory]) -> Path`.

- [ ] **Step 1: Write a fake-backend end-to-end runner test**

Use two cases, two contracts, and three variants. Assert exactly 12 unique records, stable keys, byte-identical paired messages within each contract/case, and no duplicate records after resume:

```python
def test_runner_executes_full_pairing_and_resumes_without_duplicates(tmp_path: Path):
    output = run_benchmark(config, fake_backend_factories)
    first = read_jsonl(output / "results.jsonl")
    run_benchmark(config, fake_backend_factories)
    resumed = read_jsonl(output / "results.jsonl")

    assert len(first) == 2 * 2 * 3
    assert resumed == first
    assert len({row["record_id"] for row in resumed}) == len(resumed)
```

Add tests that reject resume after contract fingerprint, dataset fingerprint, model ID, or generation settings change.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_runner.py
```

Expected: import failure for `planner_val.runner`.

- [ ] **Step 3: Implement run identity and manifest creation**

The identity hash must include dataset file hashes, training package content hash, case IDs, contract fingerprints,
variant endpoint/model identifiers, max tokens, seed, and sample size. Store environment package versions as
informational fields outside the identity hash.

Define the runner-facing types before implementing the manifest:

```python
VariantName = Literal["base", "sft", "qwen_flash"]
BackendFactory = Callable[[], CompletionBackend]

@dataclass(frozen=True)
class RunConfig:
    cases: tuple[EvalCase, ...]
    contracts: tuple[ContractSnapshot, ...]
    variants: tuple[VariantName, ...]
    model_ids: Mapping[VariantName, str]
    output_dir: Path
    max_tokens: int
    seed: int
    sample_size: int | None
    dataset_fingerprint: str
    training_package_fingerprint: str
    resume: bool = False

@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_identity: str
    created_at: str
    config: Mapping[str, Any]
    contracts: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any]

@dataclass(frozen=True)
class ResultRecord:
    record_id: str
    case_id: str
    contract_name: str
    contract_fingerprint: str
    variant: VariantName
    query: str
    level: str
    intent_count: int
    polarity: str
    language: str
    difficulty: str
    overlap: OverlapInfo
    generation: GenerationResult
    score: ScoreResult
```

Serialize dataclasses through explicit `to_dict()` helpers; do not depend on `default=str`, which would silently flatten nested evidence.

- [ ] **Step 4: Implement deterministic schedule and append-only checkpoints**

Use record IDs of the form `sha256(contract_fingerprint, variant, case_id, generation_settings)`. Iterate contracts in requested order and variants in `base`, `sft`, `qwen_flash` order. Append each scored record as one compact JSON line and flush after each backend batch. On resume, load completed record IDs before creating any backend.

- [ ] **Step 5: Implement endpoint and tunnel lifecycle**

Create and close one OpenAI-compatible backend per variant. When configured, start one SSH tunnel before endpoint calls
and close it after the run. A backend failure creates scored `backend_error` records for affected requests and does not
discard completed pairs.

- [ ] **Step 6: Run focused tests and quality checks**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_runner.py
.venv/bin/ruff check planner_val/runner.py planner_val/tests/test_runner.py
.venv/bin/ruff format --check planner_val/runner.py planner_val/tests/test_runner.py
```

Expected: all tests and checks pass.

---

### Task 6: Aggregate Metrics, Paired Bootstrap, And Failure Reports

**Files:**
- Create: `planner_val/reporting.py`
- Create: `planner_val/tests/test_reporting.py`

**Interfaces:**
- Consumes: `manifest.json` and `results.jsonl`.
- Produces: `build_summary(records: Sequence[ResultRecord], seed: int, bootstrap_samples: int = 10_000) -> dict[str, Any]`.
- Produces: `write_reports(run_dir: Path) -> None`.

- [ ] **Step 1: Write aggregate and paired-statistic tests**

Create a fixed record matrix where paired outcomes are known. Assert rates, failure counts, contamination slices, paired cells, and deterministic confidence intervals. Include a slice with zero cases and require `None`, not division-by-zero or a fabricated 0%.

- [ ] **Step 2: Run reporting tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_reporting.py
```

Expected: import failure for `planner_val.reporting`.

- [ ] **Step 3: Implement aggregate dimensions**

Aggregate each metric by contract and variant for `all`, level, intent count, polarity, tool, language, difficulty, and overlap slice. Keep numerators and denominators alongside rates. `parameter_accuracy` uses only cases with checked parameters.

- [ ] **Step 4: Implement paired comparisons and confidence intervals**

Join records by `(contract_fingerprint, case_id)`. For each variant pair, calculate both-pass, left-only, right-only, both-fail, and accuracy delta. Bootstrap paired per-case pass differences with `numpy.random.default_rng(seed)`, 10,000 samples, and percentile bounds at 2.5 and 97.5.

- [ ] **Step 5: Write JSON, Markdown, and failures artifacts**

Write `summary.json` with sorted keys and `summary.md` with:

1. run identity and contract hashes;
2. top-line controlled and production tables;
3. paired comparisons;
4. contamination slices;
5. L1-L5 breakdown;
6. failure-reason table;
7. latency/token table.

Write every non-passing record to `failures.jsonl`, preserving raw output, parsed output, expected tasks, and parameter errors.

- [ ] **Step 6: Run focused tests and quality checks**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_reporting.py
.venv/bin/ruff check planner_val/reporting.py planner_val/tests/test_reporting.py
.venv/bin/ruff format --check planner_val/reporting.py planner_val/tests/test_reporting.py
```

Expected: all tests and checks pass.

---

### Task 7: CLI, Documentation, And Full Offline Verification

**Files:**
- Create: `planner_val/run_benchmark.py`
- Create: `planner_val/README.md`
- Create: `planner_val/.gitignore`
- Modify: `planner_val/tests/test_runner.py`

**Interfaces:**
- Produces CLI entry: `.venv/bin/python -m planner_val.run_benchmark`.
- Produces modes: `--sample-size 250` and `--full`.
- Produces contract selection: `--contracts controlled production`.
- Produces variant selection: `--variants base sft qwen_flash`.

- [ ] **Step 1: Write CLI parsing and no-network integration tests**

Test required-path errors, mutually exclusive `--sample-size/--full`, explicit production source requirements,
environment-only provider credentials, optional SSH tunnel arguments, and fake backend injection. Assert `--help`
documents every required source and output artifact.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests/test_runner.py -k cli
```

Expected: CLI tests fail because `planner_val.run_benchmark` does not exist.

- [ ] **Step 3: Implement CLI validation and orchestration**

Expose these arguments with no machine-specific absolute defaults:

```text
--dataset-root PATH
--training-package PATH
--base-url URL
--base-model MODEL
--sft-url URL
--sft-model MODEL
--qwen-url https://dashscope.aliyuncs.com/compatible-mode/v1
--qwen-model qwen-flash
--contracts controlled production
--production-agent-root PATH
--production-tool-capture PATH
--sample-size 250 | --full
--seed 20260919
--concurrency 8
--max-tokens 1024
--ssh-host HOST
--ssh-user root
--ssh-port 1024
--ssh-local-port 8001
--ssh-remote-port 8001
--no-ssh-tunnel
--output-dir PATH
--resume
```

Require `DASHSCOPE_API_KEY` only when `qwen_flash` is selected. The local vLLM endpoints use `EMPTY` unless explicitly
configured. Start the SSH tunnel by default when `--ssh-host` is supplied; `--no-ssh-tunnel` uses endpoints directly.
Print progress as completed records over total records and print the final report path.

- [ ] **Step 4: Write operator documentation**

Document setup, contract meaning, contamination caveat, artifact schema, resume behavior, and commands. Include this 250-case smoke command with relative placeholders resolved by shell variables:

```bash
AGENT_ROOT=/path/to/cosa_personification
BENCHMARK_ROOT=/path/to/benchmark.dev

.venv/bin/python -m planner_val.run_benchmark \
  --dataset-root "$BENCHMARK_ROOT/omni_benchmark/data/V3.1.2" \
  --training-package data/v3-2-0-1.0.0-134feaba-parquet.zip \
    --base-url http://127.0.0.1:8001/v1 \
    --base-model gemma-3-270m-it \
    --sft-url http://127.0.0.1:8002/v1 \
    --sft-model gemma3-270m-full-sft \
    --ssh-host 8.130.18.173 \
  --contracts controlled production \
  --production-agent-root "$AGENT_ROOT" \
  --production-tool-capture "$BENCHMARK_ROOT/omni_benchmark/data/tool_captures/<capture>.json" \
  --sample-size 250 \
  --output-dir planner_val/results/smoke-250
```

Also document the full-run replacement `--full` and the expected API call volume before users incur qwen-flash cost.

- [ ] **Step 5: Ignore generated results only**

Write:

```gitignore
results/
```

Do not ignore source snapshots, tests, or documentation.

- [ ] **Step 6: Run the complete offline suite**

Run:

```bash
.venv/bin/python -m pytest -q planner_val/tests
.venv/bin/ruff check planner_val
.venv/bin/ruff format --check planner_val
```

Expected: all tests pass, Ruff reports no issues, and no model/network access occurs.

- [ ] **Step 7: Run a fake-backend CLI smoke test**

Use the integration-test fixture through the test-only backend injection and verify the generated directory contains:

```text
manifest.json
results.jsonl
summary.json
summary.md
failures.jsonl
```

Expected: record count equals `cases × contracts × variants`, rerunning with resume adds zero records, and both summary files are byte-identical.
