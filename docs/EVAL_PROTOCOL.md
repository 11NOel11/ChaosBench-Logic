# Evaluation Protocol

This document describes the evaluation procedure, metrics, and reporting format for ChaosBench-Logic.

## Overview

ChaosBench-Logic evaluates LLM reasoning capabilities on dynamical systems through zero-shot binary classification. Models receive questions in natural language and must respond with TRUE/FALSE.

**Default evaluation target**: v2 dataset (40,886 questions in `data/v22_*.jsonl`, 10 files). v1 (621 questions, archived in `data/archive/v1/`) available for baseline comparison. Dataset version: 2.0.0.

## Evaluation Procedure

### Zero-Shot Binary Classification

The standard evaluation mode uses zero-shot prompting:

1. **Input**: Natural language question from dataset
2. **Prompt**: Minimal instruction (no examples or hints)
3. **Output**: Binary answer (TRUE/FALSE or YES/NO)
4. **Scoring**: Exact match with ground truth

**Example Prompt:**

```
Answer the following question about dynamical systems with TRUE or FALSE.

Question: The Lorenz-63 system exhibits sensitive dependence on initial conditions. TRUE or FALSE?

Answer:
```

### Chain-of-Thought Mode

Optional mode for explicit reasoning:

1. **Input**: Same question as zero-shot
2. **Prompt**: Instruction to show reasoning steps
3. **Output**: Reasoning chain followed by final answer
4. **Scoring**: Extract final answer from reasoning chain

**Example Prompt:**

```
Think step-by-step about the following question, then provide your final answer as TRUE or FALSE.

Question: If a system is chaotic, must it be deterministic?

Reasoning:
```

### Answer Normalization and 3-Way Outcomes

Model outputs are parsed into one of three outcomes:

| Outcome | Meaning |
|---------|---------|
| `VALID_TRUE` | Parsed unambiguously as TRUE (includes YES normalisation) |
| `VALID_FALSE` | Parsed unambiguously as FALSE (includes NO normalisation) |
| `INVALID` | Could not extract a binary label; counted against coverage |

Parsing uses an 8-step cascade (`chaosbench/eval/parsing.py`):

1. Extract explicit `FINAL_ANSWER:` markers
2. Identify answer phrases ("Answer: TRUE", "the answer is FALSE")
3. Parse conclusion signals ("Therefore:", "Conclusion:")
4. Final-line token extraction
5. YES/NO → TRUE/FALSE normalisation
6. Ambiguity detection (reject "it depends", "cannot determine")
7. Confidence scoring
8. Return INVALID if unparseable

**Retry policy**: if INVALID on first attempt, one reprompt is issued:
`"Reply with EXACTLY one word: TRUE or FALSE."` If still INVALID, the outcome
remains INVALID. Retries are configurable (`--retries 0` to disable).

## Metrics

### Primary Metrics

#### Coverage and 3-Way Metrics

```
coverage          = valid / total
invalid_rate      = invalid / total
accuracy_valid    = correct / valid          (only over parseable responses)
effective_accuracy = coverage × accuracy_valid  (primary paper metric)
balanced_accuracy  = (TPR + TNR) / 2          (robust to class imbalance)
mcc               = (TP·TN − FP·FN) / sqrt(…) (Matthews Correlation Coefficient)
```

**Reporting convention**: report `effective_accuracy` as the headline number; also
report `coverage` and `accuracy_valid` separately so formatting artifacts can be
distinguished from reasoning errors.

#### Per-Split Accuracy

Accuracy computed separately for each data split:
- core
- robustness
- heldout_systems
- heldout_templates
- hard

#### Per-Task-Family Accuracy

Accuracy for each of the 10 task families:
- atomic
- multi_hop
- indicator_diagnostics
- regime_transition
- fol_inference
- cross_indicator
- extended_systems
- consistency_paraphrase
- adversarial
- perturbation_robustness

### Secondary Metrics

#### FOL Violation Rate

Measures logical consistency in multi-turn dialogues. A violation occurs when model responses contradict FOL axioms.

**Example Violation:**

Model says:
- "Chaotic=YES"
- "Deterministic=NO"

This violates axiom: Chaotic → Deterministic

**Computation:**

```
fol_violation_rate = total_violations / total_dialogues
```

#### Contradiction Rate

Binary flag: Did the model give both YES and NO for the same (system, task) pair?

```
contradiction_rate = dialogues_with_contradictions / total_dialogues
```

#### Coverage

Proportion of questions with valid model responses:

```
coverage = (total_questions - unknown_answers) / total_questions
```

### Task-Specific Metrics

#### Dialogue Accuracy

For multi-turn dialogues, accuracy computed at dialogue level:

```
dialogue_correct = all_turns_in_dialogue_correct
dialogue_accuracy = correct_dialogues / total_dialogues
```

A single incorrect turn causes the entire dialogue to be marked incorrect.

#### Bias Error Rate

For adversarial/bias tasks, measures susceptibility to common misconceptions:

```
bias_error_rate = incorrect_bias_questions / total_bias_questions
```

## Running Evaluations

### Canonical Eval Commands (v2 runner)

```bash
# Freeze dataset first (produces artifacts/freeze/v2_freeze_manifest.json)
chaosbench freeze

# Smoke test with mock provider (no network, 50 items)
chaosbench eval --provider mock --dataset canonical --max-items 50

# Local Ollama — 1k balanced subset (primary baseline)
chaosbench eval --provider ollama --model qwen2.5:7b \
  --subset data/subsets/api_balanced_1k.jsonl \
  --workers 6 --retries 1

# Local Ollama — 5k subset
chaosbench eval --provider ollama --model qwen2.5:14b \
  --subset data/subsets/api_balanced_5k.jsonl \
  --workers 2 --retries 1

# Full canonical (40,886 items) — slow
chaosbench eval --provider ollama --model qwen2.5:7b \
  --dataset canonical --workers 6
```

All run outputs go to `runs/<run_id>/` (gitignored).

### Create / Verify Locked Subsets

```bash
# Create 1k subset (seed=42, balanced)
python scripts/make_api_subset.py \
  --data_dir data/ --out_path data/subsets/api_balanced_1k.jsonl \
  --size 1000 --balance --seed 42

# Create 5k subset
python scripts/make_api_subset.py \
  --data_dir data/ --out_path data/subsets/api_balanced_5k.jsonl \
  --size 5000 --balance --seed 42
```

Manifest SHA256 (`api_balanced_1k`): `1ac9f1af107b626a`
Manifest SHA256 (`api_balanced_5k`): `de9b10fca80b6dae`

### Legacy Usage

### Configuration Options

```bash
# Chain-of-thought reasoning
python run_benchmark.py --model gpt4 --mode cot

# Limit number of workers (for rate limiting)
python run_benchmark.py --model llama3 --mode zeroshot --workers 2

# Limit number of questions (for testing)
python run_benchmark.py --model gpt4 --mode zeroshot --max-items 100

# Custom output directory
python run_benchmark.py --model gpt4 --mode zeroshot --out-dir results_custom
```

### Sharded Execution

For large-scale runs, shard the dataset across multiple workers:

```bash
# Run shard 0 of 4
python run_benchmark.py --model gpt4 --mode zeroshot --num-shards 4 --shard-index 0

# Run shard 1 of 4
python run_benchmark.py --model gpt4 --mode zeroshot --num-shards 4 --shard-index 1

# ... repeat for shards 2 and 3

# Merge results
python scripts/merge_sharded_runs.py --model gpt4 --mode zeroshot --results-dir results
```

Sharding divides the dataset by question ID and ensures no overlap between shards.

## Reporting Format

### Output Directory Structure

Each run creates a directory: `results/<model>_<mode>/`

```
results/gpt4_zeroshot/
├── summary.json           # Overall results
├── run_meta.json          # Run configuration
├── accuracy_by_task.csv   # Task-level accuracy
├── metrics_overview.csv   # Summary metrics table
├── completions.jsonl      # Raw model outputs (optional)
└── run_manifest.json      # Reproducibility manifest
```

### Summary JSON Schema

**File: `summary.json`**

```json
{
  "overall_accuracy": 0.94,
  "coverage": 0.998,
  "total_evaluated": 620,
  "total_questions": 41507,
  "num_unknown": 1,
  "split_accuracy": {
    "core": 0.95,
    "robustness": 0.92,
    "heldout_systems": 0.91,
    "heldout_templates": 0.93,
    "hard": 0.88
  },
  "task_accuracy": {
    "atomic": 0.96,
    "multi_hop": 0.93,
    "indicator_diagnostics": 0.94,
    "regime_transition": 0.90,
    "fol_inference": 0.91,
    "cross_indicator": 0.89,
    "extended_systems": 0.92,
    "consistency_paraphrase": 0.95,
    "adversarial": 0.87,
    "perturbation_robustness": 0.94
  },
  "fol_violation_rate": 0.08,
  "contradiction_rate": 0.02,
  "dialogue_accuracy": 0.75
}
```

### Run Metadata Schema

**File: `run_meta.json`**

```json
{
  "model": "gpt4",
  "mode": "zeroshot",
  "timestamp": "2026-02-17T10:30:00Z",
  "num_workers": 4,
  "total_questions": 41507,
  "num_items_no_gold": 1,
  "dataset_version": "2.0.0",
  "config_hash": "a3f7b9c1e4d8...",
  "execution_time_seconds": 1245.3,
  "api_success_rate": 0.999
}
```

### Reproducibility Manifest

**File: `run_manifest.json`**

Generated for each run to enable exact reproduction:

```json
{
  "run_id": "gpt4_zeroshot_20260217_103000",
  "model_config": {
    "model": "gpt4",
    "mode": "zeroshot",
    "temperature": 0.0,
    "max_tokens": 100
  },
  "dataset_config": {
    "version": "2.0.0",
    "manifest_hash": "fd3a1e056e90...",
    "total_questions": 621
  },
  "environment": {
    "python_version": "3.11.5",
    "platform": "Linux-5.15.0-x86_64",
    "seed": 42
  },
  "execution": {
    "num_workers": 4,
    "sharded": false,
    "start_time": "2026-02-17T10:30:00Z",
    "end_time": "2026-02-17T10:50:45Z"
  }
}
```

## Validation

### Pre-Evaluation Checks

Before running evaluation:

1. Verify dataset integrity:
   ```bash
   python scripts/validate_v2.py --strict
   ```

2. Check API credentials:
   ```bash
   # Ensure .env file is configured
   cat .env | grep API_KEY
   ```

3. Test with small sample:
   ```bash
   python run_benchmark.py --model gpt4 --mode zeroshot --max-items 10
   ```

### Post-Evaluation Validation

After evaluation completes:

1. Verify summary.json exists and parses
2. Check coverage ≥ 0.95 (less than 5% UNKNOWN)
3. Validate no missing ground truth beyond expected count
4. Compare results across splits for consistency

## Error Handling

### API Errors

- **Rate Limits**: Automatically retry with exponential backoff
- **Timeout**: Skip question and mark as UNKNOWN
- **Authentication**: Fail immediately with clear error message

### Invalid Responses

- **Unparseable**: Mark as UNKNOWN
- **Empty**: Mark as UNKNOWN
- **Non-binary**: Attempt normalization, else UNKNOWN

### Checkpoint Recovery

Evaluations save checkpoints every 50 questions:

```bash
# Resume from checkpoint
python run_benchmark.py --model gpt4 --mode zeroshot --resume
```

Checkpoints are saved to: `results/<model>_<mode>/checkpoints/`

## Best Practices

### Worker Configuration

Recommended worker counts by provider:
- OpenAI (GPT-4): 4-8 workers
- Anthropic (Claude): 2-4 workers
- Google (Gemini): 4-8 workers
- HuggingFace: 2 workers

Higher worker counts increase throughput but may trigger rate limits.

### Evaluation Order

For complete benchmark evaluation:

1. Run zero-shot first (baseline)
2. Run chain-of-thought (comparison)
3. Compare metrics across modes
4. Analyze task-specific differences

### Reproducibility

To ensure reproducible results:

1. Use fixed temperature (0.0 recommended)
2. Record model version/checkpoint
3. Save complete run_manifest.json
4. Archive completions.jsonl for debugging

## References

- Dataset specification: `docs/V2_SPEC.md`
- Metric definitions: `eval_chaosbench.py`
- Result aggregation: `scripts/aggregate_results.py`
