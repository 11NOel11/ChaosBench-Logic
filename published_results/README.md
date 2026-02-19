# Published Results

**Note:** These results are from the **v1 dataset** (621 questions, now archived in `data/archive/v1/`). Full evaluation on the v2 dataset (40,886 questions) is pending.

This directory contains the **minimal, verifiable artifacts** used to generate the results tables reported in the main README and documentation.

## Contents

Each configuration subdirectory contains:

- **`summary.json`** - Overall accuracy metrics and per-task-type breakdown
- **`run_meta.json`** - Run configuration metadata (model, mode, workers, item counts)
- **`accuracy_by_task.csv`** - Accuracy breakdown by task type (CSV format)
- **`metrics_overview.csv`** - Summary metrics table (CSV format)

## Schema

### summary.json
```json
{
  "overall_accuracy": float,           // Overall accuracy (0-1)
  "task_accuracy": {                   // Per-task-type accuracy
    "atomic": float,
    "implication": float,
    "bias": float,
    "counterfactual": float,
    "multi_turn": float,
    ...
  },
  "dialogue_accuracy": float,          // Multi-turn dialogue accuracy
  "contradiction_rate": float,         // Contradiction rate (lower is better)
  "bias_error": {}                     // Bias error breakdown (if applicable)
}
```

### run_meta.json
```json
{
  "model_name": string,                // Model identifier
  "mode": string,                      // "zeroshot" or "cot"
  "max_workers": int,                  // Parallel workers used
  "num_items_total": int,              // Total items in dataset
  "num_items_evaluated": int,          // Items successfully evaluated
  "num_items_unanswered": int,         // Items with no answer
  "num_items_no_gold": int,            // Items without ground truth
  "timestamp": string                  // Run timestamp (ISO format)
}
```

## What's NOT Included

The following artifacts are **intentionally excluded** to keep the repository lightweight:

- Raw model completions (`per_item_results.jsonl`) - typically 175KB-1.3MB per run
- Debug samples (`debug_samples.jsonl`) - 4-24KB per run
- Checkpoint files (`checkpoint.json`) - 181KB-965KB, used for run resumption
- Run logs (`run.log`) - execution logs
- Visualization charts (`*.png`) - generated plots
- Empty/debug CSVs (`bias_errors.csv`) - internal debugging artifacts

## Reproducing Results

To verify the published results:

```bash
# View summary for any configuration
cat published_results/gpt4_zeroshot/summary.json | python3 -m json.tool

# Aggregate all results into a markdown table
python scripts/aggregate_results.py
```

To regenerate full raw outputs locally:

```bash
# Run benchmark (requires API keys in .env)
./run_benchmark.py --model gpt-4 --mode zeroshot

# Raw outputs will be written to results/ (gitignored)
```

## Configurations Included

- `claude3_zeroshot/` - Claude-3.5-Sonnet (zero-shot)
- `gemini_zeroshot/` - Gemini-2.0-Flash (zero-shot)
- `gpt4_cot/` - GPT-4-Turbo (chain-of-thought)
- `gpt4_zeroshot/` - GPT-4-Turbo (zero-shot)
- `llama3_cot/` - LLaMA-3-70B (chain-of-thought)
- `llama3_zeroshot/` - LLaMA-3-70B (zero-shot)

## Notes

- **No timing/cost data**: Execution time and API costs vary by infrastructure (API load, network latency, worker configuration) and are not reported to avoid misleading comparisons.
- **Offline reproducibility**: All validation scripts (`pytest`, `dataset_stats.py`, `validate_repo_claims.py`) work offline without requiring API access.
- **Ground truth only**: All metrics are computed from the published artifacts in this directory.
