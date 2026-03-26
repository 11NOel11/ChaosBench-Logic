# Evaluation Protocol (v2.0.0)

This document defines the official evaluation workflow for ChaosBench-Logic v2.

## Scope

- Canonical dataset target: `data/v22_*.jsonl` (40,886 items).
- Canonical selector: `data/canonical_v2_files.json`.
- Dataset identity source of truth: `data/v2_manifest.json`.
- Runtime entrypoint: `chaosbench eval`.

## Evaluation Objective

Each item is evaluated as binary reasoning over natural-language questions with
ground truth in `{TRUE, FALSE}`.

Runner-level outcomes are:

- `VALID_TRUE`
- `VALID_FALSE`
- `INVALID` (unparseable response)

If first-pass parsing is `INVALID`, the runner issues one reprompt by default
(`--retries 1`).

## Standard Run Settings

Unless a study explicitly states otherwise, use:

- `temperature = 0.0`
- strict parsing enabled (default)
- `retries = 1`
- deterministic seeds (`--seed 42`)
- canonical order (unless using explicit `--shuffle-seed`)

## Primary Metrics

Official reporting should include:

- `coverage = valid / total`
- `accuracy_valid = correct / valid`
- `effective_accuracy = coverage * accuracy_valid`
- `balanced_accuracy = (TPR + TNR) / 2`
- `mcc` (Matthews correlation coefficient)

Use `effective_accuracy` as the top-line robustness metric, while reporting
`coverage` and `accuracy_valid` alongside it.

## Per-Group Metrics

The runner also computes:

- Per-family metrics (`metrics.json -> per_family`)
- Per-split metrics (`metrics.json -> per_split`)

Core families for v2 reporting:

- `atomic`
- `multi_hop`
- `consistency_paraphrase`
- `perturbation_robustness`
- `adversarial`
- `fol_inference`
- `indicator_diagnostics`
- `regime_transition`
- `cross_indicator`
- `extended_systems`

## Run Tiers

Use tiered execution for cost control and reproducibility:

1. `Smoke` — `--provider mock --max-items 50`
2. `Subset` — fixed subsets (e.g., 1k / 5k)
3. `Canonical Full` — full 40,886-item evaluation

## Canonical Commands

```bash
# 1) Freeze/check dataset fingerprint
uv run chaosbench freeze

# 2) Smoke validation (no network)
uv run chaosbench eval --provider mock --dataset canonical --max-items 50

# 3) Provider run (example)
uv run chaosbench eval --provider openai --model gpt-4o --dataset canonical --workers 4

# 4) Resume interrupted run
uv run chaosbench eval --provider openai --model gpt-4o --dataset canonical --resume <run_id>
```

## Output Artifacts

Each run writes to `runs/<run_id>/` (gitignored):

- `predictions.jsonl`
- `metrics.json`
- `summary.md`
- `run_manifest.json`
- `.eval_checkpoint.jsonl` (temporary; removed on successful completion)

## Official Run Criteria

A run is considered official when it satisfies policy in `docs/RUNS_POLICY.md`,
including canonical selector and SHA alignment against freeze artifacts.

## Publishing and Audit

```bash
# Publish lightweight artifacts for tracking
uv run chaosbench publish-run --run runs/<run_id>

# Audit local + published runs and regenerate paper assets
uv run python scripts/analyze_runs.py --runs_dir runs --out_dir artifacts/runs_audit --paper_assets_dir artifacts/paper_assets
```

Published artifacts are stored in `published_results/runs/`.

## Reporting Rules

- Do not mix archived v1 results with v2 in the same headline table.
- Report subset results as subset results; do not present them as full-canonical.
- Include run ID, provider/model, and dataset scope in every table.
- For paper-facing claims, point to `published_results/runs/README.md` and
  `artifacts/runs_audit/RUNS_AUDIT.md`.

## Related Documents

- `docs/RUNS_POLICY.md`
- `docs/API_SETUP.md`
- `docs/RESULTS.md`
