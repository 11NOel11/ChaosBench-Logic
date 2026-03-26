# Results (v2.0.0)

This page defines the official results surface for ChaosBench-Logic v2.

## Authoritative Sources

- Published run index: `published_results/runs/README.md`
- Published run artifacts: `published_results/runs/<run_id>/`
- Run audit report: `artifacts/runs_audit/RUNS_AUDIT.md`
- Paper asset tables: `artifacts/paper_assets/`

## What Is Reported

For each run, the official metrics are:

- `coverage`
- `accuracy_valid`
- `effective_accuracy`
- `balanced_accuracy`
- `mcc`

Run-level artifacts include:

- `run_manifest.json` (reproducibility metadata)
- `metrics.json` (machine-readable metrics)
- `summary.md` (human-readable snapshot)
- `publish_receipt.json` (publish provenance)

Subset runs may additionally include `predictions_subset.jsonl.gz`.

## Official v2 Result Policy

Use `docs/RUNS_POLICY.md` for official vs exploratory labeling. In short:

- Must use canonical selector and matching dataset SHA.
- Must be traceable to a reproducible run manifest.
- Must state scope clearly (subset vs full canonical).

## Current v2 Result Registry

The maintained registry is the auto-generated table in:

- `published_results/runs/README.md`

That index is updated by `chaosbench publish-run` and should be treated as the
single source for run listing.

## Reproducing Result Tables

```bash
# Audit runs and regenerate tables/reports
uv run python scripts/analyze_runs.py \
  --runs_dir runs \
  --out_dir artifacts/runs_audit \
  --paper_assets_dir artifacts/paper_assets
```

Generated assets include:

- `artifacts/runs_audit/summary.json`
- `artifacts/runs_audit/RUNS_AUDIT.md`
- `artifacts/paper_assets/baselines_table.csv`
- `artifacts/paper_assets/baselines_by_family.csv`
- `artifacts/paper_assets/baselines_table.md`

## v1 vs v2 Separation

- v2 is the official benchmark line (`40,886` canonical questions).
- v1 (archived `621`-item set) is historical and must not be merged into v2
  headline comparisons without explicit labeling.

## Practical Review Workflow

1. Open `published_results/runs/README.md` to identify target runs.
2. Inspect `<run_id>/run_manifest.json` for provenance and SHA data.
3. Inspect `<run_id>/metrics.json` for per-family and per-split metrics.
4. Confirm audit verdict in `artifacts/runs_audit/RUNS_AUDIT.md`.

## Related Documents

- `docs/EVAL_PROTOCOL.md`
- `docs/RUNS_POLICY.md`
- `docs/DATASET.md`
