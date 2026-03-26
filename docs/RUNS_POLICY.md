# Runs Policy (v2.0.0)

This document defines storage, publication, and audit rules for evaluation runs
in ChaosBench-Logic v2.

## Purpose

- Keep live evaluation outputs reproducible but out of git.
- Keep published artifacts lightweight and auditable.
- Separate official benchmark evidence from exploratory experiments.

## Official Run Criteria

A run is considered official only if all conditions below are satisfied.

| Criterion | Requirement |
|----------|-------------|
| Dataset selector | `canonical_selector == data/canonical_v2_files.json` |
| Dataset fingerprint | `dataset_global_sha256` matches `artifacts/freeze/v2_freeze_manifest.json -> global_sha256` |
| Prompt provenance | `prompt_version` and `prompt_hash` recorded in `run_manifest.json` |
| Reproducibility metadata | `run_manifest.json` includes `git_commit`, runtime settings, and run scope |
| Scope labeling | run is clearly marked as full canonical or documented subset |

Runs that fail any criterion are exploratory and must not be used as headline
paper or leaderboard evidence.

## Storage Layout

### Live Runs (gitignored)

`runs/` is for active/local execution only.

```text
runs/<run_id>/
  predictions.jsonl
  metrics.json
  summary.md
  run_manifest.json
  .eval_checkpoint.jsonl   # transient, removed on successful completion
```

### Published Runs (tracked)

Only lightweight artifacts are tracked in git.

```text
published_results/runs/<run_id>/
  run_manifest.json
  metrics.json
  summary.md
  publish_receipt.json
  [predictions_subset.jsonl.gz]   # optional, subset runs only
```

The maintained run registry is `published_results/runs/README.md`.

## Standard Workflow

```bash
# 1) Freeze and fingerprint canonical dataset
uv run chaosbench freeze

# 2) Execute evaluation (example)
uv run chaosbench eval --provider openai --model gpt-4o --dataset canonical --workers 4

# 3) Publish lightweight run artifacts
uv run chaosbench publish-run --run runs/<run_id>

# 4) Audit runs and regenerate paper assets
uv run python scripts/analyze_runs.py --runs_dir runs --out_dir artifacts/runs_audit --paper_assets_dir artifacts/paper_assets
```

## Dataset Fingerprint Rule

Canonical SHA is computed by `chaosbench.data.hashing.dataset_global_sha256`.

Formula:

```text
sha256(concat over sorted canonical files of "<rel_path>:<file_sha256>:<line_count>\n")
```

This same rule is used by both freeze artifacts and runtime manifests.

## Resume and Checkpointing

- The runner checkpoints to `.eval_checkpoint.jsonl` during execution.
- Resume an interrupted run with `--resume <run_id>`.
- On successful completion, checkpoint files are removed.

## Reporting Rules

- Do not mix archived v1 results with v2 headline reporting.
- Do not present subset runs as full-canonical runs.
- Always report run ID, provider/model, and dataset scope.
- Use audit outputs in `artifacts/runs_audit/` for verification narratives.

## Commit Checklist for Published Runs

1. Confirm run meets official criteria.
2. Publish run with `chaosbench publish-run`.
3. Regenerate audit outputs.
4. Stage only `published_results/runs/<run_id>/` and related docs.
5. Use a clear commit message (for example: `results(<model>): add official v2 canonical run`).

## Related Documents

- `docs/EVAL_PROTOCOL.md`
- `docs/RESULTS.md`
- `docs/DATASET.md`
