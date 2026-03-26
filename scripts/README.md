# Scripts Overview

This directory contains operational scripts for dataset builds, evaluation runs,
run auditing, and camera-ready packaging.

## Structure

- `build_*.py` - dataset/result/pack builders
- `validate_*.py` - dataset and artifact validators
- `run_*.py` - evaluation and experiment launchers
- `analyze_*.py` - metrics and audit analysis
- `cluster/` - SLURM template and cluster helper assets

## Common Workflows

```bash
# Validate canonical v2 dataset
uv run python scripts/validate_v2.py --strict --max-duplicate-questions 200

# Analyze published runs
uv run chaosbench analyze-runs --runs-dir published_results/runs --out-dir artifacts/runs_audit

# Build camera-ready repo bundle
uv run python scripts/build_camera_ready_repo.py --force
```

## Cluster Use

Use `scripts/run_cluster_eval.py` with the template in `scripts/cluster/slurm_template.sh`.
