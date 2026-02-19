# Repository Policy

This document defines what belongs in the git repository, where generated
outputs should go, and how to regenerate paper assets and reports.

---

## What Belongs in Git

### Always Tracked

| Category | Examples |
|----------|---------|
| Source code | `chaosbench/**/*.py`, `scripts/*.py`, `tests/*.py` |
| Canonical dataset files | `data/v22_*.jsonl`, `data/v2_manifest.json` |
| Archived dataset batches | `data/archive/**/*.jsonl` |
| CI smoke data | `data/ci_smoke/` |
| Configuration | `configs/**/*.yaml`, `pyproject.toml`, `uv.lock` |
| System definitions | `systems/**/*.json` |
| Canonical documentation | `docs/*.md` (not `docs/archive/`) |
| Root documentation | `README.md`, `DATASET_CARD.md`, `CHANGELOG.md`, `CITATION.cff` |
| Licensing | `LICENSE`, `LICENSE_DATA` |
| Published results | `published_results/` |
| CI configuration | `.github/workflows/` |

### Historical Notes (Tracked in docs/archive/)

Files documenting decisions, summaries, and upgrade notes that have historical
value but are not canonical documentation:

- `docs/archive/PRE_FREEZE_SUMMARY.md`
- `docs/archive/V2_COMPLETION_SUMMARY.md`
- `docs/archive/V2_UPGRADE_SUMMARY.md`
- `docs/archive/ONTOLOGY_V2_EXTENSION.md`

Add new archive entries when completing a major version cycle.

---

## What Must NOT Be Tracked

All generated outputs go into `artifacts/` (gitignored) or other ignored
directories. Never `git add` these:

| Path | Contents |
|------|---------|
| `artifacts/` | All generated outputs |
| `reports/` | Pipeline reports, AI-generated summaries |
| `runs/` | Evaluation run outputs |
| `workspace/` | Local working materials, paper drafts |
| `scratch/` | Scratch computations |
| `tmp/` | Temporary files |
| `figures/` | Generated figures (regenerate from scripts) |
| `results/` | Local evaluation results |
| `data/backup_*/` | Backup snapshots |
| `*.log`, `*.out` | Log files |

### Special Rules for Data

- `data/batch*.jsonl` — **ignored** (v1/intermediate generated batches)
- `data/v22_*.jsonl` — **tracked** (canonical v2 dataset files)
- `data/v2_manifest.json` — **tracked** (canonical manifest)
- `data/archive/` — **tracked** (historical archived batches)
- `data/subsets/` — **tracked** (reproducible API subsets)
- `data/ci_smoke/` — **tracked** (CI smoke test data)
- `data/backup_*/` — **ignored** (local backups)

---

## Generated Outputs Directory Structure

```
artifacts/
├── reports/          # Pipeline and quality reports
├── paper_assets/     # Paper figures, tables (regenerate via scripts)
├── logs/             # Script and evaluation logs
├── tmp/              # Temporary computation outputs
├── repo_cleanup/     # Hygiene audit outputs
└── heavy_verify/     # Dataset verification reports
```

---

## How to Regenerate Paper Assets and Reports

### Dataset

```bash
# Regenerate v2 dataset
python scripts/build_v2_dataset.py --config configs/generation/v2_2_scale_full.yaml

# Run CI smoke build
python scripts/build_v2_dataset.py --config configs/generation/ci_smoke.yaml
```

### Quality Reports

```bash
# Pre-freeze quality gates
python scripts/pre_freeze_check.py

# Heavy verification (schema, duplicates, splits, ontology)
python scripts/heavy_verify_dataset.py
python scripts/heavy_verify_splits.py
python scripts/heavy_verify_ontology.py
```

### Paper Tables and Figures

```bash
# Generate paper tables (outputs to artifacts/paper_assets/)
python scripts/generate_paper_tables.py --output artifacts/paper_assets/

# Generate figures (outputs to artifacts/paper_assets/)
python scripts/generate_figures.py --output artifacts/paper_assets/
```

### Evaluation Reports

```bash
# Aggregate evaluation results
python scripts/aggregate_results.py --results-dir results/ --output artifacts/reports/

# Merge sharded evaluation runs
python scripts/merge_sharded_runs.py --runs-dir runs/ --output artifacts/reports/
```

---

## Naming Conventions

### Dataset Files

- Canonical v2: `data/v22_<family>.jsonl`
- Archived v1: `data/archive/v1/batch<N>_<name>.jsonl`
- Archived intermediate: `data/archive/v21_intermediate/batch<N>_<name>.jsonl`
- CI smoke: `data/ci_smoke/<name>.jsonl`
- API subsets: `data/subsets/<name>.jsonl`

### Scripts

- Generation: `scripts/build_*.py`
- Validation: `scripts/validate_*.py`, `scripts/heavy_verify_*.py`
- Analysis: `scripts/analyze_*.py`, `scripts/report_*.py`
- Utilities: `scripts/*_utils.py`

### Tests

- Unit tests: `tests/test_<module>.py`
- Integration tests: `tests/test_*_integration.py`
- CI smoke tests: `tests/test_*_smoke.py`

### Documentation

- Canonical: `docs/<TOPIC>.md` (UPPERCASE topic name)
- Historical: `docs/archive/<TOPIC>_<context>.md`

---

## Hygiene Enforcement

Run the hygiene checker before any commit that touches `docs/` or the root:

```bash
# Dry run (shows what would be flagged)
python scripts/repo_hygiene.py

# Apply moves
python scripts/repo_hygiene.py --apply

# CI check (exits non-zero if violations found)
python -m pytest tests/test_repo_hygiene.py -v
```

The hygiene checker enforces:
1. No `*SUMMARY*.md` or `*REPORT*.md` files at repo root
2. `artifacts/` is gitignored
3. `reports/` and `runs/` are not tracked
4. No `/tmp` paths in tracked documents
5. Results files not tracked outside `published_results/`
