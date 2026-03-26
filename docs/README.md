# ChaosBench-Logic Documentation

This directory contains canonical documentation for ChaosBench-Logic v2.

---

## Canonical Documentation

| Document | Description |
|----------|-------------|
| [DATASET.md](DATASET.md) | Dataset structure, fields, statistics, and validation |
| [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) | Evaluation protocol, metrics, and reporting standards |
| [ONTOLOGY.md](ONTOLOGY.md) | Predicate definitions and first-order logic axioms |
| [V2_SPEC.md](V2_SPEC.md) | Complete v2 dataset specification: schema, splits, generation protocol |
| [QUALITY_STANDARD.md](QUALITY_STANDARD.md) | Quality gates and validation standards |
| [FREEZE_PLAN.md](FREEZE_PLAN.md) | Dataset freezing plan and criteria |
| [RELEASE_NOTES_V2.md](RELEASE_NOTES_V2.md) | v2 release notes: what changed, dataset hash, baseline results |
| [CAMERA_READY_SUBMISSION.md](CAMERA_READY_SUBMISSION.md) | Final camera-ready submission text package |
| [CAMERA_READY_LINKS.md](CAMERA_READY_LINKS.md) | Final repo and dataset links for camera-ready insertion |
| [CAMERA_READY_REPO.md](CAMERA_READY_REPO.md) | Build command and outputs for camera-ready repository bundle |
| [CAMERA_READY_REPO_STATUS.md](CAMERA_READY_REPO_STATUS.md) | Current readiness status and output artifact locations |
| [CAMERA_READY_PUSH_PLAN.md](CAMERA_READY_PUSH_PLAN.md) | Exact include/exclude scope for camera-ready push |
| [CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md) | Mapping from paper claims to verifiable artifacts |
| [OFFICIAL_REPO_V2_PLAN.md](OFFICIAL_REPO_V2_PLAN.md) | Planned official public layout for v2 repository release |
| [REPO_POLICY.md](REPO_POLICY.md) | Repository hygiene: what belongs in git, where outputs go |
| [FUTURE_WORK.md](FUTURE_WORK.md) | Deferred items and v3 candidates |

## Technical Guides

| Document | Description |
|----------|-------------|
| [API_SETUP.md](API_SETUP.md) | Setting up API keys for model evaluation |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines and development workflow |
| [RESULTS.md](RESULTS.md) | Published baseline results and analysis |
| [INDICATOR_COMPUTATION.md](INDICATOR_COMPUTATION.md) | Chaos indicator computation methods |
| [INDICATOR_THRESHOLDS.md](INDICATOR_THRESHOLDS.md) | Empirically validated indicator thresholds |
| [CACHE_USAGE.md](CACHE_USAGE.md) | Response caching system for evaluation |
| [SCALING_ROADMAP.md](SCALING_ROADMAP.md) | Scaling roadmap and future architecture |

## Archive

The [`archive/`](archive/) directory contains historical notes that document
decisions and transitions between versions. These files are tracked for
reference but are **not** canonical documentation:

| Document | Description |
|----------|-------------|
| [archive/PRE_FREEZE_SUMMARY.md](archive/PRE_FREEZE_SUMMARY.md) | Pre-freeze quality summary (v2) |
| [archive/V2_COMPLETION_SUMMARY.md](archive/V2_COMPLETION_SUMMARY.md) | V2 completion notes |
| [archive/V2_UPGRADE_SUMMARY.md](archive/V2_UPGRADE_SUMMARY.md) | V2 upgrade notes |
| [archive/ONTOLOGY_V2_EXTENSION.md](archive/ONTOLOGY_V2_EXTENSION.md) | Ontology v2 extension design notes |
| [archive/CHANGELOG_LEGACY.md](archive/CHANGELOG_LEGACY.md) | Legacy changelog moved from repo root |

---

## Quick Reference

**Build dataset:**
```bash
uv run python scripts/build_v2_dataset.py --config configs/generation/v2_2_scale_full.yaml
```

**Validate dataset:**
```bash
uv run python scripts/heavy_verify_dataset.py
uv run python scripts/heavy_verify_splits.py
uv run python scripts/heavy_verify_ontology.py
```

**Run pre-freeze check:**
```bash
uv run python scripts/pre_freeze_check.py
```

**Run evaluation:**
```bash
uv run chaosbench eval --provider mock --dataset canonical --max-items 50
```

**Check repo hygiene:**
```bash
uv run python scripts/repo_hygiene.py
uv run python -m pytest tests/test_repo_hygiene.py -v
```

See [../README.md](../README.md) for main repository documentation.
