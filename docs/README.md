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

---

## Quick Reference

**Build dataset:**
```bash
python scripts/build_v2_dataset.py --config configs/generation/v2_2_scale_full.yaml
```

**Validate dataset:**
```bash
python scripts/heavy_verify_dataset.py
python scripts/heavy_verify_splits.py
python scripts/heavy_verify_ontology.py
```

**Run pre-freeze check:**
```bash
python scripts/pre_freeze_check.py
```

**Run evaluation:**
```bash
python eval_chaosbench.py --config configs/eval/gpt4_zeroshot.yaml
```

**Check repo hygiene:**
```bash
python scripts/repo_hygiene.py
python -m pytest tests/test_repo_hygiene.py -v
```

See [../README.md](../README.md) for main repository documentation.
