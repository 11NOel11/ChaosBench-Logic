# ChaosBench-Logic Dataset (v2.0.0)

This document is the canonical in-repo reference for the released v2 dataset.

## Release Identity

| Field | Value |
|-------|-------|
| Dataset release | v2.0.0 |
| v2 questions | 40,886 |
| v1 archived questions | 621 |
| Total questions | 41,507 |
| Canonical files | 10 (`data/v22_*.jsonl`) |
| Selector file | `data/canonical_v2_files.json` |
| Manifest | `data/v2_manifest.json` |
| Global SHA256 | `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279` |

## Canonical v2 Files

| File | Count |
|------|------:|
| `data/v22_atomic.jsonl` | 25,000 |
| `data/v22_multi_hop.jsonl` | 6,000 |
| `data/v22_consistency_paraphrase.jsonl` | 4,139 |
| `data/v22_perturbation_robustness.jsonl` | 1,994 |
| `data/v22_adversarial.jsonl` | 1,285 |
| `data/v22_fol_inference.jsonl` | 1,758 |
| `data/v22_indicator_diagnostics.jsonl` | 530 |
| `data/v22_regime_transition.jsonl` | 68 |
| `data/v22_cross_indicator.jsonl` | 67 |
| `data/v22_extended_systems.jsonl` | 45 |

## Schema

Each JSONL record uses this schema:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | string | yes | unique question identifier |
| `question` | string | yes | natural language prompt |
| `ground_truth` | string | yes | `TRUE` or `FALSE` for v2 |
| `type` | string | yes | task family label |
| `system_id` | string or null | yes | null for system-agnostic ontology/FOL items |
| `template` | string | yes | template version (for v2, typically `V2`) |

## Coverage

- Systems: 165 total (30 manually curated core + 135 imported from `dysts`).
- Predicates: 27 logical predicates used in ontology and reasoning chains.
- Task families: 10 families spanning atomic, multi-hop, adversarial, indicator-based, and robustness reasoning.

## Archive Policy

- v2 files (`v22_*.jsonl`) are the default evaluation target.
- v1 is preserved at `data/archive/v1/` for historical baseline comparisons only.

## Validation and Reproducibility

- Manifest and per-file hashes: `data/v2_manifest.json`.
- Canonical file selector for tooling: `data/canonical_v2_files.json`.
- Freeze artifact generation: `uv run python scripts/freeze_v2_dataset.py`.
- Strict validation command: `uv run python scripts/validate_v2.py --strict --max-duplicate-questions 200`.

## External Publication

- Repository: `https://github.com/11NOel11/ChaosBench-Logic`
- Hugging Face dataset: `https://huggingface.co/datasets/11NOel11/ChaosBench-Logic`

For expanded narrative metadata used in external publishing workflows, see `../DATASET_CARD.md`.
