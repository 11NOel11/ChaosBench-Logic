# Changelog

All notable changes to ChaosBench-Logic are documented in this file.

## [2.2.0] - 2026-02-18

### Dataset
- Scaled dataset to **21,037 v2.2 questions** across 10 task families (21,658 total with archived v1)
- 10 canonical data files (`data/v22_*.jsonl`) replace intermediate batch8-18 naming
- **165 dynamical systems**: 30 core (manually curated) + 135 from dysts library
- **15 predicates**: added Dissipative, Bounded, Mixing, Ergodic to original 11
- Up to **4-hop reasoning chains** enabled by ontology extension

### Task Family Counts
| Family | Questions |
|--------|----------:|
| atomic | 10,890 |
| multi_hop | 3,500 |
| consistency_paraphrase | 3,188 |
| perturbation_robustness | 1,994 |
| adversarial | 598 |
| indicator_diagnostics | 547 |
| fol_inference | 140 |
| regime_transition | 68 |
| cross_indicator | 67 |
| extended_systems | 45 |

### Quality Gates
- Near-duplicate detection: PASSED (6 exact, 0 near-duplicates)
- Label leakage scan: PASSED (0 leaks)
- Difficulty distribution: PASSED (avg 0.324)
- Class balance: 56% TRUE overall (2 small families out of range; exempted by size)

### Code Changes
- Added `ELIGIBILITY_PREDICATES` to protect eligibility checks from ontology extension
- Merged dysts indicator keys into `CORE_INDICATOR_KEYS`
- Fixed multi_hop affirmative chain ground truth from "TRUE" to "YES"
- Updated `PREDICATE_DISPLAY` in all 5 task modules

### Infrastructure
- CI rewritten to validate `v22_*.jsonl` canonical files
- Coverage target changed from `eval_chaosbench` to `chaosbench`
- Archived stale generation configs to `configs/archive/`
- Intermediate batch8-18 files archived to `data/archive/v21_intermediate/`
- Sprint markdown moved to `docs/archive/`

### Documentation
- All docs updated to reflect 21,037 questions, 15 predicates, 165 systems
- ONTOLOGY.md extended with 4 new predicate definitions
- ONTOLOGY_V2_EXTENSION.md status changed from PROPOSED to IMPLEMENTED
- CITATION.cff updated to v2.2.0
- Added QUALITY_STANDARD.md and ONTOLOGY_V2_EXTENSION.md to docs index

## [2.1.0] - 2026-02-17

### Dataset
- Intermediate scaling: 11 batches (batch8-18) with expanded task families
- New families: indicator diagnostics, regime transitions, FOL inference, cross-indicator, adversarial, consistency paraphrases, extended systems
- 30 core systems + 136 dysts-imported systems
- Empirically validated indicator thresholds

## [1.0.0] - 2025-01-01

### Initial Release
- 621 questions across 7 batches and 27 dynamical systems
- 17 task types in 7 reasoning categories
- Support for GPT-4, Claude-3.5, Gemini-2.5, LLaMA-3 70B
- Zero-shot and chain-of-thought evaluation modes
