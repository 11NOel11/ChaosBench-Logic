# CARE-v3 Repair Protocol

## Goal

Define a deterministic, model-agnostic post-hoc repair layer for ChaosBench-Logic
predictions that improves logical coherence without using reference labels during
repair decisions.

## Inputs

- `predictions.jsonl` records with `id`, `question`, `parsed_label`, `task_family`
- Canonical selector index (`data/canonical_v2_files.json`) for `system_id` lookup
- Ontology axioms from `chaosbench.logic.axioms.get_fol_rules()`
- Optional group invariants for:
  - `consistency_paraphrase` groups
  - perturbation paraphrase/distractor groups

## Outputs

- `repaired_predictions.jsonl`
  - adds `repaired_label`, `was_flipped`, `flip_reason`
- `repaired_metrics.json`
  - pre/post metrics and deltas
- `repair_manifest.json`
  - config, constraint hash, flip counts, violation counts

## Constraints

1. Ontology constraints (hard)
   - implication and exclusion rules from FOL axioms
2. Group consistency constraints (optional)
   - enforce majority predicate truth within eligible groups
3. Invalid output policy
   - keep invalid rows unchanged when `leave_invalid_unchanged=true`

## Objective

Minimize label flips while satisfying ontology consistency. The current
implementation uses MaxSAT (`repair_assignment`) with unit soft penalties,
plus optional deterministic group-majority post-processing.

## No-Ground-Truth Rule

Repair decisions must not read `ground_truth`. Reference labels are allowed only
for post-repair evaluation metrics.

## Acceptance Criteria

- Deterministic for fixed config and input order
- Reproducible via `constraint_hash(config + fol_rules + negation patterns)`
- Auditable with manifest-level counts:
  - system assignments
  - predicate flips
  - row flips
  - axiom violations pre/post
  - group inconsistency pre/post

## M5 Transfer Calibration

The instance-level guardrail (`M5`) supports provider-conditioned thresholds for
prompt-variant transfer.

- Stable threshold map:
  - `chaosbench/repair/m5_provider_thresholds_crossfit_v1.json`
- One-command calibration/apply/compare cycle:
  - `python scripts/run_m5_crossfit_cycle.py`
- Calibrate only (cross-fit):
  - `python scripts/calibrate_m5_provider_thresholds.py`
- Apply M5 with explicit provider map:
  - `python scripts/run_m5_instance_guardrail.py --policy-json <path> --provider-thresholds-json <path> ...`

When `--policy-json` is provided without `--provider-thresholds-json`,
`run_m5_instance_guardrail.py` auto-loads the stable map above (if present).
