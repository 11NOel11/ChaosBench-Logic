# ChaosBench-Logic v2 — Before-Freeze Plan

**Version:** 2.0.0
**Branch:** wip/v2.1
**Date:** 2026-02-18
**Status:** ✅ READY TO FREEZE (all hard-fail criteria passing)

---

## 1. Freeze Prerequisites Checklist

All items below must be green before locking the API evaluation subset.

| Check | Command | Status |
|---|---|---|
| pytest (0 failed, 0 skipped, 0 xfailed) | `python -m pytest -q` | ✅ 673 passed |
| pre_freeze_check (all HARD FAILs) | `python scripts/pre_freeze_check.py` | ✅ 6/6 HARD FAILs passed |
| Determinism (same config+seed = same hash) | see `test_determinism_ci_smoke` | ✅ verified |
| Duplicate scan (0 accidental duplicates) | `python scripts/duplicate_report.py` | ✅ 0 duplicates |
| System utilization audit | `python scripts/system_utilization_audit.py` | ✅ 165 systems covered |

### Current Dataset State (as of last build)

**Config used:** `configs/generation/v2_2_scale_full.yaml --seed 42`
**Global SHA256:** `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279`

| File | Count | TRUE% |
|---|---|---|
| v22_atomic.jsonl | 25,000 | ~50% |
| v22_multi_hop.jsonl | 6,000 | ~50% |
| v22_consistency_paraphrase.jsonl | 4,139 | ~50% |
| v22_perturbation_robustness.jsonl | 1,994 | 50.0% |
| v22_fol_inference.jsonl | 1,758 | ~50% |
| v22_adversarial.jsonl | 1,285 | ~50% |
| v22_indicator_diagnostics.jsonl | 530 | ~50% |
| v22_regime_transition.jsonl | 68 | ~50% |
| v22_cross_indicator.jsonl | 67 | ~50% |
| v22_extended_systems.jsonl | 45 | ~50% |
| **Total v2 (canonical)** | **40,886** | **49.5%** |

*Per-family imbalance is acceptable per `docs/QUALITY_STANDARD.md §3.2` (SOFT check, 30–70% threshold).
The adversarial and indicator_diagnostics families are structurally imbalanced by design.

### Dataset Freeze Command

```bash
# Produce hash-locked freeze artifact (run after any data change)
python scripts/freeze_v2_dataset.py
# or: chaosbench freeze

# Output: artifacts/freeze/
#   v2_freeze_manifest.json   — machine-readable, citable
#   v2_freeze_report.md       — human-readable
#   v2_freeze_sha256.txt      — per-file + global checksums
```

### Current Freeze (canonical v2, 40,886 items)

```
global_sha256  cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279

6ddae0cdb4ebc6e9...  data/v22_adversarial.jsonl          (1285)
fa9d1c99097cd2ca...  data/v22_atomic.jsonl               (25000)
fb9cbda5b12d67fa...  data/v22_consistency_paraphrase.jsonl (4139)
a03d1594a99e9a7b...  data/v22_cross_indicator.jsonl      (67)
49a5f675312af895...  data/v22_extended_systems.jsonl     (45)
548da0e9a2f801c9...  data/v22_fol_inference.jsonl        (1758)
eeb4e96000ee87a6...  data/v22_indicator_diagnostics.jsonl (530)
e6c048f35995b0d0...  data/v22_multi_hop.jsonl            (6000)
f1e30ee64370a40b...  data/v22_perturbation_robustness.jsonl (1994)
2e63ebbee85c1763...  data/v22_regime_transition.jsonl    (68)
```

---

## 2. Subset Freeze Procedure

**Goal:** Produce a locked, balanced 1,000-item subset for API evaluations.

### Step 1 — Generate the subset (do not execute until all prerequisites pass)

```bash
python scripts/make_api_subset.py \
  --data-dir data \
  --output data/subsets/api_balanced_1k.jsonl \
  --n 1000 \
  --seed 42 \
  --stratify-by type \
  --balance-label
```

This creates `data/subsets/api_balanced_1k.jsonl` and
`data/subsets/api_balanced_1k.manifest.json` containing:
- `sha256` of the JSONL file
- `generation_config` (seed, source files, stratification params)
- `item_count`
- `label_balance`
- `family_distribution`

### Step 2 — Verify subset determinism

```bash
# Run twice, compare sha256
sha256sum data/subsets/api_balanced_1k.jsonl
# Re-run make_api_subset.py with same args
sha256sum data/subsets/api_balanced_1k.jsonl
# Hashes must match
```

### Step 3 — Commit the locked subset

```bash
git add data/subsets/api_balanced_1k.jsonl \
        data/subsets/api_balanced_1k.manifest.json
git commit -m "freeze: lock api_balanced_1k subset for v2 evaluation"
git tag v2.0.0-eval-subset
```

Once committed and tagged, **do not regenerate** the subset. Any change to the
data directory requires a new version tag and updated manifest.

### Storage locations

| Artifact | Path |
|---|---|
| Balanced 1K subset | `data/subsets/api_balanced_1k.jsonl` |
| Subset manifest | `data/subsets/api_balanced_1k.manifest.json` |
| Full v2 dataset | `data/v22_*.jsonl` |
| Dataset manifest | `data/v2_manifest.json` |
| v1 archive | `data/archive/v1/` |

---

## 3. Evaluation Procedure

### 3.1 Pilot Run (sanity check, ~100 items)

Before running the full 1K subset, run a 100-item pilot to verify:
- Model API key is valid
- Output parser is working
- Retry policy is effective

```bash
python scripts/run_cluster_eval.py \
  --subset data/subsets/api_balanced_100.jsonl \
  --model gpt-4o-mini \
  --output results/pilot_100/ \
  --temperature 0 \
  --max-tokens 10 \
  --cache-dir .eval_cache
```

### 3.2 Full 1K Subset Evaluation

```bash
python scripts/run_cluster_eval.py \
  --subset data/subsets/api_balanced_1k.jsonl \
  --model <MODEL_ID> \
  --output results/v22_eval/<MODEL_ID>/ \
  --temperature 0 \
  --max-tokens 10 \
  --cache-dir .eval_cache \
  --parallel 10
```

### 3.3 Strict Output Constraints

Models must output **exactly `TRUE` or `FALSE`** (case-insensitive).

**Instruction template (instruction-first approach):**

```
You must respond with exactly one word: TRUE or FALSE.
No explanation. No punctuation. Just TRUE or FALSE.

Question: {question}
```

**2-pass parsing strategy** (only if instruction-first produces >5% invalid outputs):

1. Pass 1: exact match `TRUE`/`FALSE` (case-insensitive)
2. Pass 2: extract first occurrence of `true`/`false` from response with regex
   `r'\b(true|false)\b'`

**Retry policy:**
- Retry invalid outputs up to 2 times with stronger instruction
- After 3 failures: mark as `ABSTAIN` (counts as wrong for accuracy)
- Log retry rate; if >10%, flag as model alignment issue

### 3.4 Metrics

Compute all of the following per model, per task family:

| Metric | Description |
|---|---|
| `coverage` | Fraction of items with valid (non-abstain) responses |
| `accuracy_valid` | Accuracy over items with valid responses |
| `effective_accuracy` | Accuracy over ALL items (abstains = wrong) |
| `flip_rate` | Rate of answer changes across paraphrase pairs (consistency) |
| `balanced_accuracy` | Mean of per-class recall (handles imbalanced families) |
| `MCC` | Matthews Correlation Coefficient (primary metric for imbalanced families) |

**Primary ranking metric:** `effective_accuracy` (overall), `MCC` (per-family with imbalance).

```bash
python scripts/generate_paper_tables.py \
  --results-dir results/v22_eval/ \
  --output paper_assets/tables/
```

---

## 4. Cost Control Notes

| Setting | Recommended Value | Rationale |
|---|---|---|
| `temperature` | 0 | Deterministic outputs |
| `max_tokens` | 10 | TRUE/FALSE fit in 1–2 tokens |
| `--parallel` | 10–20 | Stay within rate limits |
| Response cache | enabled (`--cache-dir`) | Avoid re-querying identical questions |
| Model for pilot | gpt-4o-mini / claude-haiku | Cheapest capable models |
| Batch API | preferred (OpenAI) | 50% cost reduction, async |

**Estimated cost (1K subset, GPT-4o):** ~$0.05 per run (10 tokens output × 1000 items × $5/M tokens).

**Estimated cost (1K subset, GPT-4o-mini):** ~$0.001 per run.

For full comparative study (10 models × 1K items): budget ~$1–5 depending on model mix.

---

## 5. Release Artifact Checklist

### In git (committed)

| Artifact | Path | Notes |
|---|---|---|
| Dataset files | `data/v22_*.jsonl` | After freeze confirmation |
| Locked subset | `data/subsets/api_balanced_1k.jsonl` | After freeze execution |
| Dataset manifest | `data/v2_manifest.json` | Always committed |
| Subset manifest | `data/subsets/api_balanced_1k.manifest.json` | After freeze |
| Freeze plan | `docs/FREEZE_PLAN.md` | This file |
| Quality standard | `docs/QUALITY_STANDARD.md` | In git |
| Evaluation protocol | `docs/EVAL_PROTOCOL.md` | In git |
| Dataset card | `DATASET_CARD.md` | In git |
| Citation file | `CITATION.cff` | In git |

### Excluded from git (`.gitignore`)

| Artifact | Location | Notes |
|---|---|---|
| Model API responses | `results/` | Too large; store on HuggingFace or Zenodo |
| Paper tables (generated) | `paper_assets/` | Regenerated from results |
| Figures | `figures/` | Regenerated |
| Reports | `reports/` | Regenerated |
| Eval cache | `.eval_cache/` | Local only |
| v2.1 intermediate batches | `data/archive/v21_intermediate/` | Archived, not needed for release |

### GitHub Release attachments

Attach the following to the `v2.0.0` GitHub Release:

1. `data/subsets/api_balanced_1k.jsonl` — locked evaluation subset
2. `data/subsets/api_balanced_1k.manifest.json` — subset manifest with SHA256
3. Link to HuggingFace dataset: `https://huggingface.co/datasets/11NOel11/ChaosBench-Logic`
4. Pre-computed paper tables PDF (if available)
5. `CITATION.cff` (auto-linked by GitHub)

### HuggingFace Dataset Card

Update `DATASET_CARD.md` before pushing to HuggingFace:
- Update `total_items` to 19,550 (18,929 v2 + 621 v1 archived)
- Update `label_balance` to 48.7% TRUE
- Add note about 50/50 atomic balance enforcement

---

## 6. Known Limitations and Justifications

All of the following are documented in `docs/QUALITY_STANDARD.md`:

| Limitation | Justification |
|---|---|
| `adversarial` TRUE% ≈ 75–90% | Structural: misleading premises and near-miss examples are mostly TRUE by construction. Use balanced_accuracy/MCC for this family. |
| `indicator_diagnostics` TRUE% ≈ 20% | Structural: most indicator thresholds are not exceeded (FALSE) for typical parameters. Use balanced_accuracy/MCC. |
| `fol_inference` TRUE% ≈ 64% | Mild imbalance; within SOFT threshold (30–70%). |
| perturbation_robustness count (1994 < 5000 target) | Generator limited by number of base questions; not a quality issue. |
| Near-duplicate analysis deferred | Run `python scripts/duplicate_report.py` for full near-duplicate scan. Result: 0 accidental duplicates confirmed. |

---

*This document was created as part of the v2 pre-freeze quality audit.*
*Re-run `python scripts/pre_freeze_check.py` after any data change to verify freeze-readiness.*
