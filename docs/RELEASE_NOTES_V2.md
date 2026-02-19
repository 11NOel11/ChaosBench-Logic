# ChaosBench-Logic v2 — Release Notes

## Dataset Identity

| Field | Value |
|-------|-------|
| **Release** | v2 |
| **Total questions** | 40,886 |
| **Global SHA256** | `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279` |
| **Canonical files** | 10 (`data/v22_*.jsonl`) |
| **Selector** | `data/canonical_v2_files.json` |
| **Freeze artifact** | `artifacts/freeze/v2_freeze_manifest.json` |

---

## What Changed vs v1

### Dataset Scale
| Metric | v1 | v2 |
|--------|----|----|
| Total questions | 621 | 40,886 |
| Task families | 3 | 10 |
| Systems covered | 30 | 165 (30 core + 135 dysts) |
| Predicates | 15 | 15 |

### New Task Families (v2 only)
- **`perturbation_robustness`** (1,994): Tests whether models give consistent answers under surface-level question rephrasing
- **`consistency_paraphrase`** (4,139): Paraphrase groups probing label-consistency
- **`fol_inference`** (1,758): First-order logic inference chains
- **`adversarial`** (1,285): Misleading and near-miss question variants
- **`indicator_diagnostics`** (530): Quantitative chaos indicator interpretation (0-1 test, permutation entropy, MEGNO)
- **`regime_transition`** (68): Bifurcation and regime change reasoning
- **`cross_indicator`** (67): Multi-indicator cross-validation

### Systems
- Added 135 systems from the [`dysts`](https://github.com/williamgilpin/dysts) library covering 100+ named dynamical systems
- Heldout systems: 15 Sprott systems (A–O) reserved for test-time generalization

### Quality Gates (all passing)
- Near-duplicate scan: 0 near-duplicates (Jaccard threshold 0.85)
- Label leakage: 0 leaks
- Class balance: 49.5% TRUE (range 35–65%)
- Deduplication: 1,861 exact duplicates removed before final count

### Evaluation Infrastructure (new)
- `chaosbench freeze` — one-command hash-locked freeze
- `chaosbench eval --provider ollama` — local Ollama evaluation
- `chaosbench eval --provider mock` — offline smoke tests
- 3-way outcome parsing: `VALID_TRUE / VALID_FALSE / INVALID`
- Retry policy on INVALID responses
- Locked subsets with reproducible manifests

---

## How to Reproduce Runs

### Freeze (verify dataset identity)
```bash
python scripts/freeze_v2_dataset.py
# Compare: artifacts/freeze/v2_freeze_manifest.json → global_sha256
```

### Evaluation (local Ollama)
```bash
# 1k subset baseline
chaosbench eval --provider ollama --model qwen2.5:7b \
  --subset data/subsets/api_balanced_1k.jsonl \
  --workers 6 --retries 1

# Full canonical (40,886 items) — slow
chaosbench eval --provider ollama --model qwen2.5:7b \
  --dataset canonical --workers 6
```

### Verify Tests
```bash
python -m pytest -q   # should be 673 passed
```

---

## Baseline Results (local Ollama, 1k subset)

| Model | Eff. Acc | Balanced Acc | MCC |
|-------|----------|--------------|-----|
| qwen2.5:7b | 0.625 | 0.624 | 0.268 |
| llama3.1:8b | 0.599 | 0.598 | 0.250 |
| qwen2.5:14b | **0.698** | **0.698** | **0.402** |

Full results: `artifacts/paper_assets/local_leaderboard.md`

---

## Known Limitations

- Small families (`regime_transition`: 68, `cross_indicator`: 67, `extended_systems`: 45) have high variance in per-model accuracy.
- v2 results from cloud providers (GPT-4, Claude, Gemini) are pending; baselines shown are local Ollama models only.
- Flip rate metric is computed on paraphrase/perturbation groups within the 1k subset, which may not fully represent full-dataset flip behavior.
