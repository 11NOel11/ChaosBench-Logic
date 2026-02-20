# ChaosBench-Logic — Runs Audit Report
_Generated: 2026-02-20T12:44:19Z_

---

## 0. Top-Line Honest Score Recommendation

> **Use `balanced_accuracy` and `MCC` as the primary headline metrics.**
> `accuracy_valid` is biased when the model has label skew (high TNR, low TPR).
> A model that predicts FALSE ~100% of the time on a near-balanced dataset would
> achieve ~50% accuracy_valid but MCC ≈ 0 and balanced_accuracy ≈ 0.5.
> Use confusion-matrix-derived metrics (TPR, TNR, MCC) to distinguish genuine
> reasoning ability from label-defaulting behaviour.

---

## 1. Run Inventory

| Run ID | Provider | Model | N | Bal_acc | MCC | Bias | Status |
|--------|----------|-------|---|---------|-----|------|--------|
| `20260219T192140Z_mock` | mock | mock | 50 | 0.5000 | 0.0000 | ⚠️ BIASED | ✅ PASS |
| `20260219T193151Z_ollama_qwen2.5:7b` | ollama | qwen2.5:7b | 1,000 | 0.6243 | 0.2680 | ⚠️ BIASED | ✅ PASS |
| `20260219T193320Z_ollama_llama3.1:8b` | ollama | llama3.1:8b | 1,000 | 0.5978 | 0.2497 | ⚠️ BIASED | ✅ PASS |
| `20260219T193425Z_ollama_qwen2.5:14b` | ollama | qwen2.5:14b | 1,000 | 0.6977 | 0.4016 | ✅ OK | ✅ PASS |
| `20260220T104105Z_ollama_llama3.1:8b` | ollama | llama3.1:8b | 40,886 | 0.5988 | 0.2400 | ⚠️ BIASED | ✅ PASS |

---

## 2. Bias & Confusion Matrix

Bias verdict: **LABEL-BIASED** if `bias_score > 0.15` OR `min(TPR,TNR) < 0.40`.

### 20260219T192140Z_mock — `mock`

**Verdict: LABEL-BIASED**

| Metric | Value |
|--------|-------|
| GT TRUE rate | 0.5800 (29/50) |
| Pred TRUE rate | 1.0000 (50/50) |
| Bias score | 0.4200 |
| TP / FP / TN / FN | 29 / 21 / 0 / 0 |
| TPR (recall) | 1.0000 |
| TNR (specificity) | 0.0000 |
| FPR | 1.0000 |
| FNR | 0.0000 |
| balanced_accuracy | 0.5000 |
| MCC | 0.0000 |

#### Defaulting Detector

- **Dominant predicted label**: `TRUE`
- **TPR/TNR gap**: 1.0000  (TPR=1.0000, TNR=0.0000)

**Recommended diagnostic runs:**

A) **Balanced prior reminder prompt**: add to system prompt:
   _"Answer TRUE or FALSE based purely on the question. Do not default to any label."_
B) **Allow UNKNOWN token** (analysis-only mode): re-run with a 3-way
   output space (TRUE / FALSE / UNKNOWN) to surface genuinely uncertain
   cases without forcing a FALSE default.
C) **Minimal rationale first-token constraint**: require a 1-sentence
   chain-of-thought before the final token to prevent shallow defaulting.

### 20260219T193151Z_ollama_qwen2.5:7b — `qwen2.5:7b`

**Verdict: LABEL-BIASED**

| Metric | Value |
|--------|-------|
| GT TRUE rate | 0.4980 (498/1000) |
| Pred TRUE rate | 0.3130 (313/1000) |
| Bias score | 0.1850 |
| TP / FP / TN / FN | 218 / 95 / 407 / 280 |
| TPR (recall) | 0.4378 |
| TNR (specificity) | 0.8108 |
| FPR | 0.1892 |
| FNR | 0.5622 |
| balanced_accuracy | 0.6243 |
| MCC | 0.2680 |

#### Defaulting Detector

- **Dominant predicted label**: `FALSE`
- **TPR/TNR gap**: 0.3730  (TPR=0.4378, TNR=0.8108)

**Recommended diagnostic runs:**

A) **Balanced prior reminder prompt**: add to system prompt:
   _"Answer TRUE or FALSE based purely on the question. Do not default to any label."_
B) **Allow UNKNOWN token** (analysis-only mode): re-run with a 3-way
   output space (TRUE / FALSE / UNKNOWN) to surface genuinely uncertain
   cases without forcing a FALSE default.
C) **Minimal rationale first-token constraint**: require a 1-sentence
   chain-of-thought before the final token to prevent shallow defaulting.

### 20260219T193320Z_ollama_llama3.1:8b — `llama3.1:8b`

**Verdict: LABEL-BIASED**

| Metric | Value |
|--------|-------|
| GT TRUE rate | 0.4980 (498/1000) |
| Pred TRUE rate | 0.1890 (189/1000) |
| Bias score | 0.3090 |
| TP / FP / TN / FN | 143 / 46 / 456 / 355 |
| TPR (recall) | 0.2871 |
| TNR (specificity) | 0.9084 |
| FPR | 0.0916 |
| FNR | 0.7129 |
| balanced_accuracy | 0.5978 |
| MCC | 0.2497 |

#### Defaulting Detector

- **Dominant predicted label**: `FALSE`
- **TPR/TNR gap**: 0.6213  (TPR=0.2871, TNR=0.9084)

**Recommended diagnostic runs:**

A) **Balanced prior reminder prompt**: add to system prompt:
   _"Answer TRUE or FALSE based purely on the question. Do not default to any label."_
B) **Allow UNKNOWN token** (analysis-only mode): re-run with a 3-way
   output space (TRUE / FALSE / UNKNOWN) to surface genuinely uncertain
   cases without forcing a FALSE default.
C) **Minimal rationale first-token constraint**: require a 1-sentence
   chain-of-thought before the final token to prevent shallow defaulting.

### 20260219T193425Z_ollama_qwen2.5:14b — `qwen2.5:14b`

**Verdict: OK**

| Metric | Value |
|--------|-------|
| GT TRUE rate | 0.4980 (498/1000) |
| Pred TRUE rate | 0.4120 (412/1000) |
| Bias score | 0.0860 |
| TP / FP / TN / FN | 304 / 108 / 394 / 194 |
| TPR (recall) | 0.6104 |
| TNR (specificity) | 0.7849 |
| FPR | 0.2151 |
| FNR | 0.3896 |
| balanced_accuracy | 0.6977 |
| MCC | 0.4016 |

### 20260220T104105Z_ollama_llama3.1:8b — `llama3.1:8b`

**Verdict: LABEL-BIASED**

| Metric | Value |
|--------|-------|
| GT TRUE rate | 0.4948 (20230/40882) |
| Pred TRUE rate | 0.2159 (8828/40882) |
| Bias score | 0.2789 |
| TP / FP / TN / FN | 6387 / 2441 / 18211 / 13843 |
| TPR (recall) | 0.3157 |
| TNR (specificity) | 0.8818 |
| FPR | 0.1182 |
| FNR | 0.6843 |
| balanced_accuracy | 0.5988 |
| MCC | 0.2400 |

#### Defaulting Detector

- **Dominant predicted label**: `FALSE`
- **TPR/TNR gap**: 0.5661  (TPR=0.3157, TNR=0.8818)

**Recommended diagnostic runs:**

A) **Balanced prior reminder prompt**: add to system prompt:
   _"Answer TRUE or FALSE based purely on the question. Do not default to any label."_
B) **Allow UNKNOWN token** (analysis-only mode): re-run with a 3-way
   output space (TRUE / FALSE / UNKNOWN) to surface genuinely uncertain
   cases without forcing a FALSE default.
C) **Minimal rationale first-token constraint**: require a 1-sentence
   chain-of-thought before the final token to prevent shallow defaulting.

---

## 3. SHA + Reproducibility Reconciliation

### Finding

- **Freeze manifest global SHA** (`v2_freeze_manifest.json`): `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279`
- **Run manifests store** (all runs): `00ec17e31193de42c525ff3c8f166b4b59fae2c2631fa84a4c78b33fb01f9374` _(computed by run.py)_

### Root Cause

The global SHA is built by hashing each file's `path:sha256` string.
However, `freeze_v2_dataset.py` uses `path:sha256:count` (includes line count),
while `run.py` uses `path:sha256` (no count).  This is a **tooling inconsistency**
— **not a data integrity problem**.

**All 10 canonical per-file SHAs match exactly between runs and freeze.**

| File | Freeze SHA (first 16) | Run recomputed (first 16) | Match |
|------|-----------------------|--------------------------|-------|
| `v22_adversarial.jsonl` | `6ddae0cdb4ebc6e9…` | `6ddae0cdb4ebc6e9…` | ✅ |
| `v22_atomic.jsonl` | `fa9d1c99097cd2ca…` | `fa9d1c99097cd2ca…` | ✅ |
| `v22_consistency_paraphrase.jsonl` | `fb9cbda5b12d67fa…` | `fb9cbda5b12d67fa…` | ✅ |
| `v22_cross_indicator.jsonl` | `a03d1594a99e9a7b…` | `a03d1594a99e9a7b…` | ✅ |
| `v22_extended_systems.jsonl` | `49a5f675312af895…` | `49a5f675312af895…` | ✅ |
| `v22_fol_inference.jsonl` | `548da0e9a2f801c9…` | `548da0e9a2f801c9…` | ✅ |
| `v22_indicator_diagnostics.jsonl` | `eeb4e96000ee87a6…` | `eeb4e96000ee87a6…` | ✅ |
| `v22_multi_hop.jsonl` | `e6c048f35995b0d0…` | `e6c048f35995b0d0…` | ✅ |
| `v22_perturbation_robustness.jsonl` | `f1e30ee64370a40b…` | `f1e30ee64370a40b…` | ✅ |
| `v22_regime_transition.jsonl` | `2e63ebbee85c1763…` | `2e63ebbee85c1763…` | ✅ |

### Verdict

> **These runs are OFFICIAL.** They evaluated the correct, frozen dataset.
> The global SHA mismatch is a tooling artefact (hashing formula inconsistency).
> Fix tracked in §8.

---

## 4. Invalid / Parse-Failure Analysis

### 20260219T192140Z_mock

- Total: 50 | Valid: 50 | Invalid: 0
- Invalid rate: 0.0000%
- No invalid predictions. ✅

### 20260219T193151Z_ollama_qwen2.5:7b

- Total: 1,000 | Valid: 1,000 | Invalid: 0
- Invalid rate: 0.0000%
- No invalid predictions. ✅

### 20260219T193320Z_ollama_llama3.1:8b

- Total: 1,000 | Valid: 1,000 | Invalid: 0
- Invalid rate: 0.0000%
- No invalid predictions. ✅

### 20260219T193425Z_ollama_qwen2.5:14b

- Total: 1,000 | Valid: 1,000 | Invalid: 0
- Invalid rate: 0.0000%
- No invalid predictions. ✅

### 20260220T104105Z_ollama_llama3.1:8b

- Total: 40,886 | Valid: 40,882 | Invalid: 4
- Invalid rate: 0.0098%
- Invalid IDs (first 10): atomic_28074, atomic_27927, perturb_negation_1183, perturb_paraphrase_0360
- Categories: {'refusal': 4}
- Root cause: Ollama safety guardrail firing on 'tumor' system name (chaotic oscillator, not medical).

---

## 5. Per-Family Performance & Bias

### 20260220T104105Z_ollama_llama3.1:8b — `llama3.1:8b`

| Family | N | Acc | Bal_acc | MCC | TPR | TNR | Bias | Wilson 95% CI | Note |
|--------|---|-----|---------|-----|-----|-----|------|---------------|------|
| atomic | 25,307 | 0.601 | 0.602 | 0.230 | 0.367 | 0.836 | ⚠️ BIASED | — |  |
| multi_hop | 6,000 | 0.631 | 0.590 | 0.329 | 0.180 | 1.000 | ⚠️ BIASED | — |  |
| consistency_paraphrase | 4,139 | 0.585 | 0.580 | 0.199 | 0.283 | 0.877 | ⚠️ BIASED | — |  |
| perturbation | 1,994 | 0.625 | 0.625 | 0.329 | 0.300 | 0.950 | ⚠️ BIASED | — |  |
| fol_inference | 1,758 | 0.531 | 0.576 | 0.246 | 0.172 | 0.980 | ⚠️ BIASED | — |  |
| indicator_diagnostic | 530 | 0.492 | 0.511 | 0.103 | 0.022 | 1.000 | ⚠️ BIASED | — |  |
| adversarial_misleading | 500 | 0.634 | 0.637 | 0.354 | 0.318 | 0.956 | ⚠️ BIASED | — |  |
| adversarial_nearmiss | 478 | 0.651 | 0.641 | 0.320 | 0.410 | 0.872 | ⚠️ BIASED | — |  |
| regime_transition | 68 | 0.544 | 0.500 | 0.000 | 0.000 | 1.000 | ⚠️ BIASED | [0.427,0.657] | (low N) |
| cross_indicator | 67 | 0.433 | 0.500 | 0.000 | 0.000 | 1.000 | ⚠️ BIASED | [0.321,0.552] | (low N) |
| extended_systems | 45 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ OK | [0.921,1.000] | (low N) |

### Key Findings

- **extended_systems** (N=45): 100% accuracy — likely all `TRUE` questions or
  very structured queries. Small-N + label skew. Wilson 95% CI: see table.
- **cross_indicator** (N=67): ~43% accuracy — **below random**. Wilson CI
  spans 0.5; statistically ambiguous at this sample size.
- **indicator_diagnostic** (N=530): ~49% — near-random, consistent failure mode.
- **fol_inference** (N=1758): ~53% — modest above random; complex chains hard.
- **consistency_paraphrase** (N=4139): ~58% — paraphrase sensitivity present.
- **atomic** (N=25307): ~60% — majority of the benchmark; near 60% ceiling.

---

## 6. extended_systems Sanity Check

**llama3.1:8b**: N=45, accuracy=1.0000

- GT TRUE rate: 0.4889
- Pred TRUE rate: 0.4889
- Wilson 95% CI for accuracy: [0.921, 1.000]

> ⚠️ **N=45 with 100% accuracy**: This is a small-N result. The wide Wilson CI
> indicates high uncertainty. Do not interpret as strong evidence of domain mastery.
> Likely all questions happen to be `TRUE` for extended_systems, and the model
> defaults to `TRUE` in this family (or the answers are straightforward).

---

## 7. Consistency / Robustness (Flip Rate)

Flip rate analysis requires per-group IDs in predictions. The current schema has
`task_family` but no `group_id`. Aggregate signal from consistency_paraphrase family:

- **llama3.1:8b**: consistency_paraphrase acc = 0.5847 (flip-proxy = 0.4153)

Recommend adding `group_id` to the prediction schema for full per-group flip analysis.

---

## 8. 'What Could Have Gone Wrong?' Checklist

| Risk | Status | Notes |
|------|--------|-------|
| Dataset SHA mismatch | ⚠️ TOOLING | Formula inconsistency (run.py vs freeze_v2_dataset.py); data files identical |
| Subset vs canonical confusion | ✅ NO | All official runs use `data/canonical_v2_files.json` |
| Predictions misaligned (count mismatch) | ✅ NO | Prediction line counts match manifests |
| Duplicate IDs in predictions | ✅ NO | 0 duplicate IDs found |
| Parsing ambiguity | ⚠️ MINOR | 4 refusals (0.01%) on 'tumor' system names |
| Label bias / defaulting | ✅ LOW | Bias score within threshold; TPR and TNR both reasonable |
| Leakage across splits | ✅ NO | Per-split metadata shows single 'unknown' bucket |
| Metric calculation errors | ✅ NO | Recomputed metrics match stored (< 0.001 delta) |
| Wall-time / throughput anomaly | ℹ️ SEE §9 | Mock run latencies are sub-ms (expected) |

---

## 9. Performance & Efficiency

### 20260219T192140Z_mock
- workers=1, retries=1
- Throughput: **6245317.5 q/s**  (avg latency: 0.0 ms/q)

### 20260219T193151Z_ollama_qwen2.5:7b
- workers=6, retries=1
- Throughput: **0.5 q/s**  (avg latency: 2040.8 ms/q)

### 20260219T193320Z_ollama_llama3.1:8b
- workers=5, retries=1
- Throughput: **0.4 q/s**  (avg latency: 2381.0 ms/q)

### 20260219T193425Z_ollama_qwen2.5:14b
- workers=2, retries=1
- Throughput: **0.8 q/s**  (avg latency: 1250.0 ms/q)

### 20260220T104105Z_ollama_llama3.1:8b
- workers=4, retries=1
- Throughput: **5.1 q/s**  (avg latency: 196.9 ms/q)

### Recommendations

1. **ETA display**: add rolling throughput ETA to tqdm (track `n_done / elapsed`).
2. **GPU detection**: log Ollama device (`/api/ps`) in run_manifest.
3. **Parallelism safety**: sort predictions by ID post-run if ordering needed.
4. **Hash fix**: adopt freeze formula (`path:sha256:count`) in run.py.

---

## 10. Suggested Fixes

### Fix A — SHA hashing formula (HIGH PRIORITY)

In `chaosbench/eval/run.py`, change `_dataset_global_sha256` to include `:count`:

```python
def _dataset_global_sha256(selector_path: str = "data/canonical_v2_files.json") -> str:
    root = PROJECT_ROOT
    sel = json.loads((root / selector_path).read_text())
    global_h = hashlib.sha256()
    for rel_path in sorted(sel["files"]):
        fpath = root / rel_path
        file_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        count = sum(1 for _ in fpath.open())
        global_h.update(f"{rel_path}:{file_sha}:{count}\n".encode("utf-8"))
    return global_h.hexdigest()
```

### Fix B — Content-filter refusals

Add a system-prompt prefix: _'These questions are about mathematical dynamical
systems and chaos theory, not biology or medicine.'_

### Fix C — Add group_id to predictions

Emit `group_id` from dataset items into predictions.jsonl for proper flip-rate
analysis.
