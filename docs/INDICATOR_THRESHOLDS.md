# Indicator Thresholds Documentation

This document explains how indicator thresholds were empirically validated for the ChaosBench-Logic v2 benchmark.

## Overview

Chaos indicators require thresholds to classify systems as chaotic vs. non-chaotic. While theoretical values exist, empirical validation on the ChaosBench dataset reveals significant differences.

**Key Finding:** Empirically optimized thresholds provide better classification accuracy than theoretical values.

---

## Methodology

### Dataset

- **Total systems:** 30 benchmark dynamical systems
- **Chaotic systems:** 19 (ground truth from `truth_assignment`)
- **Non-chaotic systems:** 11

### Threshold Optimization

For each indicator, we:
1. Computed indicator values for all 30 systems
2. Separated by ground truth (chaotic vs. non-chaotic)
3. Computed descriptive statistics (mean, median, quartiles)
4. Found optimal threshold minimizing classification error
5. Reported classification accuracy

**Script:** `scripts/compute_indicator_thresholds.py`

---

## Results Summary

| Indicator | Theoretical Threshold | Empirical Optimal | Accuracy | Status |
|-----------|----------------------|-------------------|----------|---------|
| Zero-One K | 0.5 | -0.1 | 63.3% | ⚠️ Poor |
| Permutation Entropy | 0.7 | 0.40 | 70.0% | ⚡ Fair |
| MEGNO | 2.0 | 0.55 | 95.5% | ✅ Excellent |

---

## Zero-One K Test

### Distribution Analysis

**Chaotic systems (n=19):**
```
Mean:     0.28
Median:   0.002
Std Dev:  0.43
Range:    [0.00, 1.00]
Q1-Q3:    [0.00, 0.62]
```

**Non-chaotic systems (n=11):**
```
Mean:     0.17
Median:   0.00
Std Dev:  0.37
Range:    [0.00, 1.00]
Q1-Q3:    [0.00, 0.004]
```

### Threshold Selection

**Theoretical threshold:** K = 0.5
- Based on Gottwald & Melbourne (2004) definition
- K ≈ 1 for chaotic, K ≈ 0 for regular

**Empirical optimal:** K = -0.1
- **Classification accuracy:** 63.3%
- Effectively classifies almost everything as chaotic
- Indicates poor discriminative power

### Analysis

❌ **Zero-One K test is NOT a reliable indicator for this dataset.**

**Issues:**
1. **Large overlap:** Both chaotic and non-chaotic systems show K values near 0
2. **Counter-intuitive:** Chaotic systems have LOWER mean K (0.28) than expected
3. **High variance:** Standard deviation ~0.4 for both groups
4. **Poor separation:** Only 63% accuracy, barely better than random

**Possible explanations:**
- Time series length insufficient (10,000 points may be too short)
- Parameter selection (translation parameter c) not optimal for these systems
- Some systems exhibit mixed dynamics (transient vs. asymptotic behavior)
- Indicator more suitable for low-dimensional systems

### Recommendation

⚠️ **Use with caution.** Consider combining with other indicators or increasing time series length to 50,000+ points.

**Current implementation:** Keeps theoretical threshold (0.5) with warning about limitations.

---

## Permutation Entropy

### Distribution Analysis

**Chaotic systems (n=19):**
```
Mean:     0.51
Median:   0.43
Std Dev:  0.27
Range:    [0.02, 1.00]
Q1-Q3:    [0.39, 0.66]
```

**Non-chaotic systems (n=11):**
```
Mean:     0.43
Median:   0.40
Std Dev:  0.21
Range:    [0.05, 0.97]
Q1-Q3:    [0.39, 0.41]
```

### Threshold Selection

**Theoretical threshold:** PE = 0.7
- Based on empirical practice in literature
- High PE suggests high complexity/chaos

**Empirical optimal:** PE = 0.40
- **Classification accuracy:** 70.0%
- Moderate improvement over random guessing

### Analysis

⚡ **Permutation Entropy provides moderate discriminative power.**

**Observations:**
1. **Some separation:** Chaotic systems have slightly higher mean (0.51 vs. 0.43)
2. **Significant overlap:** Ranges overlap substantially
3. **Lower threshold:** Empirical 0.40 vs. theoretical 0.70
4. **Quartile overlap:** Q1-Q3 ranges nearly identical

**Interpretation:**
- Many non-chaotic systems have relatively high PE due to quasi-periodic or transient behavior
- Some chaotic systems have moderate PE when sampled during transient phase
- Embedding dimension (order=3) may be suboptimal for some systems

### Recommendation

✅ **Acceptable as secondary indicator.** Works best in combination with other indicators.

**Current implementation:** Updated threshold to 0.40 based on empirical optimization.

---

## MEGNO (Mean Exponential Growth of Nearby Orbits)

### Distribution Analysis

**Chaotic systems (n=13 valid, 6 failed computation):**
```
Mean:     6.65
Median:   1.92
Std Dev:  11.54
Range:    [-11.18, 40.01]
Q1-Q3:    [1.02, 8.12]
```

**Non-chaotic systems (n=9 valid, 2 failed computation):**
```
Mean:     -5.14
Median:   -0.33
Std Dev:  9.46
Range:    [-25.23, 0.46]
Q1-Q3:    [-0.67, -0.00]
```

**Missing/invalid:** 8 systems (validation rejected or computation failed)

### Threshold Selection

**Theoretical threshold:** MEGNO = 2.0
- Based on Cincotta & Simó (2000)
- MEGNO ≈ 2 for regular motion, MEGNO > 2 for chaos

**Empirical optimal:** MEGNO = 0.55
- **Classification accuracy:** 95.5%
- Excellent separation between groups

### Analysis

✅ **MEGNO is by far the best indicator when it can be reliably computed.**

**Strengths:**
1. **Excellent separation:** Chaotic systems mostly positive, non-chaotic mostly negative
2. **High accuracy:** 95.5% classification accuracy
3. **Clear threshold:** 0.55 provides sharp decision boundary
4. **Robust:** Works across different system types (ODEs and maps)

**Issues:**
1. **Computation failures:** 8/30 systems (27%) have invalid MEGNO
2. **Numerical instability:** Wide range [-25, 40] indicates numerical issues
3. **Negative values:** Theoretically shouldn't occur, suggests integration problems
4. **Different threshold:** Empirical 0.55 vs. theoretical 2.0

### Validation (v2 Implementation)

After adding validation (`|MEGNO| > 50` rejected):
- **Filtered extreme outliers:** chen_system (203.29), double_pendulum (117.35), lorenz63 (89.09)
- **Improved stability:** Remaining values in reasonable range
- **Missing data documented:** `megno_failure_reason` field explains why

### Recommendation

✅ **Use MEGNO as primary indicator when available.** Fall back to K or PE when MEGNO fails.

**Current implementation:** Updated threshold to 0.55 based on empirical optimization. Validation rejects extreme outliers.

---

## Threshold Usage in Code

### Before (Hardcoded Theoretical Values)

```python
# chaosbench/tasks/indicator_diagnostics.py
INDICATOR_RANGES = {
    "zero_one_K": {"chaotic_threshold": 0.5},
    "permutation_entropy": {"chaotic_threshold": 0.7},
    "megno": {"regular_value": 2.0},
}
```

### After (Empirically Validated)

```python
# chaosbench/tasks/indicator_diagnostics.py
INDICATOR_RANGES = {
    "zero_one_K": {"chaotic_threshold": 0.5},  # Theoretical (poor discriminator)
    "permutation_entropy": {"chaotic_threshold": 0.40},  # Empirical optimal
    "megno": {"chaotic_threshold": 0.55, "regular_value": 2.0},  # Empirical optimal
}
```

**Files updated:**
- `chaosbench/tasks/indicator_diagnostics.py`
- `chaosbench/tasks/cross_indicator.py`

---

## Visualization

### Distribution Comparison (ASCII Art)

```
Zero-One K Distribution:
Non-chaotic:  ████████▒▒░░░░░░░░░░  (mean=0.17)
Chaotic:      ███████▒▒▒▒▒▒░░░░░░░  (mean=0.28)
              0.0    0.5    1.0
              ↑
          threshold (poor separation)

Permutation Entropy Distribution:
Non-chaotic:  ░░░░████████▒▒▒░░░░░  (mean=0.43)
Chaotic:      ░░░░▒▒███████▒▒▒░░░░  (mean=0.51)
              0.0    0.5    1.0
                ↑
            threshold (moderate separation)

MEGNO Distribution:
Non-chaotic:  ██████████░░░░░░░░░░  (mean=-5.14, all < 0.55)
Chaotic:      ░░░░░░░░░░██████████  (mean=6.65, most > 0.55)
              -10    0    10   20
                      ↑
                  threshold (excellent separation)
```

---

## Recommendations for Usage

### Single Indicator

If using only one indicator:
1. **Best:** MEGNO (when computable) - 95.5% accuracy
2. **Acceptable:** Permutation Entropy - 70% accuracy
3. **Avoid:** Zero-One K alone - 63.3% accuracy

### Multiple Indicators (Ensemble)

For robust classification, use voting scheme:
```python
def classify_chaotic(K, PE, MEGNO):
    votes = 0

    if MEGNO is not None:
        # MEGNO is most reliable, weight it heavily
        if MEGNO > 0.55:
            votes += 2
        else:
            votes -= 2

    if PE > 0.40:
        votes += 1
    else:
        votes -= 1

    # Use K only as tiebreaker (low confidence)
    if K > 0.5:
        votes += 0.5
    else:
        votes -= 0.5

    return votes > 0
```

### Confidence Levels

In `chaosbench/tasks/cross_indicator.py`, questions include confidence metadata:

- **High confidence:** Indicator strongly agrees with ground truth (|diff| > 0.4)
- **Medium confidence:** Moderate agreement (0.2 < |diff| < 0.4)
- **Low confidence:** Ambiguous or disagrees (|diff| < 0.2)

---

## Known Limitations

1. **Dataset size:** 30 systems is relatively small for robust threshold optimization
2. **Ground truth uncertainty:** Some systems near bifurcation points may have ambiguous classification
3. **Parameter dependence:** Thresholds may need adjustment if indicator parameters change
4. **System-specific behavior:** Optimal thresholds may vary for different classes of systems (ODEs vs. maps, low-dim vs. high-dim)

---

## Future Work

### Short Term
1. Increase time series length for K test (50,000+ points)
2. Try different embedding dimensions for PE (order=4, 5)
3. System-specific MEGNO integration parameters

### Long Term
1. Develop adaptive thresholds based on system type
2. Uncertainty quantification (confidence intervals)
3. Machine learning approach to combine indicators
4. Add alternative indicators (Lyapunov exponents, correlation dimension)

---

## References

- Gottwald, G. A., & Melbourne, I. (2004). A new test for chaos in deterministic systems. *Proceedings of the Royal Society A*, 460(2042), 603-611.
- Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.
- Cincotta, P. M., & Simó, C. (2000). Simple tools to study global dynamics in non-axisymmetric galactic potentials-I. *Astronomy and Astrophysics Supplement Series*, 147(2), 205-228.

---

## Reproducibility

To recompute thresholds:

```bash
# Regenerate indicator data
python3 -c "
from chaosbench.data.indicators.populate import populate_all_systems
populate_all_systems('systems', 'systems/indicators', seed=42)
"

# Compute optimal thresholds
python3 scripts/compute_indicator_thresholds.py
```

Expected output:
```
Zero-One K Test           -0.1000 (accuracy: 63.3%)
Permutation Entropy        0.3997 (accuracy: 70.0%)
MEGNO                      0.5526 (accuracy: 95.5%)
```
