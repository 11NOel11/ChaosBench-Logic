# Indicator Computation Documentation

This document describes how chaos indicators are computed for the ChaosBench-Logic v2 benchmark.

## Overview

Three main chaos indicators are computed for each dynamical system:

1. **Zero-One K Test** - Distinguishes between regular and chaotic dynamics via correlation analysis
2. **Permutation Entropy** - Measures complexity via ordinal pattern distributions
3. **MEGNO** (Mean Exponential Growth of Nearby Orbits) - Detects chaos via variational equations

## Zero-One K Test

### Algorithm

The 0-1 test (Gottwald & Melbourne, 2004, 2009) transforms a time series into a 2D random walk and measures its asymptotic growth rate.

### Implementation

**File:** `chaosbench/data/indicators/zero_one_test.py`

**Parameters:**
- Series length: 10,000 points
- Translation parameter `c`: Random in [π/5, 6π/5] (as recommended by authors)
- Number of bins: 100 for mean square displacement calculation

**Procedure:**
1. Generate trajectory with 10,000 time points
2. Extract first state variable as 1D time series
3. Apply 0-1 test transformation with random translation parameter
4. Compute asymptotic mean square displacement
5. Return K value ∈ [0, 1]

### Interpretation

- **K ≈ 0:** Regular dynamics (periodic, quasi-periodic, or stable)
- **K ≈ 1:** Chaotic dynamics
- **K ∈ (0.3, 0.7):** Ambiguous (mixed dynamics or insufficient data)

### Known Limitations

- Requires long time series (≥1000 points) for reliable results
- Transient behavior can contaminate results if trajectory too short
- **Empirical finding:** On ChaosBench systems, K test has limited discriminative power (63% classification accuracy)
- Many chaotic systems show K values near 0, suggesting the test may not be optimal for this dataset

### References

- Gottwald, G. A., & Melbourne, I. (2004). A new test for chaos in deterministic systems. *Proceedings of the Royal Society A*, 460(2042), 603-611.
- Gottwald, G. A., & Melbourne, I. (2009). On the implementation of the 0-1 test for chaos. *SIAM Journal on Applied Dynamical Systems*, 8(1), 129-145.

---

## Permutation Entropy

### Algorithm

Permutation entropy (Bandt & Pompe, 2002) quantifies the complexity of a time series by analyzing the relative frequencies of ordinal patterns.

### Implementation

**File:** `chaosbench/data/indicators/permutation_entropy.py`

**Parameters:**
- Series length: 10,000 points
- Embedding dimension (order): 3
- Time delay: 1
- Normalization: Log₂ for entropy in range [0, 1]

**Procedure:**
1. Generate trajectory with 10,000 time points
2. Extract first state variable as 1D time series
3. Create ordinal patterns of length 3 (3! = 6 possible patterns)
4. Count frequency of each pattern
5. Compute Shannon entropy: H = -Σ p(π) log₂ p(π)
6. Normalize by log₂(3!) to get PE ∈ [0, 1]

### Interpretation

- **PE ≈ 0:** Regular dynamics (highly predictable patterns)
- **PE ≈ 1:** Chaotic/random dynamics (unpredictable patterns)
- **PE ∈ (0.3, 0.7):** Intermediate complexity

**Empirical threshold:** 0.40 provides 70% classification accuracy on ChaosBench systems.

### Known Limitations

- Sensitive to noise and measurement precision
- Choice of embedding dimension affects results
- Short time series may not capture all patterns
- Some non-chaotic systems (quasi-periodic) can have high PE

### References

- Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.

---

## MEGNO (Mean Exponential Growth of Nearby Orbits)

### Algorithm

MEGNO (Cincotta & Simó, 2000) uses variational equations to measure the average divergence rate of nearby trajectories over time.

### Implementation

**Files:**
- `chaosbench/data/indicators/megno.py`
- `chaosbench/data/indicators/populate.py`

**Parameters for ODEs:**
- Integration time (`t_max`): 100 seconds
- Number of output points (`n_points`): 2000
- Integration method: RK45 (adaptive Runge-Kutta)
- Tolerances: `rtol=1e-8`, `atol=1e-8`
- Tangent vector: Random initial direction, renormalized

**Parameters for Maps:**
- Number of iterations (`n_points`): 2000
- Finite difference step: `ε=1e-7`
- Renormalization interval: every 10 iterations

**Procedure (ODEs):**
1. Construct augmented ODE system: [dy/dt, dw/dt] where w is tangent vector
2. Compute Jacobian via central finite differences
3. Integrate augmented system with initial conditions y₀ and random w₀
4. Compute time-averaged MEGNO: Y(t) = (2/t) ∫₀ᵗ (s · ẇ·w / w·w) ds
5. Return final MEGNO value

**Procedure (Maps):**
1. Initialize random tangent vector
2. At each iteration, compute Jacobian via finite differences
3. Evolve tangent vector: w_{n+1} = J(x_n) · w_n
4. Compute weighted sum of log growth rates
5. Return time-averaged MEGNO-like value

### Interpretation

**Theoretical:**
- **MEGNO ≈ 2:** Regular (quasi-periodic) motion
- **MEGNO > 2:** Chaotic motion (linear growth)
- **MEGNO < 2:** Super-stable periodic motion

**Empirical (ChaosBench systems):**
- **MEGNO > 0.55:** Chaotic (95.5% accuracy)
- **MEGNO < 0.55:** Non-chaotic
- Many systems show negative MEGNO values (numerical issues)

### Validation and Error Handling

**Validation (implemented in v2):**
- Reject |MEGNO| > 50 as computation errors
- Check for finite values (reject NaN, Inf)
- Document failure reason in `megno_failure_reason` field

**Common failure modes:**
- Numerical instability in variational equations (extreme outliers)
- Integration failure for stiff systems
- Tangent vector underflow/overflow

**Unsupported systems:**
- `mackey_glass` (delay differential equation)
- `stochastic_ou` (stochastic system)
- `kuramoto_sivashinsky` (PDE)
- `sine_gordon` (PDE)

### Known Limitations

- Computationally expensive (requires Jacobian at each time step)
- Numerical issues common for stiff systems
- Large discrepancy between theoretical (≈2) and empirical (≈0.55) thresholds
- Negative values observed (theoretically should be positive)
- Integration parameters may need system-specific tuning

### Recommendations

1. **When MEGNO fails:** Use Zero-One K or Permutation Entropy as fallback
2. **For new systems:** Validate MEGNO values are in reasonable range (|MEGNO| < 50)
3. **For production:** Consider using ensemble of all three indicators for robustness

### References

- Cincotta, P. M., & Simó, C. (2000). Simple tools to study global dynamics in non-axisymmetric galactic potentials-I. *Astronomy and Astrophysics Supplement Series*, 147(2), 205-228.
- Cincotta, P. M., Giordano, C. M., & Simó, C. (2003). Phase space structure of multi-dimensional systems by means of the mean exponential growth factor of nearby orbits. *Physica D*, 182(3-4), 151-178.

---

## Implementation Notes

### Trajectory Generation

All indicators work from time series generated via:
- **ODEs:** `scipy.integrate.solve_ivp` with RK45 method
- **Maps:** Iterative application of map function
- **Parameters:** System-specific default parameters from `_DEFAULT_PARAMS`
- **Initial conditions:** Default ICs with small random perturbation (1e-10)

### Reproducibility

- All computations use fixed seed (default: 42)
- Random perturbations are deterministic
- Integration tolerances fixed for consistent results

### Performance Considerations

Approximate computation times (single system):
- Zero-One K: ~0.5 seconds
- Permutation Entropy: ~0.2 seconds
- MEGNO: ~2-5 seconds (ODEs), ~1 second (maps)

Total time to compute all indicators for all 30 systems: ~3-5 minutes

---

## Quality Assurance

### Validation Checks

All indicator computations include:
1. Exception handling for failed computations
2. Finite value checks (reject NaN, Inf)
3. Range validation (system-specific)
4. Fallback to None on failure

### Testing

See `tests/test_indicators.py` for comprehensive unit tests covering:
- Basic functionality for each indicator
- Edge cases (empty series, constant values)
- Integration with full pipeline
- Reproducibility with fixed seeds

---

## Future Improvements

Potential enhancements for future versions:

1. **Adaptive MEGNO parameters:** System-specific integration times and tolerances
2. **Alternative indicators:** Lyapunov exponents, correlation dimension, Poincaré sections
3. **Ensemble methods:** Combine multiple indicators for robust classification
4. **GPU acceleration:** Parallelize computations across systems
5. **Uncertainty quantification:** Bootstrap estimates of indicator reliability
