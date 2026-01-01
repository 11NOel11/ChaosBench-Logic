# ChaosBench-Logic Ontology

## Overview

This document defines the formal ontology used in ChaosBench-Logic: the 11 logical predicates that characterize dynamical systems and the First-Order Logic (FOL) axioms that govern their relationships.

---

## The 11 Predicates

Each dynamical system in ChaosBench-Logic is characterized by 11 boolean predicates representing key mathematical and behavioral properties.

### 1. Chaotic

**Definition:** The system exhibits deterministic chaos — long-term unpredictable behavior arising from sensitive dependence on initial conditions, despite being governed by deterministic equations.

**Formal properties:**
- Has positive Lyapunov exponent
- Exhibits sensitive dependence on initial conditions
- Deterministic (not stochastic)
- Aperiodic dynamics

**Example:** Lorenz-63 system, Hénon map, logistic map at r=4

---

### 2. Deterministic

**Definition:** The system's future state is uniquely determined by its current state and the governing equations, with no random or stochastic components.

**Formal properties:**
- Evolution described by deterministic equations (ODEs, maps, PDEs)
- No noise terms or random variables
- Same initial conditions always produce the same trajectory

**Example:** All non-stochastic ODEs, discrete maps without noise

**Counter-example:** Ornstein-Uhlenbeck process (has stochastic term)

---

### 3. PosLyap (Positive Lyapunov Exponent)

**Definition:** The system has at least one positive Lyapunov exponent, meaning nearby trajectories diverge exponentially on average.

**Formal properties:**
- Largest Lyapunov exponent λ₁ > 0
- Quantifies exponential divergence rate: d(t) ≈ d₀ e^(λ₁t)

**Example:** Lorenz-63 (λ₁ ≈ 0.9), Hénon map (λ₁ ≈ 0.42)

**Note:** Positive Lyapunov exponent is necessary but not sufficient for chaos (needs bounded dynamics).

---

### 4. Sensitive (Sensitive Dependence on Initial Conditions)

**Definition:** Arbitrarily small differences in initial conditions lead to large differences in trajectories after sufficient time.

**Formal properties:**
- Related to positive Lyapunov exponent
- Makes long-term prediction impossible in practice
- Butterfly effect

**Example:** Weather systems, double pendulum

---

### 5. StrangeAttr (Strange Attractor)

**Definition:** The system's long-term behavior is confined to an attractor with fractal structure and non-integer dimension.

**Formal properties:**
- Fractal dimension (not integer)
- Self-similar structure at different scales
- Bounded in phase space
- Attracts nearby trajectories

**Example:** Lorenz attractor (dimension ≈ 2.06), Hénon attractor

**Note:** Strange attractor is *sufficient* but *not necessary* for chaos. Some chaotic systems lack attractors (e.g., Arnold cat map is area-preserving).

---

### 6. PointUnpredictable (Pointwise Unpredictable)

**Definition:** Precise point prediction of the system state at a specific future time is impossible beyond a finite horizon, even with arbitrarily accurate initial conditions.

**Formal properties:**
- Due to sensitive dependence and finite-precision measurements
- Lyapunov time: τ = 1/λ₁ (time scale for losing one bit of information)
- Individual trajectories cannot be predicted long-term

**Example:** Weather (can't predict specific temperature two weeks ahead)

---

### 7. StatPredictable (Statistically Predictable)

**Definition:** While individual trajectories are unpredictable, statistical properties (ensemble averages, distributions, moments) remain predictable over long times.

**Formal properties:**
- Ergodicity: time averages equal ensemble averages
- Invariant measure and attractor statistics are stable
- Climate vs. weather distinction

**Example:** Can predict average summer temperature (climate) but not specific day's weather

**Note:** Chaotic systems are pointwise unpredictable but statistically predictable.

---

### 8. QuasiPeriodic

**Definition:** The system exhibits motion that is almost periodic but never exactly repeats, typically characterized by multiple incommensurate frequencies.

**Formal properties:**
- Dynamics on a torus in phase space
- Multiple independent frequencies: ω₁, ω₂, ..., ωₙ
- Ratios ωᵢ/ωⱼ are irrational (incommensurate)
- All Lyapunov exponents are zero or negative

**Example:** Circle map (quasiperiodic regime), coupled oscillators with incommensurate frequencies

**Counter-example:** Chaotic systems (not quasiperiodic)

---

### 9. Random (Stochastic)

**Definition:** The system includes intrinsic randomness or noise in its governing equations.

**Formal properties:**
- Contains stochastic terms (Wiener process, random variables)
- Described by stochastic differential equations (SDEs)
- Multiple realizations from same initial conditions differ

**Example:** Ornstein-Uhlenbeck process, Brownian motion, systems with thermal noise

**Counter-example:** All deterministic systems

---

### 10. FixedPointAttr (Fixed-Point Attractor)

**Definition:** The system's long-term behavior converges to a single equilibrium point in phase space.

**Formal properties:**
- Attractor is a 0-dimensional point
- All trajectories approach the fixed point asymptotically
- All Lyapunov exponents are negative
- Stable equilibrium

**Example:** Damped harmonic oscillator, systems below bifurcation threshold

**Counter-example:** Chaotic, periodic, or quasiperiodic systems

---

### 11. Periodic

**Definition:** The system exhibits perfectly repeating behavior with a fixed period.

**Formal properties:**
- Attractor is a closed orbit (limit cycle)
- State returns exactly to starting point after period T
- One Lyapunov exponent is zero, others negative
- All Lyapunov exponents ≤ 0

**Example:** Limit cycles, periodic orbits in maps, undamped harmonic oscillator

**Counter-example:** Chaotic or quasiperiodic systems

---

## First-Order Logic (FOL) Axioms

The predicates are not independent — they obey logical constraints formalized as FOL axioms.

### Axiom Structure

Each axiom has two components:
- **Requires:** If predicate A is true, these predicates *must* also be true
- **Excludes:** If predicate A is true, these predicates *must* be false

### Axiom 1: Chaotic Systems

```
Chaotic(s) → Deterministic(s) ∧ PosLyap(s) ∧ Sensitive(s) ∧ PointUnpredictable(s) ∧ StatPredictable(s)
Chaotic(s) → ¬Random(s) ∧ ¬Periodic(s) ∧ ¬QuasiPeriodic(s) ∧ ¬FixedPointAttr(s)
```

**Interpretation:** Chaos requires determinism, positive Lyapunov exponent, sensitivity, pointwise unpredictability, and statistical predictability. Chaos excludes randomness, periodicity, quasiperiodicity, and fixed-point attractors.

**Design choice:** `StrangeAttr` is *not* in the `requires` list because strange attractors are sufficient but not necessary for chaos (e.g., Arnold cat map is chaotic without an attractor).

---

### Axiom 2: Random Systems

```
Random(s) → ¬Deterministic(s) ∧ ¬Chaotic(s) ∧ ¬QuasiPeriodic(s) ∧ ¬Periodic(s)
```

**Interpretation:** Stochastic systems are not deterministic and cannot be chaotic, quasiperiodic, or periodic (which require determinism).

---

### Axiom 3: QuasiPeriodic Systems

```
QuasiPeriodic(s) → Deterministic(s)
QuasiPeriodic(s) → ¬Chaotic(s) ∧ ¬Random(s) ∧ ¬Periodic(s) ∧ ¬FixedPointAttr(s)
```

**Interpretation:** Quasiperiodicity requires determinism and excludes chaos, randomness, periodicity, and fixed-point attractors.

---

### Axiom 4: Periodic Systems

```
Periodic(s) → Deterministic(s)
Periodic(s) → ¬Chaotic(s) ∧ ¬Random(s) ∧ ¬QuasiPeriodic(s) ∧ ¬StrangeAttr(s)
```

**Interpretation:** Periodicity requires determinism and excludes chaos, randomness, quasiperiodicity, and strange attractors.

---

### Axiom 5: Fixed-Point Attractors

```
FixedPointAttr(s) → Deterministic(s)
FixedPointAttr(s) → ¬Chaotic(s) ∧ ¬Random(s) ∧ ¬QuasiPeriodic(s) ∧ ¬Periodic(s) ∧ ¬StrangeAttr(s)
```

**Interpretation:** Fixed-point attractors require determinism and exclude chaos, randomness, quasiperiodicity, periodicity, and strange attractors.

---

### Axiom 6: Deterministic Systems

```
Deterministic(s) → ¬Random(s)
```

**Interpretation:** Determinism and randomness are mutually exclusive.

---

## Design Choices & Rationale

### 1. Unidirectional Implications

Axioms are **one-way implications**. For example:
- `Chaotic → Deterministic` (chaos implies determinism)
- But NOT: `Deterministic → Chaotic` (determinism doesn't imply chaos)

**Rationale:** Many deterministic systems are not chaotic (e.g., simple harmonic motion).

---

### 2. StrangeAttr Not Required for Chaos

`StrangeAttr` is **not** in the `requires` list for `Chaotic`.

**Rationale:** 
- Strange attractors are *sufficient* for chaos (if you have a strange attractor, the system is chaotic)
- But they are *not necessary* (some chaotic systems lack attractors)
- Example: Arnold cat map is chaotic but area-preserving (no attractor)

This design allows the ontology to handle both dissipative and conservative chaotic systems.

---

### 3. Symmetric Exclusions

If A excludes B, then B excludes A. For example:
- `Chaotic → ¬Random`
- `Random → ¬Chaotic`

**Rationale:** Exclusion is inherently symmetric. This ensures logical consistency.

---

### 4. Incomplete Specification

The axioms specify **necessary conditions** and **exclusions** but do not fully constrain all relationships.

**Example:** `Deterministic` has no `requires` list (only excludes `Random`). A deterministic system could be:
- Chaotic (Lorenz)
- Periodic (limit cycle)
- Quasiperiodic (torus)
- Fixed-point (damped oscillator)

**Rationale:** This reflects the mathematical reality — determinism alone doesn't determine the type of dynamics.

---

## Validation & Consistency

### Logical Consistency Checks

The evaluation pipeline (`eval_chaosbench.py`) checks model predictions against these axioms to detect **FOL violations**.

Example violation:
```
Model predicts: Chaotic=YES, Deterministic=NO
Violation: Chaotic → Deterministic
```

This is counted as a **logical inconsistency** even if the model doesn't contradict itself (didn't give both YES and NO to the same question).

### Ground Truth Assignment

Each system in `systems/*.json` has a `truth_assignment` field with boolean values for all 11 predicates:

```json
{
  "system_id": "lorenz63",
  "truth_assignment": {
    "Chaotic": true,
    "Deterministic": true,
    "PosLyap": true,
    "Sensitive": true,
    "StrangeAttr": true,
    "PointUnpredictable": true,
    "StatPredictable": true,
    "QuasiPeriodic": false,
    "Random": false,
    "FixedPointAttr": false,
    "Periodic": false
  }
}
```

All truth assignments are verified to satisfy the FOL axioms.

---

## Predicate Extraction from Questions

The evaluation pipeline maps natural language questions to predicates using keyword matching:

| Keywords | Predicate |
|----------|-----------|
| "chaotic", "chaos" | `Chaotic` |
| "deterministic" | `Deterministic` |
| "positive lyapunov", "lyapunov exponent" | `PosLyap` |
| "sensitive dependence", "sensitivity" | `Sensitive` |
| "strange attractor" | `StrangeAttr` |
| "pointwise prediction", "point-wise predictable" | `PointUnpredictable` |
| "statistically predictable", "statistical prediction" | `StatPredictable` |
| "quasi-periodic", "quasiperiodic" | `QuasiPeriodic` |
| "random", "randomness", "stochastic" | `Random` |
| "fixed point", "fixedpoint" | `FixedPointAttr` |
| "periodic" | `Periodic` |

See `eval_chaosbench.py:extract_predicate_from_question()` for implementation.

---

## Example System Definitions

### Chaotic System: Lorenz-63

```json
{
  "system_id": "lorenz63",
  "name": "Lorenz-63 system",
  "category": "chaotic",
  "equations": "dx/dt = σ (y - x); dy/dt = x (ρ - z) - y; dz/dt = x y - β z",
  "parameters": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666667
  },
  "truth_assignment": {
    "Chaotic": true,
    "Deterministic": true,
    "PosLyap": true,
    "Sensitive": true,
    "StrangeAttr": true,
    "PointUnpredictable": true,
    "StatPredictable": true,
    "QuasiPeriodic": false,
    "Random": false,
    "FixedPointAttr": false,
    "Periodic": false
  }
}
```

**Satisfies:** All Chaotic axioms — deterministic, positive Lyapunov, sensitive, etc.

---

### Stochastic System: Ornstein-Uhlenbeck Process

```json
{
  "system_id": "stochastic_ou",
  "name": "Ornstein-Uhlenbeck process",
  "category": "stochastic",
  "equations": "dX = θ(μ - X)dt + σ dW",
  "truth_assignment": {
    "Random": true,
    "Deterministic": false,
    "Chaotic": false,
    "PosLyap": false,
    "Sensitive": false,
    "StrangeAttr": false,
    "PointUnpredictable": false,
    "StatPredictable": false,
    "QuasiPeriodic": false,
    "FixedPointAttr": false,
    "Periodic": false
  }
}
```

**Satisfies:** Random axioms — not deterministic, not chaotic, etc.

---

### QuasiPeriodic System: Circle Map

```json
{
  "system_id": "circle_map_quasiperiodic",
  "name": "Circle map (quasiperiodic regime)",
  "category": "quasiperiodic",
  "equations": "θₙ₊₁ = θₙ + Ω - (K/2π) sin(2π θₙ) mod 1",
  "truth_assignment": {
    "QuasiPeriodic": true,
    "Deterministic": true,
    "Chaotic": false,
    "Random": false,
    "PosLyap": false,
    "Sensitive": false,
    "StrangeAttr": false,
    "PointUnpredictable": false,
    "StatPredictable": false,
    "FixedPointAttr": false,
    "Periodic": false
  }
}
```

**Satisfies:** QuasiPeriodic axioms — deterministic but not chaotic, periodic, or random.

---

## References

1. **Chaos Theory:** Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*. Westview Press.
2. **Lyapunov Exponents:** Wolf, A., et al. (1985). "Determining Lyapunov exponents from a time series." *Physica D*.
3. **Strange Attractors:** Ott, E. (2002). *Chaos in Dynamical Systems*. Cambridge University Press.
4. **Ergodic Theory:** Walters, P. (1982). *An Introduction to Ergodic Theory*. Springer.

---

## Citation

If you use this ontology in your research, please cite:

```bibtex
@software{thomas2025chaosbench,
  author = {Thomas, Noel},
  title = {ChaosBench-Logic: A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems},
  year = {2025},
  url = {https://github.com/11NOel11/ChaosBench-Logic}
}
```

---

## Contact

For questions about the ontology or to report errors:
- Open an issue on [GitHub](https://github.com/11NOel11/ChaosBench-Logic/issues)
- Contact: Noel Thomas (MBZUAI)
