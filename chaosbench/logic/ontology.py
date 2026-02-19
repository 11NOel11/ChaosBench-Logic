"""Predicate definitions and keyword mappings for dynamical systems ontology."""

from typing import Dict, List, Optional, Tuple


PREDICATES = [
    # Core predicates (v2.0 / v2.2)
    "Chaotic",
    "Deterministic",
    "PosLyap",
    "Sensitive",
    "StrangeAttr",
    "PointUnpredictable",
    "StatPredictable",
    "QuasiPeriodic",
    "Random",
    "FixedPointAttr",
    "Periodic",
    "Dissipative",
    "Bounded",
    "Mixing",
    "Ergodic",
    # v2.3 Extension: 12 new predicates from metadata dimensions
    "HyperChaotic",     # >= 2 positive Lyapunov exponents
    "Conservative",     # sum of Lyapunov exponents ≈ 0 (Hamiltonian)
    "HighDimensional",  # Kaplan-Yorke dimension >= 3.0
    "Multifractal",     # |KY_dim - corr_dim| >= 0.5
    "HighDimSystem",    # state space dimension >= 4
    "ContinuousTime",   # continuous-time ODE (not a map)
    "DiscreteTime",     # discrete-time map
    "DelaySystem",      # delay differential equation
    "Forced",           # externally forced / non-autonomous
    "Autonomous",       # no explicit time dependence
    "StrongMixing",     # strong mixing (ergodic hierarchy)
    "WeakMixing",       # weak mixing (ergodic hierarchy)
]

KEYWORD_MAP: List[Tuple[List[str], str]] = [
    (["chaotic", "chaos"], "Chaotic"),
    (["deterministic"], "Deterministic"),
    (["positive lyapunov", "poslyap", "largest lyapunov exponent"], "PosLyap"),
    (
        [
            "sensitive dependence",
            "sensitivity to initial conditions",
            "sensitive",
        ],
        "Sensitive",
    ),
    (["strange attractor"], "StrangeAttr"),
    (
        [
            "pointwise prediction",
            "point-wise prediction",
            "point-wise predictable",
            "pointunpredictable",
            "long-term pointwise",
        ],
        "PointUnpredictable",
    ),
    (
        [
            "statistically predictable",
            "statistical prediction",
            "statpredictable",
        ],
        "StatPredictable",
    ),
    (["quasi-periodic", "quasiperiodic"], "QuasiPeriodic"),
    (["random", "randomness", "stochastic"], "Random"),
    (["fixed point", "fixedpoint"], "FixedPointAttr"),
    (["periodic"], "Periodic"),
    (["dissipative", "volume-contracting", "volume contracting"], "Dissipative"),
    (["bounded", "bounded attractor", "bounded trajectory"], "Bounded"),
    (["mixing", "topological mixing"], "Mixing"),
    (["ergodic", "ergodicity"], "Ergodic"),
    # v2.3 Extension: new predicate keywords
    (["hyperchaotic", "hyper-chaotic", "hyperchaos"], "HyperChaotic"),
    (["conservative", "hamiltonian", "volume-preserving", "area-preserving"], "Conservative"),
    (["high-dimensional chaos", "high dimensional chaos", "high kaplan-yorke"], "HighDimensional"),
    (["multifractal", "multi-fractal", "multifractal structure"], "Multifractal"),
    (["high-dimensional system", "high dimensional state space", "large state space"], "HighDimSystem"),
    (["continuous-time", "continuous time", "ordinary differential equation", "ode"], "ContinuousTime"),
    (["discrete-time", "discrete time", "iterated map", "map"], "DiscreteTime"),
    (["delay", "delay differential", "time-delay", "dde"], "DelaySystem"),
    (["forced", "non-autonomous", "driven", "externally driven"], "Forced"),
    (["autonomous", "unforced", "no external forcing"], "Autonomous"),
    (["strong mixing", "strongly mixing", "strong ergodic mixing"], "StrongMixing"),
    (["weak mixing", "weakly mixing", "weak ergodic mixing"], "WeakMixing"),
]
