"""Predicate definitions and keyword mappings for dynamical systems ontology."""

from typing import Dict, List, Optional, Tuple


PREDICATES = [
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
]
