"""Hardcoded bifurcation data from dynamical systems literature.

Provides regime transition data for 7 well-studied systems:
logistic map, Lorenz system, Henon map, Rössler system,
Duffing oscillator, Chua's circuit, and Van der Pol oscillator.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Transition:
    """A single regime transition point.

    Attributes:
        param_value: Parameter value where transition occurs.
        regime: Regime label after this transition (e.g. "fixed_point", "periodic", "chaotic").
        transition_type: Type of bifurcation (e.g. "period_doubling", "pitchfork", "hopf").
        description: Brief description of what happens.
    """

    param_value: float
    regime: str
    transition_type: str
    description: str


@dataclass
class BifurcationInfo:
    """Bifurcation data for a dynamical system.

    Attributes:
        system_id: Identifier matching benchmark system.
        parameter_name: Name of the bifurcation parameter.
        transitions: Ordered list of transitions (sorted by param_value).
    """

    system_id: str
    parameter_name: str
    transitions: List[Transition] = field(default_factory=list)


BIFURCATION_DATA: Dict[str, BifurcationInfo] = {
    "logistic": BifurcationInfo(
        system_id="logistic",
        parameter_name="r",
        transitions=[
            Transition(0.0, "extinction", "trivial", "All orbits converge to x=0"),
            Transition(1.0, "fixed_point", "transcritical", "Stable fixed point at x=1-1/r appears"),
            Transition(3.0, "period_2", "period_doubling", "Fixed point loses stability, period-2 cycle appears"),
            Transition(3.449, "period_4", "period_doubling", "Period-2 cycle loses stability, period-4 appears"),
            Transition(3.544, "period_8", "period_doubling", "Period-4 loses stability, period-8 appears"),
            Transition(3.5699, "chaos_onset", "accumulation", "Feigenbaum accumulation point, onset of chaos"),
            Transition(3.8284, "period_3_window", "tangent", "Period-3 window via tangent bifurcation"),
            Transition(4.0, "full_chaos", "boundary_crisis", "Full chaos, ergodic on [0,1]"),
        ],
    ),
    "lorenz63": BifurcationInfo(
        system_id="lorenz63",
        parameter_name="rho",
        transitions=[
            Transition(0.0, "origin_stable", "trivial", "Origin is globally stable"),
            Transition(1.0, "pitchfork", "pitchfork", "Origin loses stability, two symmetric fixed points appear"),
            Transition(13.926, "subcritical_hopf", "subcritical_hopf", "Fixed points undergo subcritical Hopf bifurcation"),
            Transition(24.06, "transient_chaos", "homoclinic", "Transient chaos via homoclinic explosion"),
            Transition(24.74, "sustained_chaos", "crisis", "Sustained chaotic attractor appears"),
            Transition(28.0, "classical_chaos", "none", "Classical parameter setting, butterfly attractor"),
        ],
    ),
    "henon": BifurcationInfo(
        system_id="henon",
        parameter_name="a",
        transitions=[
            Transition(0.0, "fixed_point", "trivial", "Stable fixed point"),
            Transition(0.3675, "period_doubling_cascade", "period_doubling", "Period-doubling cascade begins"),
            Transition(1.06, "chaos_onset", "accumulation", "Onset of chaos after period-doubling cascade"),
            Transition(1.4, "strange_attractor", "none", "Classical strange attractor parameter"),
        ],
    ),
    "rossler": BifurcationInfo(
        system_id="rossler",
        parameter_name="a",
        transitions=[
            Transition(0.0, "fixed_point", "trivial", "Stable fixed point"),
            Transition(0.15, "period_doubling", "period_doubling", "Period-doubling bifurcation begins"),
            Transition(0.2867, "chaos_onset", "accumulation", "Onset of chaos via period-doubling accumulation"),
            Transition(0.35, "full_chaos", "none", "Fully developed chaotic attractor"),
        ],
    ),
    "duffing_chaotic": BifurcationInfo(
        system_id="duffing_chaotic",
        parameter_name="gamma",
        transitions=[
            Transition(0.0, "fixed_point", "trivial", "Unforced equilibrium"),
            Transition(0.2, "period_doubling", "period_doubling", "Period-doubling under increasing forcing amplitude"),
            Transition(0.28, "chaos_onset", "accumulation", "Onset of chaos via period-doubling cascade"),
            Transition(0.35, "sustained_chaos", "none", "Sustained chaotic motion"),
        ],
    ),
    "chua_circuit": BifurcationInfo(
        system_id="chua_circuit",
        parameter_name="alpha",
        transitions=[
            Transition(0.0, "fixed_point", "trivial", "Stable equilibrium"),
            Transition(8.0, "period_doubling", "period_doubling", "Period-doubling route to chaos begins"),
            Transition(9.0, "chaos_onset", "accumulation", "Onset of chaotic behavior"),
            Transition(12.0, "double_scroll_chaos", "none", "Double-scroll chaotic attractor"),
            Transition(15.6, "classical_chaos", "none", "Classical Chua chaos parameter"),
        ],
    ),
    "vdp": BifurcationInfo(
        system_id="vdp",
        parameter_name="mu",
        transitions=[
            Transition(0.0, "fixed_point", "trivial", "Stable equilibrium at origin"),
            Transition(0.01, "limit_cycle", "hopf", "Supercritical Hopf bifurcation, stable limit cycle appears"),
            Transition(5.0, "relaxation", "none", "Relaxation oscillations regime"),
        ],
    ),
}


def get_regime_at_param(system_id: str, param_name: str, param_value: float) -> str:
    """Look up the dynamical regime at a given parameter value.

    Finds the last transition with param_value <= the query value.

    Args:
        system_id: System identifier (e.g. "logistic", "lorenz63", "henon").
        param_name: Parameter name (must match BifurcationInfo.parameter_name).
        param_value: Query parameter value.

    Returns:
        Regime label string.

    Raises:
        KeyError: If system_id not found in BIFURCATION_DATA.
        ValueError: If param_name doesn't match the system's bifurcation parameter.
    """
    if system_id not in BIFURCATION_DATA:
        raise KeyError(f"Unknown system: {system_id}")

    info = BIFURCATION_DATA[system_id]

    if param_name != info.parameter_name:
        raise ValueError(
            f"Parameter '{param_name}' does not match system "
            f"'{system_id}' bifurcation parameter '{info.parameter_name}'"
        )

    matched_regime = "unknown"
    for transition in info.transitions:
        if transition.param_value <= param_value:
            matched_regime = transition.regime
        else:
            break

    return matched_regime


def is_chaotic_regime(regime: str) -> bool:
    """Check if a regime label indicates chaotic dynamics.

    Args:
        regime: Regime label string.

    Returns:
        True if the regime name contains "chaos", False otherwise.
    """
    return "chaos" in regime


def get_all_transitions(system_id: str) -> List[Transition]:
    """Get all transitions for a system.

    Args:
        system_id: System identifier (e.g. "logistic", "lorenz63", "henon").

    Returns:
        List of Transition objects ordered by param_value.

    Raises:
        KeyError: If system_id not found in BIFURCATION_DATA.
    """
    if system_id not in BIFURCATION_DATA:
        raise KeyError(f"Unknown system: {system_id}")

    return list(BIFURCATION_DATA[system_id].transitions)
