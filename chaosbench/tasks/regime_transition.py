"""Regime transition task: probing model knowledge of bifurcation structure."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from chaosbench.data.bifurcations import (
    BIFURCATION_DATA,
    BifurcationInfo,
    Transition,
    get_regime_at_param,
    is_chaotic_regime,
)
from chaosbench.data.schemas import Question


SYSTEM_DISPLAY_NAMES: Dict[str, str] = {
    "logistic": "logistic map",
    "lorenz63": "Lorenz system",
    "henon": "Henon map",
    "rossler": "Rössler system",
    "duffing_chaotic": "Duffing oscillator",
    "chua_circuit": "Chua's circuit",
    "vdp": "Van der Pol oscillator",
}


def _format_param_eq(param_name: str, value: float) -> str:
    """Format a parameter=value string.

    Args:
        param_name: Parameter name.
        value: Parameter value.

    Returns:
        String like 'r=3.8'.
    """
    if value == int(value):
        return f"{param_name}={int(value)}"
    return f"{param_name}={value}"


def _pick_probe_values(
    transitions: List[Transition],
    rng: random.Random,
) -> List[float]:
    """Select parameter values to probe, including transition points and midpoints.

    Args:
        transitions: Ordered transitions for a system.
        rng: Seeded random generator.

    Returns:
        List of parameter values to query.
    """
    values: List[float] = []
    for t in transitions:
        values.append(t.param_value)

    for i in range(len(transitions) - 1):
        lo = transitions[i].param_value
        hi = transitions[i + 1].param_value
        if hi - lo > 0.01:
            mid = round(lo + (hi - lo) * rng.uniform(0.3, 0.7), 4)
            values.append(mid)

    return sorted(set(values))


def _make_chaotic_question(
    system_id: str,
    info: BifurcationInfo,
    param_value: float,
    regime: str,
    idx: int,
) -> Question:
    """Generate a YES/NO question about whether the system is chaotic.

    Args:
        system_id: System identifier.
        info: Bifurcation info.
        param_value: Parameter value to query.
        regime: Ground truth regime at that value.
        idx: Question index for item_id.

    Returns:
        Question with ground_truth YES if chaotic, NO otherwise.
    """
    display = SYSTEM_DISPLAY_NAMES[system_id]
    peq = _format_param_eq(info.parameter_name, param_value)
    text = f"At {peq}, is the {display} chaotic?"
    answer = "YES" if is_chaotic_regime(regime) else "NO"

    return Question(
        item_id=f"regime_{system_id}_chaotic_{idx:03d}",
        question_text=text,
        system_id=system_id,
        task_family="regime_transition",
        ground_truth=answer,
        predicates=["Chaotic"],
        metadata={
            "param_name": info.parameter_name,
            "param_value": param_value,
            "regime": regime,
            "question_type": "is_chaotic",
        },
    )


def _make_persistence_question(
    system_id: str,
    info: BifurcationInfo,
    threshold: float,
    direction: str,
    regime_at_threshold: str,
    idx: int,
) -> Question:
    """Generate a question about whether chaos persists past a threshold.

    Args:
        system_id: System identifier.
        info: Bifurcation info.
        threshold: Parameter threshold value.
        direction: 'below' or 'above'.
        regime_at_threshold: Regime just past the threshold in the given direction.
        idx: Question index for item_id.

    Returns:
        Question about chaos persistence.
    """
    display = SYSTEM_DISPLAY_NAMES[system_id]
    peq = _format_param_eq(info.parameter_name, threshold)

    if direction == "below":
        text = (
            f"If {info.parameter_name} drops below {threshold}, "
            f"does {display} chaos persist?"
        )
    else:
        text = (
            f"If {info.parameter_name} rises above {threshold}, "
            f"does {display} chaos persist?"
        )

    answer = "YES" if is_chaotic_regime(regime_at_threshold) else "NO"

    return Question(
        item_id=f"regime_{system_id}_persist_{idx:03d}",
        question_text=text,
        system_id=system_id,
        task_family="regime_transition",
        ground_truth=answer,
        predicates=["Chaotic"],
        metadata={
            "param_name": info.parameter_name,
            "threshold": threshold,
            "direction": direction,
            "regime_at_threshold": regime_at_threshold,
            "question_type": "chaos_persistence",
        },
    )


def _make_periodic_question(
    system_id: str,
    info: BifurcationInfo,
    param_value: float,
    regime: str,
    idx: int,
) -> Question:
    """Generate a YES/NO question about whether the system is in a periodic regime.

    Args:
        system_id: System identifier.
        info: Bifurcation info.
        param_value: Parameter value to query.
        regime: Ground truth regime at that value.
        idx: Question index for item_id.

    Returns:
        Question with ground_truth YES if periodic, NO otherwise.
    """
    display = SYSTEM_DISPLAY_NAMES[system_id]
    peq = _format_param_eq(info.parameter_name, param_value)
    text = f"Is the {display} in a periodic regime at {peq}?"

    periodic_regimes = {
        "period_2", "period_4", "period_8", "period_3_window",
        "fixed_point", "pitchfork", "period_doubling",
        "period_doubling_cascade", "limit_cycle", "relaxation",
    }
    answer = "YES" if regime in periodic_regimes else "NO"

    return Question(
        item_id=f"regime_{system_id}_periodic_{idx:03d}",
        question_text=text,
        system_id=system_id,
        task_family="regime_transition",
        ground_truth=answer,
        predicates=["Periodic"],
        metadata={
            "param_name": info.parameter_name,
            "param_value": param_value,
            "regime": regime,
            "question_type": "is_periodic",
        },
    )


def _make_stable_question(
    system_id: str,
    info: BifurcationInfo,
    param_value: float,
    regime: str,
    idx: int,
) -> Question:
    """Generate a YES/NO question about whether the system has a stable fixed point.

    Args:
        system_id: System identifier.
        info: Bifurcation info.
        param_value: Parameter value to query.
        regime: Ground truth regime at that value.
        idx: Question index for item_id.

    Returns:
        Question with ground_truth YES if fixed point regime, NO otherwise.
    """
    display = SYSTEM_DISPLAY_NAMES[system_id]
    peq = _format_param_eq(info.parameter_name, param_value)
    text = f"Does the {display} have a stable fixed point at {peq}?"

    stable_regimes = {"fixed_point", "origin_stable", "pitchfork"}
    answer = "YES" if regime in stable_regimes else "NO"

    return Question(
        item_id=f"regime_{system_id}_stable_{idx:03d}",
        question_text=text,
        system_id=system_id,
        task_family="regime_transition",
        ground_truth=answer,
        predicates=["FixedPointAttr"],
        metadata={
            "param_name": info.parameter_name,
            "param_value": param_value,
            "regime": regime,
            "question_type": "is_stable",
        },
    )


def generate_regime_questions(
    bifurcation_data: Dict[str, BifurcationInfo],
    seed: int = 42,
) -> List[Question]:
    """Generate regime transition questions for all systems.

    Produces approximately 30 questions spanning chaotic detection,
    chaos persistence, periodicity, and stability across the logistic,
    Lorenz, and Henon systems.

    Args:
        bifurcation_data: Dict mapping system_id to BifurcationInfo.
        seed: Random seed for deterministic generation.

    Returns:
        List of Question objects with deterministic ordering.
    """
    rng = random.Random(seed)
    questions: List[Question] = []
    chaotic_idx = 0
    persist_idx = 0
    periodic_idx = 0
    stable_idx = 0

    for system_id in sorted(bifurcation_data.keys()):
        info = bifurcation_data[system_id]
        transitions = info.transitions
        all_probes = _pick_probe_values(transitions, rng)
        transition_values = [t.param_value for t in transitions]

        chaotic_probes = rng.sample(all_probes, min(5, len(all_probes)))
        for pv in sorted(chaotic_probes):
            regime = get_regime_at_param(system_id, info.parameter_name, pv)
            questions.append(_make_chaotic_question(
                system_id, info, pv, regime, chaotic_idx,
            ))
            chaotic_idx += 1

        chaotic_transitions = [
            t for t in transitions
            if is_chaotic_regime(t.regime) and t.param_value > 0
        ]
        for t in chaotic_transitions:
            below_val = round(t.param_value - 0.01, 4)
            if below_val >= 0:
                regime_below = get_regime_at_param(
                    system_id, info.parameter_name, below_val,
                )
                questions.append(_make_persistence_question(
                    system_id, info, t.param_value, "below",
                    regime_below, persist_idx,
                ))
                persist_idx += 1

        periodic_probes = rng.sample(all_probes, min(2, len(all_probes)))
        for pv in sorted(periodic_probes):
            regime = get_regime_at_param(system_id, info.parameter_name, pv)
            questions.append(_make_periodic_question(
                system_id, info, pv, regime, periodic_idx,
            ))
            periodic_idx += 1

        stable_pv = transition_values[0] if transition_values else 0.0
        regime = get_regime_at_param(system_id, info.parameter_name, stable_pv)
        questions.append(_make_stable_question(
            system_id, info, stable_pv, regime, stable_idx,
        ))
        stable_idx += 1

    return questions


@dataclass
class RegimeTransitionTask:
    """Task for probing model understanding of dynamical regime transitions.

    Generates YES/NO questions about chaotic behavior, chaos persistence,
    periodicity, and fixed point stability across bifurcation parameter
    ranges for well-studied dynamical systems.

    Attributes:
        task_family: Task type identifier.
        bifurcation_data: Bifurcation data per system.
        seed: Random seed for deterministic generation.
    """

    task_family: str = "regime_transition"
    bifurcation_data: Dict[str, BifurcationInfo] = field(
        default_factory=lambda: dict(BIFURCATION_DATA),
    )
    seed: int = 42

    def generate_items(self) -> List[Question]:
        """Generate regime transition questions for all configured systems.

        Returns:
            List of Question objects covering chaotic detection,
            persistence, periodicity, and stability.
        """
        return generate_regime_questions(self.bifurcation_data, self.seed)

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions against ground truth.

        Args:
            predictions: Dict mapping item_id to predicted label ("YES"/"NO").

        Returns:
            Dict with overall accuracy, per-system accuracy, and
            per-question-type accuracy breakdowns.
        """
        items = self.generate_items()
        item_map = {q.item_id: q for q in items}

        total = 0
        correct = 0
        by_system: Dict[str, List[bool]] = {}
        by_type: Dict[str, List[bool]] = {}

        for item_id, pred in predictions.items():
            if item_id not in item_map:
                continue

            q = item_map[item_id]
            total += 1
            is_correct = pred.upper() == q.ground_truth

            if is_correct:
                correct += 1

            sid = q.system_id
            if sid not in by_system:
                by_system[sid] = []
            by_system[sid].append(is_correct)

            qtype = q.metadata.get("question_type", "unknown")
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(is_correct)

        accuracy = correct / total if total > 0 else 0.0

        system_accuracy = {}
        for sid, results in sorted(by_system.items()):
            system_accuracy[sid] = sum(results) / len(results)

        type_accuracy = {}
        for qtype, results in sorted(by_type.items()):
            type_accuracy[qtype] = sum(results) / len(results)

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "system_accuracy": system_accuracy,
            "type_accuracy": type_accuracy,
        }
