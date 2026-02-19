"""FOL axiom engine for checking logical consistency of predicate assignments."""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


def get_fol_rules() -> Dict[str, Dict[str, List[str]]]:
    """Returns FOL axioms defining relationships between dynamical system predicates.

    v2.2 Extension: Added 4 new predicates (Dissipative, Bounded, Mixing, Ergodic)
    and 12 new edges to enable 4-5 hop reasoning chains.

    v2.3 Extension: Added 12 new predicates (HyperChaotic, Conservative,
    HighDimensional, Multifractal, HighDimSystem, ContinuousTime, DiscreteTime,
    DelaySystem, Forced, Autonomous, StrongMixing, WeakMixing) and ~40 new
    requires/excludes edges to enable 5-6 hop reasoning chains.

    Returns:
        Dict mapping predicate names to {'requires': [...], 'excludes': [...]}.
    """
    return {
        "Chaotic": {
            "requires": [
                "Deterministic",
                "PosLyap",
                "Sensitive",
                "PointUnpredictable",
                "StatPredictable",
                # NOTE: Chaotic → Dissipative removed (conservative chaotic systems exist)
                "Mixing",       # Chaotic systems exhibit mixing
                # NOTE: Chaotic → StrongMixing removed — conservative chaotic systems
                # (Arnold cat map) are strongly mixing but the proxy computation may
                # not detect this. Avoid breaking the axiom for edge cases.
            ],
            "excludes": ["Random", "Periodic", "QuasiPeriodic", "FixedPointAttr"],
        },
        "Random": {
            "requires": [],
            "excludes": ["Deterministic", "Chaotic", "QuasiPeriodic", "Periodic"],
        },
        "QuasiPeriodic": {
            "requires": [
                "Deterministic",
                "Bounded",
            ],
            "excludes": ["Chaotic", "Random", "Periodic", "FixedPointAttr"],
        },
        "Periodic": {
            "requires": [
                "Deterministic",
                "Bounded",
            ],
            "excludes": ["Chaotic", "Random", "QuasiPeriodic", "StrangeAttr"],
        },
        "FixedPointAttr": {
            "requires": ["Deterministic"],
            "excludes": [
                "Chaotic",
                "Random",
                "QuasiPeriodic",
                "Periodic",
                "StrangeAttr",
            ],
        },
        "Deterministic": {
            "requires": [],
            "excludes": ["Random"],
        },
        "PosLyap": {
            "requires": ["Sensitive"],
            "excludes": ["Periodic", "FixedPointAttr"],
        },
        "Sensitive": {
            "requires": ["PointUnpredictable"],
            "excludes": ["Periodic", "FixedPointAttr"],
        },
        "StrangeAttr": {
            "requires": [
                "Dissipative",
                "Bounded",
            ],
            "excludes": [
                "Periodic",
                "FixedPointAttr",
                "Conservative",  # v2.3: strange attractors are not conservative
            ],
        },
        "PointUnpredictable": {
            "requires": [
                "Bounded",
            ],
            "excludes": [],
        },
        "StatPredictable": {
            "requires": [
                "Bounded",
            ],
            "excludes": [],
        },
        # v2.2 Extension predicates
        "Dissipative": {
            "requires": [
                "Bounded",
            ],
            "excludes": [
                "Conservative",  # v2.3: conservative ↔ dissipative are mutually exclusive
            ],
        },
        "Bounded": {
            "requires": [],
            "excludes": [],
        },
        "Mixing": {
            "requires": [
                "Ergodic",
                # NOTE: Mixing → WeakMixing removed to avoid circular dependency
                # (StrongMixing → Mixing creates a cycle if we also have Mixing → WeakMixing
                # and StrongMixing → WeakMixing). Keep it simple: Mixing → Ergodic only.
            ],
            "excludes": ["Periodic", "QuasiPeriodic"],
        },
        "Ergodic": {
            "requires": [
                "Bounded",
            ],
            "excludes": ["Periodic"],
        },
        # v2.3 Extension predicates
        "HyperChaotic": {
            "requires": [
                "Chaotic",
                "PosLyap",
                "Sensitive",
                "StrangeAttr",
                "Dissipative",
            ],
            "excludes": [
                "Periodic",
                "QuasiPeriodic",
                "FixedPointAttr",
                "Conservative",  # hyperchaotic → dissipative → not conservative
            ],
        },
        "Conservative": {
            "requires": [
                "Bounded",       # bounded Hamiltonian systems
                "Ergodic",       # ergodic on energy surfaces (KAM theory)
            ],
            "excludes": [
                "Dissipative",
                "StrangeAttr",   # Hamiltonian systems don't have strange attractors
            ],
        },
        "HighDimensional": {
            "requires": [],
            "excludes": [],
        },
        "Multifractal": {
            "requires": [],
            "excludes": [],
        },
        "HighDimSystem": {
            "requires": [],
            "excludes": [],
        },
        "ContinuousTime": {
            "requires": [],
            # NOTE: ContinuousTime → Deterministic removed; SDEs (stochastic_ou) are
            # continuous-time but not deterministic. The implication only holds for
            # pure ODEs, not for stochastic processes.
            "excludes": [
                "DiscreteTime",
            ],
        },
        "DiscreteTime": {
            "requires": [],
            "excludes": [
                "ContinuousTime",
            ],
        },
        "DelaySystem": {
            "requires": [
                "ContinuousTime",  # delay systems are continuous-time DDEs
            ],
            "excludes": [],
        },
        "Forced": {
            "requires": [],
            "excludes": [
                "Autonomous",
            ],
        },
        "Autonomous": {
            "requires": [],
            "excludes": [
                "Forced",
            ],
        },
        "StrongMixing": {
            "requires": [
                "WeakMixing",   # StrongMixing ⊃ WeakMixing (ergodic hierarchy)
                "Ergodic",      # StrongMixing ⊃ Ergodic
                # NOTE: StrongMixing → Mixing removed to avoid circular dependency
                # (Mixing → Ergodic and StrongMixing → Mixing would create long
                # chains but also StrongMixing → Mixing → Ergodic vs
                # StrongMixing → Ergodic directly is redundant)
            ],
            "excludes": [
                "Periodic",
                "QuasiPeriodic",
            ],
        },
        "WeakMixing": {
            "requires": [
                "Ergodic",      # WeakMixing ⊃ Ergodic (ergodic hierarchy)
            ],
            "excludes": [
                "Periodic",
                "QuasiPeriodic",
            ],
        },
    }


def check_fol_violations(
    predictions: Dict[str, str],
    ground_truth: Optional[Dict[str, bool]] = None,
) -> List[str]:
    """Check for FOL violations in a set of predicate predictions.

    Args:
        predictions: Model predictions as {predicate: "YES"|"NO"}.
        ground_truth: Optional ground truth as {predicate: bool}.

    Returns:
        List of violated implication strings, e.g. ["Chaotic -> Deterministic"].
    """
    violations: List[str] = []
    fol_rules = get_fol_rules()

    pred_bool: Dict[str, bool] = {}
    for pred_name, pred_value in predictions.items():
        if pred_value == "YES":
            pred_bool[pred_name] = True
        elif pred_value == "NO":
            pred_bool[pred_name] = False

    for predicate, is_true in pred_bool.items():
        if predicate not in fol_rules:
            continue

        rules = fol_rules[predicate]

        if is_true:
            for required_pred in rules.get("requires", []):
                if required_pred in pred_bool:
                    if not pred_bool[required_pred]:
                        violations.append(f"{predicate} \u2192 {required_pred}")

            for excluded_pred in rules.get("excludes", []):
                if excluded_pred in pred_bool:
                    if pred_bool[excluded_pred]:
                        violations.append(
                            f"{predicate} \u2192 \u00ac{excluded_pred}"
                        )

    return violations


def load_system_ontology(
    systems_dir: str = "systems",
) -> Dict[str, Dict[str, bool]]:
    """Load ground truth ontology for all dynamical systems.

    Args:
        systems_dir: Path to directory containing system JSON files.

    Returns:
        Dict mapping system_id to {predicate_name: bool_value}.
    """
    ontology: Dict[str, Dict[str, bool]] = {}

    if not os.path.exists(systems_dir):
        print(f"[WARNING] Systems directory not found: {systems_dir}")
        return ontology

    for filename in os.listdir(systems_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(systems_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                system_id = data.get("system_id")
                truth_assignment = data.get("truth_assignment", {})

                if system_id and truth_assignment:
                    ontology[system_id] = truth_assignment
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] Failed to parse {filename}: {e}")
            continue

    return ontology
