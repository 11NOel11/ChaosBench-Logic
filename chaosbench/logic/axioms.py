"""FOL axiom engine for checking logical consistency of predicate assignments."""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


def get_fol_rules() -> Dict[str, Dict[str, List[str]]]:
    """Returns FOL axioms defining relationships between dynamical system predicates.

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
            ],
            "excludes": ["Random", "Periodic", "QuasiPeriodic", "FixedPointAttr"],
        },
        "Random": {
            "requires": [],
            "excludes": ["Deterministic", "Chaotic", "QuasiPeriodic", "Periodic"],
        },
        "QuasiPeriodic": {
            "requires": ["Deterministic"],
            "excludes": ["Chaotic", "Random", "Periodic", "FixedPointAttr"],
        },
        "Periodic": {
            "requires": ["Deterministic"],
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
