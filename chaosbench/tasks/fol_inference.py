"""FOL inference task for testing logical reasoning about dynamical systems.

This module generates questions testing first-order logic (FOL) reasoning
capabilities by probing understanding of ontological relationships between
predicates in the dynamical systems domain.

The task tests whether models can correctly reason about logical rules such as:
- Implications: "Chaotic systems require positive Lyapunov exponents"
- Exclusions: "A system cannot be both chaotic and periodic"
- Contrapositives: "If a system doesn't have positive Lyapunov, it can't be chaotic"
- Chains: Multi-hop reasoning through multiple implications
- Consistency: Determining if a set of predicate assignments is logically valid

Question Types
--------------
1. **Implication** (30-50 questions, all YES):
   Tests understanding of P → Q rules from the ontology.
   Example: "Given the Lorenz-63 system is chaotic, must it be deterministic?"

2. **Exclusion** (13 questions, all NO):
   Tests understanding of mutually exclusive predicates.
   Example: "Can a system be both chaotic and periodic?"

3. **Contrapositive** (8 questions, all NO):
   Tests understanding of ¬Q → ¬P rules.
   Example: "If a system is not deterministic, can it be chaotic?"

4. **Chain** (20-30 questions, mixed):
   Tests multi-step reasoning: P → Q and Q → R, therefore P → R.
   Example: "If the Rössler system is chaotic, and chaotic systems require
   being sensitive to initial conditions, does the Rössler system exhibit
   this property?"

5. **Consistency** (20 questions, 50/50 YES/NO):
   Tests ability to identify valid vs. invalid predicate assignments.
   Example: "Is the assignment {Chaotic: True, Random: True} logically
   consistent under the dynamical systems ontology?"

Expected Distribution
---------------------
- Total: ~121 questions (expanded from 91 in v2.1)
- Ground truth balance: ~70% YES, ~30% NO (varies by question type)
- Coverage: All 30 benchmark systems for system-specific questions
- Generic questions: Based solely on ontology rules (system-independent)

Implementation Notes
--------------------
- Uses ontology from `chaosbench.logic.axioms.get_fol_rules()`
- All questions have deterministic answers based on FOL reasoning
- Questions are shuffled within each type for variety
- Slice limits control how many questions of each type are included

See Also
--------
- `chaosbench.logic.axioms` : Defines the FOL ontology
- `tests/test_fol_inference.py` : Comprehensive test suite
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from chaosbench.data.schemas import Question
from chaosbench.logic.axioms import get_fol_rules


PREDICATE_DISPLAY = {
    "Chaotic": "chaotic",
    "Deterministic": "deterministic",
    "PosLyap": "having a positive Lyapunov exponent",
    "Sensitive": "sensitive to initial conditions",
    "StrangeAttr": "having a strange attractor",
    "PointUnpredictable": "pointwise unpredictable",
    "StatPredictable": "statistically predictable",
    "QuasiPeriodic": "quasi-periodic",
    "Random": "random",
    "FixedPointAttr": "having a fixed point attractor",
    "Periodic": "periodic",
}


def _load_systems(systems_dir: str = "systems") -> Dict[str, Dict]:
    """Load all system JSONs from directory.

    Args:
        systems_dir: Path to systems directory.

    Returns:
        Dict mapping system_id to system data dict.
    """
    systems = {}
    if not os.path.isdir(systems_dir):
        return systems
    for fname in sorted(os.listdir(systems_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(systems_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            systems[sid] = data
    return systems


def _generate_implication_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate implication questions: if X is P, must it have Q?"""
    questions: List[Question] = []
    system_ids = sorted(systems.keys())

    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)

        for pred, rule in rules.items():
            if not truth.get(pred, False):
                continue
            for req in rule.get("requires", []):
                counter[0] += 1
                pred_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
                req_disp = PREDICATE_DISPLAY.get(req, req.lower())
                questions.append(Question(
                    item_id=f"fol_impl_{counter[0]:04d}",
                    question_text=(
                        f"Given {name} is {pred_disp}, must it be "
                        f"{req_disp}?"
                    ),
                    system_id=sid,
                    task_family="fol_inference",
                    ground_truth="YES",
                    predicates=[pred, req],
                    metadata={
                        "question_type": "implication",
                        "antecedent": pred,
                        "consequent": req,
                    },
                ))

    rng.shuffle(questions)
    # Expanded from 30 to 50 to increase batch size (105 questions available)
    return questions[:50]


def _generate_exclusion_questions(
    rules: Dict[str, Dict[str, List[str]]],
    counter: List[int],
) -> List[Question]:
    """Generate exclusion questions: can a system be both P and Q?"""
    questions: List[Question] = []
    seen = set()

    for pred, rule in sorted(rules.items()):
        for excl in rule.get("excludes", []):
            pair = tuple(sorted([pred, excl]))
            if pair in seen:
                continue
            seen.add(pair)
            counter[0] += 1
            p_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
            e_disp = PREDICATE_DISPLAY.get(excl, excl.lower())
            questions.append(Question(
                item_id=f"fol_excl_{counter[0]:04d}",
                question_text=(
                    f"Can a system be both {p_disp} and {e_disp}?"
                ),
                system_id="generic",
                task_family="fol_inference",
                ground_truth="NO",
                predicates=[pred, excl],
                metadata={
                    "question_type": "exclusion",
                    "predicate_a": pred,
                    "predicate_b": excl,
                },
            ))

    return questions[:25]


def _generate_contrapositive_questions(
    rules: Dict[str, Dict[str, List[str]]],
    counter: List[int],
) -> List[Question]:
    """Generate contrapositive questions: if not Q, can it be P?"""
    questions: List[Question] = []

    for pred, rule in sorted(rules.items()):
        for req in rule.get("requires", []):
            counter[0] += 1
            pred_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
            req_disp = PREDICATE_DISPLAY.get(req, req.lower())
            questions.append(Question(
                item_id=f"fol_contra_{counter[0]:04d}",
                question_text=(
                    f"If a system is not {req_disp}, can it be "
                    f"{pred_disp}?"
                ),
                system_id="generic",
                task_family="fol_inference",
                ground_truth="NO",
                predicates=[pred, req],
                metadata={
                    "question_type": "contrapositive",
                    "predicate": pred,
                    "required": req,
                },
            ))

    return questions[:25]


def _generate_chain_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate multi-step chain questions."""
    questions: List[Question] = []
    system_ids = sorted(systems.keys())

    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)

        if not truth.get("Chaotic", False):
            continue

        for req in rules.get("Chaotic", {}).get("requires", []):
            counter[0] += 1
            req_disp = PREDICATE_DISPLAY.get(req, req.lower())
            questions.append(Question(
                item_id=f"fol_chain_{counter[0]:04d}",
                question_text=(
                    f"If {name} is chaotic, and chaotic systems require "
                    f"being {req_disp}, does {name} exhibit this property?"
                ),
                system_id=sid,
                task_family="fol_inference",
                ground_truth="YES",
                predicates=["Chaotic", req],
                metadata={
                    "question_type": "chain",
                    "chain": ["Chaotic", req],
                },
            ))

    rng.shuffle(questions)
    # Expanded from 20 to 30 to increase batch size
    return questions[:30]


def _generate_consistency_questions(
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate consistency check questions about predicate assignments."""
    questions: List[Question] = []

    inconsistent_sets = [
        ({"Chaotic": True, "Periodic": True, "Deterministic": True},
         "Chaotic, Periodic, and Deterministic"),
        ({"Chaotic": True, "Random": True},
         "Chaotic and Random"),
        ({"Chaotic": True, "Deterministic": False},
         "Chaotic but not Deterministic"),
        ({"QuasiPeriodic": True, "Periodic": True},
         "Quasi-Periodic and Periodic"),
        ({"Random": True, "Deterministic": True},
         "Random and Deterministic"),
        ({"Chaotic": True, "FixedPointAttr": True},
         "Chaotic and Fixed Point Attractor"),
        ({"Chaotic": True, "QuasiPeriodic": True},
         "Chaotic and Quasi-Periodic"),
        ({"FixedPointAttr": True, "Periodic": True},
         "Fixed Point Attractor and Periodic"),
        ({"Periodic": True, "StrangeAttr": True},
         "Periodic and Strange Attractor"),
        ({"Chaotic": True, "PosLyap": False},
         "Chaotic but without positive Lyapunov exponent"),
    ]

    for assignment, desc in inconsistent_sets:
        counter[0] += 1
        questions.append(Question(
            item_id=f"fol_consist_{counter[0]:04d}",
            question_text=(
                f"Is the assignment {{{desc}}} logically consistent "
                f"under the dynamical systems ontology?"
            ),
            system_id="generic",
            task_family="fol_inference",
            ground_truth="NO",
            predicates=list(assignment.keys()),
            metadata={
                "question_type": "consistency",
                "assignment": {k: v for k, v in assignment.items()},
            },
        ))

    consistent_sets = [
        ({"Chaotic": True, "Deterministic": True, "PosLyap": True,
          "Sensitive": True},
         "Chaotic, Deterministic, Positive Lyapunov, and Sensitive"),
        ({"Periodic": True, "Deterministic": True},
         "Periodic and Deterministic"),
        ({"QuasiPeriodic": True, "Deterministic": True},
         "Quasi-Periodic and Deterministic"),
        ({"FixedPointAttr": True, "Deterministic": True},
         "Fixed Point Attractor and Deterministic"),
        ({"Random": True, "Chaotic": False, "Deterministic": False},
         "Random but not Chaotic and not Deterministic"),
        ({"Deterministic": True, "Chaotic": False, "Periodic": True},
         "Deterministic, not Chaotic, and Periodic"),
        ({"Chaotic": True, "Deterministic": True, "Sensitive": True,
          "PosLyap": True, "PointUnpredictable": True, "StatPredictable": True},
         "Chaotic with all required properties"),
        ({"Deterministic": True, "Periodic": False, "Chaotic": False,
          "QuasiPeriodic": True},
         "Deterministic and Quasi-Periodic but not Periodic or Chaotic"),
        ({"FixedPointAttr": True, "Deterministic": True, "Chaotic": False},
         "Fixed Point Attractor, Deterministic, and not Chaotic"),
        ({"Deterministic": True, "Chaotic": False, "Random": False},
         "Deterministic, not Chaotic, not Random"),
    ]

    for assignment, desc in consistent_sets:
        counter[0] += 1
        questions.append(Question(
            item_id=f"fol_consist_{counter[0]:04d}",
            question_text=(
                f"Is the assignment {{{desc}}} logically consistent "
                f"under the dynamical systems ontology?"
            ),
            system_id="generic",
            task_family="fol_inference",
            ground_truth="YES",
            predicates=list(assignment.keys()),
            metadata={
                "question_type": "consistency",
                "assignment": {k: v for k, v in assignment.items()},
            },
        ))

    rng.shuffle(questions)
    return questions[:20]


def generate_fol_questions(
    systems: Dict[str, Dict],
    seed: int = 42,
) -> List[Question]:
    """Generate all FOL inference questions.

    Args:
        systems: Dict mapping system_id to system data with truth_assignment.
        seed: Random seed for reproducibility.

    Returns:
        List of Question objects across all FOL question types.
    """
    rng = random.Random(seed)
    rules = get_fol_rules()
    counter = [0]

    questions: List[Question] = []
    questions.extend(_generate_implication_questions(systems, rules, rng, counter))
    questions.extend(_generate_exclusion_questions(rules, counter))
    questions.extend(_generate_contrapositive_questions(rules, counter))
    questions.extend(_generate_chain_questions(systems, rules, rng, counter))
    questions.extend(_generate_consistency_questions(rules, rng, counter))

    return questions


@dataclass
class FOLInferenceTask:
    """Task for testing first-order logic reasoning about dynamical systems.

    Attributes:
        task_family: Always "fol_inference".
        systems: Dict mapping system_id to system data.
        seed: Random seed.
    """

    task_family: str = "fol_inference"
    systems: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42

    def generate_items(self) -> List[Question]:
        """Generate FOL inference questions.

        Returns:
            List of Question objects.
        """
        if not self.systems:
            self.systems = _load_systems()
        return generate_fol_questions(self.systems, self.seed)

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions against ground truth.

        Args:
            predictions: Dict mapping item_id to predicted label.

        Returns:
            Dict with accuracy and per-type breakdowns.
        """
        items = self.generate_items()
        correct = 0
        total = 0
        by_type: Dict[str, List[bool]] = {}

        for q in items:
            pred = predictions.get(q.item_id)
            if pred is None:
                continue
            total += 1
            is_correct = pred.upper() == q.ground_truth
            if is_correct:
                correct += 1
            qtype = q.metadata.get("question_type", "unknown")
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(is_correct)

        type_accuracy = {
            k: sum(v) / len(v) for k, v in sorted(by_type.items())
        }

        return {
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "total": total,
            "type_accuracy": type_accuracy,
        }
