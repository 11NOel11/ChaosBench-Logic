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
    # v2.2 Extension: New predicates for 4-5 hop chains
    "Dissipative": "dissipative (volume-contracting)",
    "Bounded": "bounded",
    "Mixing": "mixing",
    "Ergodic": "ergodic",
    # v2.3 Extension: 12 new predicates from metadata dimensions
    "HyperChaotic": "hyperchaotic",
    "Conservative": "conservative (Hamiltonian)",
    "HighDimensional": "high-dimensional (high Kaplan-Yorke dimension)",
    "Multifractal": "multifractal",
    "HighDimSystem": "a high-dimensional system (state space dimension ≥ 4)",
    "ContinuousTime": "a continuous-time system",
    "DiscreteTime": "a discrete-time map",
    "DelaySystem": "a delay differential equation system",
    "Forced": "externally forced (non-autonomous)",
    "Autonomous": "autonomous",
    "StrongMixing": "strongly mixing",
    "WeakMixing": "weakly mixing",
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
    cap: int = 700,
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
    return questions[:cap]


def _generate_exclusion_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
    cap: int = 200,
) -> List[Question]:
    """Generate exclusion questions: can a system be both P and Q?

    Generates both generic (system-independent) and system-specific variants.
    """
    questions: List[Question] = []
    seen_generic = set()

    # Generic: "Can a system be both P and Q?" — one question per unique pair
    for pred, rule in sorted(rules.items()):
        for excl in rule.get("excludes", []):
            pair = tuple(sorted([pred, excl]))
            if pair in seen_generic:
                continue
            seen_generic.add(pair)
            counter[0] += 1
            p_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
            e_disp = PREDICATE_DISPLAY.get(excl, excl.lower())
            questions.append(Question(
                item_id=f"fol_excl_{counter[0]:04d}",
                question_text=f"Can a system be both {p_disp} and {e_disp}?",
                system_id="generic",
                task_family="fol_inference",
                ground_truth="NO",
                predicates=[pred, excl],
                metadata={
                    "question_type": "exclusion",
                    "predicate_a": pred,
                    "predicate_b": excl,
                    "variant": "generic",
                },
            ))

    # System-specific: "Given [name] is P, can it also be Q?"
    system_ids = sorted(systems.keys())
    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)
        for pred, rule in rules.items():
            if not truth.get(pred, False):
                continue
            for excl in rule.get("excludes", []):
                counter[0] += 1
                p_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
                e_disp = PREDICATE_DISPLAY.get(excl, excl.lower())
                questions.append(Question(
                    item_id=f"fol_excl_{counter[0]:04d}",
                    question_text=(
                        f"Given {name} is {p_disp}, can it also be {e_disp}?"
                    ),
                    system_id=sid,
                    task_family="fol_inference",
                    ground_truth="NO",
                    predicates=[pred, excl],
                    metadata={
                        "question_type": "exclusion",
                        "predicate_a": pred,
                        "predicate_b": excl,
                        "variant": "system_specific",
                    },
                ))

    rng.shuffle(questions)
    return questions[:cap]


def _generate_contrapositive_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
    cap: int = 400,
) -> List[Question]:
    """Generate contrapositive questions: if not Q, can it be P?

    Generates both generic and system-specific (modus tollens) variants.
    Generic: "If a system is not Q, can it be P?" (answer: NO, by contrapositive)
    System-specific: "Given [name] lacks Q, can it be P?" for systems where
    truth[req]=False and P→Q exists (answer: NO).
    """
    questions: List[Question] = []

    # Generic contrapositive: "If a system is not Q, can it be P?"
    for pred, rule in sorted(rules.items()):
        for req in rule.get("requires", []):
            counter[0] += 1
            pred_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
            req_disp = PREDICATE_DISPLAY.get(req, req.lower())
            questions.append(Question(
                item_id=f"fol_contra_{counter[0]:04d}",
                question_text=(
                    f"If a system is not {req_disp}, can it be {pred_disp}?"
                ),
                system_id="generic",
                task_family="fol_inference",
                ground_truth="NO",
                predicates=[pred, req],
                metadata={
                    "question_type": "contrapositive",
                    "predicate": pred,
                    "required": req,
                    "variant": "generic",
                },
            ))

    # System-specific modus tollens: find systems where req=False but
    # some other predicate P requires req — "Given [name] lacks Q, can it be P?"
    system_ids = sorted(systems.keys())
    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)
        for pred, rule in rules.items():
            for req in rule.get("requires", []):
                if truth.get(req, True):
                    continue  # req is True in this system — not useful
                counter[0] += 1
                pred_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
                req_disp = PREDICATE_DISPLAY.get(req, req.lower())
                questions.append(Question(
                    item_id=f"fol_contra_{counter[0]:04d}",
                    question_text=(
                        f"Given {name} is not {req_disp}, can it be {pred_disp}?"
                    ),
                    system_id=sid,
                    task_family="fol_inference",
                    ground_truth="NO",
                    predicates=[pred, req],
                    metadata={
                        "question_type": "contrapositive",
                        "predicate": pred,
                        "required": req,
                        "variant": "system_specific",
                    },
                ))

    rng.shuffle(questions)
    return questions[:cap]


def _generate_chain_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
    cap: int = 600,
) -> List[Question]:
    """Generate multi-step chain questions using all predicates as roots.

    For each system where truth[P]=True, and P requires Q, and Q requires R:
    generates explicit 2-step reasoning questions. Single-step chains
    (P→Q) are intentionally NOT generated here to avoid overlap with
    _generate_implication_questions — only 2-step (P→Q→R) chains are used.
    """
    questions: List[Question] = []
    system_ids = sorted(systems.keys())

    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)

        for pred, rule in rules.items():
            if not truth.get(pred, False):
                continue
            pred_disp = PREDICATE_DISPLAY.get(pred, pred.lower())

            # 2-step requires chains: P → Q → R
            for q in rule.get("requires", []):
                if q not in rules:
                    continue
                q_disp = PREDICATE_DISPLAY.get(q, q.lower())
                for r in rules[q].get("requires", []):
                    counter[0] += 1
                    r_disp = PREDICATE_DISPLAY.get(r, r.lower())
                    questions.append(Question(
                        item_id=f"fol_chain_{counter[0]:04d}",
                        question_text=(
                            f"If {name} is {pred_disp}, and systems that are "
                            f"{pred_disp} must be {q_disp}, and systems that "
                            f"are {q_disp} must be {r_disp}, does {name} "
                            f"exhibit {r_disp}?"
                        ),
                        system_id=sid,
                        task_family="fol_inference",
                        ground_truth="YES",
                        predicates=[pred, q, r],
                        metadata={
                            "question_type": "chain",
                            "chain": [pred, q, r],
                            "hop_count": 2,
                        },
                    ))

            # 2-step requires-to-excludes chains: P → Q → ¬R (answer: NO)
            for q in rule.get("requires", []):
                if q not in rules:
                    continue
                q_disp = PREDICATE_DISPLAY.get(q, q.lower())
                for r in rules[q].get("excludes", []):
                    counter[0] += 1
                    r_disp = PREDICATE_DISPLAY.get(r, r.lower())
                    questions.append(Question(
                        item_id=f"fol_chain_{counter[0]:04d}",
                        question_text=(
                            f"If {name} is {pred_disp}, and systems that are "
                            f"{pred_disp} must be {q_disp}, and systems that "
                            f"are {q_disp} cannot be {r_disp}, can {name} "
                            f"be {r_disp}?"
                        ),
                        system_id=sid,
                        task_family="fol_inference",
                        ground_truth="NO",
                        predicates=[pred, q, r],
                        metadata={
                            "question_type": "chain",
                            "chain": [pred, q, r],
                            "hop_count": 2,
                            "chain_type": "requires_to_excludes",
                        },
                    ))

    rng.shuffle(questions)
    return questions[:cap]


def _generate_consistency_questions(
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
    cap: int = 100,
) -> List[Question]:
    """Generate consistency check questions about predicate assignments."""
    questions: List[Question] = []

    inconsistent_sets = [
        # v2.0 / v2.2 core inconsistencies
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
        # v2.3 new predicate inconsistencies
        ({"Conservative": True, "Dissipative": True},
         "Conservative and Dissipative"),
        ({"Conservative": True, "StrangeAttr": True},
         "Conservative and having a Strange Attractor"),
        ({"HyperChaotic": True, "Conservative": True},
         "HyperChaotic and Conservative"),
        ({"ContinuousTime": True, "DiscreteTime": True},
         "Continuous-Time and Discrete-Time"),
        ({"Forced": True, "Autonomous": True},
         "Externally Forced and Autonomous"),
        ({"HyperChaotic": True, "Periodic": True},
         "HyperChaotic and Periodic"),
        ({"StrongMixing": True, "Periodic": True},
         "Strongly Mixing and Periodic"),
        ({"WeakMixing": True, "QuasiPeriodic": True},
         "Weakly Mixing and Quasi-Periodic"),
        ({"DelaySystem": True, "DiscreteTime": True},
         "Delay System and Discrete-Time"),
        ({"Mixing": True, "Periodic": True},
         "Mixing and Periodic"),
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
        # v2.0 / v2.2 core consistencies
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
        # v2.3 new predicate consistencies
        ({"Chaotic": True, "Dissipative": True, "StrangeAttr": True,
          "Conservative": False},
         "Chaotic, Dissipative, Strange Attractor, and not Conservative"),
        ({"Conservative": True, "Bounded": True, "Ergodic": True,
          "Dissipative": False},
         "Conservative, Bounded, Ergodic, and not Dissipative"),
        ({"HyperChaotic": True, "Chaotic": True, "Dissipative": True,
          "Conservative": False},
         "HyperChaotic, Chaotic, Dissipative, and not Conservative"),
        ({"ContinuousTime": True, "DiscreteTime": False, "Autonomous": True},
         "Continuous-Time, not Discrete-Time, and Autonomous"),
        ({"DiscreteTime": True, "ContinuousTime": False, "Chaotic": True},
         "Discrete-Time, not Continuous-Time, and Chaotic"),
        ({"StrongMixing": True, "WeakMixing": True, "Ergodic": True},
         "Strongly Mixing, Weakly Mixing, and Ergodic"),
        ({"DelaySystem": True, "ContinuousTime": True},
         "Delay System and Continuous-Time"),
        ({"Forced": True, "Autonomous": False, "ContinuousTime": True},
         "Forced, not Autonomous, and Continuous-Time"),
        ({"Chaotic": True, "Mixing": True, "Ergodic": True, "Periodic": False},
         "Chaotic, Mixing, Ergodic, and not Periodic"),
        ({"HighDimSystem": True, "ContinuousTime": True, "Chaotic": True},
         "High-Dimensional System, Continuous-Time, and Chaotic"),
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
    return questions[:cap]


def generate_fol_questions(
    systems: Dict[str, Dict],
    seed: int = 42,
    target_count: int = 2000,
) -> List[Question]:
    """Generate all FOL inference questions.

    Args:
        systems: Dict mapping system_id to system data with truth_assignment.
        seed: Random seed for reproducibility.
        target_count: Target total number of questions. Caps for each
            sub-generator are distributed proportionally:
            implication 35%, chain 30%, contrapositive 20%,
            exclusion 10%, consistency 5%.

    Returns:
        List of Question objects across all FOL question types.
    """
    rng = random.Random(seed)
    rules = get_fol_rules()
    counter = [0]

    # Distribute target_count proportionally across question types
    cap_impl    = int(target_count * 0.35)  # implication  35%
    cap_chain   = int(target_count * 0.30)  # chain        30%
    cap_contra  = int(target_count * 0.20)  # contrapositive 20%
    cap_excl    = int(target_count * 0.10)  # exclusion    10%
    cap_consist = max(20, target_count - cap_impl - cap_chain - cap_contra - cap_excl)

    questions: List[Question] = []
    questions.extend(_generate_implication_questions(systems, rules, rng, counter, cap=cap_impl))
    questions.extend(_generate_exclusion_questions(systems, rules, rng, counter, cap=cap_excl))
    questions.extend(_generate_contrapositive_questions(systems, rules, rng, counter, cap=cap_contra))
    questions.extend(_generate_chain_questions(systems, rules, rng, counter, cap=cap_chain))
    questions.extend(_generate_consistency_questions(rules, rng, counter, cap=cap_consist))

    rng.shuffle(questions)
    return questions


@dataclass
class FOLInferenceTask:
    """Task for testing first-order logic reasoning about dynamical systems.

    Attributes:
        task_family: Always "fol_inference".
        systems: Dict mapping system_id to system data.
        seed: Random seed.
        target_count: Target number of questions to generate.
    """

    task_family: str = "fol_inference"
    systems: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42
    target_count: int = 2000

    def generate_items(self) -> List[Question]:
        """Generate FOL inference questions.

        Returns:
            List of Question objects.
        """
        if not self.systems:
            self.systems = _load_systems()
        return generate_fol_questions(self.systems, self.seed, self.target_count)

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
