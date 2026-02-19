"""Multi-hop FOL chain reasoning task for dynamical systems.

This module generates questions testing multi-hop logical reasoning through
transitive chains of FOL implications and exclusions. It extends the basic
FOL inference task by requiring models to chain together multiple logical
steps to reach a conclusion.

Question Types
--------------
1. **2-hop transitive requires** (YES):
   Tests P→Q and Q→R chains where both are "requires" implications.
   Example: "Given that the Lorenz system is chaotic, and chaotic systems
   must be deterministic, does it follow that the Lorenz system is deterministic?"

2. **2-hop requires-to-excludes** (NO):
   Tests P→Q (requires) and Q→¬R (excludes) chains.
   Example: "If the Lorenz system is chaotic, which requires being deterministic,
   can it also be random?"

3. **3-hop transitive chain** (YES):
   Tests P→Q→R→S chains through three consecutive implications.
   Example: "The Lorenz system is chaotic. Systems that are chaotic must be
   deterministic. Systems that are deterministic cannot be random. Therefore,
   can the Lorenz system be random?"

4. **Contrapositive fallacy** (NO):
   Tests understanding that ¬P does not imply ¬Q even if P→Q.
   Example: "If the damped pendulum is NOT chaotic, and chaotic systems require
   positive Lyapunov exponent, does this tell us anything definitive about
   whether it has a positive Lyapunov exponent?"

5. **Modus tollens** (NO):
   Tests contrapositive: if ¬Q and P→Q, then ¬P.
   Example: "If a system lacks positive Lyapunov exponent, and being chaotic
   requires positive Lyapunov exponent, can it be chaotic?"

Expected Distribution
---------------------
- Total: ~100-150 questions depending on target_count
- Ground truth balance: Mixed YES/NO based on reasoning type
- 2-hop questions: majority of dataset
- 3-hop questions: only for chaotic systems (longest chains available)
- Contrapositive questions: for balance and fallacy detection

Implementation Notes
--------------------
- Uses ontology from `chaosbench.logic.axioms.get_fol_rules()`
- Chains are extracted by traversing the FOL rule graph
- Questions are shuffled and balanced for YES/NO distribution
- All reasoning paths are validated against system truth assignments

See Also
--------
- `chaosbench.logic.axioms` : Defines the FOL ontology
- `chaosbench.tasks.fol_inference` : Basic FOL reasoning task
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

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


def _find_2hop_chains(
    rules: Dict[str, Dict[str, List[str]]]
) -> List[Tuple[str, str, str, str]]:
    """Find all valid 2-hop reasoning chains in the FOL rules.

    Args:
        rules: FOL rules from get_fol_rules().

    Returns:
        List of (P, Q, R, chain_type) tuples where:
        - P→Q and Q→R (chain_type="requires_requires") → YES
        - P→Q and Q→¬R (chain_type="requires_excludes") → NO
    """
    chains = []

    for p, p_rules in rules.items():
        # P requires Q
        for q in p_rules.get("requires", []):
            if q not in rules:
                continue
            q_rules = rules[q]

            # Q requires R → P→Q→R (transitive YES)
            for r in q_rules.get("requires", []):
                chains.append((p, q, r, "requires_requires"))

            # Q excludes R → P→Q→¬R (transitive NO)
            for r in q_rules.get("excludes", []):
                chains.append((p, q, r, "requires_excludes"))

    return chains


def _find_3hop_chains(
    rules: Dict[str, Dict[str, List[str]]]
) -> List[Tuple[str, str, str, str, str]]:
    """Find all valid 3-hop reasoning chains in the FOL rules.

    Args:
        rules: FOL rules from get_fol_rules().

    Returns:
        List of (P, Q, R, S, chain_type) tuples where:
        - P→Q→R→S through requires (chain_type="requires_3hop") → YES
        - P→Q→R→¬S through requires then excludes (chain_type="mixed_3hop") → NO
    """
    chains = []

    for p, p_rules in rules.items():
        for q in p_rules.get("requires", []):
            if q not in rules:
                continue
            q_rules = rules[q]

            for r in q_rules.get("requires", []):
                if r not in rules:
                    continue
                r_rules = rules[r]

                # P→Q→R→S (all requires)
                for s in r_rules.get("requires", []):
                    chains.append((p, q, r, s, "requires_3hop"))

                # P→Q→R→¬S (last step excludes)
                for s in r_rules.get("excludes", []):
                    chains.append((p, q, r, s, "mixed_3hop"))

    return chains


def _generate_2hop_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate 2-hop chain reasoning questions.

    Args:
        systems: Dict mapping system_id to system data.
        rules: FOL rules from get_fol_rules().
        rng: Random number generator.
        counter: Counter for item_id generation.

    Returns:
        List of 2-hop Question objects.
    """
    questions: List[Question] = []
    chains = _find_2hop_chains(rules)
    system_ids = sorted(systems.keys())

    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)

        for p, q, r, chain_type in chains:
            # Only generate if P is true for this system
            if not truth.get(p, False):
                continue

            counter[0] += 1
            p_disp = PREDICATE_DISPLAY.get(p, p.lower())
            q_disp = PREDICATE_DISPLAY.get(q, q.lower())
            r_disp = PREDICATE_DISPLAY.get(r, r.lower())

            if chain_type == "requires_requires":
                # P→Q→R (YES)
                question_text = (
                    f"Given that {name} is {p_disp}, and {p_disp} systems must be "
                    f"{q_disp}, does it follow that {name} is {r_disp}?"
                )
                ground_truth = "YES"
            else:  # requires_excludes
                # P→Q→¬R (NO)
                question_text = (
                    f"If {name} is {p_disp}, which requires being {q_disp}, "
                    f"can it also be {r_disp}?"
                )
                ground_truth = "NO"

            questions.append(Question(
                item_id=f"mhop_{counter[0]:04d}",
                question_text=question_text,
                system_id=sid,
                task_family="multi_hop",
                ground_truth=ground_truth,
                predicates=[p, q, r],
                metadata={
                    "hop_count": 2,
                    "chain": [p, q, r],
                    "reasoning_type": chain_type,
                },
            ))

    rng.shuffle(questions)
    return questions


def _generate_3hop_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate 3-hop chain reasoning questions.

    Args:
        systems: Dict mapping system_id to system data.
        rules: FOL rules from get_fol_rules().
        rng: Random number generator.
        counter: Counter for item_id generation.

    Returns:
        List of 3-hop Question objects.
    """
    questions: List[Question] = []
    chains = _find_3hop_chains(rules)
    system_ids = sorted(systems.keys())

    # Only generate for chaotic systems (have longest chains)
    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)

        if not truth.get("Chaotic", False):
            continue

        for p, q, r, s, chain_type in chains:
            # Only generate if P is true for this system
            if not truth.get(p, False):
                continue

            counter[0] += 1
            p_disp = PREDICATE_DISPLAY.get(p, p.lower())
            q_disp = PREDICATE_DISPLAY.get(q, q.lower())
            r_disp = PREDICATE_DISPLAY.get(r, r.lower())
            s_disp = PREDICATE_DISPLAY.get(s, s.lower())

            if chain_type == "requires_3hop":
                # P→Q→R→S (YES)
                question_text = (
                    f"{name} is {p_disp}. Systems that are {p_disp} must be {q_disp}. "
                    f"Systems that are {q_disp} must be {r_disp}. Therefore, is {name} {s_disp}?"
                )
                ground_truth = "YES"
            else:  # mixed_3hop
                # P→Q→R→¬S (NO)
                question_text = (
                    f"{name} is {p_disp}. Systems that are {p_disp} must be {q_disp}. "
                    f"Systems that are {q_disp} must be {r_disp}. Systems that are {r_disp} "
                    f"cannot be {s_disp}. Therefore, can {name} be {s_disp}?"
                )
                ground_truth = "NO"

            questions.append(Question(
                item_id=f"mhop_{counter[0]:04d}",
                question_text=question_text,
                system_id=sid,
                task_family="multi_hop",
                ground_truth=ground_truth,
                predicates=[p, q, r, s],
                metadata={
                    "hop_count": 3,
                    "chain": [p, q, r, s],
                    "reasoning_type": chain_type,
                },
            ))

    rng.shuffle(questions)
    return questions


def _generate_contrapositive_fallacy_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate questions testing contrapositive fallacy (affirming consequent).

    Tests understanding that ¬P does not imply ¬Q even if P→Q.

    Args:
        systems: Dict mapping system_id to system data.
        rules: FOL rules from get_fol_rules().
        rng: Random number generator.
        counter: Counter for item_id generation.

    Returns:
        List of contrapositive fallacy Question objects (all NO).
    """
    questions: List[Question] = []
    system_ids = sorted(systems.keys())

    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)

        for p, p_rules in rules.items():
            # Only if P is FALSE for this system
            if truth.get(p, False):
                continue

            for q in p_rules.get("requires", []):
                counter[0] += 1
                p_disp = PREDICATE_DISPLAY.get(p, p.lower())
                q_disp = PREDICATE_DISPLAY.get(q, q.lower())

                question_text = (
                    f"If {name} is NOT {p_disp}, and {p_disp} systems require "
                    f"being {q_disp}, does this tell us anything definitive about "
                    f"whether it is {q_disp}?"
                )

                questions.append(Question(
                    item_id=f"mhop_{counter[0]:04d}",
                    question_text=question_text,
                    system_id=sid,
                    task_family="multi_hop",
                    ground_truth="NO",
                    predicates=[p, q],
                    metadata={
                        "hop_count": 2,
                        "chain": [p, q],
                        "reasoning_type": "contrapositive_fallacy",
                    },
                ))

    rng.shuffle(questions)
    return questions


def _generate_modus_tollens_questions(
    systems: Dict[str, Dict],
    rules: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate questions testing modus tollens (valid contrapositive).

    Tests understanding that if ¬Q and P→Q, then ¬P.

    Args:
        systems: Dict mapping system_id to system data.
        rules: FOL rules from get_fol_rules().
        rng: Random number generator.
        counter: Counter for item_id generation.

    Returns:
        List of modus tollens Question objects (all NO).
    """
    questions: List[Question] = []
    system_ids = sorted(systems.keys())

    for sid in system_ids:
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)

        for p, p_rules in rules.items():
            for q in p_rules.get("requires", []):
                # Only if Q is FALSE for this system
                if truth.get(q, False):
                    continue

                counter[0] += 1
                p_disp = PREDICATE_DISPLAY.get(p, p.lower())
                q_disp = PREDICATE_DISPLAY.get(q, q.lower())

                question_text = (
                    f"If {name} lacks being {q_disp}, and being {p_disp} requires "
                    f"being {q_disp}, can it be {p_disp}?"
                )

                questions.append(Question(
                    item_id=f"mhop_{counter[0]:04d}",
                    question_text=question_text,
                    system_id=sid,
                    task_family="multi_hop",
                    ground_truth="NO",
                    predicates=[p, q],
                    metadata={
                        "hop_count": 2,
                        "chain": [p, q],
                        "reasoning_type": "modus_tollens",
                    },
                ))

    rng.shuffle(questions)
    return questions


def generate_multi_hop_questions(
    systems: Dict[str, Dict],
    seed: int = 42,
    target_count: int = None,
) -> List[Question]:
    """Generate multi-hop FOL reasoning questions.

    Args:
        systems: Dict mapping system_id to system data with truth_assignment.
        seed: Random seed for reproducibility.
        target_count: Optional target number of questions. If specified,
            questions are truncated to this count after shuffling.

    Returns:
        List of Question objects with multi-hop reasoning chains.
    """
    rng = random.Random(seed)
    rules = get_fol_rules()
    counter = [0]

    questions: List[Question] = []

    # Generate all question types
    questions.extend(_generate_2hop_questions(systems, rules, rng, counter))
    questions.extend(_generate_3hop_questions(systems, rules, rng, counter))
    questions.extend(_generate_contrapositive_fallacy_questions(systems, rules, rng, counter))
    questions.extend(_generate_modus_tollens_questions(systems, rules, rng, counter))

    # Shuffle all questions together
    rng.shuffle(questions)

    # Balance YES/NO by alternating if target_count is specified
    if target_count is not None:
        yes_questions = [q for q in questions if q.ground_truth == "YES"]
        no_questions = [q for q in questions if q.ground_truth == "NO"]

        balanced = []
        for i in range(max(len(yes_questions), len(no_questions))):
            if i < len(yes_questions):
                balanced.append(yes_questions[i])
            if i < len(no_questions):
                balanced.append(no_questions[i])
            if len(balanced) >= target_count:
                break

        questions = balanced[:target_count]

        # Final shuffle to avoid strict alternation pattern
        rng.shuffle(questions)

    return questions


@dataclass
class MultiHopTask:
    """Task for testing multi-hop FOL reasoning about dynamical systems.

    Attributes:
        task_family: Always "multi_hop".
        systems: Dict mapping system_id to system data.
        seed: Random seed for reproducibility.
        target_count: Optional target number of questions to generate.
    """

    task_family: str = "multi_hop"
    systems: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42
    target_count: int = None

    def generate_items(self) -> List[Question]:
        """Generate multi-hop reasoning questions.

        Returns:
            List of Question objects.
        """
        if not self.systems:
            self.systems = _load_systems()
        return generate_multi_hop_questions(
            self.systems,
            self.seed,
            self.target_count,
        )

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
        by_hop: Dict[int, List[bool]] = {}

        for q in items:
            pred = predictions.get(q.item_id)
            if pred is None:
                continue
            total += 1
            is_correct = pred.upper() == q.ground_truth
            if is_correct:
                correct += 1

            # Score by reasoning type
            reasoning_type = q.metadata.get("reasoning_type", "unknown")
            if reasoning_type not in by_type:
                by_type[reasoning_type] = []
            by_type[reasoning_type].append(is_correct)

            # Score by hop count
            hop_count = q.metadata.get("hop_count", 0)
            if hop_count not in by_hop:
                by_hop[hop_count] = []
            by_hop[hop_count].append(is_correct)

        type_accuracy = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in sorted(by_type.items())
        }

        hop_accuracy = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in sorted(by_hop.items())
        }

        return {
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "total": total,
            "type_accuracy": type_accuracy,
            "hop_accuracy": hop_accuracy,
        }
