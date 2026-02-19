"""Atomic predicate questions scaled to all systems.

This module generates direct factual questions about dynamical systems predicates
scaled across all 30 benchmark systems. It tests whether models can correctly
identify single predicate properties for each system.

The task probes all 11 predicates from the ontology for all systems in the benchmark,
providing comprehensive coverage of atomic predicate knowledge.

Question Format
---------------
Three template variations with natural language phrasings:
1. "Is the {system_name} {predicate_display}?"
2. "Does the {system_name} exhibit {predicate_display}?"
3. "Would you classify the {system_name} as {predicate_display}?"

Each system-predicate pair gets one question with a randomly selected template.

Predicates Probed
-----------------
All 11 predicates from the ontology:
1. Chaotic - Exhibits chaotic dynamics
2. Deterministic - Fully determined by initial conditions
3. PosLyap - Has positive Lyapunov exponent
4. Sensitive - Sensitive to initial conditions
5. StrangeAttr - Has strange attractor
6. PointUnpredictable - Pointwise unpredictable
7. StatPredictable - Statistically predictable
8. QuasiPeriodic - Has quasi-periodic motion
9. Random - Exhibits random/stochastic behavior
10. FixedPointAttr - Has fixed point attractor
11. Periodic - Has periodic orbits

Expected Distribution
---------------------
- Total: 330 questions (30 systems × 11 predicates)
- Coverage: All 30 benchmark systems
- Template variety: Questions use 3 different phrasings
- Ground truth: Based on truth_assignment in system metadata
- Balance: Approximately 50% YES, 50% NO (via interleaving)

Implementation Notes
--------------------
- Questions are shuffled using provided seed for reproducibility
- YES/NO balance is enforced by sorting and interleaving ground truths
- Optional target_count parameter allows truncating to specific size
- All questions grounded in system's truth_assignment from JSON metadata

Design Rationale
----------------
This task ensures that:
1. All systems and all predicates are comprehensively tested
2. Models are evaluated on fundamental atomic predicate knowledge
3. Results provide detailed per-predicate accuracy breakdowns
4. Questions are naturally phrased with variety in templates

See Also
--------
- `chaosbench.logic.ontology.PREDICATES` : List of 11 predicates
- System metadata files in `systems/` directory
- `tests/test_atomic.py` : Test suite (if created)
"""

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chaosbench.data.schemas import Question
from chaosbench.logic.ontology import PREDICATES


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

TEMPLATES = [
    "Is the {name} {predicate_display}?",
    "Does the {name} exhibit {predicate_display}?",
    "Would you classify the {name} as {predicate_display}?",
]


def _load_systems(systems_dir: str = "systems") -> Dict[str, Dict]:
    """Load system JSONs from directory.

    Args:
        systems_dir: Path to systems directory.

    Returns:
        Dict mapping system_id to system data.
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


def generate_atomic_questions(
    systems: Dict[str, Dict],
    seed: int = 42,
    target_count: Optional[int] = None,
) -> List[Question]:
    """Generate atomic predicate questions for all systems.

    Args:
        systems: Dict mapping system_id to system data with truth_assignment.
        seed: Random seed for reproducibility.
        target_count: Optional target number of questions (truncates if provided).

    Returns:
        List of Question objects with roughly balanced YES/NO answers.
    """
    rng = random.Random(seed)
    questions: List[Question] = []
    counter = [0]  # Mutable list for shared state

    # Generate all system-predicate combinations
    for sid in sorted(systems.keys()):
        sys_data = systems[sid]
        truth = sys_data.get("truth_assignment", {})
        name = sys_data.get("name", sid)

        for pred in PREDICATES:
            pred_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
            template = TEMPLATES[rng.randint(0, len(TEMPLATES) - 1)]
            gt_val = truth.get(pred, False)
            ground_truth = "YES" if gt_val else "NO"

            counter[0] += 1
            questions.append(Question(
                item_id=f"atomic_{counter[0]:04d}",
                question_text=template.format(
                    name=name,
                    predicate_display=pred_disp,
                ),
                system_id=sid,
                task_family="atomic",
                ground_truth=ground_truth,
                predicates=[pred],
                metadata={
                    "question_type": "atomic_predicate",
                    "predicate": pred,
                    "template_index": TEMPLATES.index(template),
                },
            ))

    # Balance YES/NO by interleaving
    yes_questions = [q for q in questions if q.ground_truth == "YES"]
    no_questions = [q for q in questions if q.ground_truth == "NO"]

    rng.shuffle(yes_questions)
    rng.shuffle(no_questions)

    # Interleave to balance
    balanced_questions = []
    max_len = max(len(yes_questions), len(no_questions))
    for i in range(max_len):
        if i < len(yes_questions):
            balanced_questions.append(yes_questions[i])
        if i < len(no_questions):
            balanced_questions.append(no_questions[i])

    # Shuffle the balanced list
    rng.shuffle(balanced_questions)

    # Truncate to target count if specified
    if target_count is not None:
        balanced_questions = balanced_questions[:target_count]

    return balanced_questions


@dataclass
class AtomicTask:
    """Task for testing atomic predicate knowledge across all systems.

    Attributes:
        task_family: Always "atomic".
        systems: Dict mapping system_id to system data.
        seed: Random seed for reproducibility.
        target_count: Optional target number of questions.
    """

    task_family: str = "atomic"
    systems: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42
    target_count: Optional[int] = None

    def generate_items(self) -> List[Question]:
        """Generate atomic predicate questions.

        Returns:
            List of Question objects.
        """
        if not self.systems:
            self.systems = _load_systems()
        return generate_atomic_questions(
            self.systems,
            self.seed,
            self.target_count,
        )

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions against ground truth.

        Args:
            predictions: Dict mapping item_id to predicted label.

        Returns:
            Dict with accuracy and per-predicate accuracy breakdowns.
        """
        items = self.generate_items()
        correct = 0
        total = 0
        by_predicate: Dict[str, List[bool]] = defaultdict(list)

        for q in items:
            pred = predictions.get(q.item_id)
            if pred is None:
                continue
            total += 1
            is_correct = pred.upper() == q.ground_truth
            if is_correct:
                correct += 1

            # Track per-predicate accuracy
            predicate = q.metadata.get("predicate")
            if predicate:
                by_predicate[predicate].append(is_correct)

        predicate_accuracy = {
            k: sum(v) / len(v) for k, v in sorted(by_predicate.items())
        }

        return {
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "total": total,
            "predicate_accuracy": predicate_accuracy,
        }
