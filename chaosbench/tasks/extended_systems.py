"""Extended systems task: atomic predicate questions for underrepresented systems.

This module generates direct factual questions about dynamical systems that are
underrepresented in the base benchmark. It focuses on testing whether models have
learned basic properties of these systems.

The task probes atomic predicates (single predicate questions) about 15 systems
that appear less frequently in other task families, ensuring comprehensive coverage
across all 30 benchmark systems.

Target Systems
--------------
The following 15 systems are targeted:
- PDEs: sine_gordon, kuramoto_sivashinsky
- Chemical oscillators: oregonator, brusselator
- Dynamos: rikitake_dynamo
- Maps: circle_map_quasiperiodic, standard_map, ikeda_map, arnold_cat_map, bakers_map
- Biological: mackey_glass, hindmarsh_rose, fitzhugh_nagumo
- Mechanical: damped_driven_pendulum_nonchaotic
- Stochastic: stochastic_ou

Predicates Probed
-----------------
Seven predicates are sampled per system:
1. Chaotic - Exhibits chaotic dynamics
2. Deterministic - Fully determined by initial conditions
3. Periodic - Has periodic orbits
4. Sensitive - Sensitive to initial conditions
5. PosLyap - Has positive Lyapunov exponent
6. QuasiPeriodic - Has quasi-periodic motion
7. Random - Exhibits random/stochastic behavior

Question Format
---------------
Three template variations:
1. "Is the {name} {predicate}?"
2. "Does the {name} exhibit {predicate} behavior?"
3. "Can the {name} be characterized as {predicate}?"

Each system gets 3 predicates (randomly sampled), resulting in:
- Total: ~45 questions (15 systems × 3 predicates)
- Ground truth: Based on truth_assignment in system metadata
- Distribution: Approximately balanced YES/NO (varies by system)

Expected Distribution
---------------------
- Total: 45 questions (expanded from 30 in v2.1)
- Coverage: All 15 target systems
- Template variety: Questions rotated through 3 templates
- Ground truth balance: ~50% YES, ~50% NO

Implementation Notes
--------------------
- Predicates are shuffled per system for variety
- Questions use mutable list counter pattern for unique IDs
- Template selection rotates based on counter value
- All questions grounded in system's truth_assignment

Design Rationale
----------------
This task ensures that:
1. All 30 benchmark systems are adequately represented
2. Models are tested on systems beyond the common examples (Lorenz, Rössler)
3. Basic factual knowledge is assessed for underrepresented systems
4. Coverage includes diverse system types (ODEs, maps, PDEs, stochastic)

See Also
--------
- `tests/test_extended_systems.py` : Comprehensive test suite
- System metadata files in `systems/` directory
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from chaosbench.data.schemas import Question


PREDICATE_DISPLAY = {
    "Chaotic": "chaotic",
    "Deterministic": "deterministic",
    "PosLyap": "having a positive Lyapunov exponent",
    "Sensitive": "sensitive to initial conditions",
    "StrangeAttr": "possessing a strange attractor",
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
}

TARGET_SYSTEMS = [
    "sine_gordon",
    "kuramoto_sivashinsky",
    "oregonator",
    "brusselator",
    "rikitake_dynamo",
    "circle_map_quasiperiodic",
    "standard_map",
    "mackey_glass",
    "hindmarsh_rose",
    "fitzhugh_nagumo",
    "ikeda_map",
    "arnold_cat_map",
    "bakers_map",
    "damped_driven_pendulum_nonchaotic",
    "stochastic_ou",
]

TEMPLATES = [
    "Is the {name} {predicate_display}?",
    "Does the {name} exhibit {predicate_display} behavior?",
    "Can the {name} be characterized as {predicate_display}?",
]

# Expanded from 4 to 7 predicates to increase batch size
PREDICATES_TO_PROBE = [
    "Chaotic",
    "Deterministic",
    "Periodic",
    "Sensitive",
    "PosLyap",
    "QuasiPeriodic",
    "Random",
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


def generate_extended_system_questions(
    systems: Dict[str, Dict],
    seed: int = 42,
) -> List[Question]:
    """Generate atomic predicate questions for underrepresented systems.

    Args:
        systems: Dict mapping system_id to system data with truth_assignment.
        seed: Random seed for reproducibility.

    Returns:
        List of Question objects.
    """
    rng = random.Random(seed)
    questions: List[Question] = []
    counter = [0]  # Mutable list for shared state (consistent with other task modules)

    for sid in sorted(TARGET_SYSTEMS):
        if sid not in systems:
            continue

        sys_data = systems[sid]
        truth = sys_data.get("truth_assignment", {})
        name = sys_data.get("name", sid)

        probes = list(PREDICATES_TO_PROBE)
        rng.shuffle(probes)

        # Sample 3 predicates per system (increased from 2 to expand batch size)
        for pred in probes[:3]:
            pred_disp = PREDICATE_DISPLAY.get(pred, pred.lower())
            template = TEMPLATES[counter[0] % len(TEMPLATES)]
            gt_val = truth.get(pred, False)
            ground_truth = "YES" if gt_val else "NO"

            counter[0] += 1
            questions.append(Question(
                item_id=f"ext_sys_{counter[0]:04d}",
                question_text=template.format(
                    name=name,
                    predicate_display=pred_disp,
                ),
                system_id=sid,
                task_family="extended_systems",
                ground_truth=ground_truth,
                predicates=[pred],
                metadata={
                    "question_type": "atomic_predicate",
                    "target_predicate": pred,
                },
            ))

    return questions


@dataclass
class ExtendedSystemsTask:
    """Task for testing knowledge of underrepresented dynamical systems.

    Attributes:
        task_family: Always "extended_systems".
        systems: Dict mapping system_id to system data.
        seed: Random seed.
    """

    task_family: str = "extended_systems"
    systems: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42

    def generate_items(self) -> List[Question]:
        """Generate extended system questions.

        Returns:
            List of Question objects.
        """
        if not self.systems:
            self.systems = _load_systems()
        return generate_extended_system_questions(self.systems, self.seed)

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions.

        Args:
            predictions: Dict mapping item_id to predicted label.

        Returns:
            Dict with accuracy breakdown.
        """
        items = self.generate_items()
        correct = 0
        total = 0

        for q in items:
            pred = predictions.get(q.item_id)
            if pred is None:
                continue
            total += 1
            if pred.upper() == q.ground_truth:
                correct += 1

        return {
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "total": total,
        }
