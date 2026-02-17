"""Systematic perturbation framework for robustness testing.

This module provides a comprehensive perturbation testing framework that builds on
the existing perturbation primitives in chaosbench.data.perturb. It generates
controlled variants of atomic questions to test model robustness across four
perturbation types:

1. **Paraphrase Perturbation** - Semantic-preserving rewording
2. **Negation Perturbation** - Logical sense flipping
3. **Entity Swap Perturbation** - System substitution with ground truth update
4. **Distractor Insertion** - Addition of irrelevant context

Architecture
------------
The framework operates in two phases:

Phase 1: Base Question Generation
    Generate atomic questions (one per system per predicate) using a subset
    of predicates for computational efficiency.

Phase 2: Perturbation Application
    Apply each perturbation type to create variants with proper ground truth
    preservation or transformation.

Perturbation Types
------------------

**Paraphrase Perturbation**
    Uses existing paraphrase() from perturb.py to reword questions while
    preserving semantic meaning and ground truth.

    Example:
        Base: "Is the Lorenz system chaotic?"
        Perturbed: "Does the Lorenz system exhibit chaos?"
        Ground truth: PRESERVED

**Negation Perturbation**
    Flips the logical sense of the question, inverting ground truth.

    Templates:
        - "Is it false that {question}?"
        - "Is it NOT the case that {question}?"
        - "Would you say it is incorrect that {question}?"

    Example:
        Base: "Is the Lorenz system chaotic?" (YES)
        Perturbed: "Is it false that the Lorenz system is chaotic?" (NO)

**Entity Swap Perturbation**
    Replaces the system name with a different system, updating ground truth
    based on the new system's truth_assignment.

    Strategy:
        Only swap to systems where the answer differs, ensuring the
        perturbation creates a meaningful test case.

    Example:
        Base: "Is the Lorenz system chaotic?" (YES)
        Perturbed: "Is the simple harmonic oscillator chaotic?" (NO)

**Distractor Insertion**
    Prepends irrelevant but plausible context before the question.

    Example:
        Base: "Is the Lorenz system chaotic?"
        Perturbed: "The system has 3 degrees of freedom and was first
                    studied in 1963. Is the Lorenz system chaotic?"
        Ground truth: PRESERVED

Metadata Tracking
-----------------
Each perturbed question includes metadata:
    - perturbation_type: "paraphrase" | "negation" | "entity_swap" | "distractor"
    - base_item_id: Original question identifier
    - base_system_id: Original system identifier
    - (entity_swap only) target_system_id: New system identifier

Usage Example
-------------
```python
from chaosbench.tasks.perturbation_robustness import (
    generate_perturbation_questions,
    PerturbationRobustnessTask,
)

# Generate perturbed questions
questions = generate_perturbation_questions(
    systems=systems_dict,
    seed=42,
    target_count=500,
    perturbation_types=["paraphrase", "negation"],
)

# Or use the task dataclass
task = PerturbationRobustnessTask(
    systems=systems_dict,
    seed=42,
    target_count=500,
    perturbation_types=["paraphrase", "negation"],
)
questions = task.generate_items()
results = task.score(predictions)
```

Expected Distribution
---------------------
With all 30 systems and 5 predicates per system:
    - Base questions: ~150
    - Paraphrase variants: ~150
    - Negation variants: ~150
    - Entity swap variants: ~150 (varies by system compatibility)
    - Distractor variants: ~150
    - Total: ~750 questions (before truncation)

The framework ensures approximately balanced YES/NO distribution and supports
flexible perturbation type selection for targeted robustness testing.

See Also
--------
- chaosbench.data.perturb : Core perturbation primitives
- chaosbench.tasks.consistency : Paraphrase consistency testing
- tests/test_perturbations.py : Comprehensive test suite
"""

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from chaosbench.data.schemas import Question
from chaosbench.data.perturb import paraphrase, DISTRACTOR_TEMPLATES
from chaosbench.logic.ontology import PREDICATES


# Negation templates for logical sense flipping
NEGATION_TEMPLATES = [
    "Is it false that {question_core}?",
    "Is it NOT the case that {question_core}?",
    "Would you say it is incorrect that {question_core}?",
]


def _strip_question_mark(text: str) -> str:
    """Remove trailing question mark from text.

    Args:
        text: Question text.

    Returns:
        Text without trailing question mark.
    """
    text = text.strip()
    if text.endswith("?"):
        return text[:-1].strip()
    return text


def _extract_question_core(question_text: str) -> str:
    """Extract the core statement from a question for negation.

    Converts "Is the Lorenz system chaotic?" to "the Lorenz system is chaotic"

    Args:
        question_text: Original question text.

    Returns:
        Core statement suitable for negation template insertion.
    """
    text = _strip_question_mark(question_text)

    # Handle "Is X..." pattern
    if text.lower().startswith("is "):
        core = text[3:].strip()
        # Insert "is" after the subject
        # Simple heuristic: assume subject ends at first major word boundary
        words = core.split(None, 2)
        if len(words) >= 2:
            return f"{words[0]} is {' '.join(words[1:])}"
        return f"{core} is true"

    # Handle "Does X..." pattern
    if text.lower().startswith("does "):
        core = text[5:].strip()
        return core

    # Fallback: return as-is
    return text


def _generate_base_atomic_questions(
    systems: Dict[str, Dict],
    num_predicates: int = 5,
    seed: int = 42,
) -> List[Question]:
    """Generate atomic base questions for perturbation.

    Creates one question per system per predicate (using first num_predicates).

    Args:
        systems: Dict mapping system_id to system info with truth_assignment.
        num_predicates: Number of predicates to use per system.
        seed: Random seed for reproducibility.

    Returns:
        List of atomic Question objects.
    """
    rng = random.Random(seed)
    questions: List[Question] = []
    counter = 0

    # Use first num_predicates for efficiency
    predicates_subset = PREDICATES[:num_predicates]

    for system_id in sorted(systems.keys()):
        system_info = systems[system_id]
        name = system_info.get("name", system_id)
        truth = system_info.get("truth_assignment", {})

        for predicate in predicates_subset:
            if predicate not in truth:
                continue

            ground_truth = "YES" if truth[predicate] else "NO"

            # Generate simple atomic question
            question_text = f"Is {name} {predicate.lower()}?"

            counter += 1
            questions.append(Question(
                item_id=f"base_atomic_{counter:04d}",
                question_text=question_text,
                system_id=system_id,
                task_family="perturbation",
                ground_truth=ground_truth,
                predicates=[predicate],
                metadata={
                    "perturbation_type": "base",
                    "base_system_id": system_id,
                },
            ))

    return questions


def _apply_paraphrase_perturbation(
    base_questions: List[Question],
    counter_start: int,
    seed: int,
) -> List[Question]:
    """Apply paraphrase perturbation to base questions.

    Args:
        base_questions: List of base atomic questions.
        counter_start: Starting counter for item IDs.
        seed: Random seed.

    Returns:
        List of paraphrased Question variants.
    """
    perturbed: List[Question] = []
    counter = counter_start

    for base_q in base_questions:
        # Use paraphrase() from perturb.py
        para_q, record = paraphrase(base_q, seed=seed + counter)

        # Update IDs and metadata
        counter += 1
        para_q.item_id = f"perturb_paraphrase_{counter:04d}"
        para_q.task_family = "perturbation"
        para_q.metadata = {
            "perturbation_type": "paraphrase",
            "base_item_id": base_q.item_id,
            "base_system_id": base_q.system_id,
            "perturbation_record": record.details,
        }

        perturbed.append(para_q)

    return perturbed


def _apply_negation_perturbation(
    base_questions: List[Question],
    counter_start: int,
    seed: int,
) -> List[Question]:
    """Apply negation perturbation to base questions.

    Args:
        base_questions: List of base atomic questions.
        counter_start: Starting counter for item IDs.
        seed: Random seed.

    Returns:
        List of negated Question variants with flipped ground truth.
    """
    rng = random.Random(seed)
    perturbed: List[Question] = []
    counter = counter_start

    for base_q in base_questions:
        # Extract question core for negation
        core = _extract_question_core(base_q.question_text)

        # Select random negation template
        template = rng.choice(NEGATION_TEMPLATES)
        negated_text = template.format(question_core=core)

        # Flip ground truth
        flipped_truth = "NO" if base_q.ground_truth == "YES" else "YES"

        counter += 1
        negated_q = Question(
            item_id=f"perturb_negation_{counter:04d}",
            question_text=negated_text,
            system_id=base_q.system_id,
            task_family="perturbation",
            ground_truth=flipped_truth,
            predicates=base_q.predicates.copy(),
            metadata={
                "perturbation_type": "negation",
                "base_item_id": base_q.item_id,
                "base_system_id": base_q.system_id,
                "original_ground_truth": base_q.ground_truth,
            },
        )

        perturbed.append(negated_q)

    return perturbed


def _apply_entity_swap_perturbation(
    base_questions: List[Question],
    systems: Dict[str, Dict],
    counter_start: int,
    seed: int,
) -> List[Question]:
    """Apply entity swap perturbation to base questions.

    Replaces system name with a different system, updating ground truth
    based on new system's truth_assignment. Only swaps to systems where
    the answer differs.

    Args:
        base_questions: List of base atomic questions.
        systems: Dict mapping system_id to system info.
        counter_start: Starting counter for item IDs.
        seed: Random seed.

    Returns:
        List of entity-swapped Question variants.
    """
    rng = random.Random(seed)
    perturbed: List[Question] = []
    counter = counter_start

    # Pre-index systems by predicate truth values for efficient lookup
    system_ids = sorted(systems.keys())

    for base_q in base_questions:
        if not base_q.predicates:
            continue

        predicate = base_q.predicates[0]
        base_system_id = base_q.system_id
        base_truth = base_q.ground_truth

        # Find systems with different answer for this predicate
        candidates: List[str] = []
        for sid in system_ids:
            if sid == base_system_id:
                continue

            truth = systems[sid].get("truth_assignment", {})
            if predicate not in truth:
                continue

            new_truth = "YES" if truth[predicate] else "NO"
            if new_truth != base_truth:
                candidates.append(sid)

        if not candidates:
            continue

        # Select random swap target
        target_system_id = rng.choice(candidates)
        target_info = systems[target_system_id]
        target_name = target_info.get("name", target_system_id)
        target_truth = target_info.get("truth_assignment", {})
        new_ground_truth = "YES" if target_truth[predicate] else "NO"

        # Replace system name in question text
        base_name = systems[base_system_id].get("name", base_system_id)
        swapped_text = base_q.question_text.replace(base_name, target_name)

        counter += 1
        swapped_q = Question(
            item_id=f"perturb_entity_swap_{counter:04d}",
            question_text=swapped_text,
            system_id=target_system_id,
            task_family="perturbation",
            ground_truth=new_ground_truth,
            predicates=base_q.predicates.copy(),
            metadata={
                "perturbation_type": "entity_swap",
                "base_item_id": base_q.item_id,
                "base_system_id": base_system_id,
                "target_system_id": target_system_id,
                "original_ground_truth": base_truth,
            },
        )

        perturbed.append(swapped_q)

    return perturbed


def _apply_distractor_perturbation(
    base_questions: List[Question],
    counter_start: int,
    seed: int,
) -> List[Question]:
    """Apply distractor insertion perturbation to base questions.

    Prepends irrelevant but plausible context from DISTRACTOR_TEMPLATES
    before the question.

    Args:
        base_questions: List of base atomic questions.
        counter_start: Starting counter for item IDs.
        seed: Random seed.

    Returns:
        List of distractor-enhanced Question variants.
    """
    rng = random.Random(seed)
    perturbed: List[Question] = []
    counter = counter_start

    for base_q in base_questions:
        # Generate distractor text
        template = rng.choice(DISTRACTOR_TEMPLATES)
        distractor = template.format(
            n=rng.randint(2, 10),
            dt=rng.choice(["0.01", "0.001", "0.1"]),
            year=rng.randint(1960, 2020),
            dim=rng.randint(2, 6),
        )

        # Prepend distractor to question
        enhanced_text = f"{distractor} {base_q.question_text}"

        counter += 1
        distractor_q = Question(
            item_id=f"perturb_distractor_{counter:04d}",
            question_text=enhanced_text,
            system_id=base_q.system_id,
            task_family="perturbation",
            ground_truth=base_q.ground_truth,
            predicates=base_q.predicates.copy(),
            metadata={
                "perturbation_type": "distractor",
                "base_item_id": base_q.item_id,
                "base_system_id": base_q.system_id,
                "distractor_text": distractor,
            },
        )

        perturbed.append(distractor_q)

    return perturbed


def generate_perturbation_questions(
    systems: Dict[str, Dict],
    seed: int = 42,
    target_count: Optional[int] = None,
    perturbation_types: Optional[List[str]] = None,
) -> List[Question]:
    """Generate perturbation robustness questions.

    Creates base atomic questions then applies selected perturbation types
    to generate variants. Balances YES/NO distribution and shuffles before
    optional truncation.

    Args:
        systems: Dict mapping system_id to system info with truth_assignment
            and name.
        seed: Random seed for reproducibility.
        target_count: Optional maximum number of questions to return.
        perturbation_types: Optional list of perturbation types to include.
            Defaults to all four: ["paraphrase", "negation", "entity_swap",
            "distractor"].

    Returns:
        List of Question objects with balanced YES/NO and shuffled order.

    Example:
        >>> questions = generate_perturbation_questions(
        ...     systems=systems_dict,
        ...     seed=42,
        ...     target_count=500,
        ...     perturbation_types=["paraphrase", "negation"],
        ... )
    """
    rng = random.Random(seed)

    # Default to all perturbation types
    if perturbation_types is None:
        perturbation_types = ["paraphrase", "negation", "entity_swap", "distractor"]

    # Phase 1: Generate base atomic questions
    base_questions = _generate_base_atomic_questions(systems, num_predicates=5, seed=seed)

    # Phase 2: Apply perturbations
    all_questions: List[Question] = []
    counter = 0

    if "paraphrase" in perturbation_types:
        paraphrase_qs = _apply_paraphrase_perturbation(base_questions, counter, seed)
        all_questions.extend(paraphrase_qs)
        counter += len(paraphrase_qs)

    if "negation" in perturbation_types:
        negation_qs = _apply_negation_perturbation(base_questions, counter, seed)
        all_questions.extend(negation_qs)
        counter += len(negation_qs)

    if "entity_swap" in perturbation_types:
        swap_qs = _apply_entity_swap_perturbation(base_questions, systems, counter, seed)
        all_questions.extend(swap_qs)
        counter += len(swap_qs)

    if "distractor" in perturbation_types:
        distractor_qs = _apply_distractor_perturbation(base_questions, counter, seed)
        all_questions.extend(distractor_qs)
        counter += len(distractor_qs)

    # Balance YES/NO distribution
    yes_questions = [q for q in all_questions if q.ground_truth == "YES"]
    no_questions = [q for q in all_questions if q.ground_truth == "NO"]

    # Equalize counts
    min_count = min(len(yes_questions), len(no_questions))
    rng.shuffle(yes_questions)
    rng.shuffle(no_questions)

    balanced = yes_questions[:min_count] + no_questions[:min_count]
    rng.shuffle(balanced)

    # Apply target count if specified
    if target_count is not None:
        balanced = balanced[:target_count]

    return balanced


@dataclass
class PerturbationRobustnessTask:
    """Task for systematic perturbation robustness testing.

    Attributes:
        task_family: Always "perturbation".
        systems: Dict mapping system_id to system info including
            truth_assignment and name.
        seed: Random seed for reproducible generation.
        target_count: Optional maximum number of questions.
        perturbation_types: List of perturbation types to include.
            Defaults to all four types.
    """

    task_family: str = "perturbation"
    systems: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42
    target_count: Optional[int] = None
    perturbation_types: Optional[List[str]] = None

    def generate_items(self) -> List[Question]:
        """Generate perturbation robustness questions.

        Returns:
            List of Question objects with perturbations applied.
        """
        return generate_perturbation_questions(
            systems=self.systems,
            seed=self.seed,
            target_count=self.target_count,
            perturbation_types=self.perturbation_types,
        )

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions with per-perturbation-type breakdown.

        Args:
            predictions: Dict mapping item_id to predicted label
                ("YES" or "NO").

        Returns:
            Dict with overall accuracy and per-perturbation-type metrics:
                - accuracy: Overall accuracy across all questions
                - correct: Total correct predictions
                - total: Total questions with predictions
                - by_perturbation_type: Dict mapping perturbation type to metrics
                    - accuracy: Type-specific accuracy
                    - correct: Correct predictions for this type
                    - total: Total questions for this type
        """
        items = self.generate_items()

        if not items:
            return {
                "accuracy": None,
                "correct": 0,
                "total": 0,
                "by_perturbation_type": {},
            }

        type_buckets: Dict[str, List[tuple]] = {}
        correct = 0
        total = 0

        for q in items:
            pred = predictions.get(q.item_id)
            if pred is None:
                continue

            total += 1
            is_correct = pred == q.ground_truth
            if is_correct:
                correct += 1

            # Track by perturbation type
            p_type = q.metadata.get("perturbation_type", "unknown")
            if p_type not in type_buckets:
                type_buckets[p_type] = []
            type_buckets[p_type].append((q.item_id, q.ground_truth, pred))

        # Compute per-type metrics
        by_perturbation_type: Dict[str, Dict[str, Any]] = {}
        for p_type, entries in sorted(type_buckets.items()):
            type_correct = sum(1 for _, gt, p in entries if p == gt)
            by_perturbation_type[p_type] = {
                "accuracy": type_correct / len(entries) if entries else None,
                "correct": type_correct,
                "total": len(entries),
            }

        return {
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "total": total,
            "by_perturbation_type": by_perturbation_type,
        }
