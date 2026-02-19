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
from chaosbench.data.grouping import _normalize_text


# Negation templates for logical sense flipping
NEGATION_TEMPLATES = [
    "Is it false that {question_core}?",
    "Is it NOT the case that {question_core}?",
    "Would you say it is incorrect that {question_core}?",
]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Edit distance (number of insertions/deletions/substitutions).
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _generate_name_aliases(system_id: str, system_name: str) -> List[str]:
    """Generate common name variants for a system (v2.2 entity swap fix).

    Creates aliases to improve entity swap success rate by handling
    common name format differences.

    Args:
        system_id: System identifier.
        system_name: System display name.

    Returns:
        List of name variants to try during entity swap.
    """
    aliases = []

    # Always include original name
    aliases.append(system_name)

    # Remove leading "the"
    if system_name.lower().startswith("the "):
        aliases.append(system_name[4:])

    # Add/remove "system", "map", "oscillator", etc.
    for suffix in [" system", " map", " oscillator", " model", " equation"]:
        if suffix in system_name.lower():
            # Remove suffix
            aliases.append(system_name.replace(suffix, "").replace(suffix.title(), "").strip())
        else:
            # Add suffix
            aliases.append(f"{system_name}{suffix}")

    # Add system_id as fallback
    aliases.append(system_id)

    # Handle unicode variants (Rössler → Rossler, Poincaré → Poincare)
    unicode_map = {
        'ö': 'o', 'ü': 'u', 'ä': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e',
        'á': 'a', 'à': 'a', 'â': 'a',
        'ñ': 'n',
    }
    ascii_name = system_name
    for unicode_char, ascii_char in unicode_map.items():
        ascii_name = ascii_name.replace(unicode_char, ascii_char)
    if ascii_name != system_name:
        aliases.append(ascii_name)

    # Extract first word (often the key identifier: "Lorenz", "Rössler")
    first_word = system_name.split()[0] if system_name else ""
    if first_word and len(first_word) > 3:  # Avoid short words like "The"
        aliases.append(first_word)

    # Remove duplicates while preserving order
    seen = set()
    unique_aliases = []
    for alias in aliases:
        alias_lower = alias.lower()
        if alias_lower not in seen:
            seen.add(alias_lower)
            unique_aliases.append(alias)

    return unique_aliases


def _try_entity_swap_with_fallback(
    base_text: str,
    base_system_id: str,
    base_name: str,
    target_name: str,
    systems: Dict[str, Dict],
) -> tuple[bool, str]:
    """Try entity swap with multiple fallback strategies (v2.2 robustness fix).

    Attempts entity swap using multiple strategies to maximize success rate:
    1. Exact name match (current behavior)
    2. Name aliases (with/without "system", unicode variants, etc.)
    3. Fuzzy match (Levenshtein distance ≤2)
    4. First word match (e.g., "Lorenz" in "Lorenz system")

    Args:
        base_text: Original question text.
        base_system_id: Original system identifier.
        base_name: Original system name.
        target_name: Target system name to swap to.
        systems: Dict of all systems.

    Returns:
        Tuple of (success: bool, swapped_text: str).
    """
    # Generate aliases for base system
    base_aliases = _generate_name_aliases(base_system_id, base_name)

    # Strategy 1: Try exact match with each alias
    for alias in base_aliases:
        alias_escaped = re.escape(alias)
        swapped = re.sub(
            r'\b' + alias_escaped + r'\b',
            target_name,
            base_text,
            flags=re.IGNORECASE,
            count=1,  # Only replace first occurrence
        )
        if swapped != base_text:
            return True, swapped

    # Strategy 2: Try case-insensitive partial match (remove word boundaries)
    for alias in base_aliases:
        alias_escaped = re.escape(alias)
        swapped = re.sub(
            alias_escaped,
            target_name,
            base_text,
            flags=re.IGNORECASE,
            count=1,
        )
        if swapped != base_text:
            return True, swapped

    # Strategy 3: Fuzzy match (Levenshtein distance ≤2)
    # Find words in base_text that are close to base_name
    words = base_text.split()
    for i, word in enumerate(words):
        word_clean = word.strip('.,?!;:"\'')
        for alias in base_aliases:
            if _levenshtein_distance(word_clean.lower(), alias.lower()) <= 2 and len(alias) > 3:
                # Replace this word
                words[i] = word.replace(word_clean, target_name)
                swapped = ' '.join(words)
                if swapped != base_text:
                    return True, swapped

    # Strategy 4: Try replacing just the first significant word (e.g., "Lorenz" in "Lorenz system")
    first_word = base_name.split()[0] if base_name else ""
    if first_word and len(first_word) > 3:
        # Replace first word, case-insensitive
        first_word_escaped = re.escape(first_word)
        swapped = re.sub(
            r'\b' + first_word_escaped + r'\b',
            target_name.split()[0] if target_name else target_name,
            base_text,
            flags=re.IGNORECASE,
            count=1,
        )
        if swapped != base_text:
            return True, swapped

    # Strategy 5: Try replacing system_id literally
    if base_system_id in base_text:
        swapped = base_text.replace(base_system_id, target_name, 1)
        if swapped != base_text:
            return True, swapped

    # Strategy 6: Aggressive fallback - find the longest common substring
    # between base_name and text, replace it with target_name
    for alias in base_aliases:
        if len(alias) > 4:  # Avoid very short matches
            alias_lower = alias.lower()
            text_lower = base_text.lower()
            if alias_lower in text_lower:
                # Find position and replace
                start = text_lower.index(alias_lower)
                swapped = base_text[:start] + target_name + base_text[start + len(alias):]
                if swapped != base_text:
                    return True, swapped

    # All strategies failed
    return False, base_text


def _compute_uniqueness_key(question_text: str, system_id: str, ground_truth: str, family: str) -> str:
    """Compute uniqueness key for duplicate detection.

    Args:
        question_text: Question text.
        system_id: System identifier.
        ground_truth: Ground truth label.
        family: Task family.

    Returns:
        Uniqueness key string.
    """
    norm_text = _normalize_text(question_text)
    return f"{family}:{system_id}:{norm_text}:{ground_truth}"


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

    Uses regex-based name replacement to ensure actual text change occurs.
    Skips swaps where replacement fails to produce different text.

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

        # Replace system name using fallback strategy (v2.2 fix for 64% failure rate)
        base_name = systems[base_system_id].get("name", base_system_id)

        # Try entity swap with multiple fallback strategies
        success, swapped_text = _try_entity_swap_with_fallback(
            base_q.question_text,
            base_system_id,
            base_name,
            target_name,
            systems,
        )

        # Skip if all strategies failed (text unchanged)
        if not success or swapped_text == base_q.question_text:
            continue

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

    # Phase 2: Apply perturbations with uniqueness tracking
    all_questions: List[Question] = []
    counter = 0
    seen_keys: Set[str] = set()

    def add_unique_questions(questions: List[Question]) -> int:
        """Add questions to all_questions, skipping accidental duplicates."""
        added = 0
        for q in questions:
            key = _compute_uniqueness_key(q.question_text, q.system_id, q.ground_truth, q.task_family)
            if key not in seen_keys:
                all_questions.append(q)
                seen_keys.add(key)
                added += 1
        return added

    if "paraphrase" in perturbation_types:
        paraphrase_qs = _apply_paraphrase_perturbation(base_questions, counter, seed)
        added = add_unique_questions(paraphrase_qs)
        counter += added

    if "negation" in perturbation_types:
        negation_qs = _apply_negation_perturbation(base_questions, counter, seed)
        added = add_unique_questions(negation_qs)
        counter += added

    if "entity_swap" in perturbation_types:
        swap_qs = _apply_entity_swap_perturbation(base_questions, systems, counter, seed)
        added = add_unique_questions(swap_qs)
        counter += added

    if "distractor" in perturbation_types:
        distractor_qs = _apply_distractor_perturbation(base_questions, counter, seed)
        added = add_unique_questions(distractor_qs)
        counter += added

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
