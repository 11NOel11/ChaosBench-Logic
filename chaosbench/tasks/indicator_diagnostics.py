"""Indicator diagnostic task for testing model reasoning about chaos indicators.

This module generates questions that test understanding of how chaos indicators
relate to system properties. It requires models to interpret numerical indicator
values and reason about their implications for classifying systems as chaotic
or non-chaotic.

The task focuses on three main chaos indicators:
- Zero-One K test
- Permutation Entropy
- MEGNO (Mean Exponential Growth of Nearby Orbits)

Question Types
--------------
1. **Direct Interpretation** (~30-40 questions):
   Given an indicator value, determine if the system is chaotic.
   Example: "The 0-1 test gives K=0.95 for Lorenz-63. Is this system chaotic?"
   → YES (ground truth from truth_assignment)

   Tests direct indicator → classification reasoning.

2. **Multi-Indicator** (~20-30 questions):
   Given multiple indicator values, determine system classification.
   Example: "System X has K=0.92, PE=0.88. Is it chaotic?"
   → Based on ground truth (not simple threshold rule)

   Tests ability to integrate multiple sources of evidence.

3. **Comparison** (~15-20 questions):
   Compare indicator values across systems to rank chaos.
   Example: "System A has K=0.92, System B has K=0.15. Which is more chaotic?"
   → System A (higher K value)

   Tests relative reasoning about indicator values.

Indicator Ranges and Thresholds
--------------------------------
Empirically validated thresholds (v2.1):

**Zero-One K:**
- Range: [0.0, 1.0]
- Chaotic threshold: 0.5 (theoretical, limited discriminative power)
- Interpretation: K ≈ 1 suggests chaos, K ≈ 0 suggests regular dynamics

**Permutation Entropy:**
- Range: [0.0, 1.0]
- Chaotic threshold: 0.40 (empirical optimal, 70% accuracy)
- Regular threshold: 0.30
- Interpretation: Higher PE suggests more complex/chaotic behavior

**MEGNO:**
- Range: [-30.0, 50.0] (with validation rejecting |MEGNO| > 50)
- Chaotic threshold: 0.55 (empirical optimal, 95.5% accuracy)
- Regular value: ~2.0 (theoretical)
- Interpretation: MEGNO > 0.55 suggests chaos, MEGNO < 0.55 suggests regular

See `docs/INDICATOR_THRESHOLDS.md` for detailed threshold analysis.

Expected Distribution
---------------------
- Total: ~550 questions (largest batch in ChaosBench-Logic v2)
- Ground truth: Based on system truth_assignment (not indicator thresholds)
- Coverage: All 30 systems with valid indicator values
- Balance: Approximately 60% chaotic, 40% non-chaotic (reflects dataset composition)

Implementation Notes
--------------------
- Questions generated for systems with non-None indicator values
- Ground truth based on truth_assignment, NOT computed from indicators
- This tests whether models understand indicator interpretation correctly
- Indicator values loaded from pre-computed `systems/indicators/` files

Key Design Principle
--------------------
**Important:** Ground truth is the system's actual chaotic/non-chaotic status
from `truth_assignment`, NOT a simple threshold check on indicators.

This tests whether models can:
1. Interpret indicator values correctly
2. Understand that indicators are imperfect measurements
3. Reason about relationship between measurements and properties

For example:
- A chaotic system might have K=0.3 (below threshold but still chaotic)
- A non-chaotic system might have PE=0.6 (above threshold but still regular)

Models must learn that indicators are evidence, not definitions.

Confidence and Reliability
---------------------------
Indicators have varying reliability:
- **MEGNO**: Most reliable (95.5% accuracy) when computable
- **Permutation Entropy**: Moderate reliability (70% accuracy)
- **Zero-One K**: Limited reliability (63.3% accuracy)

Models should ideally learn these reliability differences from the data.

See Also
--------
- `chaosbench.data.indicators` : Indicator computation modules
- `scripts/compute_indicator_thresholds.py` : Threshold validation
- `tests/test_indicators.py` : Comprehensive test suite
- `docs/INDICATOR_COMPUTATION.md` : Indicator algorithms documentation
- `docs/INDICATOR_THRESHOLDS.md` : Threshold analysis
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from chaosbench.data.schemas import Question


# Empirically validated thresholds (computed via scripts/compute_indicator_thresholds.py)
# Based on analysis of 30 benchmark systems (19 chaotic, 11 non-chaotic)
INDICATOR_RANGES = {
    # Zero-One K: Theoretical threshold 0.5, but shows poor discrimination (63% accuracy)
    # on this dataset. Keeping theoretical threshold but note limited utility.
    "zero_one_K": {"min": 0.0, "max": 1.0, "chaotic_threshold": 0.5},

    # Permutation Entropy: Empirical optimal ~0.40 (70% accuracy)
    # Lower than theoretical ~0.7, updated to reflect empirical performance
    "permutation_entropy": {"min": 0.0, "max": 1.0, "chaotic_threshold": 0.40, "regular_threshold": 0.3},

    # MEGNO: Empirical optimal ~0.55 (95.5% accuracy)
    # Note: Theoretical MEGNO ~2.0 for regular motion, but empirical data shows
    # significant numerical issues. Using empirical threshold for better discrimination.
    "megno": {"min": -30.0, "max": 50.0, "chaotic_threshold": 0.55, "regular_value": 2.0},
}


def _is_chaotic(truth_assignment: Dict[str, bool]) -> bool:
    """Check whether a system is chaotic from its truth assignment.

    Args:
        truth_assignment: Dict mapping predicate names to boolean values.

    Returns:
        True if the system is chaotic.
    """
    return truth_assignment.get("Chaotic", False)


def _megno_indicates_regular(megno_val: float) -> bool:
    """Check whether a MEGNO value indicates a regular regime.

    Args:
        megno_val: The MEGNO indicator value.

    Returns:
        True if the value is close to 2 (regular regime).
    """
    return abs(megno_val - 2.0) < 0.5


def _both_indicators_suggest_chaotic(
    zero_one_K: float,
    permutation_entropy: float,
) -> bool:
    """Check whether both K and PE suggest chaotic dynamics.

    Args:
        zero_one_K: The 0-1 test K value.
        permutation_entropy: The permutation entropy value.

    Returns:
        True if both indicators point toward chaotic behavior.
    """
    k_chaotic = zero_one_K > INDICATOR_RANGES["zero_one_K"]["chaotic_threshold"]
    pe_chaotic = permutation_entropy > INDICATOR_RANGES["permutation_entropy"]["chaotic_threshold"]
    return k_chaotic and pe_chaotic


def _generate_direct_questions(
    system_id: str,
    name: str,
    truth_assignment: Dict[str, bool],
    ind: Dict[str, float],
    counter: List[int],
) -> List[Question]:
    """Generate direct interpretation questions for a single system.

    Args:
        system_id: System identifier.
        name: Human-readable system name.
        truth_assignment: Ground truth predicate assignments.
        ind: Indicator values for this system.
        counter: Mutable list with a single int for item numbering.

    Returns:
        List of direct interpretation Question objects.
    """
    chaotic = _is_chaotic(truth_assignment)
    questions: List[Question] = []

    if "zero_one_K" in ind:
        k_val = ind["zero_one_K"]
        counter[0] += 1
        questions.append(Question(
            item_id=f"ind_direct_{counter[0]:04d}",
            question_text=(
                f"The 0-1 test gives K={k_val:.2f} for {name}. "
                f"Is this system chaotic?"
            ),
            system_id=system_id,
            task_family="indicator_diagnostic",
            ground_truth="YES" if chaotic else "NO",
            predicates=["Chaotic"],
            metadata={
                "question_type": "direct",
                "indicator": "zero_one_K",
                "indicator_value": k_val,
            },
        ))

    if "permutation_entropy" in ind:
        pe_val = ind["permutation_entropy"]
        counter[0] += 1
        questions.append(Question(
            item_id=f"ind_direct_{counter[0]:04d}",
            question_text=(
                f"The permutation entropy of {name} is {pe_val:.2f}. "
                f"Does this indicate chaotic dynamics?"
            ),
            system_id=system_id,
            task_family="indicator_diagnostic",
            ground_truth="YES" if chaotic else "NO",
            predicates=["Chaotic"],
            metadata={
                "question_type": "direct",
                "indicator": "permutation_entropy",
                "indicator_value": pe_val,
            },
        ))

    if "megno" in ind and ind["megno"] is not None:
        megno_val = ind["megno"]
        regular = not chaotic
        counter[0] += 1
        questions.append(Question(
            item_id=f"ind_direct_{counter[0]:04d}",
            question_text=(
                f"The MEGNO value for {name} is {megno_val:.2f}. "
                f"Is this system in a regular regime?"
            ),
            system_id=system_id,
            task_family="indicator_diagnostic",
            ground_truth="YES" if regular else "NO",
            predicates=["Chaotic"],
            metadata={
                "question_type": "direct",
                "indicator": "megno",
                "indicator_value": megno_val,
            },
        ))

    return questions


def _generate_comparative_questions(
    systems: Dict[str, Dict],
    indicators: Dict[str, Dict],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate comparative questions between pairs of systems.

    Only generates cross-regime pairs (one chaotic, one non-chaotic) and
    produces BOTH orderings of each pair so the TRUE/FALSE balance is ~50/50.
    Same-regime pairs are excluded because the answer depends on indicator
    thresholds that vary, making reliable ground truth ambiguous.

    Args:
        systems: Dict mapping system_id to system info with truth_assignment.
        indicators: Dict mapping system_id to indicator values.
        rng: Seeded random number generator.
        counter: Mutable list with a single int for item numbering.

    Returns:
        List of comparative Question objects (~50% YES, ~50% NO).
    """
    questions: List[Question] = []
    system_ids = sorted(indicators.keys())

    # Collect all cross-regime pairs (chaotic vs non-chaotic, both orderings)
    ordered_pairs: List[Tuple[str, str]] = []
    for sid_a in system_ids:
        if sid_a not in systems or "zero_one_K" not in indicators[sid_a]:
            continue
        chaotic_a = _is_chaotic(systems[sid_a].get("truth_assignment", {}))
        for sid_b in system_ids:
            if sid_b == sid_a:
                continue
            if sid_b not in systems or "zero_one_K" not in indicators[sid_b]:
                continue
            chaotic_b = _is_chaotic(systems[sid_b].get("truth_assignment", {}))
            # Only cross-regime pairs: one chaotic, one non-chaotic
            if chaotic_a and not chaotic_b:
                ordered_pairs.append((sid_a, sid_b))  # YES
            elif not chaotic_a and chaotic_b:
                ordered_pairs.append((sid_a, sid_b))  # NO

    rng.shuffle(ordered_pairs)

    for sid_a, sid_b in ordered_pairs:
        truth_a = systems[sid_a].get("truth_assignment", {})
        truth_b = systems[sid_b].get("truth_assignment", {})
        k_a = indicators[sid_a]["zero_one_K"]
        k_b = indicators[sid_b]["zero_one_K"]
        chaotic_a = _is_chaotic(truth_a)

        name_a = systems[sid_a].get("name", sid_a)
        name_b = systems[sid_b].get("name", sid_b)

        # YES iff A is chaotic and B is not (clear directional comparison)
        answer = "YES" if chaotic_a else "NO"

        counter[0] += 1
        questions.append(Question(
            item_id=f"ind_comp_{counter[0]:04d}",
            question_text=(
                f"System {name_a} has K={k_a:.2f} and system {name_b} has "
                f"K={k_b:.2f}. Does {name_a} show more chaotic behavior "
                f"than {name_b}?"
            ),
            system_id=f"{sid_a}_vs_{sid_b}",
            task_family="indicator_diagnostic",
            ground_truth=answer,
            predicates=["Chaotic"],
            metadata={
                "question_type": "comparative",
                "indicator": "zero_one_K",
                "system_a": sid_a,
                "system_b": sid_b,
                "K_a": k_a,
                "K_b": k_b,
            },
        ))

    return questions


def _generate_multi_indicator_questions(
    systems: Dict[str, Dict],
    indicators: Dict[str, Dict],
    counter: List[int],
) -> List[Question]:
    """Generate multi-indicator questions combining K and PE.

    Args:
        systems: Dict mapping system_id to system info with truth_assignment.
        indicators: Dict mapping system_id to indicator values.
        counter: Mutable list with a single int for item numbering.

    Returns:
        List of multi-indicator Question objects.
    """
    questions: List[Question] = []

    for sid in sorted(indicators.keys()):
        ind = indicators[sid]
        if "zero_one_K" not in ind or "permutation_entropy" not in ind:
            continue
        if sid not in systems:
            continue

        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)
        k_val = ind["zero_one_K"]
        pe_val = ind["permutation_entropy"]
        chaotic = _is_chaotic(truth)
        both_suggest = _both_indicators_suggest_chaotic(k_val, pe_val)

        # Ground truth is the system's actual chaotic status (not threshold logic).
        # The question presents indicator values as evidence; the model must reason
        # about whether the evidence matches the system's actual classification.
        answer = "YES" if chaotic else "NO"

        counter[0] += 1
        questions.append(Question(
            item_id=f"ind_multi_{counter[0]:04d}",
            question_text=(
                f"System {name} has K={k_val:.2f} and PE={pe_val:.2f}. "
                f"Based on these indicators, is this system chaotic?"
            ),
            system_id=sid,
            task_family="indicator_diagnostic",
            ground_truth=answer,
            predicates=["Chaotic"],
            metadata={
                "question_type": "multi_indicator",
                "indicators": {"zero_one_K": k_val, "permutation_entropy": pe_val},
                "both_suggest_chaotic": both_suggest,
            },
        ))

    return questions


def generate_indicator_questions(
    systems: Dict[str, Dict],
    indicators: Dict[str, Dict],
    seed: int = 42,
) -> List[Question]:
    """Generate questions about chaos indicators.

    Question types:
    1. Direct interpretation: "The 0-1 test gives K=0.98 for system X.
       Is the system chaotic?"
    2. Comparative: "System A has PE=0.92, System B has PE=0.15.
       Which shows more chaotic behavior?"
    3. Multi-indicator: "System X has K=0.05 and PE=0.2. Together these
       suggest the system is periodic. Is this correct?"

    Args:
        systems: Dict mapping system_id to system info (must have
            truth_assignment).
        indicators: Dict mapping system_id to indicator values
            (zero_one_K, permutation_entropy, megno).
        seed: Random seed.

    Returns:
        List of Question objects with correct ground truth.
    """
    rng = random.Random(seed)
    counter = [0]
    questions: List[Question] = []

    for sid in sorted(indicators.keys()):
        if sid not in systems:
            continue
        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)
        ind = indicators[sid]
        questions.extend(
            _generate_direct_questions(sid, name, truth, ind, counter)
        )

    questions.extend(
        _generate_comparative_questions(systems, indicators, rng, counter)
    )

    questions.extend(
        _generate_multi_indicator_questions(systems, indicators, counter)
    )

    return questions


@dataclass
class IndicatorDiagnosticTask:
    """Task for testing model reasoning about chaos indicators.

    Attributes:
        task_family: Always "indicator_diagnostic".
        systems: Dict mapping system_id to system info including
            truth_assignment.
        indicators: Dict mapping system_id to indicator values.
        seed: Random seed.
    """

    task_family: str = "indicator_diagnostic"
    systems: Dict[str, Dict] = field(default_factory=dict)
    indicators: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42

    def generate_items(self) -> List[Question]:
        """Generate indicator diagnostic questions for all configured systems.

        Returns:
            List of Question objects covering direct, comparative, and
            multi-indicator question types.
        """
        return generate_indicator_questions(
            self.systems, self.indicators, self.seed
        )

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions against ground truth.

        Args:
            predictions: Dict mapping item_id to predicted label
                ("YES" or "NO").

        Returns:
            Dict with accuracy, correct count, total count, and
            per-type breakdowns.
        """
        items = self.generate_items()

        if not items:
            return {
                "accuracy": None,
                "correct": 0,
                "total": 0,
                "by_type": {},
            }

        type_buckets: Dict[str, List[Tuple[str, str, str]]] = {}
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

            q_type = q.metadata.get("question_type", "unknown")
            if q_type not in type_buckets:
                type_buckets[q_type] = []
            type_buckets[q_type].append((q.item_id, q.ground_truth, pred))

        by_type: Dict[str, Dict[str, Any]] = {}
        for q_type, entries in sorted(type_buckets.items()):
            type_correct = sum(1 for _, gt, p in entries if p == gt)
            by_type[q_type] = {
                "accuracy": type_correct / len(entries) if entries else None,
                "correct": type_correct,
                "total": len(entries),
            }

        return {
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "total": total,
            "by_type": by_type,
        }
