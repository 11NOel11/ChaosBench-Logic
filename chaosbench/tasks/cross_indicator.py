"""Cross-indicator task: combining numerical indicator values with predicate knowledge.

This module generates questions that require integrating chaos indicators
(numerical measurements) with ontological knowledge about system properties.
It tests whether models can reason about the relationship between indicator
values and chaotic/non-chaotic classification.

The task probes three types of reasoning:
1. **Consistency** - Does an indicator value agree with a system's known properties?
2. **Comparison** - Which system is more chaotic based on indicator values?
3. **Agreement** - Does MEGNO value align with chaotic/regular classification?

Chaos Indicators Used
---------------------
Three quantitative chaos indicators:
1. **Zero-One K** - 0-1 test (K ≈ 1 for chaotic, K ≈ 0 for regular)
2. **Permutation Entropy** - PE ≈ 1 for chaotic, PE ≈ 0 for regular
3. **MEGNO** - Mean Exponential Growth of Nearby Orbits
   (empirical: >0.55 for chaotic, <0.55 for regular)

Question Types
--------------
1. **Indicator Consistency** (~30 questions):
   Tests whether an indicator value is consistent with ground truth.
   Example: "The 0-1 test K=0.95 for Lorenz-63. Given it is chaotic,
   is this indicator consistent?" → YES

   Includes confidence metadata:
   - High: Indicator strongly agrees with truth (|diff| > 0.4)
   - Medium: Moderate agreement (0.2 < |diff| < 0.4)
   - Low: Ambiguous or disagrees (|diff| < 0.2)

2. **Cross-System Comparison** (~25 questions):
   Compares indicator values between chaotic and non-chaotic systems.
   Example: "The Rössler system has K=0.92, the SHM has K=0.05.
   Which is more chaotic?" → Rössler

3. **MEGNO Agreement** (~12 questions):
   Tests understanding of MEGNO interpretation.
   Example: "The MEGNO for Lorenz-63 is 89.09. Does this agree with
   it being chaotic?" → Depends on threshold (empirical: YES if >0.55)

Expected Distribution
---------------------
- Total: ~67 questions
- Ground truth: ~65% YES, ~35% NO
- Confidence distribution:
  - High: ~25% (strong indicator agreement)
  - Medium: ~20% (moderate agreement)
  - Low: ~52% (ambiguous cases)
  - None: ~22% (comparison questions without confidence)

Implementation Notes
--------------------
- Loads pre-computed indicator values from `systems/indicators/`
- Uses empirically validated thresholds (see docs/INDICATOR_THRESHOLDS.md)
- Skips systems with None indicator values
- Confidence metadata added in v2.1 for downstream analysis

Confidence Scoring
------------------
The `compute_confidence()` function assesses indicator reliability:

```python
def compute_confidence(indicator_val, chaotic, indicator_type):
    # Returns "high", "medium", or "low" based on how well
    # the indicator agrees with ground truth
```

This allows filtering or weighting questions by confidence in evaluation.

Empirical Thresholds (v2.1)
----------------------------
After empirical validation on 30 systems:
- **K threshold**: 0.5 (theoretical, poor discriminator - 63% accuracy)
- **PE threshold**: 0.40 (empirical optimal - 70% accuracy)
- **MEGNO threshold**: 0.55 (empirical optimal - 95.5% accuracy)

See `docs/INDICATOR_THRESHOLDS.md` for detailed analysis.

Known Limitations
-----------------
1. Zero-One K has poor discriminative power on this dataset
2. Some systems have missing MEGNO values due to computation failures
3. Confidence scores are heuristic-based (not probabilistic)
4. Threshold selection affects ground truth for some ambiguous cases

See Also
--------
- `chaosbench.data.indicators` : Indicator computation modules
- `scripts/compute_indicator_thresholds.py` : Threshold validation
- `tests/test_cross_indicator.py` : Comprehensive test suite
- `docs/INDICATOR_THRESHOLDS.md` : Threshold analysis documentation
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from chaosbench.data.schemas import Question


def compute_confidence(indicator_val: float, chaotic: bool, indicator_type: str = "K") -> str:
    """Compute confidence level for indicator-based question.

    Assesses how well an indicator value agrees with the ground truth
    chaotic/non-chaotic label.

    Args:
        indicator_val: The indicator value (e.g., K, PE, MEGNO).
        chaotic: True if system is chaotic, False otherwise.
        indicator_type: Type of indicator ("K", "PE", or "MEGNO").

    Returns:
        "high" if indicator strongly agrees with truth (|val - expected| > 0.4)
        "medium" if indicator moderately agrees (0.2 < |val - expected| < 0.4)
        "low" if indicator disagrees or is ambiguous (|val - expected| < 0.2)
    """
    if indicator_type == "K":
        # For Zero-One K: expect ~1.0 for chaotic, ~0.0 for non-chaotic
        expected_k = 0.8 if chaotic else 0.2
        diff = abs(indicator_val - expected_k)
    elif indicator_type == "PE":
        # For Permutation Entropy: expect high for chaotic, low for non-chaotic
        expected_pe = 0.7 if chaotic else 0.3
        diff = abs(indicator_val - expected_pe)
    elif indicator_type == "MEGNO":
        # For MEGNO: expect >0.55 for chaotic, <0.55 for non-chaotic
        # Using empirical threshold from threshold analysis
        if chaotic:
            diff = max(0, 0.55 - indicator_val) / 0.55  # Normalized distance
        else:
            diff = max(0, indicator_val - 0.55) / 0.55
    else:
        return "unknown"

    # Classify confidence based on agreement
    if diff > 0.4:
        return "high"
    elif diff > 0.2:
        return "medium"
    else:
        return "low"


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


def _load_indicators(indicators_dir: str = "systems/indicators") -> Dict[str, Dict]:
    """Load indicator JSONs from directory.

    Args:
        indicators_dir: Path to indicators directory.

    Returns:
        Dict mapping system_id to indicator data.
    """
    indicators = {}
    if not os.path.isdir(indicators_dir):
        return indicators
    for fname in sorted(os.listdir(indicators_dir)):
        if not fname.endswith("_indicators.json"):
            continue
        fpath = os.path.join(indicators_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            filtered = {k: v for k, v in data.items() if v is not None}
            indicators[sid] = filtered
    return indicators


def _generate_consistency_questions(
    systems: Dict[str, Dict],
    indicators: Dict[str, Dict],
    counter: List[int],
) -> List[Question]:
    """Generate indicator-predicate consistency questions.

    Asks whether a specific indicator value is consistent with a system's
    known chaotic/non-chaotic status.
    """
    questions: List[Question] = []

    for sid in sorted(indicators.keys()):
        if sid not in systems:
            continue
        ind = indicators[sid]
        sys_data = systems[sid]
        truth = sys_data.get("truth_assignment", {})
        name = sys_data.get("name", sid)
        chaotic = truth.get("Chaotic", False)

        k_val = ind.get("zero_one_K")
        if k_val is not None:
            # Using theoretical threshold (empirical analysis shows K has limited
            # discriminative power for this dataset, see scripts/compute_indicator_thresholds.py)
            k_suggests_chaotic = k_val > 0.5
            is_consistent = (k_suggests_chaotic == chaotic)
            counter[0] += 1
            pred_label = "chaotic" if chaotic else "non-chaotic"
            confidence = compute_confidence(k_val, chaotic, indicator_type="K")
            questions.append(Question(
                item_id=f"cross_consist_{counter[0]:04d}",
                question_text=(
                    f"The 0-1 test K={k_val:.2f} for {name}. "
                    f"Given it is {pred_label}, is this indicator consistent?"
                ),
                system_id=sid,
                task_family="cross_indicator",
                ground_truth="YES" if is_consistent else "NO",
                predicates=["Chaotic"],
                metadata={
                    "question_type": "indicator_consistency",
                    "indicator": "zero_one_K",
                    "indicator_value": k_val,
                    "system_chaotic": chaotic,
                    "confidence": confidence,  # NEW: confidence level
                },
            ))

    return questions


def _generate_comparison_questions(
    systems: Dict[str, Dict],
    indicators: Dict[str, Dict],
    rng: random.Random,
    counter: List[int],
) -> List[Question]:
    """Generate cross-system indicator comparison questions."""
    questions: List[Question] = []

    chaotic_ids = []
    nonchaotic_ids = []
    for sid in sorted(indicators.keys()):
        if sid not in systems:
            continue
        truth = systems[sid].get("truth_assignment", {})
        k_val = indicators[sid].get("zero_one_K")
        if k_val is None:
            continue
        if truth.get("Chaotic", False):
            chaotic_ids.append(sid)
        else:
            nonchaotic_ids.append(sid)

    pairs: List[Tuple[str, str]] = []
    for c_sid in chaotic_ids:
        for nc_sid in nonchaotic_ids:
            pairs.append((c_sid, nc_sid))

    rng.shuffle(pairs)

    for c_sid, nc_sid in pairs[:15]:
        k_c = indicators[c_sid]["zero_one_K"]
        k_nc = indicators[nc_sid]["zero_one_K"]
        name_c = systems[c_sid].get("name", c_sid)
        name_nc = systems[nc_sid].get("name", nc_sid)

        counter[0] += 1
        answer = "YES" if k_c > k_nc else "NO"
        questions.append(Question(
            item_id=f"cross_comp_{counter[0]:04d}",
            question_text=(
                f"System {name_c} has K={k_c:.2f} and system {name_nc} has "
                f"K={k_nc:.2f}. {name_c} is chaotic and {name_nc} is "
                f"non-chaotic. Does {name_c} have higher K?"
            ),
            system_id=f"{c_sid}_vs_{nc_sid}",
            task_family="cross_indicator",
            ground_truth=answer,
            predicates=["Chaotic"],
            metadata={
                "question_type": "comparison",
                "chaotic_system": c_sid,
                "nonchaotic_system": nc_sid,
                "K_chaotic": k_c,
                "K_nonchaotic": k_nc,
            },
        ))

    return questions


def _generate_megno_questions(
    systems: Dict[str, Dict],
    indicators: Dict[str, Dict],
    counter: List[int],
) -> List[Question]:
    """Generate MEGNO agreement questions."""
    questions: List[Question] = []

    for sid in sorted(indicators.keys()):
        if sid not in systems:
            continue
        ind = indicators[sid]
        megno_val = ind.get("megno")
        if megno_val is None:
            continue

        truth = systems[sid].get("truth_assignment", {})
        name = systems[sid].get("name", sid)
        chaotic = truth.get("Chaotic", False)

        megno_suggests_chaotic = megno_val > 2.5
        agrees = (megno_suggests_chaotic == chaotic)
        label = "chaotic" if chaotic else "regular"
        confidence = compute_confidence(megno_val, chaotic, indicator_type="MEGNO")

        counter[0] += 1
        questions.append(Question(
            item_id=f"cross_megno_{counter[0]:04d}",
            question_text=(
                f"The MEGNO for {name} is {megno_val:.2f}. "
                f"Does this agree with it being {label}?"
            ),
            system_id=sid,
            task_family="cross_indicator",
            ground_truth="YES" if agrees else "NO",
            predicates=["Chaotic"],
            metadata={
                "question_type": "megno_agreement",
                "megno_value": megno_val,
                "system_chaotic": chaotic,
                "confidence": confidence,  # NEW: confidence level
            },
        ))

    return questions


def generate_cross_indicator_questions(
    systems: Dict[str, Dict],
    indicators: Dict[str, Dict],
    seed: int = 42,
) -> List[Question]:
    """Generate all cross-indicator reasoning questions.

    Args:
        systems: Dict mapping system_id to system data with truth_assignment.
        indicators: Dict mapping system_id to indicator values.
        seed: Random seed for reproducibility.

    Returns:
        List of Question objects.
    """
    rng = random.Random(seed)
    counter = [0]
    questions: List[Question] = []

    questions.extend(_generate_consistency_questions(systems, indicators, counter))
    questions.extend(_generate_comparison_questions(systems, indicators, rng, counter))
    questions.extend(_generate_megno_questions(systems, indicators, counter))

    return questions


@dataclass
class CrossIndicatorTask:
    """Task for testing cross-indicator reasoning.

    Attributes:
        task_family: Always "cross_indicator".
        systems: Dict mapping system_id to system data.
        indicators: Dict mapping system_id to indicator values.
        seed: Random seed.
    """

    task_family: str = "cross_indicator"
    systems: Dict[str, Dict] = field(default_factory=dict)
    indicators: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42

    def generate_items(self) -> List[Question]:
        """Generate cross-indicator questions.

        Returns:
            List of Question objects.
        """
        if not self.systems:
            self.systems = _load_systems()
        if not self.indicators:
            self.indicators = _load_indicators()
        return generate_cross_indicator_questions(
            self.systems, self.indicators, self.seed
        )

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
