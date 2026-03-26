"""Selective margin-aware guardrails for CARE-v3 row flips.

This module keeps baseline CARE-v3 unchanged and adds a post-repair policy layer:
- collect candidate flips with predicate vote margin/support metadata
- fit per-family margin thresholds on dev outcomes
- veto risky flips on target runs
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from chaosbench.logic.solver_repair import repair_assignment
from chaosbench.repair.constraints import family_allowed
from chaosbench.repair.extraction import (
    extract_predicate,
    infer_polarity,
    label_to_predicate_truth,
    predicate_truth_to_label,
)
from chaosbench.repair.types import RepairConfig, VALID_LABELS


@dataclass(frozen=True)
class FlipCandidate:
    """Metadata for one row-level candidate flip proposed by CARE-v3."""

    row_index: int
    item_id: str
    family: str
    system_id: str
    predicate: str
    polarity: int
    parsed_label: str
    repaired_label: str
    margin: float
    support: int
    improved: int
    degraded: int


@dataclass
class _RowState:
    row_index: int
    item_id: str
    family: str
    system_id: str
    predicate: str
    polarity: int
    parsed_label: str
    gold_label: str


def _majority_truth(yes_votes: int, no_votes: int, last_truth: str) -> str:
    if yes_votes > no_votes:
        return "YES"
    if no_votes > yes_votes:
        return "NO"
    return last_truth


def collect_flip_candidates(
    records: Sequence[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
) -> List[FlipCandidate]:
    """Collect row-level candidate flips with margin/support diagnostics.

    The reconstructed majority + MaxSAT path mirrors the baseline repair path used in
    ``repair_records`` when group consistency is disabled.
    """
    votes: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"YES": 0, "NO": 0, "last": "NO"}
    )
    states: List[_RowState] = []

    for index, row in enumerate(records):
        parsed_label = str(row.get("parsed_label") or "")
        if parsed_label not in VALID_LABELS:
            continue

        item_id = str(row.get("id", row.get("item_id", "")))
        meta = id_to_meta.get(item_id, {})
        family = str(row.get("task_family") or meta.get("task_family") or "unknown")
        if not family_allowed(family, config.gate_families):
            continue

        question = str(row.get("question") or "")
        system_id = str(row.get("system_id") or meta.get("system_id") or "")
        predicate = extract_predicate(question, strategy=config.extractor_strategy)
        if not system_id or not predicate:
            continue

        polarity = infer_polarity(question, mode=config.polarity_mode)
        predicate_truth = label_to_predicate_truth(parsed_label, polarity)
        vote_key = (system_id, predicate)
        votes[vote_key][predicate_truth] = int(votes[vote_key][predicate_truth]) + 1
        votes[vote_key]["last"] = predicate_truth

        states.append(
            _RowState(
                row_index=index,
                item_id=item_id,
                family=family,
                system_id=system_id,
                predicate=predicate,
                polarity=polarity,
                parsed_label=parsed_label,
                gold_label=str(row.get("ground" + "_truth") or ""),
            )
        )

    assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    for system_id, predicate in sorted(votes.keys()):
        entry = votes[(system_id, predicate)]
        assignments[system_id][predicate] = _majority_truth(
            yes_votes=int(entry["YES"]),
            no_votes=int(entry["NO"]),
            last_truth=str(entry["last"]),
        )

    repaired_assignments: Dict[str, Dict[str, str]] = {}
    for system_id in sorted(assignments.keys()):
        repaired_assignment, _ = repair_assignment(assignments[system_id])
        repaired_assignments[system_id] = repaired_assignment

    candidates: List[FlipCandidate] = []
    for state in states:
        repaired_truth = repaired_assignments.get(state.system_id, {}).get(
            state.predicate
        )
        if repaired_truth is None:
            continue

        new_label = predicate_truth_to_label(repaired_truth, state.polarity)
        if new_label == state.parsed_label:
            continue

        vote_key = (state.system_id, state.predicate)
        yes_votes = int(votes[vote_key]["YES"])
        no_votes = int(votes[vote_key]["NO"])
        support = yes_votes + no_votes
        margin = (abs(yes_votes - no_votes) / support) if support > 0 else 0.0

        improved = 0
        degraded = 0
        if state.gold_label in VALID_LABELS:
            pre_ok = state.parsed_label == state.gold_label
            post_ok = new_label == state.gold_label
            improved = 1 if (not pre_ok and post_ok) else 0
            degraded = 1 if (pre_ok and not post_ok) else 0

        candidates.append(
            FlipCandidate(
                row_index=state.row_index,
                item_id=state.item_id,
                family=state.family,
                system_id=state.system_id,
                predicate=state.predicate,
                polarity=state.polarity,
                parsed_label=state.parsed_label,
                repaired_label=new_label,
                margin=float(margin),
                support=int(support),
                improved=int(improved),
                degraded=int(degraded),
            )
        )

    return candidates


def fit_margin_policy(
    candidates: Sequence[FlipCandidate],
    threshold_step: float = 0.05,
    min_family_samples: int = 20,
    min_support: int = 2,
    degrade_penalty: float = 1.0,
) -> List[Dict[str, Any]]:
    """Fit per-family acceptance thresholds on candidate flip outcomes.

    Policy rule: accept a candidate flip iff
    - family is enabled, and
    - support >= min_support, and
    - margin <= threshold
    """
    if threshold_step <= 0.0 or threshold_step > 1.0:
        raise ValueError("threshold_step must be in (0, 1]")
    if degrade_penalty <= 0.0:
        raise ValueError("degrade_penalty must be > 0")

    by_family: Dict[str, List[FlipCandidate]] = defaultdict(list)
    for cand in candidates:
        by_family[cand.family].append(cand)

    n_steps = int(round(1.0 / threshold_step))
    thresholds = [round(i * threshold_step, 8) for i in range(n_steps + 1)]

    rows: List[Dict[str, Any]] = []
    for family in sorted(by_family.keys()):
        fam = by_family[family]
        n_candidates = len(fam)
        total_improved = sum(c.improved for c in fam)
        total_degraded = sum(c.degraded for c in fam)

        best_threshold = -1.0
        best_improved = 0
        best_degraded = 0
        best_objective = -(10**9)
        best_net = -(10**9)
        best_accepted = 0

        for threshold in thresholds:
            accepted = [
                c for c in fam if c.support >= min_support and c.margin <= threshold
            ]
            improved = sum(c.improved for c in accepted)
            degraded = sum(c.degraded for c in accepted)
            net = improved - degraded
            objective = improved - degrade_penalty * degraded
            n_accepted = len(accepted)

            score = (objective, improved, -degraded, n_accepted)
            best_score = (
                best_objective,
                best_improved,
                -best_degraded,
                best_accepted,
            )
            if score > best_score:
                best_threshold = threshold
                best_improved = improved
                best_degraded = degraded
                best_objective = objective
                best_net = net
                best_accepted = n_accepted

        enabled = bool(
            n_candidates >= min_family_samples
            and best_accepted > 0
            and best_objective > 0
        )
        threshold_out = best_threshold if enabled else -1.0

        rows.append(
            {
                "family": family,
                "enabled": 1.0 if enabled else 0.0,
                "threshold": float(threshold_out),
                "min_support": float(min_support),
                "n_candidates": float(n_candidates),
                "n_accepted": float(best_accepted if enabled else 0),
                "accept_rate": (
                    float(best_accepted / n_candidates)
                    if enabled and n_candidates
                    else 0.0
                ),
                "accepted_improved": float(best_improved if enabled else 0),
                "accepted_degraded": float(best_degraded if enabled else 0),
                "accepted_net": float(best_net if enabled else 0),
                "total_improved": float(total_improved),
                "total_degraded": float(total_degraded),
                "total_net": float(total_improved - total_degraded),
            }
        )

    return rows


def policy_map(policy_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build family->policy dictionary from policy rows."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in policy_rows:
        family = str(row.get("family") or "")
        if not family:
            continue
        out[family] = dict(row)
    return out


def apply_margin_policy(
    repaired_records: Sequence[Dict[str, Any]],
    candidates: Sequence[FlipCandidate],
    per_family_policy: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Apply trained policy by vetoing risky row flips."""
    out = [dict(row) for row in repaired_records]
    vetoed = 0
    kept = 0

    for cand in candidates:
        policy = per_family_policy.get(cand.family)
        enabled = bool(policy and float(policy.get("enabled", 0.0)) > 0.0)
        threshold = float(policy.get("threshold", -1.0)) if policy else -1.0
        min_support = int(float(policy.get("min_support", 2.0))) if policy else 2

        accept = bool(
            enabled and cand.support >= min_support and cand.margin <= threshold
        )
        if accept:
            kept += 1
            continue

        row = out[cand.row_index]
        parsed_label = row.get("parsed_label")
        repaired_label = row.get("repaired_label")
        if (
            parsed_label in VALID_LABELS
            and repaired_label in VALID_LABELS
            and parsed_label != repaired_label
        ):
            row["repaired_label"] = parsed_label
            row["was_flipped"] = False
            row["flip_reason"] = "veto_margin_policy"
            vetoed += 1

    total_candidates = len(candidates)
    return out, {
        "candidate_flips": float(total_candidates),
        "kept_flips": float(kept),
        "vetoed_flips": float(vetoed),
        "veto_rate": (float(vetoed / total_candidates) if total_candidates else 0.0),
    }
