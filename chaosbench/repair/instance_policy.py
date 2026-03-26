"""Instance-level acceptance policy for CARE-v3 row flips."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from chaosbench.repair.selective import FlipCandidate
from chaosbench.repair.types import VALID_LABELS


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _margin_bucket(margin: float, margin_step: float) -> int:
    clipped = min(1.0, max(0.0, float(margin)))
    n_steps = int(round(1.0 / margin_step))
    bucket = int(clipped / margin_step)
    return min(bucket, n_steps)


def _support_bucket(support: int, support_cap: int) -> int:
    return min(max(0, int(support)), support_cap)


def _cell_key(family: str, margin_bucket: int, support_bucket: int) -> str:
    return f"{family}|m{margin_bucket}|s{support_bucket}"


def _candidate_utility(candidate: FlipCandidate, degrade_penalty: float) -> float:
    return float(candidate.improved) - float(degrade_penalty) * float(
        candidate.degraded
    )


def score_instance_candidate(candidate: FlipCandidate, policy: Dict[str, Any]) -> float:
    """Return predicted utility score for one candidate flip."""
    family = str(candidate.family or "unknown")
    families = policy.get("families", {})
    family_state = families.get(family)
    global_mean = _as_float(policy.get("global_mean_utility", 0.0), 0.0)
    if not isinstance(family_state, dict):
        return global_mean

    margin_step = _as_float(policy.get("margin_step", 0.05), 0.05)
    support_cap = _as_int(policy.get("support_cap", 8), 8)
    mb = _margin_bucket(candidate.margin, margin_step)
    sb = _support_bucket(candidate.support, support_cap)
    key = _cell_key(family, mb, sb)

    cell_state = policy.get("cells", {}).get(key)
    if isinstance(cell_state, dict):
        return _as_float(
            cell_state.get("smoothed_utility"),
            _as_float(family_state.get("mean_utility"), global_mean),
        )

    return _as_float(family_state.get("mean_utility"), global_mean)


def fit_instance_policy(
    candidates: Sequence[FlipCandidate],
    margin_step: float = 0.05,
    support_cap: int = 8,
    shrinkage: float = 20.0,
    min_family_samples: int = 20,
    degrade_penalty: float = 1.0,
) -> Tuple[
    Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Fit an instance-level policy from dev candidate outcomes.

    The policy estimates expected utility per candidate using family +
    (margin, support)-bucket evidence, then learns one global threshold.
    """
    if margin_step <= 0.0 or margin_step > 1.0:
        raise ValueError("margin_step must be in (0, 1]")
    if support_cap < 1:
        raise ValueError("support_cap must be >= 1")
    if shrinkage < 0.0:
        raise ValueError("shrinkage must be >= 0")
    if min_family_samples < 1:
        raise ValueError("min_family_samples must be >= 1")
    if degrade_penalty <= 0.0:
        raise ValueError("degrade_penalty must be > 0")

    family_agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "n_candidates": 0.0,
            "improved": 0.0,
            "degraded": 0.0,
            "utility_sum": 0.0,
        }
    )
    cell_agg: Dict[Tuple[str, int, int], Dict[str, float]] = defaultdict(
        lambda: {
            "n_candidates": 0.0,
            "improved": 0.0,
            "degraded": 0.0,
            "utility_sum": 0.0,
        }
    )

    total_candidates = 0.0
    total_utility = 0.0
    for candidate in candidates:
        family = str(candidate.family or "unknown")
        utility = _candidate_utility(candidate, degrade_penalty)
        mb = _margin_bucket(candidate.margin, margin_step)
        sb = _support_bucket(candidate.support, support_cap)

        fam_row = family_agg[family]
        fam_row["n_candidates"] += 1.0
        fam_row["improved"] += float(candidate.improved)
        fam_row["degraded"] += float(candidate.degraded)
        fam_row["utility_sum"] += utility

        cell_row = cell_agg[(family, mb, sb)]
        cell_row["n_candidates"] += 1.0
        cell_row["improved"] += float(candidate.improved)
        cell_row["degraded"] += float(candidate.degraded)
        cell_row["utility_sum"] += utility

        total_candidates += 1.0
        total_utility += utility

    global_mean_utility = (
        float(total_utility / total_candidates) if total_candidates > 0 else 0.0
    )

    family_rows: List[Dict[str, Any]] = []
    families: Dict[str, Dict[str, Any]] = {}
    for family in sorted(family_agg.keys()):
        agg = family_agg[family]
        n = _as_int(agg["n_candidates"], 0)
        improved = _as_int(agg["improved"], 0)
        degraded = _as_int(agg["degraded"], 0)
        neutral = max(0, n - improved - degraded)
        mean_utility = float(agg["utility_sum"] / n) if n else 0.0
        enabled = 1.0 if n >= min_family_samples and mean_utility > 0.0 else 0.0

        row = {
            "family": family,
            "enabled": float(enabled),
            "n_candidates": float(n),
            "improved": float(improved),
            "degraded": float(degraded),
            "neutral": float(neutral),
            "mean_utility": float(mean_utility),
        }
        family_rows.append(row)
        families[family] = dict(row)

    cell_rows: List[Dict[str, Any]] = []
    cells: Dict[str, Dict[str, Any]] = {}
    for family, mb, sb in sorted(cell_agg.keys()):
        agg = cell_agg[(family, mb, sb)]
        n = _as_int(agg["n_candidates"], 0)
        improved = _as_int(agg["improved"], 0)
        degraded = _as_int(agg["degraded"], 0)
        neutral = max(0, n - improved - degraded)
        mean_utility = float(agg["utility_sum"] / n) if n else 0.0

        family_mean = _as_float(families.get(family, {}).get("mean_utility"), 0.0)
        smoothed = (
            float((agg["utility_sum"] + shrinkage * family_mean) / (n + shrinkage))
            if (n + shrinkage) > 0.0
            else family_mean
        )

        row = {
            "family": family,
            "margin_bucket": float(mb),
            "support_bucket": float(sb),
            "n_candidates": float(n),
            "improved": float(improved),
            "degraded": float(degraded),
            "neutral": float(neutral),
            "mean_utility": float(mean_utility),
            "smoothed_utility": float(smoothed),
        }
        cell_rows.append(row)
        cells[_cell_key(family, mb, sb)] = dict(row)

    base_policy: Dict[str, Any] = {
        "policy_type": "instance_v1",
        "margin_step": float(margin_step),
        "support_cap": int(support_cap),
        "shrinkage": float(shrinkage),
        "min_family_samples": int(min_family_samples),
        "degrade_penalty": float(degrade_penalty),
        "global_mean_utility": float(global_mean_utility),
        "families": families,
        "cells": cells,
    }

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        family = str(candidate.family or "unknown")
        family_enabled = (
            _as_float(base_policy["families"].get(family, {}).get("enabled"), 0.0) > 0.0
        )
        score = score_instance_candidate(candidate, base_policy)
        scored.append(
            {
                "score": float(score),
                "utility": _candidate_utility(candidate, degrade_penalty),
                "improved": int(candidate.improved),
                "degraded": int(candidate.degraded),
                "enabled": bool(family_enabled),
            }
        )

    if scored:
        unique_thresholds = sorted({row["score"] for row in scored})
        reject_all_threshold = float(max(unique_thresholds) + 1.0)
        thresholds = unique_thresholds + [reject_all_threshold]
    else:
        reject_all_threshold = 1.0
        thresholds = [reject_all_threshold]

    threshold_rows: List[Dict[str, Any]] = []
    best_tuple = (-(10**9), -(10**9), -(10**9), -(10**9))
    best_threshold = reject_all_threshold
    best_objective = 0.0

    for threshold in thresholds:
        accepted = [
            row
            for row in scored
            if row["enabled"] is True and float(row["score"]) >= float(threshold)
        ]
        objective = sum(float(row["utility"]) for row in accepted)
        improved = sum(int(row["improved"]) for row in accepted)
        degraded = sum(int(row["degraded"]) for row in accepted)
        accepted_count = len(accepted)

        threshold_rows.append(
            {
                "threshold": float(threshold),
                "n_accepted": float(accepted_count),
                "accepted_improved": float(improved),
                "accepted_degraded": float(degraded),
                "accepted_objective": float(objective),
            }
        )

        score_tuple = (objective, improved, -degraded, accepted_count)
        if score_tuple > best_tuple:
            best_tuple = score_tuple
            best_threshold = float(threshold)
            best_objective = float(objective)

    if best_objective <= 0.0:
        best_threshold = reject_all_threshold

    policy = dict(base_policy)
    policy["threshold"] = float(best_threshold)
    policy["fit_summary"] = {
        "n_candidates": float(len(candidates)),
        "best_objective": float(max(0.0, best_objective)),
        "reject_all_threshold": float(reject_all_threshold),
    }

    return policy, family_rows, cell_rows, threshold_rows


def policy_family_rows(policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return stable family-level rows from a serialized policy."""
    out: List[Dict[str, Any]] = []
    families = policy.get("families", {})
    for family in sorted(families.keys()):
        entry = families.get(family, {})
        out.append(
            {
                "family": family,
                "enabled": _as_float(entry.get("enabled"), 0.0),
                "n_candidates": _as_float(entry.get("n_candidates"), 0.0),
                "improved": _as_float(entry.get("improved"), 0.0),
                "degraded": _as_float(entry.get("degraded"), 0.0),
                "neutral": _as_float(entry.get("neutral"), 0.0),
                "mean_utility": _as_float(entry.get("mean_utility"), 0.0),
            }
        )
    return out


def policy_cell_rows(policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return stable cell-level rows from a serialized policy."""
    out: List[Dict[str, Any]] = []
    cells = policy.get("cells", {})
    values = list(cells.values()) if isinstance(cells, dict) else []
    values.sort(
        key=lambda row: (
            str(row.get("family") or ""),
            _as_int(row.get("margin_bucket"), 0),
            _as_int(row.get("support_bucket"), 0),
        )
    )
    for row in values:
        out.append(
            {
                "family": str(row.get("family") or ""),
                "margin_bucket": _as_float(row.get("margin_bucket"), 0.0),
                "support_bucket": _as_float(row.get("support_bucket"), 0.0),
                "n_candidates": _as_float(row.get("n_candidates"), 0.0),
                "improved": _as_float(row.get("improved"), 0.0),
                "degraded": _as_float(row.get("degraded"), 0.0),
                "neutral": _as_float(row.get("neutral"), 0.0),
                "mean_utility": _as_float(row.get("mean_utility"), 0.0),
                "smoothed_utility": _as_float(row.get("smoothed_utility"), 0.0),
            }
        )
    return out


def apply_instance_policy(
    repaired_records: Sequence[Dict[str, Any]],
    candidates: Sequence[FlipCandidate],
    policy: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Apply instance-level policy by vetoing risky row flips."""
    out = [dict(row) for row in repaired_records]
    threshold = _as_float(policy.get("threshold"), 1.0)
    families = policy.get("families", {})

    kept = 0
    vetoed = 0
    kept_scores: List[float] = []
    vetoed_scores: List[float] = []

    for candidate in candidates:
        family = str(candidate.family or "unknown")
        family_state = families.get(family, {})
        enabled = _as_float(family_state.get("enabled"), 0.0) > 0.0
        score = score_instance_candidate(candidate, policy)
        accept = bool(enabled and score >= threshold)

        if accept:
            kept += 1
            kept_scores.append(float(score))
            continue

        row = out[candidate.row_index]
        parsed_label = row.get("parsed_label")
        repaired_label = row.get("repaired_label")
        if (
            parsed_label in VALID_LABELS
            and repaired_label in VALID_LABELS
            and parsed_label != repaired_label
        ):
            row["repaired_label"] = parsed_label
            row["was_flipped"] = False
            row["flip_reason"] = "veto_instance_policy"
            vetoed += 1
            vetoed_scores.append(float(score))

    total_candidates = len(candidates)
    kept_mean = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0
    vetoed_mean = sum(vetoed_scores) / len(vetoed_scores) if vetoed_scores else 0.0
    return out, {
        "candidate_flips": float(total_candidates),
        "kept_flips": float(kept),
        "vetoed_flips": float(vetoed),
        "veto_rate": (float(vetoed / total_candidates) if total_candidates else 0.0),
        "threshold": float(threshold),
        "mean_score_kept": float(kept_mean),
        "mean_score_vetoed": float(vetoed_mean),
    }
