"""Constraint helpers for CARE-v3 repair and diagnostics."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from chaosbench.logic.axioms import check_fol_violations
from chaosbench.repair.extraction import (
    extract_predicate,
    infer_polarity,
    label_to_predicate_truth,
)

_VALID_LABELS = {"TRUE", "FALSE"}
_CONSISTENCY_GROUP_RE = re.compile(r"^(?P<base>.+)_para_[0-9]+$")
_PERTURBATION_ID_RE = re.compile(r"^perturb_(?P<ptype>[a-z_]+)_[0-9]+$")


def family_allowed(
    task_family: Optional[str], gate_families: Optional[Iterable[str]]
) -> bool:
    """Return True when family belongs to gate set."""
    if gate_families is None:
        return True
    if task_family is None:
        return False
    return task_family in set(gate_families)


def derive_group_key(
    item_id: str,
    task_family: Optional[str],
    system_id: Optional[str],
    predicate: Optional[str],
    polarity: int,
) -> Optional[str]:
    """Derive consistency-group key for eligible group constraints."""
    family = (task_family or "").lower().strip()

    if family == "consistency_paraphrase":
        match = _CONSISTENCY_GROUP_RE.match(item_id)
        if match:
            return f"consistency_paraphrase:{match.group('base')}"
        return None

    if family in {"perturbation", "perturbation_robustness"}:
        match = _PERTURBATION_ID_RE.match(item_id)
        if not match:
            return None
        perturbation_type = match.group("ptype")
        if perturbation_type not in {"paraphrase", "distractor"}:
            return None
        if not system_id or not predicate:
            return None
        polarity_tag = "neg" if polarity < 0 else "pos"
        return (
            f"perturbation:{perturbation_type}:{system_id}:{predicate}:{polarity_tag}"
        )

    return None


def _majority_truth(votes: Dict[str, int], last_seen: str) -> str:
    yes_votes = votes.get("YES", 0)
    no_votes = votes.get("NO", 0)
    if yes_votes > no_votes:
        return "YES"
    if no_votes > yes_votes:
        return "NO"
    return last_seen


def build_truth_assignments(
    records: List[Dict],
    label_key: str,
    id_to_system: Dict[str, str],
    extractor_strategy: str,
    polarity_mode: str,
    gate_families: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Dict[str, str]]:
    """Build per-system predicate truth assignments from record labels."""
    votes: Dict[Tuple[str, str], Dict[str, object]] = defaultdict(
        lambda: {"YES": 0, "NO": 0, "last": "NO"}
    )

    for record in records:
        label = record.get(label_key)
        if label not in _VALID_LABELS:
            continue

        task_family = record.get("task_family")
        if not family_allowed(task_family, gate_families):
            continue

        item_id = record.get("id", record.get("item_id", ""))
        question = record.get("question", "")
        system_id = record.get("system_id") or id_to_system.get(item_id)
        predicate = extract_predicate(question, strategy=extractor_strategy)
        if not system_id or not predicate:
            continue

        polarity = infer_polarity(question, mode=polarity_mode)
        predicate_truth = label_to_predicate_truth(label, polarity)
        key = (system_id, predicate)
        votes[key][predicate_truth] = int(votes[key][predicate_truth]) + 1
        votes[key]["last"] = predicate_truth

    assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    for system_id, predicate in sorted(votes.keys()):
        entry = votes[(system_id, predicate)]
        chosen = _majority_truth(
            {"YES": int(entry["YES"]), "NO": int(entry["NO"])},
            str(entry["last"]),
        )
        assignments[system_id][predicate] = chosen

    return dict(assignments)


def count_axiom_violations(assignments: Dict[str, Dict[str, str]]) -> Tuple[int, float]:
    """Return (total violations, average violations per system)."""
    if not assignments:
        return 0, 0.0

    total = 0
    for system_id in sorted(assignments.keys()):
        total += len(check_fol_violations(assignments[system_id]))

    rate = total / len(assignments)
    return total, rate


def compute_group_inconsistency_rate(
    records: List[Dict],
    label_key: str,
    id_to_system: Dict[str, str],
    extractor_strategy: str,
    polarity_mode: str,
) -> float:
    """Compute fraction of consistency groups with conflicting truths."""
    groups: Dict[str, List[str]] = defaultdict(list)

    for record in records:
        label = record.get(label_key)
        if label not in _VALID_LABELS:
            continue

        item_id = record.get("id", record.get("item_id", ""))
        task_family = record.get("task_family")
        question = record.get("question", "")
        system_id = record.get("system_id") or id_to_system.get(item_id)
        predicate = extract_predicate(question, strategy=extractor_strategy)
        polarity = infer_polarity(question, mode=polarity_mode)

        group_key = derive_group_key(
            item_id=item_id,
            task_family=task_family,
            system_id=system_id,
            predicate=predicate,
            polarity=polarity,
        )
        if not group_key:
            continue

        predicate_truth = label_to_predicate_truth(label, polarity)
        groups[group_key].append(predicate_truth)

    if not groups:
        return 0.0

    inconsistent = 0
    for truths in groups.values():
        if len(truths) < 2:
            continue
        if len(set(truths)) > 1:
            inconsistent += 1

    return inconsistent / len(groups)
