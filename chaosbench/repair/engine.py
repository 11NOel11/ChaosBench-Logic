"""CARE-v3 deterministic repair engine."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chaosbench.logic.solver_repair import repair_assignment
from chaosbench.repair.constraints import (
    build_truth_assignments,
    compute_group_inconsistency_rate,
    count_axiom_violations,
    derive_group_key,
    family_allowed,
)
from chaosbench.repair.extraction import (
    extract_predicate,
    infer_polarity,
    label_to_predicate_truth,
    predicate_truth_to_label,
)
from chaosbench.repair.hashing import constraint_hash
from chaosbench.repair.types import (
    RepairConfig,
    RepairResult,
    RepairStats,
    VALID_LABELS,
)


@dataclass
class _RowContext:
    index: int
    item_id: str
    task_family: Optional[str]
    system_id: Optional[str]
    predicate: Optional[str]
    polarity: int
    parsed_label: Optional[str]
    eligible: bool


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL records from file path."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write records as JSONL with deterministic key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_selector_index(
    selector_path: Path,
    project_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Load canonical selector and return item-id index with metadata."""
    root = project_root or Path(__file__).resolve().parents[2]
    selector_abs = (
        selector_path if selector_path.is_absolute() else root / selector_path
    )
    selector = json.loads(selector_abs.read_text(encoding="utf-8"))

    index: Dict[str, Dict[str, Optional[str]]] = {}
    for rel_path in selector["files"]:
        data_path = rel_path if rel_path.startswith("/") else str(root / rel_path)
        with Path(data_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_id = item.get("id") or item.get("item_id")
                if not item_id:
                    continue
                index[item_id] = {
                    "system_id": item.get("system_id"),
                    "task_family": item.get("type") or item.get("task_family"),
                }

    return index


def _majority_truth(yes_votes: int, no_votes: int, last_truth: str) -> str:
    if yes_votes > no_votes:
        return "YES"
    if no_votes > yes_votes:
        return "NO"
    return last_truth


def _apply_group_consistency(
    records: List[Dict[str, Any]],
    contexts: List[_RowContext],
) -> Counter:
    """Optional group-level majority consistency pass over repaired labels."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for context in contexts:
        if context.parsed_label not in VALID_LABELS:
            continue
        group_key = derive_group_key(
            item_id=context.item_id,
            task_family=context.task_family,
            system_id=context.system_id,
            predicate=context.predicate,
            polarity=context.polarity,
        )
        if group_key:
            groups[group_key].append(context.index)

    reason_counter: Counter = Counter()
    for group_key in sorted(groups.keys()):
        indices = groups[group_key]
        if len(indices) < 2:
            continue

        truth_values: Dict[int, str] = {}
        yes_votes = 0
        no_votes = 0
        last_truth = "NO"
        for index in indices:
            repaired_label = records[index].get("repaired_label")
            if repaired_label not in VALID_LABELS:
                continue
            polarity = contexts[index].polarity
            truth = label_to_predicate_truth(repaired_label, polarity)
            truth_values[index] = truth
            if truth == "YES":
                yes_votes += 1
            else:
                no_votes += 1
            last_truth = truth

        if len(truth_values) < 2:
            continue

        majority = _majority_truth(
            yes_votes=yes_votes, no_votes=no_votes, last_truth=last_truth
        )
        for index, current_truth in truth_values.items():
            if current_truth == majority:
                continue

            new_label = predicate_truth_to_label(majority, contexts[index].polarity)
            if records[index].get("repaired_label") == new_label:
                continue

            records[index]["repaired_label"] = new_label
            records[index]["was_flipped"] = True
            prior_reason = records[index].get("flip_reason", "")
            if prior_reason:
                records[index]["flip_reason"] = f"{prior_reason}+group_consistency"
                reason_counter[f"{prior_reason}+group_consistency"] += 1
            else:
                records[index]["flip_reason"] = "group_consistency"
                reason_counter["group_consistency"] += 1

    return reason_counter


def repair_records(
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
) -> RepairResult:
    """Apply deterministic CARE-v3 repair to prediction records.

    This function does not use reference labels during repair decisions.
    """
    repaired_records: List[Dict[str, Any]] = [dict(row) for row in records]
    contexts: List[_RowContext] = []

    votes: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"YES": 0, "NO": 0, "last": "NO", "rows": []}
    )

    valid_records = 0
    eligible_records = 0
    eligible_with_predicate = 0

    for index, row in enumerate(repaired_records):
        item_id = row.get("id", row.get("item_id", ""))
        meta = id_to_meta.get(item_id, {})
        task_family = row.get("task_family") or meta.get("task_family")
        system_id = row.get("system_id") or meta.get("system_id")
        question = row.get("question", "")
        parsed_label = row.get("parsed_label")
        polarity = infer_polarity(question, mode=config.polarity_mode)
        predicate = extract_predicate(question, strategy=config.extractor_strategy)

        is_valid = parsed_label in VALID_LABELS
        is_eligible = bool(
            is_valid and family_allowed(task_family, config.gate_families)
        )

        contexts.append(
            _RowContext(
                index=index,
                item_id=item_id,
                task_family=task_family,
                system_id=system_id,
                predicate=predicate,
                polarity=polarity,
                parsed_label=parsed_label,
                eligible=is_eligible,
            )
        )

        row["repaired_label"] = (
            parsed_label if parsed_label in VALID_LABELS else parsed_label
        )
        row["was_flipped"] = False
        row["flip_reason"] = ""

        if not is_valid:
            continue

        valid_records += 1
        if is_eligible:
            eligible_records += 1

        if not is_eligible or not system_id or not predicate:
            continue

        eligible_with_predicate += 1
        predicate_truth = label_to_predicate_truth(parsed_label, polarity)
        vote_key = (system_id, predicate)
        votes[vote_key][predicate_truth] = int(votes[vote_key][predicate_truth]) + 1
        votes[vote_key]["last"] = predicate_truth
        votes[vote_key]["rows"].append(index)

    assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    for system_id, predicate in sorted(votes.keys()):
        entry = votes[(system_id, predicate)]
        chosen_truth = _majority_truth(
            yes_votes=int(entry["YES"]),
            no_votes=int(entry["NO"]),
            last_truth=str(entry["last"]),
        )
        assignments[system_id][predicate] = chosen_truth

    assignments_dict = dict(assignments)
    axiom_violations_pre, _ = count_axiom_violations(assignments_dict)

    repaired_assignments: Dict[str, Dict[str, str]] = {}
    predicate_flips = 0
    for system_id in sorted(assignments_dict.keys()):
        repaired_assignment, flips = repair_assignment(assignments_dict[system_id])
        repaired_assignments[system_id] = repaired_assignment
        predicate_flips += flips

    axiom_violations_post, _ = count_axiom_violations(repaired_assignments)

    flip_reasons: Counter = Counter()
    for context in contexts:
        if context.parsed_label not in VALID_LABELS:
            continue
        if not context.eligible:
            continue
        if not context.system_id or not context.predicate:
            continue

        repaired_truth = repaired_assignments.get(context.system_id, {}).get(
            context.predicate
        )
        if repaired_truth is None:
            continue

        new_label = predicate_truth_to_label(repaired_truth, context.polarity)
        old_label = repaired_records[context.index].get("parsed_label")
        if new_label == old_label:
            continue

        repaired_records[context.index]["repaired_label"] = new_label
        repaired_records[context.index]["was_flipped"] = True
        repaired_records[context.index]["flip_reason"] = "axiom_repair"
        flip_reasons["axiom_repair"] += 1

    if config.enable_group_consistency:
        group_flip_reasons = _apply_group_consistency(repaired_records, contexts)
        flip_reasons.update(group_flip_reasons)

    row_flips = sum(1 for row in repaired_records if row.get("was_flipped") is True)

    id_to_system = {
        item_id: meta.get("system_id") for item_id, meta in id_to_meta.items()
    }
    group_inconsistency_pre = compute_group_inconsistency_rate(
        records=repaired_records,
        label_key="parsed_label",
        id_to_system=id_to_system,
        extractor_strategy=config.extractor_strategy,
        polarity_mode=config.polarity_mode,
    )
    group_inconsistency_post = compute_group_inconsistency_rate(
        records=repaired_records,
        label_key="repaired_label",
        id_to_system=id_to_system,
        extractor_strategy=config.extractor_strategy,
        polarity_mode=config.polarity_mode,
    )

    stats = RepairStats(
        total_records=len(repaired_records),
        valid_records=valid_records,
        eligible_records=eligible_records,
        eligible_with_predicate=eligible_with_predicate,
        systems_repaired=len(assignments_dict),
        predicate_assignments=sum(len(v) for v in assignments_dict.values()),
        predicate_flips=predicate_flips,
        row_flips=row_flips,
        axiom_violations_pre=axiom_violations_pre,
        axiom_violations_post=axiom_violations_post,
        group_inconsistency_pre=group_inconsistency_pre,
        group_inconsistency_post=group_inconsistency_post,
        flip_reasons=dict(sorted(flip_reasons.items())),
    )

    return RepairResult(
        records=repaired_records,
        stats=stats,
        constraint_hash=constraint_hash(config),
        config=config,
    )


def compute_axiom_violation_rate(
    records: List[Dict[str, Any]],
    label_key: str,
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
    gate_families: Optional[Tuple[str, ...]] = None,
) -> Tuple[int, float]:
    """Compute axiom violations from labels using the configured extractor/polarity."""
    id_to_system = {
        item_id: meta.get("system_id") for item_id, meta in id_to_meta.items()
    }
    assignments = build_truth_assignments(
        records=records,
        label_key=label_key,
        id_to_system=id_to_system,
        extractor_strategy=config.extractor_strategy,
        polarity_mode=config.polarity_mode,
        gate_families=gate_families,
    )
    return count_axiom_violations(assignments)
