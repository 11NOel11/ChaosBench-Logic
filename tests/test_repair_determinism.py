"""Determinism and hashing tests for CARE-v3."""

from __future__ import annotations

import json
from pathlib import Path

from chaosbench.repair import (
    RepairConfig,
    constraint_hash,
    load_selector_index,
    repair_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_records(limit: int = 220):
    path = PROJECT_ROOT / "data" / "subsets" / "api_balanced_1k.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            gt = item["ground_truth"]
            parsed = gt if len(rows) % 5 else ("FALSE" if gt == "TRUE" else "TRUE")
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": gt,
                    "parsed_label": parsed,
                    "task_family": item.get("type"),
                    "outcome": f"VALID_{parsed}",
                }
            )
            if len(rows) >= limit:
                break
    return rows


def test_constraint_hash_is_stable_and_sensitive():
    cfg_a = RepairConfig(
        name="cfg_a",
        gate_families=("multi_hop", "fol_inference"),
        extractor_strategy="tail_clause",
        polarity_mode="rule_based",
    )
    cfg_b = RepairConfig(
        name="cfg_b",
        gate_families=("multi_hop",),
        extractor_strategy="tail_clause",
        polarity_mode="rule_based",
    )

    hash_a_1 = constraint_hash(cfg_a)
    hash_a_2 = constraint_hash(cfg_a)
    hash_b = constraint_hash(cfg_b)

    assert hash_a_1 == hash_a_2
    assert hash_a_1 != hash_b


def test_repair_is_deterministic_for_fixed_input_and_config():
    id_to_meta = load_selector_index(
        selector_path=PROJECT_ROOT / "data" / "canonical_v2_files.json",
        project_root=PROJECT_ROOT,
    )
    records = _load_records(limit=220)
    config = RepairConfig(
        name="determinism",
        gate_families=("multi_hop", "fol_inference", "consistency_paraphrase"),
        extractor_strategy="tail_clause",
        polarity_mode="rule_based",
    )

    first = repair_records(records=records, id_to_meta=id_to_meta, config=config)
    second = repair_records(records=records, id_to_meta=id_to_meta, config=config)

    first_labels = [row.get("repaired_label") for row in first.records]
    second_labels = [row.get("repaired_label") for row in second.records]
    first_flags = [row.get("was_flipped") for row in first.records]
    second_flags = [row.get("was_flipped") for row in second.records]

    assert first.constraint_hash == second.constraint_hash
    assert first.stats.to_dict() == second.stats.to_dict()
    assert first_labels == second_labels
    assert first_flags == second_flags
