"""Smoke tests for CARE-v3 repair pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from chaosbench.repair import RepairConfig, repair_records
from chaosbench.repair.engine import compute_axiom_violation_rate, load_selector_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_sample_predictions(limit: int = 200):
    subset_path = PROJECT_ROOT / "data" / "subsets" / "api_balanced_1k.jsonl"
    rows = []
    with subset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gt = item["ground_truth"]
            parsed = gt
            if len(rows) % 7 == 0:
                parsed = "FALSE" if gt == "TRUE" else "TRUE"

            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": gt,
                    "parsed_label": parsed,
                    "outcome": f"VALID_{parsed}",
                    "task_family": item.get("type"),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def _valid_domain(records, key: str) -> bool:
    for row in records:
        value = row.get(key)
        if value is None:
            continue
        if value not in {"TRUE", "FALSE"}:
            return False
    return True


def test_repair_smoke_schema_and_invariants():
    id_to_meta = load_selector_index(
        selector_path=PROJECT_ROOT / "data" / "canonical_v2_files.json",
        project_root=PROJECT_ROOT,
    )
    records = _load_sample_predictions(limit=200)
    config = RepairConfig(
        name="smoke",
        gate_families=("multi_hop", "fol_inference"),
        extractor_strategy="last_mention",
        polarity_mode="rule_based",
    )

    result = repair_records(records=records, id_to_meta=id_to_meta, config=config)
    repaired = result.records

    assert len(repaired) == len(records)
    assert _valid_domain(repaired, "parsed_label")
    assert _valid_domain(repaired, "repaired_label")

    for row in repaired:
        assert "repaired_label" in row
        assert "was_flipped" in row
        assert "flip_reason" in row
        assert isinstance(row["was_flipped"], bool)

    # Coverage should stay constant because invalid items are not converted.
    pre_valid = sum(
        1 for row in repaired if row.get("parsed_label") in {"TRUE", "FALSE"}
    )
    post_valid = sum(
        1 for row in repaired if row.get("repaired_label") in {"TRUE", "FALSE"}
    )
    assert pre_valid == post_valid


def test_synthetic_violation_reduction():
    records = [
        {
            "id": "syn_chaotic",
            "question": "Is the synthetic system chaotic?",
            "ground_truth": "TRUE",
            "parsed_label": "TRUE",
            "task_family": "atomic",
            "outcome": "VALID_TRUE",
        },
        {
            "id": "syn_deterministic",
            "question": "Is the synthetic system deterministic?",
            "ground_truth": "TRUE",
            "parsed_label": "FALSE",
            "task_family": "atomic",
            "outcome": "VALID_FALSE",
        },
        {
            "id": "syn_random",
            "question": "Is the synthetic system random?",
            "ground_truth": "FALSE",
            "parsed_label": "TRUE",
            "task_family": "atomic",
            "outcome": "VALID_TRUE",
        },
    ]
    id_to_meta = {
        "syn_chaotic": {"system_id": "synthetic_system", "task_family": "atomic"},
        "syn_deterministic": {"system_id": "synthetic_system", "task_family": "atomic"},
        "syn_random": {"system_id": "synthetic_system", "task_family": "atomic"},
    }
    config = RepairConfig(
        name="synthetic",
        gate_families=("atomic",),
        extractor_strategy="last_mention",
        polarity_mode="none",
    )

    pre_count, _ = compute_axiom_violation_rate(
        records=records,
        label_key="parsed_label",
        id_to_meta=id_to_meta,
        config=config,
    )

    repaired = repair_records(records=records, id_to_meta=id_to_meta, config=config)
    post_count, _ = compute_axiom_violation_rate(
        records=repaired.records,
        label_key="repaired_label",
        id_to_meta=id_to_meta,
        config=config,
    )

    assert post_count <= pre_count
    assert pre_count > 0
