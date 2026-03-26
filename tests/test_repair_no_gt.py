"""Guardrails ensuring repair logic is independent of reference labels."""

from __future__ import annotations

from pathlib import Path

from chaosbench.repair import RepairConfig, repair_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GuardedRecord(dict):
    """Dictionary that fails if repair code reads ground-truth keys."""

    def get(self, key, default=None):
        if key == "ground_truth":
            raise AssertionError("repair path accessed ground_truth")
        return super().get(key, default)


def test_repair_does_not_read_ground_truth_key():
    records = [
        GuardedRecord(
            {
                "id": "guard_1",
                "question": "Is the guarded system chaotic?",
                "parsed_label": "TRUE",
                "ground_truth": "FALSE",
                "outcome": "VALID_TRUE",
                "task_family": "multi_hop",
            }
        ),
        GuardedRecord(
            {
                "id": "guard_2",
                "question": "Is the guarded system deterministic?",
                "parsed_label": "FALSE",
                "ground_truth": "TRUE",
                "outcome": "VALID_FALSE",
                "task_family": "fol_inference",
            }
        ),
    ]
    id_to_meta = {
        "guard_1": {"system_id": "guarded", "task_family": "multi_hop"},
        "guard_2": {"system_id": "guarded", "task_family": "fol_inference"},
    }
    config = RepairConfig(
        name="no_gt_guard",
        gate_families=("multi_hop", "fol_inference"),
        extractor_strategy="last_mention",
        polarity_mode="rule_based",
    )

    result = repair_records(records=records, id_to_meta=id_to_meta, config=config)
    assert len(result.records) == 2


def test_repair_package_source_contains_no_ground_truth_token():
    repair_dir = PROJECT_ROOT / "chaosbench" / "repair"
    offenders = []
    for path in sorted(repair_dir.glob("*.py")):
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "ground_truth" in line:
                offenders.append((path.name, line_no, line.strip()))

    assert not offenders, f"ground_truth token found in repair package: {offenders}"
