"""Tests for release QA gate checks in validate_v2."""

import json
from pathlib import Path

from scripts.validate_v2 import (
    check_manifest_integrity,
    check_question_contamination,
    check_unique_item_ids,
)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_manifest_integrity_passes_for_matching_counts_and_hashes(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    batch_name = "batch8_indicator_diagnostics"
    batch_path = data_dir / f"{batch_name}.jsonl"
    _write_jsonl(batch_path, [{"id": "q1", "question": "A", "ground_truth": "TRUE"}])

    import hashlib

    digest = hashlib.sha256(batch_path.read_bytes()).hexdigest()
    manifest = {
        "batches": {
            batch_name: {
                "count": 1,
                "sha256": digest,
            }
        },
        "total_new_questions": 1,
    }
    manifest_path = data_dir / "v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ok, _ = check_manifest_integrity(str(data_dir), str(manifest_path))
    assert ok


def test_unique_item_ids_fails_on_duplicate(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    _write_jsonl(
        data_dir / "batch8_indicator_diagnostics.jsonl",
        [{"id": "q1", "question": "Q1", "ground_truth": "TRUE"}],
    )
    _write_jsonl(
        data_dir / "batch9_regime_transitions.jsonl",
        [{"id": "q1", "question": "Q2", "ground_truth": "FALSE"}],
    )

    ok, msg = check_unique_item_ids(str(data_dir))
    assert not ok
    assert "duplicate id" in msg.lower()


def test_question_contamination_fails_on_duplicate_question_text(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    _write_jsonl(
        data_dir / "batch8_indicator_diagnostics.jsonl",
        [{"id": "q1", "question": "Is this chaotic?", "ground_truth": "TRUE"}],
    )
    _write_jsonl(
        data_dir / "batch9_regime_transitions.jsonl",
        [{"id": "q2", "question": "Is this chaotic?", "ground_truth": "TRUE"}],
    )

    ok, _ = check_question_contamination(str(data_dir))
    assert not ok


def test_question_contamination_respects_duplicate_threshold(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    _write_jsonl(
        data_dir / "batch8_indicator_diagnostics.jsonl",
        [
            {"id": "q1", "question": "Repeat me", "ground_truth": "TRUE"},
            {"id": "q2", "question": "Unique", "ground_truth": "TRUE"},
        ],
    )
    _write_jsonl(
        data_dir / "batch9_regime_transitions.jsonl",
        [{"id": "q3", "question": "Repeat me", "ground_truth": "FALSE"}],
    )

    ok, _ = check_question_contamination(str(data_dir), max_duplicates=1)
    assert ok
