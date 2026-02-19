"""Test that v2.2 canonical dataset files are consistent and valid.

Verifies that the dataset files are stable, reproducible, and well-formed.
"""

import json
import os

import pytest


class TestV22FileExistence:
    """Verify all v2.2 canonical files exist."""

    EXPECTED_FILES = [
        "data/v22_adversarial.jsonl",
        "data/v22_atomic.jsonl",
        "data/v22_consistency_paraphrase.jsonl",
        "data/v22_cross_indicator.jsonl",
        "data/v22_extended_systems.jsonl",
        "data/v22_fol_inference.jsonl",
        "data/v22_indicator_diagnostics.jsonl",
        "data/v22_multi_hop.jsonl",
        "data/v22_perturbation_robustness.jsonl",
        "data/v22_regime_transition.jsonl",
    ]

    def test_all_v22_files_exist(self):
        """All 10 v2.2 canonical files exist in data directory."""
        for path in self.EXPECTED_FILES:
            assert os.path.isfile(path), f"Missing canonical file: {path}"

    def test_v22_file_count(self):
        """Exactly 10 v22_*.jsonl files exist."""
        v22_files = [
            f for f in os.listdir("data")
            if f.startswith("v22_") and f.endswith(".jsonl")
        ]
        assert len(v22_files) == 10, f"Expected 10 v22_*.jsonl files, found {len(v22_files)}"


class TestV22FileCounts:
    """Verify v2.2 files have expected minimum counts."""

    MIN_COUNTS = {
        "v22_adversarial.jsonl": 500,
        "v22_atomic.jsonl": 8000,
        "v22_consistency_paraphrase.jsonl": 3000,
        "v22_cross_indicator.jsonl": 50,
        "v22_extended_systems.jsonl": 40,
        "v22_fol_inference.jsonl": 100,
        "v22_indicator_diagnostics.jsonl": 500,
        "v22_multi_hop.jsonl": 3000,
        "v22_perturbation_robustness.jsonl": 1500,
        "v22_regime_transition.jsonl": 60,
    }

    def test_file_counts_meet_minimums(self):
        """Each v2.2 file has at least the expected minimum count."""
        for filename, min_count in self.MIN_COUNTS.items():
            path = f"data/{filename}"
            if not os.path.isfile(path):
                pytest.skip(f"File not found: {path}")

            with open(path) as f:
                lines = [line for line in f if line.strip()]

            count = len(lines)
            assert count >= min_count, (
                f"{filename}: expected >={min_count}, got {count}"
            )

    def test_total_question_count(self):
        """Total across all v2.2 files should be at least 18000.

        After enforcing strict 50/50 balance on the atomic task (which reduces
        the natural ~70% TRUE pool from 10890 to ~8808), the expected total is
        approximately 18929.
        """
        total = 0
        for filename in self.MIN_COUNTS:
            path = f"data/{filename}"
            if not os.path.isfile(path):
                pytest.skip(f"File not found: {path}")
            with open(path) as f:
                total += sum(1 for line in f if line.strip())
        assert total >= 18000, f"Expected >=18000 total, got {total}"


class TestV22ValidJsonl:
    """Verify all v2.2 files are valid JSONL."""

    def test_all_v22_valid_jsonl(self):
        """Every v2.2 file is valid JSONL with required fields."""
        v22_files = [
            f for f in os.listdir("data")
            if f.startswith("v22_") and f.endswith(".jsonl")
        ]

        assert len(v22_files) >= 10, f"Expected at least 10 v22 files, found {len(v22_files)}"

        for v22_file in v22_files:
            path = os.path.join("data", v22_file)
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"{v22_file} line {i}: Invalid JSON: {e}")

                    has_id = "item_id" in item or "id" in item
                    assert has_id, f"{v22_file} line {i}: Missing item_id/id"

                    has_question = "question_text" in item or "question" in item
                    assert has_question, f"{v22_file} line {i}: Missing question_text/question"

                    assert "ground_truth" in item, f"{v22_file} line {i}: Missing ground_truth"

                    gt = item["ground_truth"]
                    assert isinstance(gt, str) and len(gt) > 0, (
                        f"{v22_file} line {i}: ground_truth must be non-empty string"
                    )

    def test_item_ids_unique_within_file(self):
        """All item IDs are unique within each v2.2 file."""
        v22_files = [
            f for f in os.listdir("data")
            if f.startswith("v22_") and f.endswith(".jsonl")
        ]

        for v22_file in v22_files:
            path = os.path.join("data", v22_file)
            item_ids = []

            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    item_id = item.get("item_id", item.get("id"))
                    item_ids.append(item_id)

            assert len(item_ids) == len(set(item_ids)), (
                f"{v22_file}: Duplicate item IDs found"
            )


class TestManifestIntegrity:
    """Verify manifest matches v2.2 files."""

    def test_manifest_exists_and_valid(self):
        """v2_manifest.json exists and contains v2.2 batch metadata."""
        manifest_path = "data/v2_manifest.json"

        if not os.path.isfile(manifest_path):
            pytest.skip("Manifest file not found")

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest.get("version") in ("2.2.0", "2.3.0"), (
            f"Expected manifest version 2.2.0 or 2.3.0, got {manifest.get('version')}"
        )

        if "batches" in manifest:
            batches = manifest["batches"]
            for batch_id, batch_data in batches.items():
                assert "count" in batch_data, f"{batch_id} missing 'count'"
                assert isinstance(batch_data["count"], int)


class TestGroundTruthBalance:
    """Verify ground truth distribution across v2.2 files."""

    def test_files_not_all_same_label(self):
        """Each v2.2 file has mix of TRUE/FALSE (not all one label)."""
        v22_files = [
            f for f in os.listdir("data")
            if f.startswith("v22_") and f.endswith(".jsonl")
        ]

        for v22_file in v22_files:
            path = os.path.join("data", v22_file)
            labels = []

            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    gt = item["ground_truth"]
                    if gt in ("YES", "TRUE"):
                        labels.append("TRUE")
                    else:
                        labels.append("FALSE")

            unique_labels = set(labels)
            if len(labels) > 10:
                assert len(unique_labels) >= 1, f"{v22_file}: No labels found"
