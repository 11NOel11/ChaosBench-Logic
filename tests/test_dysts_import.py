"""Tests for dysts import pipeline in ChaosBench-Logic v2."""

import json
import os

import pytest

from chaosbench.logic.axioms import check_fol_violations


class TestDystsImport:
    """Test the dysts import pipeline."""

    def test_dysts_registry_exists(self):
        """Verify chaosbench/data/systems/dysts_registry.py exists."""
        registry_path = "chaosbench/data/systems/dysts_registry.py"
        assert os.path.exists(registry_path), f"Registry file missing: {registry_path}"

    def test_import_script_exists(self):
        """Verify scripts/import_dysts_systems.py exists."""
        script_path = "scripts/import_dysts_systems.py"
        assert os.path.exists(script_path), f"Import script missing: {script_path}"

    @pytest.mark.skipif(
        not os.path.isdir("systems/dysts") or len(os.listdir("systems/dysts")) == 0,
        reason="systems/dysts directory is empty (dysts not installed or not imported)"
    )
    def test_dysts_json_schema(self):
        """Load a sample JSON from systems/dysts/, verify required fields."""
        dysts_dir = "systems/dysts"
        json_files = [f for f in os.listdir(dysts_dir) if f.endswith(".json")]

        assert len(json_files) > 0, "No JSON files found in systems/dysts/"

        # Load first JSON file
        sample_path = os.path.join(dysts_dir, json_files[0])
        with open(sample_path, "r") as f:
            data = json.load(f)

        # Verify required fields
        assert "system_id" in data, "Missing system_id field"
        assert "name" in data, "Missing name field"
        assert "category" in data, "Missing category field"
        assert "truth_assignment" in data, "Missing truth_assignment field"
        assert "provenance" in data, "Missing provenance field"

        # Verify system_id starts with dysts_
        assert data["system_id"].startswith("dysts_"), "system_id should start with dysts_"

    @pytest.mark.skipif(
        not os.path.isdir("systems/dysts") or len(os.listdir("systems/dysts")) == 0,
        reason="systems/dysts directory is empty (dysts not installed or not imported)"
    )
    def test_truth_assignment_complete(self):
        """Verify all 11 predicates present in truth_assignment."""
        dysts_dir = "systems/dysts"
        json_files = [f for f in os.listdir(dysts_dir) if f.endswith(".json")]

        assert len(json_files) > 0, "No JSON files found in systems/dysts/"

        # Expected predicates
        expected_predicates = {
            "Chaotic", "Deterministic", "PosLyap", "Sensitive", "StrangeAttr",
            "PointUnpredictable", "StatPredictable", "QuasiPeriodic",
            "Random", "FixedPointAttr", "Periodic"
        }

        # Load first JSON file
        sample_path = os.path.join(dysts_dir, json_files[0])
        with open(sample_path, "r") as f:
            data = json.load(f)

        truth = data["truth_assignment"]
        assert set(truth.keys()) == expected_predicates, "Missing or extra predicates in truth_assignment"

    @pytest.mark.skipif(
        not os.path.isdir("systems/dysts") or len(os.listdir("systems/dysts")) == 0,
        reason="systems/dysts directory is empty (dysts not installed or not imported)"
    )
    def test_provenance_fields(self):
        """Verify provenance has source='dysts' and cite='2110.05266'."""
        dysts_dir = "systems/dysts"
        json_files = [f for f in os.listdir(dysts_dir) if f.endswith(".json")]

        assert len(json_files) > 0, "No JSON files found in systems/dysts/"

        # Load first JSON file
        sample_path = os.path.join(dysts_dir, json_files[0])
        with open(sample_path, "r") as f:
            data = json.load(f)

        provenance = data["provenance"]
        assert "source" in provenance, "Missing source field in provenance"
        assert provenance["source"] == "dysts", "source should be 'dysts'"
        assert "cite" in provenance, "Missing cite field in provenance"
        assert "2110.05266" in provenance["cite"], "cite should contain arXiv ID 2110.05266"

    @pytest.mark.skipif(
        not os.path.isdir("systems/dysts") or len(os.listdir("systems/dysts")) == 0,
        reason="systems/dysts directory is empty (dysts not installed or not imported)"
    )
    def test_fol_consistency(self):
        """Load a dysts system, run check_fol_violations on its truth_assignment."""
        dysts_dir = "systems/dysts"
        json_files = [f for f in os.listdir(dysts_dir) if f.endswith(".json")]

        assert len(json_files) > 0, "No JSON files found in systems/dysts/"

        # Load first JSON file
        sample_path = os.path.join(dysts_dir, json_files[0])
        with open(sample_path, "r") as f:
            data = json.load(f)

        truth = data["truth_assignment"]

        # Convert to YES/NO format
        predictions = {pred: "YES" if val else "NO" for pred, val in truth.items()}

        # Check for FOL violations
        violations = check_fol_violations(predictions)

        assert len(violations) == 0, f"FOL violations found: {violations}"
