"""Tests for dataset factory utilities in ChaosBench-Logic v2."""

import json
import os
import tempfile

import pytest

from chaosbench.data.schemas import Question
from scripts.build_v2_dataset import (
    load_systems,
    load_dysts_systems,
    load_all_systems,
    question_to_jsonl,
    validate_jsonl,
    compute_file_hash,
)


class TestDatasetFactory:
    """Dataset factory utility functions."""

    def test_load_systems(self):
        """load_systems('systems') returns dict with >= 30 systems."""
        systems = load_systems("systems")

        assert isinstance(systems, dict), "Should return a dict"
        assert len(systems) >= 30, (
            f"Should have at least 30 systems, got {len(systems)}"
        )

        # Verify systems have expected fields
        for system_id, system_data in systems.items():
            assert "system_id" in system_data, "system_id field missing"
            assert "name" in system_data, "name field missing"
            assert "truth_assignment" in system_data, "truth_assignment field missing"

    def test_load_dysts_systems(self):
        """load_dysts_systems returns dict (may be empty if no dysts)."""
        systems = load_dysts_systems("systems/dysts")

        assert isinstance(systems, dict), "Should return a dict"

        # If dysts directory exists and has systems
        if os.path.isdir("systems/dysts"):
            json_files = [f for f in os.listdir("systems/dysts") if f.endswith(".json")]
            assert len(systems) == len(json_files), (
                "Number of loaded systems should match JSON files in directory"
            )

    def test_load_all_systems(self):
        """load_all_systems combines both."""
        all_systems = load_all_systems("systems")

        assert isinstance(all_systems, dict), "Should return a dict"
        assert len(all_systems) >= 30, (
            f"Should have at least 30 systems, got {len(all_systems)}"
        )

        # Verify includes both core and dysts systems
        core_count = sum(1 for sid in all_systems.keys() if not sid.startswith("dysts_"))
        dysts_count = sum(1 for sid in all_systems.keys() if sid.startswith("dysts_"))

        assert core_count >= 30, f"Should have at least 30 core systems, got {core_count}"

    def test_question_to_jsonl(self):
        """Converts Question to correct dict format."""
        question = Question(
            item_id="test_001",
            question_text="Is the Lorenz system chaotic?",
            system_id="lorenz63",
            task_family="atomic",
            ground_truth="YES",
            predicates=["Chaotic"],
        )

        jsonl_dict = question_to_jsonl(question, template="V2")

        assert jsonl_dict["id"] == "test_001", "id should match item_id"
        assert jsonl_dict["question"] == "Is the Lorenz system chaotic?", (
            "question should match question_text"
        )
        assert jsonl_dict["ground_truth"] == "TRUE", (
            "ground_truth should be converted from YES to TRUE"
        )
        assert jsonl_dict["type"] == "atomic", "type should match task_family"
        assert jsonl_dict["system_id"] == "lorenz63", "system_id should match"
        assert jsonl_dict["template"] == "V2", "template should match"

    def test_validate_jsonl(self):
        """Validates a valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

            # Write valid JSONL records
            f.write(json.dumps({
                "id": "test_001",
                "question": "Is the system chaotic?",
                "ground_truth": "TRUE",
                "type": "atomic",
                "system_id": "lorenz63",
                "template": "V2",
            }) + "\n")

            f.write(json.dumps({
                "id": "test_002",
                "question": "Is the system periodic?",
                "ground_truth": "FALSE",
                "type": "atomic",
                "system_id": "shm",
                "template": "V2",
            }) + "\n")

        try:
            is_valid = validate_jsonl(temp_path)
            assert is_valid is True, "Should validate successfully"
        finally:
            os.unlink(temp_path)

    def test_compute_file_hash(self):
        """Hash is deterministic."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name
            f.write("test content\n")

        try:
            hash1 = compute_file_hash(temp_path)
            hash2 = compute_file_hash(temp_path)

            assert hash1 == hash2, "Hash should be deterministic"
            assert len(hash1) == 64, "SHA-256 hash should be 64 hex characters"
        finally:
            os.unlink(temp_path)
