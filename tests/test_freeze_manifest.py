"""Tests for the dataset freeze workflow.

These tests run offline - no external API calls.
They verify the canonical selector, the freeze script output, and v2_manifest.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SELECTOR_PATH = PROJECT_ROOT / "data" / "canonical_v2_files.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "v2_manifest.json"
FREEZE_DIR = PROJECT_ROOT / "artifacts" / "freeze"
FREEZE_MANIFEST = FREEZE_DIR / "v2_freeze_manifest.json"


class TestCanonicalSelector:
    def test_selector_exists(self):
        assert SELECTOR_PATH.exists(), "data/canonical_v2_files.json must exist"

    def test_selector_has_files(self):
        sel = json.loads(SELECTOR_PATH.read_text())
        assert "files" in sel
        assert len(sel["files"]) > 0

    def test_selector_files_exist(self):
        sel = json.loads(SELECTOR_PATH.read_text())
        for rel_path in sel["files"]:
            fpath = PROJECT_ROOT / rel_path
            assert fpath.exists(), f"Canonical file missing: {rel_path}"

    def test_selector_files_are_jsonl(self):
        sel = json.loads(SELECTOR_PATH.read_text())
        for rel_path in sel["files"]:
            assert rel_path.endswith(".jsonl"), f"Expected .jsonl: {rel_path}"


class TestFreezeScript:
    @pytest.fixture(scope="class")
    def freeze_manifest(self, tmp_path_factory):
        """Run the freeze script in a temp output dir and return the manifest."""
        out = tmp_path_factory.mktemp("freeze")
        result = subprocess.run(
            [sys.executable, "scripts/freeze_v2_dataset.py", "--output-dir", str(out)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"freeze script failed:\n{result.stderr}"
        manifest_file = out / "v2_freeze_manifest.json"
        assert manifest_file.exists()
        return json.loads(manifest_file.read_text())

    def test_required_fields(self, freeze_manifest):
        required = [
            "dataset_release",
            "schema_version",
            "created_utc",
            "canonical_files",
            "canonical_total_questions",
            "global_sha256",
            "tool_versions",
        ]
        for field in required:
            assert field in freeze_manifest, f"Missing field: {field}"

    def test_dataset_release(self, freeze_manifest):
        assert freeze_manifest["dataset_release"] == "v2"

    def test_schema_version(self, freeze_manifest):
        assert freeze_manifest["schema_version"] == "v2"

    def test_canonical_files_non_empty(self, freeze_manifest):
        assert len(freeze_manifest["canonical_files"]) > 0

    def test_canonical_files_have_required_fields(self, freeze_manifest):
        for rec in freeze_manifest["canonical_files"]:
            assert "path" in rec
            assert "count" in rec
            assert "sha256" in rec
            assert rec["count"] > 0
            assert len(rec["sha256"]) == 64  # SHA256 hex

    def test_canonical_total_matches_sum(self, freeze_manifest):
        summed = sum(rec["count"] for rec in freeze_manifest["canonical_files"])
        assert freeze_manifest["canonical_total_questions"] == summed

    def test_global_sha256_is_64_hex(self, freeze_manifest):
        assert len(freeze_manifest["global_sha256"]) == 64

    def test_tool_versions_present(self, freeze_manifest):
        tv = freeze_manifest["tool_versions"]
        assert "python" in tv
        assert "package" in tv

    def test_output_files_written(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("freeze2")
        subprocess.run(
            [sys.executable, "scripts/freeze_v2_dataset.py", "--output-dir", str(out)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=120,
        )
        assert (out / "v2_freeze_manifest.json").exists()
        assert (out / "v2_freeze_report.md").exists()
        assert (out / "v2_freeze_sha256.txt").exists()


class TestV2Manifest:
    def test_v2_manifest_exists(self):
        assert MANIFEST_PATH.exists(), "data/v2_manifest.json must exist"

    def test_v2_manifest_has_schema_version(self):
        m = json.loads(MANIFEST_PATH.read_text())
        assert "schema_version" in m

    def test_v2_manifest_has_dataset_release(self):
        m = json.loads(MANIFEST_PATH.read_text())
        assert "dataset_release" in m

    def test_v2_manifest_has_total_questions(self):
        m = json.loads(MANIFEST_PATH.read_text())
        # Either canonical_total_questions or total_new_questions must be present
        assert "canonical_total_questions" in m or "total_new_questions" in m
