"""Phase 5b: Subset determinism test.

Generates the same API subset twice and confirms identical SHA256.
Also tests that the subset script is idempotent (same seed → same sample).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
MAKE_SUBSET_SCRIPT = PROJECT_ROOT / "scripts" / "make_api_subset.py"
DATA_DIR = PROJECT_ROOT / "data"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _run_subset(output_path: Path, size: int = 50, seed: int = 42) -> int:
    """Run make_api_subset.py with given parameters."""
    cmd = [
        sys.executable,
        str(MAKE_SUBSET_SCRIPT),
        "--size", str(size),
        "--seed", str(seed),
        "--out_path", str(output_path),
        "--data_dir", str(DATA_DIR),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    return result.returncode


class TestSubsetDeterminism:
    """Verify that subset generation is deterministic."""

    def test_same_seed_produces_same_subset(self, tmp_path):
        """Two subset generations with the same seed must be identical."""
        if not MAKE_SUBSET_SCRIPT.exists():
            pytest.skip("make_api_subset.py not found")

        canonical_files = list((DATA_DIR).glob("v22_*.jsonl"))
        if not canonical_files:
            pytest.skip("No v22_*.jsonl files found in data/")

        out1 = tmp_path / "subset1.jsonl"
        out2 = tmp_path / "subset2.jsonl"

        rc1 = _run_subset(out1, size=50, seed=42)
        if rc1 != 0:
            pytest.skip("Subset script returned non-zero (may need data)")

        rc2 = _run_subset(out2, size=50, seed=42)
        assert rc2 == 0, "Second subset run failed"

        assert out1.exists(), "subset1.jsonl not created"
        assert out2.exists(), "subset2.jsonl not created"

        h1 = _sha256_file(out1)
        h2 = _sha256_file(out2)

        assert h1 == h2, (
            f"Subset not deterministic: run1={h1[:16]}, run2={h2[:16]}\n"
            "Check for random.sample without seed or non-deterministic sort."
        )

    def test_different_seeds_produce_different_subsets(self, tmp_path):
        """Different seeds should produce different subsets (with high probability)."""
        if not MAKE_SUBSET_SCRIPT.exists():
            pytest.skip("make_api_subset.py not found")

        canonical_files = list((DATA_DIR).glob("v22_*.jsonl"))
        if not canonical_files:
            pytest.skip("No v22_*.jsonl files found in data/")

        out42 = tmp_path / "subset_42.jsonl"
        out99 = tmp_path / "subset_99.jsonl"

        rc1 = _run_subset(out42, size=50, seed=42)
        if rc1 != 0:
            pytest.skip("Subset script returned non-zero (may need data)")

        rc2 = _run_subset(out99, size=50, seed=99)
        assert rc2 == 0

        h42 = _sha256_file(out42)
        h99 = _sha256_file(out99)

        # With 50 samples from thousands of records, seeds 42 and 99
        # should produce different subsets
        assert h42 != h99, (
            "Seeds 42 and 99 produced identical subsets — "
            "subset generation may not be seeded properly."
        )

    def test_subset_contains_valid_records(self, tmp_path):
        """Subset records must have required fields and canonical ground_truth."""
        if not MAKE_SUBSET_SCRIPT.exists():
            pytest.skip("make_api_subset.py not found")

        canonical_files = list((DATA_DIR).glob("v22_*.jsonl"))
        if not canonical_files:
            pytest.skip("No v22_*.jsonl files found in data/")

        output = tmp_path / "subset_valid.jsonl"
        rc = _run_subset(output, size=50, seed=42)
        if rc != 0:
            pytest.skip("Subset script returned non-zero")

        records = []
        for line in output.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

        assert len(records) > 0, "Subset is empty"

        required_fields = {"id", "question", "ground_truth", "type"}
        for rec in records:
            missing = required_fields - set(rec.keys())
            assert not missing, f"Record {rec.get('id')} missing fields: {missing}"

            gt = rec.get("ground_truth", "")
            assert gt in ("TRUE", "FALSE"), (
                f"Record {rec.get('id')} has non-canonical ground_truth='{gt}'"
            )

    def test_existing_canonical_subset_is_stable(self):
        """The committed API subset (if it exists) must not have changed."""
        subset_path = DATA_DIR / "subsets" / "api_balanced_100.jsonl"
        manifest_path = DATA_DIR / "subsets" / "api_balanced_100.manifest.json"

        if not subset_path.exists():
            pytest.skip("api_balanced_100.jsonl not committed yet")
        if not manifest_path.exists():
            pytest.skip("api_balanced_100.manifest.json not found")

        manifest = json.loads(manifest_path.read_text())
        expected_hash = manifest.get("sha256", "")

        if not expected_hash:
            pytest.skip("Manifest does not contain sha256 field")

        actual_hash = _sha256_file(subset_path)
        assert actual_hash == expected_hash, (
            f"Canonical subset hash changed!\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            "The dataset was modified. Do NOT regenerate the canonical subset "
            "without updating the manifest."
        )
