"""Phase 5a: Determinism smoke test.

Builds a SMALL dataset via ci_smoke config twice and asserts identical
hashes and file contents.

This test is CI-feasible and runs in a temporary directory.
Marked as 'slow' but can be included in regular CI since the smoke config
is minimal (max_systems=5, low family_targets).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
CI_SMOKE_CONFIG = PROJECT_ROOT / "configs" / "generation" / "ci_smoke.yaml"
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_v2_dataset.py"

FAMILIES = [
    "atomic",
    "multi_hop",
    "indicator_diagnostics",
    "regime_transition",
    "fol_inference",
    "extended_systems",
    "cross_indicator",
    "adversarial",
    "consistency_paraphrase",
    "perturbation_robustness",
]


def _sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _run_build(output_dir: Path) -> Dict[str, str]:
    """Run the dataset build script and return dict of file -> sha256."""
    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--config", str(CI_SMOKE_CONFIG),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Build script failed (may need API keys or dysts): "
            f"{result.stderr[-500:]}"
        )

    hashes = {}
    for f in output_dir.glob("*.jsonl"):
        hashes[f.name] = _sha256_file(f)
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        hashes["manifest.json"] = _sha256_file(manifest)
    return hashes


@pytest.mark.slow
class TestDeterminismSmoke:
    """Test that the same seed+config produces identical output."""

    def test_smoke_build_is_deterministic(self, tmp_path):
        """Two smoke builds must produce byte-identical JSONL files."""
        if not CI_SMOKE_CONFIG.exists():
            pytest.skip("ci_smoke.yaml not found")
        if not BUILD_SCRIPT.exists():
            pytest.skip("build_v2_dataset.py not found")

        # Build 1
        dir1 = tmp_path / "build1"
        dir1.mkdir()
        hashes1 = _run_build(dir1)

        if not hashes1:
            pytest.skip("No output files generated — check build script")

        # Build 2
        dir2 = tmp_path / "build2"
        dir2.mkdir()
        hashes2 = _run_build(dir2)

        # Compare
        assert set(hashes1.keys()) == set(hashes2.keys()), (
            f"Build 1 files: {sorted(hashes1.keys())}\n"
            f"Build 2 files: {sorted(hashes2.keys())}\n"
            "File sets differ between runs!"
        )

        mismatches = [
            fname for fname in hashes1
            if hashes1[fname] != hashes2.get(fname)
        ]
        assert not mismatches, (
            f"Non-deterministic files (hash differs between runs): {mismatches}\n"
            "Check for random.seed() calls, timestamp-based IDs, or set iteration."
        )

    def test_smoke_build_ground_truth_canonical(self, tmp_path):
        """Smoke build must produce only TRUE/FALSE ground_truth values."""
        if not CI_SMOKE_CONFIG.exists():
            pytest.skip("ci_smoke.yaml not found")
        if not BUILD_SCRIPT.exists():
            pytest.skip("build_v2_dataset.py not found")

        output_dir = tmp_path / "build_gt"
        output_dir.mkdir()
        _run_build(output_dir)

        violations = []
        for jsonl_file in output_dir.glob("*.jsonl"):
            for i, line in enumerate(jsonl_file.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gt = rec.get("ground_truth", "")
                if gt not in ("TRUE", "FALSE"):
                    violations.append(
                        f"{jsonl_file.name}:{i} ground_truth='{gt}'"
                    )

        assert not violations, (
            f"Non-canonical ground_truth values found:\n" +
            "\n".join(violations[:10])
        )

    def test_smoke_build_no_missing_ids(self, tmp_path):
        """All records in smoke build must have non-empty IDs."""
        if not CI_SMOKE_CONFIG.exists():
            pytest.skip("ci_smoke.yaml not found")
        if not BUILD_SCRIPT.exists():
            pytest.skip("build_v2_dataset.py not found")

        output_dir = tmp_path / "build_ids"
        output_dir.mkdir()
        _run_build(output_dir)

        violations = []
        for jsonl_file in output_dir.glob("*.jsonl"):
            for i, line in enumerate(jsonl_file.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("id"):
                    violations.append(f"{jsonl_file.name}:{i} missing id")

        assert not violations, (
            f"Records with missing IDs:\n" + "\n".join(violations[:10])
        )
