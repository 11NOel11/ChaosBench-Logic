"""Tests for dataset generation determinism.

Verifies that identical config + seed yields identical output.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _compute_dir_hash(data_dir: str) -> str:
    """Compute combined hash of all JSONL files in a directory."""
    hasher = hashlib.sha256()

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()


def _run_build(config_path: str, output_dir: str, seed: int = 42) -> int:
    """Run build_v2_dataset.py with given config and output directory.

    Returns:
        Exit code of the build process.
    """
    cmd = [
        "python",
        "scripts/build_v2_dataset.py",
        "--config", config_path,
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--dedupe_exact",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    if result.returncode != 0:
        print(f"Build failed with exit code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

    return result.returncode


@pytest.mark.slow
def test_determinism_ci_smoke():
    """Verify that identical config+seed yields identical output on CI smoke test."""
    config_path = "configs/generation/ci_smoke.yaml"

    if not os.path.exists(config_path):
        pytest.skip(f"Config {config_path} not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build 1
        build1_dir = os.path.join(tmpdir, "build1")
        os.makedirs(build1_dir)

        exit_code = _run_build(config_path, build1_dir, seed=42)
        if exit_code != 0:
            pytest.fail(f"Build 1 failed with exit code {exit_code}")

        # Build 2
        build2_dir = os.path.join(tmpdir, "build2")
        os.makedirs(build2_dir)

        exit_code = _run_build(config_path, build2_dir, seed=42)
        if exit_code != 0:
            pytest.fail(f"Build 2 failed with exit code {exit_code}")

        # Compute hashes
        hash1 = _compute_dir_hash(build1_dir)
        hash2 = _compute_dir_hash(build2_dir)

        # Assert identical
        assert hash1 == hash2, \
            f"Determinism check failed: build1 hash {hash1[:16]} != build2 hash {hash2[:16]}"

        print(f"✓ Determinism verified: {hash1[:16]}...")


@pytest.mark.slow
def test_determinism_different_seeds():
    """Verify that different seeds produce different outputs."""
    config_path = "configs/generation/ci_smoke.yaml"

    if not os.path.exists(config_path):
        pytest.skip(f"Config {config_path} not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build with seed=42
        build1_dir = os.path.join(tmpdir, "build_seed42")
        os.makedirs(build1_dir)
        exit_code = _run_build(config_path, build1_dir, seed=42)
        if exit_code != 0:
            pytest.skip("Build with seed=42 failed")

        # Build with seed=123
        build2_dir = os.path.join(tmpdir, "build_seed123")
        os.makedirs(build2_dir)
        exit_code = _run_build(config_path, build2_dir, seed=123)
        if exit_code != 0:
            pytest.skip("Build with seed=123 failed")

        # Compute hashes
        hash1 = _compute_dir_hash(build1_dir)
        hash2 = _compute_dir_hash(build2_dir)

        # Assert different
        assert hash1 != hash2, \
            "Different seeds should produce different outputs"

        print(f"✓ Seed variation verified: {hash1[:16]} != {hash2[:16]}")


def test_question_ordering_determinism():
    """Verify that question ordering is deterministic within a batch."""
    # This is a lighter-weight test that just checks if questions are
    # consistently ordered without running full builds

    from chaosbench.tasks.atomic import generate_atomic_questions

    systems = {
        "lorenz63": {
            "system_id": "lorenz63",
            "name": "Lorenz system",
            "truth_assignment": {
                "Chaotic": True,
                "Deterministic": True,
                "PosLyap": True,
                "Sensitive": True,
                "StrangeAttr": True,
                "PointUnpredictable": True,
                "StatPredictable": True,
                "QuasiPeriodic": False,
                "Random": False,
                "FixedPointAttr": False,
                "Periodic": False,
            }
        }
    }

    # Generate twice with same seed
    questions1 = generate_atomic_questions(systems, seed=42, target_count=20)
    questions2 = generate_atomic_questions(systems, seed=42, target_count=20)

    # Check same length
    assert len(questions1) == len(questions2), "Question counts differ"

    # Check same question texts in same order
    for i, (q1, q2) in enumerate(zip(questions1, questions2)):
        assert q1.question_text == q2.question_text, \
            f"Question {i} differs: {q1.question_text!r} vs {q2.question_text!r}"
        assert q1.ground_truth == q2.ground_truth, \
            f"Ground truth {i} differs"

    print(f"✓ Question ordering determinism verified ({len(questions1)} questions)")
