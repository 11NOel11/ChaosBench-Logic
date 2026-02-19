"""Tests for generation statistics instrumentation."""

import json
from pathlib import Path

import pytest


def test_generation_stats_schema():
    """Verify generation_stats.json has expected schema."""
    stats_path = Path("reports/scale_diag/generation_stats.json")

    if not stats_path.exists():
        pytest.skip("generation_stats.json not found (run build first)")

    with open(stats_path) as f:
        stats = json.load(f)

    # Required top-level keys
    assert "timestamp" in stats
    assert "config_file" in stats
    assert "seed" in stats
    assert "total_requested" in stats
    assert "total_generated" in stats
    assert "total_final" in stats
    assert "total_dedupe_removed" in stats
    assert "per_family" in stats

    # Verify per_family structure
    per_family = stats["per_family"]
    assert isinstance(per_family, dict)
    assert len(per_family) > 0, "per_family should not be empty"

    # Check each family has required keys
    for family_name, family_stats in per_family.items():
        assert "requested_target" in family_stats, f"{family_name} missing requested_target"
        assert "eligible_systems" in family_stats, f"{family_name} missing eligible_systems"
        assert "generated_count" in family_stats, f"{family_name} missing generated_count"
        assert "final_count" in family_stats, f"{family_name} missing final_count"
        assert "dedupe_removed" in family_stats, f"{family_name} missing dedupe_removed"
        assert "gap_from_target" in family_stats, f"{family_name} missing gap_from_target"
        assert "achievement_pct" in family_stats, f"{family_name} missing achievement_pct"

        # Verify value types
        assert isinstance(family_stats["requested_target"], int)
        assert isinstance(family_stats["eligible_systems"], int)
        assert isinstance(family_stats["generated_count"], int)
        assert isinstance(family_stats["final_count"], int)
        assert isinstance(family_stats["dedupe_removed"], int)
        assert isinstance(family_stats["gap_from_target"], int)
        assert isinstance(family_stats["achievement_pct"], (int, float))


def test_generation_stats_consistency():
    """Verify generation stats are internally consistent."""
    stats_path = Path("reports/scale_diag/generation_stats.json")

    if not stats_path.exists():
        pytest.skip("generation_stats.json not found (run build first)")

    with open(stats_path) as f:
        stats = json.load(f)

    per_family = stats["per_family"]

    for family_name, family_stats in per_family.items():
        # final_count + dedupe_removed should equal generated_count
        expected_generated = family_stats["final_count"] + family_stats["dedupe_removed"]
        actual_generated = family_stats["generated_count"]
        assert expected_generated == actual_generated, \
            f"{family_name}: final({family_stats['final_count']}) + dedupe({family_stats['dedupe_removed']}) != generated({actual_generated})"

        # gap should be requested - final
        expected_gap = family_stats["requested_target"] - family_stats["final_count"]
        actual_gap = family_stats["gap_from_target"]
        assert expected_gap == actual_gap, \
            f"{family_name}: requested({family_stats['requested_target']}) - final({family_stats['final_count']}) != gap({actual_gap})"

        # achievement_pct should be (final / requested) * 100
        if family_stats["requested_target"] > 0:
            expected_pct = (family_stats["final_count"] / family_stats["requested_target"]) * 100
            actual_pct = family_stats["achievement_pct"]
            assert abs(expected_pct - actual_pct) < 0.2, \
                f"{family_name}: achievement_pct mismatch: {expected_pct:.1f} vs {actual_pct:.1f}"

    # Verify totals match sum of per-family
    assert stats["total_requested"] == sum(f["requested_target"] for f in per_family.values())
    assert stats["total_generated"] == sum(f["generated_count"] for f in per_family.values())
    assert stats["total_final"] == sum(f["final_count"] for f in per_family.values())
    assert stats["total_dedupe_removed"] == sum(f["dedupe_removed"] for f in per_family.values())


def test_generation_stats_determinism(tmp_path):
    """Verify per-family generation counts are deterministic for same seed/config.

    Runs the ci_smoke build twice with identical seed and compares the
    per-family question counts derived from the produced JSONL files.
    The generation_stats.json is excluded from comparison because it
    embeds a timestamp; the structural count data is what must be stable.
    """
    import subprocess
    import sys
    from collections import Counter

    config_path = "configs/generation/ci_smoke.yaml"
    if not Path(config_path).exists():
        pytest.fail(f"Config {config_path} not found")

    def _build(out_dir: Path) -> None:
        """Run one build into out_dir."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/build_v2_dataset.py",
                "--config", config_path,
                "--seed", "42",
                "--output-dir", str(out_dir),
                "--dedupe_exact",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode != 0:
            pytest.fail(
                f"Build failed (rc={result.returncode}):\n"
                f"STDOUT: {result.stdout[-2000:]}\n"
                f"STDERR: {result.stderr[-2000:]}"
            )

    def _family_counts(out_dir: Path) -> dict:
        """Return per-family question counts from JSONL files in out_dir."""
        counts: Counter = Counter()
        for jl in sorted(out_dir.glob("v22_*.jsonl")):
            with open(jl) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    counts[rec.get("type", "unknown")] += 1
        return dict(sorted(counts.items()))

    _build(tmp_path / "run1")
    _build(tmp_path / "run2")

    counts1 = _family_counts(tmp_path / "run1")
    counts2 = _family_counts(tmp_path / "run2")

    assert counts1 == counts2, (
        f"Per-family counts differ between identical builds:\n"
        f"  run1: {counts1}\n"
        f"  run2: {counts2}"
    )


def test_generation_stats_bounds():
    """Verify generation stats are within reasonable bounds."""
    stats_path = Path("reports/scale_diag/generation_stats.json")

    if not stats_path.exists():
        pytest.skip("generation_stats.json not found (run build first)")

    with open(stats_path) as f:
        stats = json.load(f)

    per_family = stats["per_family"]

    for family_name, family_stats in per_family.items():
        # All counts should be non-negative
        assert family_stats["requested_target"] >= 0
        assert family_stats["eligible_systems"] >= 0
        assert family_stats["generated_count"] >= 0
        assert family_stats["final_count"] >= 0
        assert family_stats["dedupe_removed"] >= 0

        # final_count should not exceed generated_count
        assert family_stats["final_count"] <= family_stats["generated_count"], \
            f"{family_name}: final_count > generated_count"

        # achievement_pct should be 0-100 (allowing some overshoot)
        assert 0 <= family_stats["achievement_pct"] <= 150, \
            f"{family_name}: achievement_pct out of range: {family_stats['achievement_pct']}"

        # eligible_systems should be reasonable (1-200)
        assert 0 <= family_stats["eligible_systems"] <= 200, \
            f"{family_name}: eligible_systems out of range: {family_stats['eligible_systems']}"
