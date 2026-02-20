"""
tests/test_runs_audit_smoke.py — Fast regression tests for analyze_runs.py.

Uses a small fixture run built in-memory (no disk dependency on real run data)
to ensure the analysis pipeline never silently breaks.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers to build fixture runs
# ---------------------------------------------------------------------------

_FAKE_MANIFEST = {
    "run_id": "TEST_smoke_run",
    "created_utc": "2026-01-01T00:00:00+00:00",
    "provider": "mock/smoke-model",
    "prompt_version": "v1",
    "prompt_hash": "aaaa1234bbbb5678",
    "dataset_global_sha256": "00000000000000000000000000000000000000000000000000000000000000ff",
    "canonical_selector": "data/canonical_v2_files.json",
    "total_items_evaluated": 10,
    "max_items": 10,
    "seed": 42,
    "retries": 1,
    "strict_parsing": True,
    "workers": 1,
    "git_commit": "deadbeef",
    "python_version": "3.12.0",
    "metrics_summary": {
        "coverage": 1.0,
        "accuracy_valid": 0.7,
        "effective_accuracy": 0.7,
        "balanced_accuracy": 0.68,
        "mcc": 0.36,
    },
}

_FAKE_METRICS = {
    "total": 10,
    "valid": 10,
    "invalid": 0,
    "correct": 7,
    "coverage": 1.0,
    "invalid_rate": 0.0,
    "accuracy_valid": 0.7,
    "effective_accuracy": 0.7,
    "balanced_accuracy": 0.68,
    "mcc": 0.36,
    "per_family": {
        "atomic": {"total": 8, "valid": 8, "correct": 6, "coverage": 1.0, "accuracy_valid": 0.75},
        "fol_inference": {"total": 2, "valid": 2, "correct": 1, "coverage": 1.0, "accuracy_valid": 0.5},
    },
    "per_split": {"unknown": {"total": 10, "valid": 10, "accuracy_valid": 0.7}},
}

_FAKE_PREDICTIONS = [
    {"id": f"atomic_{i}", "question": f"Q{i}", "ground_truth": "TRUE",
     "pred_text": "TRUE", "parsed_label": "TRUE",
     "outcome": "VALID_TRUE", "correct": True, "latency_s": 0.1,
     "task_family": "atomic", "split": None,
     "retry_pred_text": None, "retry_outcome": None,
     "meta": {"parse_reason": "final line token", "parse_confidence": 0.9}}
    for i in range(8)
] + [
    {"id": "fol_0", "question": "FOL Q0", "ground_truth": "FALSE",
     "pred_text": "TRUE", "parsed_label": "TRUE",
     "outcome": "VALID_TRUE", "correct": False, "latency_s": 0.15,
     "task_family": "fol_inference", "split": None,
     "retry_pred_text": None, "retry_outcome": None,
     "meta": {"parse_reason": "final line token", "parse_confidence": 0.9}},
    {"id": "fol_1", "question": "FOL Q1", "ground_truth": "TRUE",
     "pred_text": "TRUE", "parsed_label": "TRUE",
     "outcome": "VALID_TRUE", "correct": True, "latency_s": 0.12,
     "task_family": "fol_inference", "split": None,
     "retry_pred_text": None, "retry_outcome": None,
     "meta": {"parse_reason": "final line token", "parse_confidence": 0.9}},
]


def _build_fake_run_dir(tmp_path: Path) -> Path:
    """Write fixture run files into a temp directory and return path."""
    run_dir = tmp_path / "runs" / "TEST_smoke_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(_FAKE_MANIFEST))
    (run_dir / "metrics.json").write_text(json.dumps(_FAKE_METRICS))
    pred_lines = "\n".join(json.dumps(p) for p in _FAKE_PREDICTIONS)
    (run_dir / "predictions.jsonl").write_text(pred_lines + "\n")
    return run_dir


# ---------------------------------------------------------------------------
# Import helper (imports the script as a module)
# ---------------------------------------------------------------------------

def _import_analyze_runs():
    """Import scripts/analyze_runs.py as a module (cached after first load)."""
    import importlib.util
    import sys as _sys
    mod_name = "_analyze_runs_test_module"
    if mod_name in _sys.modules:
        return _sys.modules[mod_name]
    script_path = Path(__file__).parent.parent / "scripts" / "analyze_runs.py"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    _sys.modules[mod_name] = mod  # register before exec so dataclass __module__ resolves
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_perfect_accuracy(self):
        mod = _import_analyze_runs()
        lo, hi = mod.wilson_ci(10, 10)
        assert lo > 0.7
        assert hi == pytest.approx(1.0, abs=0.01)

    def test_zero_accuracy(self):
        mod = _import_analyze_runs()
        lo, hi = mod.wilson_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=0.01)
        assert hi < 0.3

    def test_midrange(self):
        mod = _import_analyze_runs()
        lo, hi = mod.wilson_ci(50, 100)
        assert lo < 0.5 < hi

    def test_zero_total_returns_zeros(self):
        mod = _import_analyze_runs()
        lo, hi = mod.wilson_ci(0, 0)
        assert lo == 0.0 and hi == 0.0


class TestCategorizeInvalid:
    def test_refusal(self):
        mod = _import_analyze_runs()
        assert mod._categorize_invalid("I cannot provide information about this") == "refusal"

    def test_empty(self):
        mod = _import_analyze_runs()
        assert mod._categorize_invalid("") == "empty_response"

    def test_hedging(self):
        mod = _import_analyze_runs()
        assert mod._categorize_invalid("It depends on the context") == "hedging_or_multi_answer"

    def test_formatting(self):
        mod = _import_analyze_runs()
        assert mod._categorize_invalid("The answer is yes, but also no, perhaps") == "formatting_failure"


class TestDiscoverRuns:
    def test_finds_fake_run(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        found = mod.discover_runs(tmp_path / "runs")
        assert run_dir in found

    def test_empty_dir(self, tmp_path):
        mod = _import_analyze_runs()
        (tmp_path / "runs").mkdir()
        found = mod.discover_runs(tmp_path / "runs")
        assert found == []


class TestAnalyzeRun:
    def test_basic_fields(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        assert s.run_id == "TEST_smoke_run"
        assert s.total_items == 10
        assert s.valid_count == 10
        assert s.invalid_count == 0
        assert s.accuracy_valid == pytest.approx(0.7, abs=0.001)
        assert s.mcc == pytest.approx(0.36, abs=0.001)

    def test_predictions_match(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        assert s.predictions_match_manifest
        assert s.predictions_line_count == 10
        assert s.duplicate_ids_count == 0

    def test_family_metrics_populated(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        families = {f.family for f in s.per_family}
        assert "atomic" in families
        assert "fol_inference" in families

    def test_low_support_flag(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        # Both families in fixture have N < 100
        for fam in s.per_family:
            if fam.family in ("atomic", "fol_inference"):
                assert fam.low_support
                assert fam.wilson_lo is not None
                assert fam.wilson_hi is not None

    def test_missing_predictions_error(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        (run_dir / "predictions.jsonl").unlink()
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        assert any("predictions.jsonl missing" in e for e in s.errors)

    def test_count_mismatch_error(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        # Add an extra line to create mismatch
        pred_path = run_dir / "predictions.jsonl"
        extra = json.dumps(_FAKE_PREDICTIONS[0].copy())
        extra_id = dict(json.loads(extra), id="atomic_extra_999")
        pred_path.write_text(pred_path.read_text() + json.dumps(extra_id) + "\n")
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        assert any("mismatch" in e.lower() for e in s.errors)

    def test_sha_mismatch_warning(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        # stored SHA won't match freeze SHA in fixture
        s = mod.analyze_run(run_dir, freeze_sha="different_freeze_sha")
        assert not s.sha_matches_freeze_method

    def test_no_hard_errors_on_valid_run(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        # Only expected errors: sha mismatch warning (not an error), everything else clean
        hard_errors = [e for e in s.errors if "predictions.jsonl" not in e and "mismatch" not in e.lower()]
        assert hard_errors == []


class TestGenerateBaselinesCsv:
    def test_csv_has_header_and_rows(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        # Tweak to look like a canonical run
        s.is_canonical_run = True
        s.provider = "mock_provider"  # not "mock"
        csv, md = mod.generate_baselines_csv([s])
        assert "model,provider" in csv
        assert "smoke-model" in csv

    def test_markdown_table_has_pipes(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        s.is_canonical_run = True
        s.provider = "mock_provider"
        _, md = mod.generate_baselines_csv([s])
        assert "|" in md


class TestGenerateFamilyCsv:
    def test_csv_has_family_rows(self, tmp_path):
        mod = _import_analyze_runs()
        run_dir = _build_fake_run_dir(tmp_path)
        s = mod.analyze_run(run_dir, freeze_sha="abc123")
        s.is_canonical_run = True
        s.provider = "mock_provider"
        csv = mod.generate_family_csv([s])
        assert "atomic" in csv
        assert "fol_inference" in csv


class TestEndToEnd:
    def test_script_runs_on_real_runs(self):
        """Invoke the script on the real runs/ directory and verify outputs exist."""
        project_root = Path(__file__).parent.parent
        runs_dir = project_root / "runs"
        if not runs_dir.exists():
            pytest.skip("runs/ directory not present")

        with tempfile.TemporaryDirectory() as tmp_out:
            result = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "analyze_runs.py"),
                    "--runs_dir", str(runs_dir),
                    "--out_dir", tmp_out,
                    "--paper_assets_dir", tmp_out,
                ],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )
            assert result.returncode == 0, f"Script failed:\n{result.stderr}"

            # Check outputs exist
            assert (Path(tmp_out) / "RUNS_AUDIT.md").exists()
            assert (Path(tmp_out) / "summary.json").exists()
            assert (Path(tmp_out) / "baselines_table.csv").exists()
            assert (Path(tmp_out) / "baselines_table.md").exists()
            assert (Path(tmp_out) / "baselines_by_family.csv").exists()

            # Check summary.json is valid JSON with expected keys
            summary = json.loads((Path(tmp_out) / "summary.json").read_text())
            assert "runs" in summary
            assert "freeze_sha" in summary
            assert len(summary["runs"]) > 0

            # Check at least one canonical run parsed correctly
            runs_data = summary["runs"]
            canonical = [r for r in runs_data if r.get("is_canonical_run")]
            assert len(canonical) > 0

            # Check no hard errors in any run
            for r in canonical:
                hard_errors = [e for e in r.get("errors", []) if "mismatch" not in e.lower() and "sha" not in e.lower()]
                assert hard_errors == [], f"Hard errors in {r['run_id']}: {hard_errors}"

    def test_audit_md_contains_key_sections(self):
        """Verify RUNS_AUDIT.md contains required sections."""
        audit_path = Path(__file__).parent.parent / "artifacts" / "runs_audit" / "RUNS_AUDIT.md"
        if not audit_path.exists():
            pytest.skip("RUNS_AUDIT.md not yet generated; run analyze_runs.py first")
        content = audit_path.read_text()
        assert "SHA" in content
        assert "Verdict" in content or "OFFICIAL" in content
        assert "Checklist" in content or "Could Have Gone Wrong" in content
        assert "Per-Family" in content or "per-family" in content.lower()
