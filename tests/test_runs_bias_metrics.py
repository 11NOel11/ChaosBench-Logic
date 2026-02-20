"""
tests/test_runs_bias_metrics.py — Unit tests for bias & confusion-matrix diagnostics.

Uses tiny synthetic prediction fixtures with known TP/FP/TN/FN values.
All tests run in < 1s.
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module import helper (shared with smoke tests; registers in sys.modules)
# ---------------------------------------------------------------------------

def _import_analyze_runs():
    mod_name = "_analyze_runs_bias_test_module"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    script_path = Path(__file__).parent.parent / "scripts" / "analyze_runs.py"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Prediction row builders
# ---------------------------------------------------------------------------

def _row(gt: str, pred: str, family: str = "atomic") -> dict:
    """Build a minimal prediction row with known ground truth and prediction."""
    outcome = f"VALID_{pred}" if pred in ("TRUE", "FALSE") else "INVALID"
    correct = (gt == pred)
    return {
        "id": f"{family}_{gt}_{pred}",
        "ground_truth": gt,
        "pred_text": pred,
        "parsed_label": pred,
        "outcome": outcome,
        "correct": correct,
        "task_family": family,
        "latency_s": 0.1,
        "split": None,
        "retry_pred_text": None,
        "retry_outcome": None,
        "meta": {},
    }


def _make_rows(tp: int, fp: int, tn: int, fn: int, family: str = "atomic") -> list:
    """Create exactly tp TRUE+correct, fp FALSE+gt+TRUE pred, tn TRUE+gt-FALSE pred, fn rows."""
    rows = []
    rows += [_row("TRUE", "TRUE", family) for _ in range(tp)]
    rows += [_row("FALSE", "TRUE", family) for _ in range(fp)]
    rows += [_row("FALSE", "FALSE", family) for _ in range(tn)]
    rows += [_row("TRUE", "FALSE", family) for _ in range(fn)]
    return rows


# ---------------------------------------------------------------------------
# Tests for _compute_mcc
# ---------------------------------------------------------------------------

class TestComputeMCC:
    def test_perfect_classifier(self):
        mod = _import_analyze_runs()
        # TP=10, FP=0, TN=10, FN=0 → MCC=1.0
        assert mod._compute_mcc(10, 0, 10, 0) == pytest.approx(1.0, abs=1e-6)

    def test_all_wrong(self):
        mod = _import_analyze_runs()
        # TP=0, FP=10, TN=0, FN=10 → MCC=-1.0
        assert mod._compute_mcc(0, 10, 0, 10) == pytest.approx(-1.0, abs=1e-6)

    def test_random_classifier_balanced(self):
        mod = _import_analyze_runs()
        # TP=5, FP=5, TN=5, FN=5 → MCC=0
        assert mod._compute_mcc(5, 5, 5, 5) == pytest.approx(0.0, abs=1e-6)

    def test_false_defaulter(self):
        mod = _import_analyze_runs()
        # Always predicts FALSE: TP=0, FP=0, TN=10, FN=10
        # MCC = (0*10 - 0*10) / sqrt(0*(0+10)*(10+0)*(10+10)) = 0 / 0 → 0
        assert mod._compute_mcc(0, 0, 10, 10) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        mod = _import_analyze_runs()
        # TP=6, FP=1, TN=3, FN=2 → known MCC value
        # MCC = (6*3 - 1*2) / sqrt((6+1)*(6+2)*(3+1)*(3+2))
        #     = (18-2) / sqrt(7*8*4*5) = 16 / sqrt(1120) ≈ 0.4781
        result = mod._compute_mcc(6, 1, 3, 2)
        assert result == pytest.approx(0.4781, abs=0.001)


# ---------------------------------------------------------------------------
# Tests for _compute_confusion
# ---------------------------------------------------------------------------

class TestComputeConfusion:
    def test_perfect_classifier(self):
        mod = _import_analyze_runs()
        rows = _make_rows(tp=10, fp=0, tn=10, fn=0)
        c = mod._compute_confusion(rows)
        assert c["tp"] == 10
        assert c["fp"] == 0
        assert c["tn"] == 10
        assert c["fn"] == 0
        assert c["tpr"] == pytest.approx(1.0, abs=1e-4)
        assert c["tnr"] == pytest.approx(1.0, abs=1e-4)
        assert c["balanced_accuracy"] == pytest.approx(1.0, abs=1e-4)
        assert c["mcc"] == pytest.approx(1.0, abs=1e-4)

    def test_false_defaulter(self):
        mod = _import_analyze_runs()
        # Model always predicts FALSE: TP=0, FP=0, TN=10, FN=10
        rows = _make_rows(tp=0, fp=0, tn=10, fn=10)
        c = mod._compute_confusion(rows)
        assert c["tp"] == 0
        assert c["fn"] == 10
        assert c["tpr"] == pytest.approx(0.0, abs=1e-4)
        assert c["tnr"] == pytest.approx(1.0, abs=1e-4)
        assert c["balanced_accuracy"] == pytest.approx(0.5, abs=1e-4)
        assert c["mcc"] == pytest.approx(0.0, abs=1e-4)

    def test_true_defaulter(self):
        mod = _import_analyze_runs()
        # Model always predicts TRUE: TP=10, FP=10, TN=0, FN=0
        rows = _make_rows(tp=10, fp=10, tn=0, fn=0)
        c = mod._compute_confusion(rows)
        assert c["tpr"] == pytest.approx(1.0, abs=1e-4)
        assert c["tnr"] == pytest.approx(0.0, abs=1e-4)
        assert c["balanced_accuracy"] == pytest.approx(0.5, abs=1e-4)

    def test_bias_score_symmetric(self):
        mod = _import_analyze_runs()
        # GT is 50/50 but model predicts mostly TRUE
        rows = _make_rows(tp=9, fp=8, tn=2, fn=1)
        c = mod._compute_confusion(rows)
        # pred_true_rate ≈ 17/20 = 0.85, gt_true_rate = 10/20 = 0.5 → bias ≈ 0.35
        assert c["bias_score"] > 0.15

    def test_balanced_no_bias(self):
        mod = _import_analyze_runs()
        rows = _make_rows(tp=5, fp=5, tn=5, fn=5)
        c = mod._compute_confusion(rows)
        assert c["bias_score"] < 0.15

    def test_gt_pred_rates(self):
        mod = _import_analyze_runs()
        # 8 TRUE predictions, 4 GT TRUE
        rows = _make_rows(tp=4, fp=4, tn=0, fn=0)
        c = mod._compute_confusion(rows)
        assert c["gt_true_rate"] == pytest.approx(0.5, abs=0.01)  # 4/8
        assert c["pred_true_rate"] == pytest.approx(1.0, abs=0.01)  # 8/8

    def test_empty_rows_returns_zeros(self):
        mod = _import_analyze_runs()
        c = mod._compute_confusion([])
        assert c["n_valid"] == 0
        assert c["mcc"] == 0.0


# ---------------------------------------------------------------------------
# Tests for bias verdict logic
# ---------------------------------------------------------------------------

class TestBiasVerdict:
    def test_label_biased_high_bias_score(self):
        mod = _import_analyze_runs()
        # Large bias: GT 50/50, model predicts ~85% TRUE
        rows = _make_rows(tp=9, fp=8, tn=2, fn=1)
        c = mod._compute_confusion(rows)
        assert c["bias_verdict"] == "LABEL-BIASED"
        assert c["label_biased"] is True

    def test_label_biased_low_tpr(self):
        mod = _import_analyze_runs()
        # Low TPR: model almost always predicts FALSE
        rows = _make_rows(tp=1, fp=0, tn=20, fn=19)
        c = mod._compute_confusion(rows)
        # TPR = 1/20 = 0.05 < 0.40 → BIASED
        assert c["tpr"] < 0.40
        assert c["bias_verdict"] == "LABEL-BIASED"

    def test_label_biased_low_tnr(self):
        mod = _import_analyze_runs()
        # Low TNR: model almost always predicts TRUE
        rows = _make_rows(tp=20, fp=19, tn=1, fn=0)
        c = mod._compute_confusion(rows)
        # TNR = 1/20 = 0.05 < 0.40 → BIASED
        assert c["tnr"] < 0.40
        assert c["bias_verdict"] == "LABEL-BIASED"

    def test_ok_verdict_balanced(self):
        mod = _import_analyze_runs()
        # Balanced and reasonably accurate
        rows = _make_rows(tp=8, fp=2, tn=8, fn=2)
        c = mod._compute_confusion(rows)
        assert c["bias_verdict"] == "OK"
        assert c["label_biased"] is False

    def test_dominant_label_false(self):
        mod = _import_analyze_runs()
        rows = _make_rows(tp=1, fp=0, tn=20, fn=19)
        c = mod._compute_confusion(rows)
        assert c["dominant_label"] == "FALSE"

    def test_dominant_label_true(self):
        mod = _import_analyze_runs()
        rows = _make_rows(tp=20, fp=15, tn=5, fn=0)
        c = mod._compute_confusion(rows)
        assert c["dominant_label"] == "TRUE"


# ---------------------------------------------------------------------------
# Tests for balanced_accuracy recompute
# ---------------------------------------------------------------------------

class TestBalancedAccuracy:
    def test_known_value(self):
        mod = _import_analyze_runs()
        # TPR = 6/8 = 0.75, TNR = 4/6 = 0.667
        rows = _make_rows(tp=6, fp=2, tn=4, fn=2)
        c = mod._compute_confusion(rows)
        expected = (6/8 + 4/6) / 2
        assert c["balanced_accuracy"] == pytest.approx(expected, abs=0.001)

    def test_false_default_gives_half(self):
        mod = _import_analyze_runs()
        rows = _make_rows(tp=0, fp=0, tn=50, fn=50)
        c = mod._compute_confusion(rows)
        assert c["balanced_accuracy"] == pytest.approx(0.5, abs=0.001)

    def test_perfect_gives_one(self):
        mod = _import_analyze_runs()
        rows = _make_rows(tp=50, fp=0, tn=50, fn=0)
        c = mod._compute_confusion(rows)
        assert c["balanced_accuracy"] == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Integration: RunSummary confusion field populated by analyze_run
# ---------------------------------------------------------------------------

class TestAnalyzeRunBiasIntegration:
    def test_confusion_populated(self, tmp_path):
        """End-to-end: analyze_run returns confusion dict with all expected keys."""
        import json

        mod = _import_analyze_runs()

        run_dir = tmp_path / "runs" / "test_bias_run"
        run_dir.mkdir(parents=True)

        # Build predictions with known confusion: TP=6, FP=2, TN=4, FN=2
        rows = _make_rows(tp=6, fp=2, tn=4, fn=2)
        n = len(rows)
        manifest = {
            "run_id": "test_bias_run",
            "created_utc": "2026-01-01T00:00:00+00:00",
            "provider": "mock/test",
            "prompt_version": "v1",
            "prompt_hash": "aabb",
            "dataset_global_sha256": "00" * 32,
            "canonical_selector": "data/canonical_v2_files.json",
            "total_items_evaluated": n,
            "max_items": n,
            "seed": 42,
            "retries": 1,
            "strict_parsing": True,
            "workers": 1,
            "git_commit": "abc",
            "python_version": "3.12.0",
            "metrics_summary": {
                "coverage": 1.0,
                "accuracy_valid": sum(r["correct"] for r in rows) / n,
                "effective_accuracy": sum(r["correct"] for r in rows) / n,
                "balanced_accuracy": 0.7,
                "mcc": 0.4,
            },
        }
        metrics = {
            "total": n, "valid": n, "invalid": 0,
            "correct": sum(r["correct"] for r in rows),
            "coverage": 1.0, "invalid_rate": 0.0,
            "accuracy_valid": manifest["metrics_summary"]["accuracy_valid"],
            "effective_accuracy": manifest["metrics_summary"]["accuracy_valid"],
            "balanced_accuracy": 0.7, "mcc": 0.4,
            "per_family": {
                "atomic": {
                    "total": n, "valid": n,
                    "correct": sum(r["correct"] for r in rows),
                    "coverage": 1.0,
                    "accuracy_valid": manifest["metrics_summary"]["accuracy_valid"],
                }
            },
            "per_split": {"unknown": {"total": n, "valid": n, "accuracy_valid": 0.7}},
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        (run_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )

        s = mod.analyze_run(run_dir, freeze_sha="dummy_freeze_sha")

        assert s.confusion, "confusion dict should be populated"
        assert s.confusion["tp"] == 6
        assert s.confusion["fp"] == 2
        assert s.confusion["tn"] == 4
        assert s.confusion["fn"] == 2
        assert s.confusion["tpr"] == pytest.approx(6 / 8, abs=0.001)
        assert s.confusion["tnr"] == pytest.approx(4 / 6, abs=0.001)
        assert "bias_verdict" in s.confusion
        assert s.confusion["bias_verdict"] in ("OK", "LABEL-BIASED")

    def test_family_confusion_populated(self, tmp_path):
        """Family-level FamilyMetrics objects should have tpr/tnr/bias_verdict."""
        import json

        mod = _import_analyze_runs()

        run_dir = tmp_path / "runs" / "test_family_bias"
        run_dir.mkdir(parents=True)

        rows_atomic = _make_rows(tp=4, fp=4, tn=4, fn=4, family="atomic")
        rows_fol = _make_rows(tp=1, fp=0, tn=9, fn=9, family="fol_inference")
        rows = rows_atomic + rows_fol
        n = len(rows)

        manifest = {
            "run_id": "test_family_bias",
            "created_utc": "2026-01-01T00:00:00+00:00",
            "provider": "mock/test2",
            "prompt_version": "v1",
            "prompt_hash": "ccdd",
            "dataset_global_sha256": "00" * 32,
            "canonical_selector": "data/canonical_v2_files.json",
            "total_items_evaluated": n,
            "max_items": n,
            "seed": 42, "retries": 1, "strict_parsing": True,
            "workers": 1, "git_commit": "abc", "python_version": "3.12.0",
            "metrics_summary": {
                "coverage": 1.0,
                "accuracy_valid": sum(r["correct"] for r in rows) / n,
                "effective_accuracy": 0.5, "balanced_accuracy": 0.5, "mcc": 0.0,
            },
        }
        metrics = {
            "total": n, "valid": n, "invalid": 0,
            "correct": sum(r["correct"] for r in rows),
            "coverage": 1.0, "invalid_rate": 0.0,
            "accuracy_valid": manifest["metrics_summary"]["accuracy_valid"],
            "effective_accuracy": 0.5, "balanced_accuracy": 0.5, "mcc": 0.0,
            "per_family": {
                "atomic": {"total": 16, "valid": 16,
                           "correct": sum(r["correct"] for r in rows_atomic),
                           "coverage": 1.0, "accuracy_valid": 0.5},
                "fol_inference": {"total": 19, "valid": 19,
                                  "correct": sum(r["correct"] for r in rows_fol),
                                  "coverage": 1.0, "accuracy_valid": 0.5},
            },
            "per_split": {"unknown": {"total": n, "valid": n, "accuracy_valid": 0.5}},
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        (run_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )

        s = mod.analyze_run(run_dir, freeze_sha="dummy_freeze_sha")

        fam_dict = {f.family: f for f in s.per_family}
        assert "atomic" in fam_dict
        assert "fol_inference" in fam_dict

        # atomic is balanced → should be OK
        atomic = fam_dict["atomic"]
        assert atomic.tpr is not None
        assert atomic.tnr is not None
        assert atomic.bias_verdict in ("OK", "LABEL-BIASED")

        # fol_inference has low TPR (1/10) → LABEL-BIASED
        fol = fam_dict["fol_inference"]
        assert fol.tpr is not None
        assert fol.tpr < 0.40
        assert fol.bias_verdict == "LABEL-BIASED"
