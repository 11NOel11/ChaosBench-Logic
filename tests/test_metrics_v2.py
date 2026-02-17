"""Tests for v2 metrics and axis computations in ChaosBench-Logic v2."""

import pytest

from chaosbench.eval.metrics import (
    EvalResult,
    AxisMetricResult,
    compute_axis_metrics,
    format_axis_report,
)


class TestMetricsV2:
    """Test v2 metrics computation."""

    @pytest.fixture
    def mock_results(self):
        """Create mock EvalResult objects for testing."""
        return [
            EvalResult(
                item_id="item_001",
                batch_file="batch8_indicator_diagnostics.jsonl",
                task_family="indicator_diagnostics",
                bias_family=None,
                dialogue_id=None,
                turn_index=None,
                system_id="lorenz63",
                gold="YES",
                pred_raw="YES",
                pred_norm="YES",
                correct=True,
            ),
            EvalResult(
                item_id="item_002",
                batch_file="batch8_indicator_diagnostics.jsonl",
                task_family="indicator_diagnostics",
                bias_family=None,
                dialogue_id=None,
                turn_index=None,
                system_id="shm",
                gold="NO",
                pred_raw="NO",
                pred_norm="NO",
                correct=True,
            ),
            EvalResult(
                item_id="item_003",
                batch_file="batch9_regime_transitions.jsonl",
                task_family="regime_transition",
                bias_family=None,
                dialogue_id=None,
                turn_index=None,
                system_id="lorenz63",
                gold="YES",
                pred_raw="NO",
                pred_norm="NO",
                correct=False,
            ),
            EvalResult(
                item_id="item_004",
                batch_file="batch12_fol_inference.jsonl",
                task_family="fol_inference",
                bias_family=None,
                dialogue_id=None,
                turn_index=None,
                system_id="rossler",
                gold="YES",
                pred_raw="YES",
                pred_norm="YES",
                correct=True,
            ),
            EvalResult(
                item_id="item_005",
                batch_file="batch12_fol_inference.jsonl",
                task_family="fol_inference",
                bias_family=None,
                dialogue_id=None,
                turn_index=None,
                system_id="shm",
                gold="NO",
                pred_raw="YES",
                pred_norm="YES",
                correct=False,
            ),
        ]

    def test_compute_axis_metrics_task_family(self, mock_results):
        """Group by task_family."""
        axis_metrics = compute_axis_metrics(mock_results, axes=["task_family"])

        assert "task_family" in axis_metrics, "Should have task_family axis"

        task_family_metrics = axis_metrics["task_family"]
        assert len(task_family_metrics) > 0, "Should have at least one task family"

        # Check we have expected task families
        task_families = {m.value for m in task_family_metrics}
        assert "indicator_diagnostics" in task_families, (
            "Should have indicator_diagnostics task family"
        )
        assert "fol_inference" in task_families, (
            "Should have fol_inference task family"
        )

    def test_compute_axis_metrics_batch_file(self, mock_results):
        """Group by batch_file."""
        axis_metrics = compute_axis_metrics(mock_results, axes=["batch_file"])

        assert "batch_file" in axis_metrics, "Should have batch_file axis"

        batch_metrics = axis_metrics["batch_file"]
        assert len(batch_metrics) > 0, "Should have at least one batch file"

        # Check we have expected batch files
        batch_files = {m.value for m in batch_metrics}
        assert "batch8_indicator_diagnostics.jsonl" in batch_files, (
            "Should have batch8_indicator_diagnostics.jsonl"
        )
        assert "batch12_fol_inference.jsonl" in batch_files, (
            "Should have batch12_fol_inference.jsonl"
        )

    def test_axis_metric_result_fields(self, mock_results):
        """Verify AxisMetricResult has all fields."""
        axis_metrics = compute_axis_metrics(mock_results, axes=["task_family"])

        for metric in axis_metrics["task_family"]:
            assert isinstance(metric, AxisMetricResult), (
                "Should be AxisMetricResult instance"
            )
            assert hasattr(metric, "axis"), "Should have axis field"
            assert hasattr(metric, "value"), "Should have value field"
            assert hasattr(metric, "accuracy"), "Should have accuracy field"
            assert hasattr(metric, "n_correct"), "Should have n_correct field"
            assert hasattr(metric, "n_total"), "Should have n_total field"
            assert hasattr(metric, "fol_violation_rate"), (
                "Should have fol_violation_rate field"
            )

            # Verify field types
            assert isinstance(metric.axis, str), "axis should be string"
            assert isinstance(metric.value, str), "value should be string"
            assert isinstance(metric.accuracy, float), "accuracy should be float"
            assert isinstance(metric.n_correct, int), "n_correct should be int"
            assert isinstance(metric.n_total, int), "n_total should be int"
            assert isinstance(metric.fol_violation_rate, float), (
                "fol_violation_rate should be float"
            )

    def test_format_axis_report(self, mock_results):
        """Returns string with markdown table."""
        axis_metrics = compute_axis_metrics(mock_results, axes=["task_family"])

        report = format_axis_report(axis_metrics)

        assert isinstance(report, str), "Should return a string"
        assert len(report) > 0, "Report should not be empty"

        # Check for markdown table formatting
        assert "|" in report, "Should contain markdown table separators"
        assert "Accuracy" in report, "Should contain Accuracy column"
        assert "Correct" in report, "Should contain Correct column"
        assert "Total" in report, "Should contain Total column"
        assert "FOL Violation Rate" in report, (
            "Should contain FOL Violation Rate column"
        )
