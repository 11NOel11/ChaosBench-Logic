"""Smoke tests for the evaluation runner using MockProvider (no network).

Runs eval on a tiny in-memory dataset and verifies:
- metrics keys exist
- coverage is computed correctly
- predictions.jsonl schema is correct
- summary.md is written
- run_manifest.json has required fields
"""

import json
from pathlib import Path

import pytest

from chaosbench.eval.providers import MockProvider
from chaosbench.eval.run import EvalRunner, RunConfig, compute_metrics, PredictionRecord
from chaosbench.eval.parsing import ParseOutcome


# ---------------------------------------------------------------------------
# Sample dataset fixture
# ---------------------------------------------------------------------------

SAMPLE_ITEMS = [
    {"id": f"item_{i}", "question": f"Is the Lorenz system chaotic? ({i})",
     "answer": "TRUE", "task_family": "atomic", "split": "test"}
    for i in range(30)
] + [
    {"id": f"item_{i+30}", "question": f"Does this system have a stable fixed point? ({i})",
     "answer": "FALSE", "task_family": "multi_hop", "split": "val"}
    for i in range(20)
]


class TestComputeMetrics:
    def test_all_correct_valid(self):
        records = [
            PredictionRecord(
                id=f"i{i}", question="q", ground_truth="TRUE",
                pred_text="TRUE", parsed_label="TRUE",
                outcome=ParseOutcome.VALID_TRUE.value, correct=True, latency_s=0.0,
            )
            for i in range(10)
        ]
        m = compute_metrics(records)
        assert m["coverage"] == 1.0
        assert m["accuracy_valid"] == 1.0
        assert m["effective_accuracy"] == 1.0
        assert m["invalid"] == 0

    def test_all_invalid(self):
        records = [
            PredictionRecord(
                id=f"i{i}", question="q", ground_truth="TRUE",
                pred_text="maybe", parsed_label=None,
                outcome=ParseOutcome.INVALID.value, correct=None, latency_s=0.0,
            )
            for i in range(5)
        ]
        m = compute_metrics(records)
        assert m["coverage"] == 0.0
        assert m["invalid"] == 5
        assert m["accuracy_valid"] == 0.0
        assert m["effective_accuracy"] == 0.0

    def test_mixed_outcomes(self):
        records = (
            [PredictionRecord(
                id=f"v{i}", question="q", ground_truth="TRUE",
                pred_text="TRUE", parsed_label="TRUE",
                outcome=ParseOutcome.VALID_TRUE.value, correct=True, latency_s=0.0,
            ) for i in range(8)]
            + [PredictionRecord(
                id=f"w{i}", question="q", ground_truth="TRUE",
                pred_text="?", parsed_label=None,
                outcome=ParseOutcome.INVALID.value, correct=None, latency_s=0.0,
            ) for i in range(2)]
        )
        m = compute_metrics(records)
        assert m["total"] == 10
        assert m["valid"] == 8
        assert m["invalid"] == 2
        assert abs(m["coverage"] - 0.8) < 1e-6

    def test_required_keys_present(self):
        records = [
            PredictionRecord(
                id="x", question="q", ground_truth="TRUE",
                pred_text="TRUE", parsed_label="TRUE",
                outcome=ParseOutcome.VALID_TRUE.value, correct=True, latency_s=0.0,
                task_family="atomic",
            )
        ]
        m = compute_metrics(records)
        for key in ["total", "valid", "invalid", "coverage", "accuracy_valid",
                    "effective_accuracy", "balanced_accuracy", "mcc",
                    "per_family", "per_split"]:
            assert key in m, f"Missing key: {key}"

    def test_per_family_metrics(self):
        records = [
            PredictionRecord(
                id=f"i{i}", question="q", ground_truth="TRUE",
                pred_text="TRUE", parsed_label="TRUE",
                outcome=ParseOutcome.VALID_TRUE.value, correct=True, latency_s=0.0,
                task_family="atomic",
            )
            for i in range(5)
        ] + [
            PredictionRecord(
                id=f"j{i}", question="q", ground_truth="FALSE",
                pred_text="FALSE", parsed_label="FALSE",
                outcome=ParseOutcome.VALID_FALSE.value, correct=True, latency_s=0.0,
                task_family="multi_hop",
            )
            for i in range(3)
        ]
        m = compute_metrics(records)
        assert "atomic" in m["per_family"]
        assert "multi_hop" in m["per_family"]
        assert m["per_family"]["atomic"]["total"] == 5
        assert m["per_family"]["multi_hop"]["total"] == 3


class TestEvalRunnerSmoke:
    def test_mock_runner_completes(self, tmp_path):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=50, seed=42)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:50])

        assert "run_id" in result
        assert "metrics" in result
        assert "predictions_path" in result
        assert "manifest_path" in result

    def test_predictions_jsonl_written(self, tmp_path):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=50)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:50])

        ppath = Path(result["predictions_path"])
        assert ppath.exists()
        lines = [json.loads(l) for l in ppath.read_text().splitlines() if l.strip()]
        assert len(lines) == 50

    def test_predictions_jsonl_schema(self, tmp_path):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=10)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:10])

        ppath = Path(result["predictions_path"])
        for line in ppath.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for field in ["id", "question", "ground_truth", "pred_text",
                          "parsed_label", "outcome", "correct", "latency_s"]:
                assert field in rec, f"Missing field in prediction: {field}"

    def test_metrics_keys_present(self, tmp_path):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=50)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:50])

        m = result["metrics"]
        for key in ["total", "valid", "invalid", "coverage", "accuracy_valid",
                    "effective_accuracy", "balanced_accuracy", "mcc"]:
            assert key in m

    def test_coverage_all_valid_mock(self, tmp_path):
        """MockProvider always returns TRUE -> all responses parseable -> coverage=1.0"""
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path))
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:20])
        assert result["metrics"]["coverage"] == 1.0

    def test_coverage_all_invalid_mock(self, tmp_path):
        """MockProvider returning garbage -> coverage=0.0"""
        provider = MockProvider(default="I cannot determine this.")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), retries=0)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:10])
        assert result["metrics"]["coverage"] == 0.0

    def test_run_manifest_schema(self, tmp_path):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=10)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:10])

        mpath = Path(result["manifest_path"])
        assert mpath.exists()
        manifest = json.loads(mpath.read_text())
        for field in ["run_id", "created_utc", "provider", "prompt_version",
                      "prompt_hash", "total_items_evaluated"]:
            assert field in manifest, f"Missing manifest field: {field}"

    def test_summary_md_written(self, tmp_path):
        provider = MockProvider(default="FALSE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=10)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS[:10])

        out_dir = Path(result["output_dir"])
        assert (out_dir / "summary.md").exists()
        content = (out_dir / "summary.md").read_text()
        assert "Coverage" in content

    def test_max_items_respected(self, tmp_path):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=15)
        runner = EvalRunner(cfg)
        result = runner.run(items=SAMPLE_ITEMS)
        assert result["metrics"]["total"] == 15
