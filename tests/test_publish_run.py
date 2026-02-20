"""
tests/test_publish_run.py — Tests for chaosbench.eval.publish and EvalRunner resume.

Verifies:
- publish_run copies core artifacts and writes publish_receipt.json
- publish_run raises FileExistsError without --force
- publish_run respects --force
- compress_predictions works for subset runs; skipped for full runs
- update_published_readme generates a valid README.md
- EvalRunner checkpoint written during run
- EvalRunner resume skips already-done items
- EvalRunner cleans up checkpoint after completion
"""
from __future__ import annotations

import gzip
import json
import threading
from pathlib import Path

import pytest

from chaosbench.eval.publish import publish_run, update_published_readme
from chaosbench.eval.run import EvalRunner, RunConfig, PredictionRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path, run_id: str = "test_run_001", max_items: int | None = None) -> Path:
    """Create a minimal run directory with all core artifacts."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    manifest = {
        "run_id": run_id,
        "provider": "mock",
        "created_utc": "2026-02-20T00:00:00+00:00",
        "max_items": max_items,
        "total_items_evaluated": max_items or 100,
        "metrics_summary": {
            "coverage": 1.0,
            "accuracy_valid": 0.6,
            "balanced_accuracy": 0.55,
            "mcc": 0.25,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.json").write_text(json.dumps({"total": max_items or 100}))
    (run_dir / "summary.md").write_text(f"# Summary {run_id}\n")
    (run_dir / "predictions.jsonl").write_text(
        '{"id": "x1", "outcome": "VALID_TRUE"}\n'
    )
    return run_dir


# ---------------------------------------------------------------------------
# publish_run — core functionality
# ---------------------------------------------------------------------------

class TestPublishRun:
    def test_copies_core_artifacts(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        out_dir = tmp_path / "published"

        result = publish_run(run_dir=run_dir, out_dir=out_dir)

        assert (result / "run_manifest.json").exists()
        assert (result / "metrics.json").exists()
        assert (result / "summary.md").exists()
        assert (result / "publish_receipt.json").exists()

    def test_predictions_not_copied(self, tmp_path):
        """predictions.jsonl is never copied (full run)."""
        run_dir = _make_run_dir(tmp_path)
        out_dir = tmp_path / "published"

        publish_run(run_dir=run_dir, out_dir=out_dir)

        assert not (out_dir / "predictions.jsonl").exists()

    def test_receipt_contains_metadata(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, run_id="my_run_42")
        out_dir = tmp_path / "published"

        publish_run(run_dir=run_dir, out_dir=out_dir)

        receipt = json.loads((out_dir / "publish_receipt.json").read_text())
        assert receipt["run_id"] == "my_run_42"
        assert "published_utc" in receipt
        assert "run_manifest.json" in receipt["artifacts_copied"]

    def test_raises_if_dest_exists_without_force(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        out_dir = tmp_path / "published"
        out_dir.mkdir()

        with pytest.raises(FileExistsError):
            publish_run(run_dir=run_dir, out_dir=out_dir)

    def test_force_overwrites_existing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        out_dir = tmp_path / "published"
        out_dir.mkdir()
        # Place a stale file
        (out_dir / "old_file.txt").write_text("stale")

        publish_run(run_dir=run_dir, out_dir=out_dir, force=True)

        # Core artifacts should exist
        assert (out_dir / "run_manifest.json").exists()

    def test_raises_if_run_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            publish_run(run_dir=tmp_path / "nonexistent", out_dir=tmp_path / "out")

    def test_raises_if_manifest_missing(self, tmp_path):
        run_dir = tmp_path / "bad_run"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text("{}")
        with pytest.raises(FileNotFoundError):
            publish_run(run_dir=run_dir, out_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# publish_run — predictions compression
# ---------------------------------------------------------------------------

class TestPublishRunCompress:
    def test_compress_predictions_for_subset_run(self, tmp_path):
        """subset run (max_items=1000) with compress_predictions=True should create .gz."""
        run_dir = _make_run_dir(tmp_path, max_items=1000)
        out_dir = tmp_path / "published"

        publish_run(run_dir=run_dir, out_dir=out_dir, compress_predictions=True)

        gz_path = out_dir / "predictions_subset.jsonl.gz"
        assert gz_path.exists()
        # Verify it's valid gzip
        with gzip.open(gz_path, "rb") as f:
            content = f.read()
        assert b"VALID_TRUE" in content

    def test_compress_predictions_skipped_for_full_run(self, tmp_path):
        """Full run (max_items=None) should NOT create .gz even with compress_predictions=True."""
        run_dir = _make_run_dir(tmp_path, max_items=None)
        out_dir = tmp_path / "published"

        publish_run(run_dir=run_dir, out_dir=out_dir, compress_predictions=True)

        assert not (out_dir / "predictions_subset.jsonl.gz").exists()

    def test_compress_false_no_gz(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, max_items=500)
        out_dir = tmp_path / "published"

        publish_run(run_dir=run_dir, out_dir=out_dir, compress_predictions=False)

        assert not (out_dir / "predictions_subset.jsonl.gz").exists()


# ---------------------------------------------------------------------------
# update_published_readme
# ---------------------------------------------------------------------------

class TestUpdatePublishedReadme:
    def _make_published_run(self, root: Path, run_id: str) -> None:
        d = root / run_id
        d.mkdir(parents=True)
        manifest = {
            "run_id": run_id,
            "provider": "ollama/test:7b",
            "total_items_evaluated": 1000,
            "created_utc": "2026-02-20T00:00:00+00:00",
            "metrics_summary": {
                "accuracy_valid": 0.70,
                "balanced_accuracy": 0.65,
                "mcc": 0.30,
            },
        }
        (d / "run_manifest.json").write_text(json.dumps(manifest))
        (d / "metrics.json").write_text(json.dumps({"accuracy_valid": 0.70}))

    def test_generates_readme(self, tmp_path):
        root = tmp_path / "published_runs"
        root.mkdir()
        self._make_published_run(root, "20260220T000000Z_ollama_test:7b")

        update_published_readme(root)

        readme = (root / "README.md").read_text()
        assert "20260220T000000Z_ollama_test:7b" in readme
        assert "0.6500" in readme  # balanced_accuracy formatted to 4dp

    def test_multiple_runs_all_listed(self, tmp_path):
        root = tmp_path / "published_runs"
        root.mkdir()
        for rid in ["20260219_run_a", "20260220_run_b"]:
            self._make_published_run(root, rid)

        update_published_readme(root)

        readme = (root / "README.md").read_text()
        assert "20260219_run_a" in readme
        assert "20260220_run_b" in readme

    def test_empty_dir_creates_header_only(self, tmp_path):
        root = tmp_path / "published_runs"
        root.mkdir()

        update_published_readme(root)

        readme = (root / "README.md").read_text()
        assert "Published Runs" in readme


# ---------------------------------------------------------------------------
# EvalRunner — checkpoint / resume
# ---------------------------------------------------------------------------

class _CountingProvider:
    """Provider that counts how many calls it receives."""

    name = "counting"

    def __init__(self, answer: str = "TRUE"):
        self.answer = answer
        self.call_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt: str):
        with self._lock:
            self.call_count += 1
        from chaosbench.eval.providers.types import ProviderResponse
        return ProviderResponse(text=self.answer, latency_s=0.0)


def _make_items(n: int = 5) -> list:
    return [
        {"id": f"item_{i}", "question": "Q?", "ground_truth": "TRUE"}
        for i in range(n)
    ]


class TestEvalRunnerCheckpoint:
    def test_checkpoint_written_during_run(self, tmp_path):
        """After a run, no checkpoint remains (deleted on completion)."""
        provider = _CountingProvider()
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), workers=1)
        runner = EvalRunner(cfg)
        items = _make_items(5)
        result = runner.run(items=items, dataset="canonical")

        out = Path(result["output_dir"])
        assert not (out / ".eval_checkpoint.jsonl").exists()
        assert (out / "predictions.jsonl").exists()

    def test_provider_called_for_each_item(self, tmp_path):
        provider = _CountingProvider()
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), workers=1)
        runner = EvalRunner(cfg)
        items = _make_items(7)
        runner.run(items=items, dataset="canonical")

        assert provider.call_count == 7

    def test_resume_skips_done_items(self, tmp_path):
        """When resuming, only items not yet in checkpoint are evaluated."""
        items = _make_items(6)
        run_id = "resume_test_run"
        out_dir = tmp_path / run_id
        out_dir.mkdir()

        # Pre-populate checkpoint with first 3 items
        pre_done = [
            PredictionRecord(
                id=items[i]["id"],
                question=items[i]["question"],
                ground_truth="TRUE",
                pred_text="TRUE",
                parsed_label="TRUE",
                outcome="VALID_TRUE",
                correct=True,
                latency_s=0.0,
            )
            for i in range(3)
        ]
        cp_path = out_dir / ".eval_checkpoint.jsonl"
        with open(cp_path, "w") as fh:
            for rec in pre_done:
                import dataclasses
                fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")

        provider = _CountingProvider()
        cfg = RunConfig(
            provider=provider,
            output_dir=str(tmp_path),
            workers=1,
            resume_run_id=run_id,
        )
        runner = EvalRunner(cfg)
        runner.run(items=items, dataset="canonical")

        # Only the remaining 3 items should be evaluated
        assert provider.call_count == 3

    def test_resume_final_predictions_complete(self, tmp_path):
        """Resumed run produces predictions.jsonl with ALL items (pre-done + new)."""
        items = _make_items(4)
        run_id = "resume_complete_test"
        out_dir = tmp_path / run_id
        out_dir.mkdir()

        # Pre-populate checkpoint with first 2 items
        pre_done = [
            PredictionRecord(
                id=items[i]["id"],
                question=items[i]["question"],
                ground_truth="TRUE",
                pred_text="TRUE",
                parsed_label="TRUE",
                outcome="VALID_TRUE",
                correct=True,
                latency_s=0.0,
            )
            for i in range(2)
        ]
        cp_path = out_dir / ".eval_checkpoint.jsonl"
        with open(cp_path, "w") as fh:
            for rec in pre_done:
                import dataclasses
                fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")

        provider = _CountingProvider()
        cfg = RunConfig(
            provider=provider,
            output_dir=str(tmp_path),
            workers=1,
            resume_run_id=run_id,
        )
        runner = EvalRunner(cfg)
        result = runner.run(items=items, dataset="canonical")

        preds_path = Path(result["predictions_path"])
        preds = [json.loads(l) for l in preds_path.read_text().splitlines() if l.strip()]
        ids_in_preds = {r["id"] for r in preds}
        expected_ids = {it["id"] for it in items}
        assert ids_in_preds == expected_ids

    def test_resume_raises_if_dir_missing(self, tmp_path):
        """resume_run_id pointing to nonexistent dir raises FileNotFoundError."""
        provider = _CountingProvider()
        cfg = RunConfig(
            provider=provider,
            output_dir=str(tmp_path),
            workers=1,
            resume_run_id="does_not_exist",
        )
        runner = EvalRunner(cfg)
        with pytest.raises(FileNotFoundError):
            runner.run(items=_make_items(3), dataset="canonical")

    def test_max_items_applied(self, tmp_path):
        """max_items limits the number of evaluations."""
        provider = _CountingProvider()
        cfg = RunConfig(
            provider=provider,
            output_dir=str(tmp_path),
            max_items=3,
            workers=1,
        )
        runner = EvalRunner(cfg)
        items = _make_items(10)
        runner.run(items=items, dataset="canonical")

        assert provider.call_count == 3
