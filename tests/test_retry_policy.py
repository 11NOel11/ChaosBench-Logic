"""Tests for the retry policy on INVALID responses.

MockProvider returns INVALID first, then valid on second call.
Verifies that retry produces a valid outcome.
"""

import json
from pathlib import Path

import pytest

from chaosbench.eval.providers import MockProvider
from chaosbench.eval.run import EvalRunner, RunConfig, _evaluate_item
from chaosbench.eval.parsing import ParseOutcome


SAMPLE_ITEM = {
    "id": "test_001",
    "question": "Is the Lorenz system chaotic?",
    "answer": "TRUE",
    "task_family": "atomic",
    "split": "test",
}


class TestRetryOnInvalid:
    def test_retry_recovers_from_invalid(self):
        """Provider: first call returns garbage (INVALID), second returns TRUE."""
        responses = ["I cannot determine this.", "TRUE"]
        provider = MockProvider(responses=responses)
        rec = _evaluate_item(SAMPLE_ITEM, provider, retries=1, strict=True)

        assert rec.outcome == ParseOutcome.VALID_TRUE.value
        assert rec.parsed_label == "TRUE"
        assert rec.retry_pred_text == "TRUE"
        assert rec.retry_outcome == ParseOutcome.VALID_TRUE.value

    def test_no_retry_when_retries_zero(self):
        """With retries=0, INVALID stays INVALID even if second call would succeed."""
        responses = ["I cannot determine this.", "TRUE"]
        provider = MockProvider(responses=responses)
        rec = _evaluate_item(SAMPLE_ITEM, provider, retries=0, strict=True)

        assert rec.outcome == ParseOutcome.INVALID.value
        assert rec.retry_pred_text is None

    def test_double_invalid_stays_invalid(self):
        """Provider returns garbage twice -> still INVALID after retry."""
        provider = MockProvider(default="I cannot determine this.")
        rec = _evaluate_item(SAMPLE_ITEM, provider, retries=1, strict=True)

        assert rec.outcome == ParseOutcome.INVALID.value
        assert rec.retry_pred_text is not None  # retry was attempted
        assert rec.retry_outcome == ParseOutcome.INVALID.value

    def test_valid_first_no_retry_needed(self):
        """Provider returns TRUE immediately -> no retry needed."""
        provider = MockProvider(default="TRUE")
        rec = _evaluate_item(SAMPLE_ITEM, provider, retries=1, strict=True)

        assert rec.outcome == ParseOutcome.VALID_TRUE.value
        assert rec.retry_pred_text is None

    def test_retry_false_recovery(self):
        """Provider: first garbage, second FALSE."""
        responses = ["maybe", "FALSE"]
        provider = MockProvider(responses=responses)
        rec = _evaluate_item(SAMPLE_ITEM, provider, retries=1, strict=True)

        assert rec.outcome == ParseOutcome.VALID_FALSE.value
        assert rec.parsed_label == "FALSE"

    def test_retry_recorded_in_meta(self):
        """retry_pred_text and retry_outcome are recorded on the record."""
        provider = MockProvider(responses=["not sure", "TRUE"])
        rec = _evaluate_item(SAMPLE_ITEM, provider, retries=1, strict=True)

        assert rec.retry_pred_text == "TRUE"
        assert rec.retry_outcome is not None

    def test_retry_in_runner_integration(self, tmp_path):
        """Full runner integration: half items get invalid first call, then valid."""
        items = [
            {"id": f"item_{i}", "question": "Is chaos present?",
             "answer": "TRUE", "task_family": "atomic", "split": "test"}
            for i in range(10)
        ]

        call_count = [0]

        def smart_response(prompt: str) -> str:
            idx = call_count[0]
            call_count[0] += 1
            # Odd calls (retries) return TRUE, even (first attempt) return garbage
            return "TRUE" if idx % 2 == 1 else "I cannot determine."

        provider = MockProvider(responses=smart_response)
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), retries=1)
        runner = EvalRunner(cfg)
        result = runner.run(items=items)

        # With retries=1, all items should become valid
        assert result["metrics"]["coverage"] == 1.0


class TestRetryConfig:
    def test_retries_field_in_run_config(self):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, retries=1)
        assert cfg.retries == 1

    def test_retries_zero_config(self):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, retries=0)
        assert cfg.retries == 0

    def test_retries_recorded_in_manifest(self, tmp_path):
        provider = MockProvider(default="TRUE")
        cfg = RunConfig(provider=provider, output_dir=str(tmp_path), max_items=5, retries=1)
        runner = EvalRunner(cfg)
        result = runner.run(items=[
            {"id": f"i{i}", "question": "q", "answer": "TRUE",
             "task_family": "atomic", "split": "test"}
            for i in range(5)
        ])
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        assert manifest["retries"] == 1
