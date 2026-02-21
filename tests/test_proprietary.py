"""Tests for proprietary model pipeline: providers, cost estimation, budget guardrails,
launch script dry-run, and pred_text truncation.

All tests run without any live API calls.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing scripts/
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Sample items for tests that need a dataset
# ---------------------------------------------------------------------------

_SAMPLE_ITEMS: List[Dict] = [
    {
        "id": f"test_{i:04d}",
        "question": f"Does the Lorenz system exhibit chaotic behaviour for item {i}?",
        "ground_truth": "TRUE" if i % 2 == 0 else "FALSE",
        "task_family": "lorenz",
        "split": "test",
    }
    for i in range(20)
]


# ---------------------------------------------------------------------------
# Helper: load a script module by path
# ---------------------------------------------------------------------------


def _load_script(name: str):
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1: Provider configs load and return errors gracefully without API keys
# ---------------------------------------------------------------------------


class TestProviderConfigsLoad:
    """Providers must instantiate without API keys and return error responses."""

    def _unset_keys(self):
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            os.environ.pop(var, None)

    def test_openai_name_and_no_key_error(self):
        from chaosbench.eval.providers.openai import OpenAIProvider

        self._unset_keys()
        provider = OpenAIProvider(model="gpt-4o")
        assert provider.name == "openai/gpt-4o"
        resp = provider.generate("test prompt")
        assert resp.ok is False
        assert resp.error is not None
        assert "OPENAI_API_KEY" in resp.error

    def test_anthropic_name_and_no_key_error(self):
        from chaosbench.eval.providers.anthropic import AnthropicProvider

        self._unset_keys()
        provider = AnthropicProvider(model="claude-sonnet-4-6")
        assert provider.name == "anthropic/claude-sonnet-4-6"
        resp = provider.generate("test prompt")
        assert resp.ok is False
        assert resp.error is not None
        assert "ANTHROPIC_API_KEY" in resp.error

    def test_gemini_name_and_no_key_error(self):
        from chaosbench.eval.providers.gemini import GeminiProvider

        self._unset_keys()
        provider = GeminiProvider(model="gemini-2.0-flash")
        assert provider.name == "gemini/gemini-2.0-flash"
        resp = provider.generate("test prompt")
        assert resp.ok is False
        assert resp.error is not None
        assert "GEMINI_API_KEY" in resp.error

    def test_providers_exported_from_package(self):
        from chaosbench.eval.providers import AnthropicProvider, GeminiProvider, OpenAIProvider

        assert OpenAIProvider is not None
        assert AnthropicProvider is not None
        assert GeminiProvider is not None


# ---------------------------------------------------------------------------
# Test 2: Cost estimator runs on subset_1k_armored
# ---------------------------------------------------------------------------


class TestCostEstimatorRunsOnSubset:
    """Cost estimator must produce valid output for the 1k armored subset."""

    SUBSET = str(_REPO_ROOT / "data" / "subsets" / "subset_1k_armored.jsonl")
    PRICING = str(_REPO_ROOT / "configs" / "pricing" / "anthropic.yaml")
    MODEL = "claude-sonnet-4-6"
    PROVIDER = "anthropic"

    @pytest.fixture(autouse=True)
    def require_subset(self):
        if not Path(self.SUBSET).exists():
            pytest.skip(f"Subset file not found: {self.SUBSET}")

    def test_estimate_cost_returns_valid_dict(self):
        mod = _load_script("estimate_api_costs")
        result = mod.estimate_cost(
            subset_path=self.SUBSET,
            provider=self.PROVIDER,
            model=self.MODEL,
            pricing_config=self.PRICING,
            assumed_output_tokens=2,
        )
        assert result["n_items"] == 1000
        assert result["estimated_cost_usd"] > 0
        assert result["upper_bound_cost_usd"] >= result["estimated_cost_usd"]
        assert result["provider"] == self.PROVIDER
        assert result["model"] == self.MODEL

    def test_estimate_cost_fields_present(self):
        mod = _load_script("estimate_api_costs")
        result = mod.estimate_cost(
            subset_path=self.SUBSET,
            provider=self.PROVIDER,
            model=self.MODEL,
            pricing_config=self.PRICING,
        )
        required_keys = [
            "timestamp",
            "n_items",
            "estimated_total_input_tokens",
            "upper_bound_total_input_tokens",
            "estimated_cost_usd",
            "upper_bound_cost_usd",
            "pricing_source",
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    def test_budget_feasibility_computed(self):
        mod = _load_script("estimate_api_costs")
        result = mod.estimate_cost(
            subset_path=self.SUBSET,
            provider=self.PROVIDER,
            model=self.MODEL,
            pricing_config=self.PRICING,
            budget_usd=50.0,
        )
        assert result["items_feasible_under_budget"] is not None
        assert result["items_feasible_under_budget"] <= 1000


# ---------------------------------------------------------------------------
# Test 3: Budget guardrails — dry_run_cost returns early without calling provider
# ---------------------------------------------------------------------------


class TestBudgetGuardrailsDryRun:
    """dry_run_cost=True must return a cost estimate dict without calling the provider."""

    def test_dry_run_returns_early(self):
        from chaosbench.eval.providers.mock import MockProvider
        from chaosbench.eval.run import EvalRunner, RunConfig

        call_count = [0]

        class CountingMock(MockProvider):
            def generate(self, prompt, **kwargs):
                call_count[0] += 1
                return super().generate(prompt, **kwargs)

        cfg = RunConfig(
            provider=CountingMock(),
            dry_run_cost=True,
            cost_per_input_token=0.000003,
            cost_per_output_token=0.000015,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            result = EvalRunner(cfg).run(items=_SAMPLE_ITEMS[:10])

        assert result.get("dry_run") is True
        assert "estimated_cost_usd" in result
        assert "n_items" in result
        assert result["n_items"] == 10
        # Provider must NOT have been called
        assert call_count[0] == 0

    def test_dry_run_budget_status_ok(self):
        from chaosbench.eval.providers.mock import MockProvider
        from chaosbench.eval.run import EvalRunner, RunConfig

        cfg = RunConfig(
            provider=MockProvider(),
            dry_run_cost=True,
            cost_per_input_token=0.000003,
            cost_per_output_token=0.000015,
            max_usd=100.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            result = EvalRunner(cfg).run(items=_SAMPLE_ITEMS[:10])

        assert result["budget_status"] in ("OK", "OVER")

    def test_dry_run_budget_status_over(self):
        from chaosbench.eval.providers.mock import MockProvider
        from chaosbench.eval.run import EvalRunner, RunConfig

        cfg = RunConfig(
            provider=MockProvider(),
            dry_run_cost=True,
            cost_per_input_token=100.0,  # absurdly high to force OVER
            cost_per_output_token=100.0,
            max_usd=0.01,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            result = EvalRunner(cfg).run(items=_SAMPLE_ITEMS[:10])

        assert result["budget_status"] == "OVER"


# ---------------------------------------------------------------------------
# Test 4: Launch script dry-run outputs correct commands
# ---------------------------------------------------------------------------


class TestLaunchScriptDryRunOutputsCommands:
    """launch_proprietary_runs.main() in dry-run mode must print valid commands."""

    def test_dry_run_prints_eval_command(self, capsys):
        mod = _load_script("launch_proprietary_runs")
        mod.main(
            [
                "--provider", "anthropic",
                "--model", "claude-sonnet-4-6",
                "--phase", "P1",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert "chaosbench eval" in captured.out
        assert "--provider anthropic" in captured.out

    def test_dry_run_no_execute_by_default(self, capsys, monkeypatch):
        """Dry-run mode must not call subprocess.run for eval commands."""
        executed = []
        mod = _load_script("launch_proprietary_runs")

        original_run = __import__("subprocess").run

        def mock_run(cmd, **kwargs):
            executed.append(cmd)
            return original_run(["true"])  # no-op subprocess

        monkeypatch.setattr("subprocess.run", mock_run)

        mod.main(
            [
                "--provider", "openai",
                "--model", "gpt-4o",
                "--phase", "P0",
                "--dry-run",
            ]
        )
        # In dry-run mode, subprocess.run should not be called for eval
        eval_calls = [c for c in executed if "chaosbench" in (c[0] if c else "")]
        assert len(eval_calls) == 0

    def test_dry_run_phase_p0_shows_subset(self, capsys):
        mod = _load_script("launch_proprietary_runs")
        mod.main(
            [
                "--provider", "gemini",
                "--model", "gemini-2.0-flash",
                "--phase", "P0",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert "P0" in captured.out or "sanity" in captured.out.lower()


# ---------------------------------------------------------------------------
# Test 5: truncate_pred_text limits stored text length
# ---------------------------------------------------------------------------


class TestTruncatePredText:
    """truncate_pred_text=N must limit pred_text in predictions.jsonl to N chars."""

    LONG_RESPONSE = "TRUE " * 100  # 500 chars

    def test_pred_text_truncated_in_output(self):
        from chaosbench.eval.providers.mock import MockProvider
        from chaosbench.eval.run import EvalRunner, RunConfig

        cfg = RunConfig(
            provider=MockProvider(default=self.LONG_RESPONSE),
            truncate_pred_text=50,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            result = EvalRunner(cfg).run(items=_SAMPLE_ITEMS[:5])

            preds_path = Path(result["predictions_path"])
            assert preds_path.exists()
            with open(preds_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    assert len(rec["pred_text"]) <= 50, (
                        f"pred_text too long: {len(rec['pred_text'])} chars"
                    )

    def test_truncate_zero_keeps_full_text(self):
        """truncate_pred_text=0 (default) must not truncate anything."""
        from chaosbench.eval.providers.mock import MockProvider
        from chaosbench.eval.run import EvalRunner, RunConfig

        cfg = RunConfig(
            provider=MockProvider(default="TRUE"),
            truncate_pred_text=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = tmp
            result = EvalRunner(cfg).run(items=_SAMPLE_ITEMS[:3])

            preds_path = Path(result["predictions_path"])
            with open(preds_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    assert rec["pred_text"] == "TRUE"
