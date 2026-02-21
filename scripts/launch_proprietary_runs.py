"""Phased launch script for proprietary model evaluations.

Orchestrates P0 → P1 → P2 → P3 evaluation phases with stop rules and
optional cost estimation. In dry-run mode (default), prints the exact
commands to execute without making any API calls.

Usage:
    # Dry-run (default — no network calls)
    python scripts/launch_proprietary_runs.py \\
      --provider anthropic --model claude-sonnet-4-6 --phase P1 --dry-run

    # Execute (requires --execute flag and API key in environment)
    python scripts/launch_proprietary_runs.py \\
      --provider anthropic --model claude-sonnet-4-6 --phase P1 --execute \\
      --budget-usd 20.0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Phase presets
# ---------------------------------------------------------------------------

PHASE_PRESETS: Dict[str, Dict[str, Any]] = {
    "P0": {
        "subset": "data/subsets/api_balanced_100.jsonl",
        "max_items": 100,
        "default_max_usd": 2.0,
        "label": "sanity",
        "description": "Sanity check — 100-item balanced subset",
    },
    "P1": {
        "subset": "data/subsets/subset_1k_armored.jsonl",
        "max_items": None,
        "default_max_usd": 20.0,
        "label": "1k_armored",
        "description": "1k armored subset — first signal run",
    },
    "P2": {
        "subset": "data/subsets/subset_5k_armored.jsonl",
        "max_items": None,
        "default_max_usd": 100.0,
        "label": "5k_armored",
        "description": "5k armored subset — main comparison subset",
    },
    "P3": {
        "subset": None,  # uses canonical dataset
        "max_items": None,
        "default_max_usd": 700.0,
        "label": "full_canonical",
        "description": "Full canonical dataset (40k+ items)",
    },
}

_PHASE_ORDER = ["P0", "P1", "P2", "P3"]

# ---------------------------------------------------------------------------
# Default stop thresholds
# ---------------------------------------------------------------------------

_DEFAULT_STOP_MCC = 0.35
_DEFAULT_STOP_INVALID_RATE = 0.02
_DEFAULT_STOP_BIAS_SCORE = 0.15

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def _pricing_config_path(provider: str) -> str:
    return f"configs/pricing/{provider}.yaml"


def _build_eval_command(
    provider: str,
    model: str,
    phase: str,
    preset: Dict[str, Any],
    max_usd: Optional[float],
    truncate_pred_text: int = 200,
) -> List[str]:
    """Build the chaosbench eval CLI command for a phase."""
    cmd = ["chaosbench", "eval", "--provider", provider, "--model", model]

    subset = preset.get("subset")
    if subset:
        cmd += ["--dataset", subset]
    else:
        cmd += ["--dataset", "canonical"]

    if preset.get("max_items"):
        cmd += ["--max-items", str(preset["max_items"])]

    cmd += ["--phase", phase]
    cmd += ["--truncate-pred-text", str(truncate_pred_text)]
    cmd += ["--strict-parsing"]

    if max_usd is not None:
        cmd += ["--max-usd", str(max_usd)]

    return cmd


def _build_cost_estimate_command(
    provider: str,
    model: str,
    phase: str,
    preset: Dict[str, Any],
) -> Optional[List[str]]:
    """Build the estimate_api_costs.py command for a phase (None if no subset)."""
    subset = preset.get("subset")
    if not subset:
        return None
    return [
        sys.executable,
        "scripts/estimate_api_costs.py",
        "--subset", subset,
        "--provider", provider,
        "--model", model,
        "--pricing_config", _pricing_config_path(provider),
    ]


# ---------------------------------------------------------------------------
# Stop rules
# ---------------------------------------------------------------------------


def _check_stop_rules(
    metrics: Dict[str, Any],
    phase: str,
    stop_mcc: float,
    stop_invalid_rate: float,
    stop_bias_score: float,
) -> Dict[str, Any]:
    """Check post-phase stop rules. Returns dict with keys: abort, proceed, warnings."""
    result = {"abort": False, "proceed": True, "warnings": []}

    invalid_rate = metrics.get("invalid_rate", 0.0)
    mcc = metrics.get("mcc", 0.0)

    # Compute TRUE prediction rate for bias check
    total = metrics.get("total", 0)
    tp = 0
    fp = 0
    if "per_family" in metrics:
        pass  # would need raw preds; approximate from top-level metrics
    # Use balanced_accuracy proxy: if BA ≈ 0.5 and accuracy_valid ≠ 0.5 → bias
    # Prefer a direct pred_true_pct if available; otherwise skip bias check
    pred_true_pct = metrics.get("pred_true_pct")

    if invalid_rate > stop_invalid_rate:
        result["abort"] = True
        result["proceed"] = False
        result["warnings"].append(
            f"ABORT: invalid_rate={invalid_rate:.3f} exceeds threshold={stop_invalid_rate:.3f}. "
            "Check prompt format and model output parsing."
        )

    if not result["abort"] and mcc < stop_mcc:
        result["proceed"] = False
        result["warnings"].append(
            f"NO-GO: MCC={mcc:.3f} is below threshold={stop_mcc:.3f}. "
            "Do not proceed to the next phase."
        )

    if pred_true_pct is not None:
        bias = abs(pred_true_pct - 0.495)
        if bias > stop_bias_score:
            result["warnings"].append(
                f"BIAS FLAG: pred_true_pct={pred_true_pct:.3f} deviates from 0.495 by {bias:.3f} "
                f"(threshold={stop_bias_score:.3f}). Recommend prompt calibration."
            )

    return result


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


def _run_dry(
    provider: str,
    model: str,
    phases: List[str],
    budget_usd: Optional[float],
) -> None:
    """Print commands for each phase without making any network calls."""
    print("=" * 70)
    print("DRY-RUN MODE — no API calls will be made")
    print(f"Provider: {provider}  Model: {model}")
    print(f"Phases: {', '.join(phases)}")
    print("=" * 70)

    for phase in phases:
        preset = PHASE_PRESETS[phase]
        max_usd = budget_usd if budget_usd is not None else preset["default_max_usd"]

        print(f"\n{'─' * 60}")
        print(f"Phase {phase}: {preset['description']}")
        print(f"{'─' * 60}")

        # Cost estimate command
        cost_cmd = _build_cost_estimate_command(provider, model, phase, preset)
        if cost_cmd:
            print("\n[1] Cost estimate:")
            print("    " + " ".join(cost_cmd))

            # Run cost estimator in-process if available
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "estimate_api_costs",
                    _REPO_ROOT / "scripts" / "estimate_api_costs.py",
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                result = mod.estimate_cost(
                    subset_path=str(_REPO_ROOT / preset["subset"]),
                    provider=provider,
                    model=model,
                    pricing_config=str(_REPO_ROOT / _pricing_config_path(provider)),
                )
                print(
                    f"    Estimated cost: ${result['estimated_cost_usd']:.4f} USD "
                    f"(upper bound: ${result['upper_bound_cost_usd']:.4f})"
                )
            except Exception as e:
                print(f"    (cost estimate unavailable: {e})")

        # Eval command
        eval_cmd = _build_eval_command(provider, model, phase, preset, max_usd)
        print("\n[2] Eval command:")
        print("    " + " ".join(eval_cmd))

        # Publish command
        print("\n[3] Publish (after reviewing results):")
        print(f"    chaosbench publish-run --run runs/<run_id>")

    print("\n" + "=" * 70)
    print("To execute, re-run with --execute flag (and ensure API key is set).")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Execute mode
# ---------------------------------------------------------------------------


def _run_execute(
    provider: str,
    model: str,
    phases: List[str],
    budget_usd: Optional[float],
    stop_mcc: float,
    stop_invalid_rate: float,
    stop_bias_score: float,
    skip_publish: bool,
) -> None:
    """Execute phases sequentially, applying stop rules between phases."""
    for i, phase in enumerate(phases):
        preset = PHASE_PRESETS[phase]
        max_usd = budget_usd if budget_usd is not None else preset["default_max_usd"]

        print(f"\n[execute] Starting phase {phase}: {preset['description']}")
        eval_cmd = _build_eval_command(provider, model, phase, preset, max_usd)
        print(f"[execute] Running: {' '.join(eval_cmd)}")

        proc = subprocess.run(eval_cmd, cwd=str(_REPO_ROOT))
        if proc.returncode != 0:
            print(f"[execute] ERROR: eval command failed (exit code {proc.returncode})", file=sys.stderr)
            sys.exit(proc.returncode)

        # Find most recent run directory (assumes run_id includes timestamp)
        runs_dir = _REPO_ROOT / "runs"
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = run_dirs[0] if run_dirs else None
        if run_dir is None:
            print("[execute] WARNING: Could not find run directory for metrics check.", file=sys.stderr)
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"[execute] WARNING: metrics.json not found at {metrics_path}", file=sys.stderr)
            continue

        metrics = json.loads(metrics_path.read_text())
        print(f"[execute] Phase {phase} metrics: MCC={metrics.get('mcc', 'N/A')}, "
              f"invalid_rate={metrics.get('invalid_rate', 'N/A')}, "
              f"coverage={metrics.get('coverage', 'N/A')}")

        stop = _check_stop_rules(metrics, phase, stop_mcc, stop_invalid_rate, stop_bias_score)
        for warn in stop["warnings"]:
            print(f"[execute] {warn}", file=sys.stderr)

        if stop["abort"]:
            print(f"[execute] Run aborted at phase {phase} due to stop rule.", file=sys.stderr)
            sys.exit(1)

        if not skip_publish:
            publish_cmd = ["chaosbench", "publish-run", "--run", str(run_dir)]
            print(f"[execute] Publishing: {' '.join(publish_cmd)}")
            subprocess.run(publish_cmd, cwd=str(_REPO_ROOT), check=True)

        if not stop["proceed"] and i < len(phases) - 1:
            print(f"[execute] Stop rule prevents advancing beyond phase {phase}.", file=sys.stderr)
            break

    print("\n[execute] All requested phases complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch proprietary model evaluations in phases."
    )
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument(
        "--phase",
        choices=_PHASE_ORDER,
        default=None,
        help="Run a single phase only (default: all phases P0→P3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print commands without executing (default: True)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually execute the eval commands (overrides --dry-run)",
    )
    parser.add_argument("--budget-usd", type=float, default=None, help="Budget cap per phase (USD)")
    parser.add_argument(
        "--stop-mcc-threshold",
        type=float,
        default=_DEFAULT_STOP_MCC,
        help=f"Minimum MCC to proceed to next phase (default: {_DEFAULT_STOP_MCC})",
    )
    parser.add_argument(
        "--stop-invalid-rate",
        type=float,
        default=_DEFAULT_STOP_INVALID_RATE,
        help=f"Maximum invalid rate before abort (default: {_DEFAULT_STOP_INVALID_RATE})",
    )
    parser.add_argument(
        "--stop-bias-score",
        type=float,
        default=_DEFAULT_STOP_BIAS_SCORE,
        help=f"Max abs(pred_true_pct - 0.495) before bias flag (default: {_DEFAULT_STOP_BIAS_SCORE})",
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        default=False,
        help="Skip publishing runs in execute mode",
    )

    args = parser.parse_args(argv)

    phases = [args.phase] if args.phase else _PHASE_ORDER

    if args.execute:
        _run_execute(
            provider=args.provider,
            model=args.model,
            phases=phases,
            budget_usd=args.budget_usd,
            stop_mcc=args.stop_mcc_threshold,
            stop_invalid_rate=args.stop_invalid_rate,
            stop_bias_score=args.stop_bias_score,
            skip_publish=args.skip_publish,
        )
    else:
        _run_dry(
            provider=args.provider,
            model=args.model,
            phases=phases,
            budget_usd=args.budget_usd,
        )


if __name__ == "__main__":
    main()
