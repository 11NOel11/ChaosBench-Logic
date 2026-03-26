#!/usr/bin/env python3
"""One-command M5 cycle: calibrate, apply, compare."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STABLE_PROVIDER_MAP_PATH = (
    PROJECT_ROOT / "chaosbench" / "repair" / "m5_provider_thresholds_crossfit_v1.json"
)


def parse_csv_list(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def parse_provider_threshold_offsets(raw: str) -> Dict[str, float]:
    payload = str(raw or "").strip()
    if not payload:
        return {}
    out: Dict[str, float] = {}
    for token in payload.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--provider-threshold-offsets entries must be provider=offset"
            )
        provider_raw, offset_raw = item.split("=", 1)
        provider = provider_raw.strip().lower()
        if not provider:
            raise ValueError("--provider-threshold-offsets has empty provider key")
        offset = as_float(offset_raw, float("nan"))
        if offset != offset:
            raise ValueError("--provider-threshold-offsets offset must be numeric")
        out[provider] = float(offset)
    return out


def apply_provider_threshold_offsets(
    provider_map: Dict[str, float],
    offsets: Dict[str, float],
    threshold_min: float,
    threshold_max: float,
) -> Dict[str, float]:
    if not offsets:
        return dict(provider_map)
    out: Dict[str, float] = {}
    for key, value in provider_map.items():
        provider = str(key).strip().lower()
        base = provider.split("/", 1)[0]
        offset = offsets.get(provider, offsets.get(base, 0.0))
        shifted = as_float(value, 0.0) + as_float(offset, 0.0)
        out[provider] = max(float(threshold_min), min(float(threshold_max), shifted))
    return out


def has_repair_run_artifacts(repair_dir: Path) -> bool:
    for base in (repair_dir / "repair_runs", repair_dir / "runs"):
        if not base.exists():
            continue
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "repair_manifest.json").exists() and (
                entry / "repaired_predictions.jsonl"
            ).exists():
                return True
    return False


def discover_ready_providers(
    providers_root: Path,
    provider_dirs: Sequence[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    ready: List[str] = []
    skipped: List[Dict[str, str]] = []
    for provider_dir in provider_dirs:
        repair_dir = providers_root / provider_dir
        if not repair_dir.exists():
            skipped.append(
                {
                    "provider_dir": provider_dir,
                    "reason": "missing_directory",
                    "path": str(repair_dir),
                }
            )
            continue
        if not has_repair_run_artifacts(repair_dir):
            skipped.append(
                {
                    "provider_dir": provider_dir,
                    "reason": "no_repair_artifacts",
                    "path": str(repair_dir),
                }
            )
            continue
        ready.append(provider_dir)
    return ready, skipped


def run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def slugify(value: str) -> str:
    out = []
    for ch in str(value or ""):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    text = "".join(out).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "item"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate_deltas(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {
            "n": 0.0,
            "mean_baseline": 0.0,
            "mean_policy": 0.0,
            "mean_diff": 0.0,
            "positives": 0.0,
            "negatives": 0.0,
            "zeros": 0.0,
            "mean_flip_policy": 0.0,
            "mean_harm_loss": 0.0,
            "alarm_rate": 0.0,
            "mean_shift_score": 0.0,
            "mean_update_applied": 0.0,
        }

    baseline = [as_float(row.get("delta_mcc_baseline"), 0.0) for row in rows]
    policy = [as_float(row.get("delta_mcc_policy"), 0.0) for row in rows]
    diffs = [as_float(row.get("policy_minus_baseline_mcc"), 0.0) for row in rows]
    flip_rates = [as_float(row.get("row_flip_rate_policy"), 0.0) for row in rows]
    harm_losses = [as_float(row.get("harm_loss"), 0.0) for row in rows]
    alarms = [as_float(row.get("alarm_triggered"), 0.0) for row in rows]
    shifts = [as_float(row.get("shift_score"), 0.0) for row in rows]
    updates = [as_float(row.get("update_applied"), 0.0) for row in rows]
    positives = sum(1 for value in diffs if value > 0)
    negatives = sum(1 for value in diffs if value < 0)
    zeros = len(diffs) - positives - negatives

    return {
        "n": float(len(rows)),
        "mean_baseline": sum(baseline) / len(baseline),
        "mean_policy": sum(policy) / len(policy),
        "mean_diff": sum(diffs) / len(diffs),
        "positives": float(positives),
        "negatives": float(negatives),
        "zeros": float(zeros),
        "mean_flip_policy": sum(flip_rates) / len(flip_rates),
        "mean_harm_loss": sum(harm_losses) / len(harm_losses),
        "alarm_rate": sum(alarms) / len(alarms),
        "mean_shift_score": sum(shifts) / len(shifts),
        "mean_update_applied": sum(updates) / len(updates),
    }


def aggregate_provider_set(
    providers_root: Path,
    provider_dirs: Sequence[str],
    folder_name: str,
    file_name: str,
) -> Dict[str, float] | None:
    rows: List[Dict[str, Any]] = []
    for provider_dir in provider_dirs:
        path = providers_root / provider_dir / folder_name / file_name
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(list(csv.DictReader(handle)))

    if not rows:
        return None

    temp_path = providers_root / "_tmp_aggregate.csv"
    fieldnames = list(rows[0].keys())
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    stats = aggregate_deltas(temp_path)
    temp_path.unlink(missing_ok=True)
    return stats


def aggregate_deltas_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "n": 0.0,
            "mean_baseline": 0.0,
            "mean_policy": 0.0,
            "mean_diff": 0.0,
            "positives": 0.0,
            "negatives": 0.0,
            "zeros": 0.0,
            "mean_flip_policy": 0.0,
            "mean_harm_loss": 0.0,
            "alarm_rate": 0.0,
            "mean_shift_score": 0.0,
            "mean_update_applied": 0.0,
        }

    baseline = [as_float(row.get("delta_mcc_baseline"), 0.0) for row in rows]
    policy = [as_float(row.get("delta_mcc_policy"), 0.0) for row in rows]
    diffs = [as_float(row.get("policy_minus_baseline_mcc"), 0.0) for row in rows]
    flip_rates = [as_float(row.get("row_flip_rate_policy"), 0.0) for row in rows]
    harm_losses = [as_float(row.get("harm_loss"), 0.0) for row in rows]
    alarms = [as_float(row.get("alarm_triggered"), 0.0) for row in rows]
    shifts = [as_float(row.get("shift_score"), 0.0) for row in rows]
    updates = [as_float(row.get("update_applied"), 0.0) for row in rows]
    positives = sum(1 for value in diffs if value > 0)
    negatives = sum(1 for value in diffs if value < 0)
    zeros = len(diffs) - positives - negatives

    return {
        "n": float(len(rows)),
        "mean_baseline": sum(baseline) / len(baseline),
        "mean_policy": sum(policy) / len(policy),
        "mean_diff": sum(diffs) / len(diffs),
        "positives": float(positives),
        "negatives": float(negatives),
        "zeros": float(zeros),
        "mean_flip_policy": sum(flip_rates) / len(flip_rates),
        "mean_harm_loss": sum(harm_losses) / len(harm_losses),
        "alarm_rate": sum(alarms) / len(alarms),
        "mean_shift_score": sum(shifts) / len(shifts),
        "mean_update_applied": sum(updates) / len(updates),
    }


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_paired_transfer(
    static_csv: Path,
    online_csv: Path,
    split_filter: str | None = None,
) -> Dict[str, float] | None:
    static_rows = load_csv_rows(static_csv)
    online_rows = load_csv_rows(online_csv)
    if not static_rows or not online_rows:
        return None

    static_by_run = {str(row.get("run_id") or ""): row for row in static_rows}
    online_by_run = {str(row.get("run_id") or ""): row for row in online_rows}
    paired_rows: List[Dict[str, Any]] = []
    for run_id in sorted(set(static_by_run.keys()) & set(online_by_run.keys())):
        if not run_id:
            continue
        s_row = static_by_run[run_id]
        o_row = online_by_run[run_id]
        split = str(o_row.get("split") or s_row.get("split") or "")
        if split_filter and split != split_filter:
            continue
        merged = dict(o_row)
        merged["run_id"] = run_id
        merged["online_minus_static_mcc"] = as_float(
            o_row.get("policy_minus_baseline_mcc"), 0.0
        ) - as_float(s_row.get("policy_minus_baseline_mcc"), 0.0)
        paired_rows.append(merged)

    if not paired_rows:
        return None
    stats = aggregate_deltas_from_rows(paired_rows)
    stats["mean_online_minus_static"] = sum(
        as_float(row.get("online_minus_static_mcc"), 0.0) for row in paired_rows
    ) / len(paired_rows)
    return stats


def list_repair_run_ids(repair_dir: Path) -> List[str]:
    run_roots = [repair_dir / "repair_runs", repair_dir / "runs"]
    run_ids: List[str] = []
    for root in run_roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            if (
                not (run_dir / "repair_manifest.json").exists()
                or not (run_dir / "repaired_predictions.jsonl").exists()
            ):
                continue
            run_id = run_dir.name
            manifest_path = run_dir / "repair_manifest.json"
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                run_id = str(payload.get("run_id") or run_dir.name)
            except (json.JSONDecodeError, OSError):
                run_id = run_dir.name
            run_ids.append(run_id)
        if run_ids:
            break
    return sorted(set(run_ids))


def build_temporal_splits(
    run_ids: Sequence[str],
    min_train_runs: int,
    min_test_runs: int,
    max_cuts: int,
) -> List[Tuple[int, List[Dict[str, str]]]]:
    n_runs = len(run_ids)
    out: List[Tuple[int, List[Dict[str, str]]]] = []
    if n_runs <= 1:
        return out
    for cut in range(0, n_runs - 1):
        train_count = cut + 1
        test_count = n_runs - train_count
        if train_count < min_train_runs or test_count < min_test_runs:
            continue
        rows: List[Dict[str, str]] = []
        for index, run_id in enumerate(run_ids):
            split = "dev" if index <= cut else "heldout"
            rows.append({"run_id": str(run_id), "split": split})
        out.append((cut, rows))
    if max_cuts > 0 and len(out) > max_cuts:
        return out[:max_cuts]
    return out


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_transfer_for_repair_dir(
    repair_dir: Path,
    base_policy_json: Path,
    provider_map_path: Path,
    default_split: str,
    out_dir_static: Path,
    run_online_controller: bool,
    out_dir_online: Path,
    args: argparse.Namespace,
    provider_reference_path: Path,
    split_map_csv: Path | None = None,
    online_update_splits_override: str | None = None,
) -> None:
    apply_cmd_static = [
        sys.executable,
        "scripts/run_m5_instance_guardrail.py",
        "--repair-dir",
        str(repair_dir),
        "--policy-json",
        str(base_policy_json),
        "--provider-thresholds-json",
        str(provider_map_path),
        "--default-split",
        str(default_split),
        "--out-dir",
        str(out_dir_static),
    ]
    if split_map_csv is not None:
        apply_cmd_static.extend(["--split-map-csv", str(split_map_csv)])
    run_command(apply_cmd_static)

    if not run_online_controller:
        return

    apply_cmd_online = [
        sys.executable,
        "scripts/run_m5_instance_guardrail.py",
        "--repair-dir",
        str(repair_dir),
        "--policy-json",
        str(base_policy_json),
        "--provider-thresholds-json",
        str(provider_map_path),
        "--controller-mode",
        "online",
        "--default-split",
        str(default_split),
        "--online-sweep-radius",
        str(args.online_sweep_radius),
        "--online-sweep-mix",
        str(args.online_sweep_mix),
        "--online-sweep-min-improvement",
        str(args.online_sweep_min_improvement),
        "--online-label-lag-runs",
        str(args.online_label_lag_runs),
        "--online-risk-budget-b0",
        str(args.online_risk_budget_b0),
        "--online-shift-kappa",
        str(args.online_shift_kappa),
        "--online-alarm-threshold",
        str(args.online_alarm_threshold),
        "--online-provider-step-mults",
        str(args.online_provider_step_mults),
        "--online-provider-step-default",
        str(args.online_provider_step_default),
        "--online-non-degrade-margin",
        str(args.online_non_degrade_margin),
        "--online-non-degrade-rollback-step",
        str(args.online_non_degrade_rollback_step),
        "--out-dir",
        str(out_dir_online),
    ]
    if split_map_csv is not None:
        apply_cmd_online.extend(["--split-map-csv", str(split_map_csv)])
    if online_update_splits_override:
        apply_cmd_online.extend(
            ["--online-update-splits", str(online_update_splits_override)]
        )
    if args.online_disable_non_degrade_guard:
        apply_cmd_online.append("--online-disable-non-degrade-guard")
    if provider_reference_path.exists():
        apply_cmd_online.extend(
            [
                "--provider-reference-json",
                str(provider_reference_path),
            ]
        )
    run_command(apply_cmd_online)


def build_report(
    out_path: Path,
    included: Sequence[str],
    skipped: Sequence[Dict[str, str]],
    map_payload: Dict[str, float],
    comparison_rows: Sequence[Dict[str, Any]],
    full_suite_stats: Dict[str, float] | None,
    full_suite_online_stats: Dict[str, float] | None,
    lopo_rows: Sequence[Dict[str, Any]],
    temporal_rows: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "# M5 Cross-Fit Cycle Results",
        "",
        "## Providers",
        "",
        f"- Included: {', '.join(included) if included else '(none)'}",
        f"- Skipped: {len(skipped)}",
        "",
        "## Deployment map",
        "",
    ]
    for key in sorted(map_payload.keys()):
        lines.append(f"- {key}: {map_payload[key]:.4f}")

    if skipped:
        lines.extend(["", "## Skipped providers", ""])
        for row in skipped:
            lines.append(f"- {row['provider_dir']}: {row['reason']} ({row['path']})")

    lines.extend(
        [
            "",
            "## Prompt-variant comparison",
            "",
            "| Variant | Mean baseline | Mean policy | Mean diff | Mean harm | Alarm rate | Pos | Neg | Mean flip policy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['variant']} | {row['mean_baseline']:+.4f} | "
            f"{row['mean_policy']:+.4f} | {row['mean_diff']:+.4f} | "
            f"{row['mean_harm_loss']:.4f} | {row['alarm_rate']:.4f} | "
            f"{int(row['positives'])} | {int(row['negatives'])} | "
            f"{row['mean_flip_policy']:.4f} |"
        )

    if full_suite_stats is not None:
        lines.extend(
            [
                "",
                "## Full suite transfer",
                "",
                f"- Mean baseline delta MCC: {full_suite_stats['mean_baseline']:+.4f}",
                f"- Mean policy delta MCC: {full_suite_stats['mean_policy']:+.4f}",
                f"- Mean (policy - baseline) delta MCC: {full_suite_stats['mean_diff']:+.4f}",
            ]
        )

    if full_suite_online_stats is not None:
        lines.extend(
            [
                "",
                "## Full suite online transfer",
                "",
                f"- Mean baseline delta MCC: {full_suite_online_stats['mean_baseline']:+.4f}",
                f"- Mean policy delta MCC: {full_suite_online_stats['mean_policy']:+.4f}",
                f"- Mean (policy - baseline) delta MCC: {full_suite_online_stats['mean_diff']:+.4f}",
            ]
        )

    if lopo_rows:
        worst_lopo = min(
            lopo_rows,
            key=lambda row: as_float(row.get("mean_online_minus_static"), 0.0),
        )
        lines.extend(
            [
                "",
                "## LOPO replay",
                "",
                f"- Rows: {len(lopo_rows)}",
                "- Worst holdout mean (online-static): "
                f"{str(worst_lopo.get('holdout_provider') or '')} "
                f"{as_float(worst_lopo.get('mean_online_minus_static'), 0.0):+.4f}",
            ]
        )

    if temporal_rows:
        worst_temporal = min(
            temporal_rows,
            key=lambda row: as_float(row.get("mean_online_minus_static"), 0.0),
        )
        lines.extend(
            [
                "",
                "## Temporal backtest replay",
                "",
                f"- Rows: {len(temporal_rows)}",
                "- Worst provider/cut mean (online-static): "
                f"{str(worst_temporal.get('provider_dir') or '')} "
                f"cut={int(as_float(worst_temporal.get('cut_index'), 0.0))} "
                f"{as_float(worst_temporal.get('mean_online_minus_static'), 0.0):+.4f}",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one-command M5 cross-fit calibration/apply/compare cycle"
    )
    parser.add_argument(
        "--providers-root",
        default=str(
            PROJECT_ROOT
            / "workspace"
            / "deep_survey_2026-03-01"
            / "prompt_variants_parallel"
        ),
    )
    parser.add_argument(
        "--provider-dirs",
        default="openai,deepseek,gemini_v2,openrouter",
    )
    parser.add_argument(
        "--base-policy-json",
        default=str(
            PROJECT_ROOT
            / "workspace"
            / "deep_survey_2026-03-01"
            / "repair_v3"
            / "m5_instance"
            / "m5_policy.json"
        ),
    )
    parser.add_argument("--selector", default="data/canonical_v2_files.json")
    parser.add_argument("--default-split", default="heldout")
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=0.6)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--calibration-out-dir",
        default=str(
            PROJECT_ROOT
            / "workspace"
            / "deep_survey_2026-03-01"
            / "repair_v3"
            / "m5_instance"
            / "crossfit_calibration"
        ),
    )
    parser.add_argument(
        "--transfer-out-tag",
        default="m5_instance_transfer_crossfit_auto",
    )
    parser.add_argument(
        "--online-transfer-out-tag",
        default=None,
        help="Output tag for online controller runs (default: <transfer-out-tag>_online)",
    )
    parser.add_argument(
        "--full-suite-repair-dir",
        default=str(
            PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "prompt_variants"
        ),
    )
    parser.add_argument(
        "--full-suite-out-dir",
        default=None,
        help="Optional output directory for full-suite transfer (default: <full-suite>/m5_instance_transfer_crossfit_auto)",
    )
    parser.add_argument(
        "--skip-full-suite",
        action="store_true",
        help="Skip full prompt-variant suite transfer step",
    )
    parser.add_argument(
        "--no-run-online-controller",
        action="store_false",
        dest="run_online_controller",
        help="Disable online controller transfer runs (keep static only)",
    )
    parser.set_defaults(run_online_controller=True)
    parser.add_argument("--online-sweep-radius", type=int, default=2)
    parser.add_argument("--online-sweep-mix", type=float, default=0.5)
    parser.add_argument("--online-sweep-min-improvement", type=float, default=0.0)
    parser.add_argument("--online-label-lag-runs", type=int, default=0)
    parser.add_argument("--online-risk-budget-b0", type=float, default=0.001)
    parser.add_argument("--online-shift-kappa", type=float, default=2.0)
    parser.add_argument("--online-alarm-threshold", type=float, default=0.02)
    parser.add_argument("--online-provider-step-mults", default="")
    parser.add_argument("--online-provider-step-default", type=float, default=1.0)
    parser.add_argument("--online-non-degrade-margin", type=float, default=0.0)
    parser.add_argument(
        "--online-non-degrade-rollback-step",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--online-disable-non-degrade-guard",
        action="store_true",
        help="Disable online provider non-degradation guard",
    )
    parser.add_argument(
        "--sync-stable-config",
        action="store_true",
        help="Sync calibrated map into stable repo config path",
    )
    parser.add_argument(
        "--provider-threshold-offsets",
        default="",
        help=(
            "Comma-separated provider=offset map to stress calibrated thresholds "
            "(example: openai=-0.05,gemini=0.03)"
        ),
    )
    parser.add_argument(
        "--run-lopo-replay",
        action="store_true",
        help="Run leave-one-provider-out replay calibration and transfer",
    )
    parser.add_argument(
        "--run-temporal-backtest",
        action="store_true",
        help="Run temporal prefix->suffix backtest replays per provider",
    )
    parser.add_argument(
        "--temporal-update-splits",
        default="dev",
        help="Comma-separated update splits for temporal online replays",
    )
    parser.add_argument("--temporal-min-train-runs", type=int, default=1)
    parser.add_argument("--temporal-min-test-runs", type=int, default=1)
    parser.add_argument(
        "--temporal-max-cuts",
        type=int,
        default=0,
        help="Limit cuts per provider (0 means all eligible cuts)",
    )
    args = parser.parse_args()
    if int(args.online_label_lag_runs) < 0:
        raise ValueError("--online-label-lag-runs must be >= 0")
    if float(args.online_provider_step_default) < 0.0:
        raise ValueError("--online-provider-step-default must be >= 0")
    if float(args.online_non_degrade_margin) < 0.0:
        raise ValueError("--online-non-degrade-margin must be >= 0")
    if float(args.online_non_degrade_rollback_step) < 0.0:
        raise ValueError("--online-non-degrade-rollback-step must be >= 0")
    if float(args.online_sweep_min_improvement) < 0.0:
        raise ValueError("--online-sweep-min-improvement must be >= 0")
    if int(args.temporal_min_train_runs) < 1:
        raise ValueError("--temporal-min-train-runs must be >= 1")
    if int(args.temporal_min_test_runs) < 1:
        raise ValueError("--temporal-min-test-runs must be >= 1")
    if int(args.temporal_max_cuts) < 0:
        raise ValueError("--temporal-max-cuts must be >= 0")

    provider_threshold_offsets = parse_provider_threshold_offsets(
        str(args.provider_threshold_offsets)
    )
    if provider_threshold_offsets and args.sync_stable_config:
        raise ValueError(
            "Refusing to sync stable provider map when --provider-threshold-offsets is set"
        )

    providers_root = Path(args.providers_root)
    provider_dirs = parse_csv_list(args.provider_dirs)
    calibration_out_dir = Path(args.calibration_out_dir)
    base_policy_json = Path(args.base_policy_json)

    included, skipped = discover_ready_providers(providers_root, provider_dirs)
    if not included:
        raise RuntimeError("No provider directories are ready for calibration")

    calibrate_cmd = [
        sys.executable,
        "scripts/calibrate_m5_provider_thresholds.py",
        "--providers-root",
        str(providers_root),
        "--provider-dirs",
        ",".join(included),
        "--base-policy-json",
        str(base_policy_json),
        "--selector",
        str(args.selector),
        "--default-split",
        str(args.default_split),
        "--threshold-min",
        str(args.threshold_min),
        "--threshold-max",
        str(args.threshold_max),
        "--threshold-step",
        str(args.threshold_step),
        "--out-dir",
        str(calibration_out_dir),
    ]
    run_command(calibrate_cmd)

    provider_map_path = calibration_out_dir / "provider_thresholds_crossfit_v1.json"
    if not provider_map_path.exists():
        raise RuntimeError(f"Missing calibrated provider map at {provider_map_path}")
    provider_reference_path = (
        calibration_out_dir / "provider_reference_dists_crossfit_v1.json"
    )

    provider_map = json.loads(provider_map_path.read_text(encoding="utf-8"))
    if provider_threshold_offsets:
        provider_map = apply_provider_threshold_offsets(
            provider_map=provider_map,
            offsets=provider_threshold_offsets,
            threshold_min=0.0,
            threshold_max=1.5,
        )
        provider_map_path = (
            calibration_out_dir / "provider_thresholds_crossfit_v1_perturbed.json"
        )
        provider_map_path.write_text(
            json.dumps(provider_map, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.sync_stable_config:
        STABLE_PROVIDER_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(provider_map_path, STABLE_PROVIDER_MAP_PATH)

    online_transfer_out_tag = (
        str(args.online_transfer_out_tag)
        if args.online_transfer_out_tag
        else f"{args.transfer_out_tag}_online"
    )

    for provider_dir in included:
        repair_dir = providers_root / provider_dir
        out_dir_static = repair_dir / args.transfer_out_tag
        out_dir_online = repair_dir / online_transfer_out_tag
        run_transfer_for_repair_dir(
            repair_dir=repair_dir,
            base_policy_json=base_policy_json,
            provider_map_path=provider_map_path,
            default_split=str(args.default_split),
            out_dir_static=out_dir_static,
            run_online_controller=bool(args.run_online_controller),
            out_dir_online=out_dir_online,
            args=args,
            provider_reference_path=provider_reference_path,
        )

    full_suite_stats: Dict[str, float] | None = None
    full_suite_online_stats: Dict[str, float] | None = None
    full_suite_online_out_dir: Path | None = None
    full_suite_repair_dir = Path(args.full_suite_repair_dir)
    full_suite_out_dir = (
        Path(args.full_suite_out_dir)
        if args.full_suite_out_dir
        else full_suite_repair_dir / args.transfer_out_tag
    )
    if not args.skip_full_suite and full_suite_repair_dir.exists():
        full_suite_online_out_dir = (
            full_suite_repair_dir / online_transfer_out_tag
            if args.full_suite_out_dir is None
            else Path(str(args.full_suite_out_dir) + "_online")
        )
        run_transfer_for_repair_dir(
            repair_dir=full_suite_repair_dir,
            base_policy_json=base_policy_json,
            provider_map_path=provider_map_path,
            default_split=str(args.default_split),
            out_dir_static=full_suite_out_dir,
            run_online_controller=bool(args.run_online_controller),
            out_dir_online=full_suite_online_out_dir,
            args=args,
            provider_reference_path=provider_reference_path,
        )
        full_suite_csv = full_suite_out_dir / "m5_run_deltas.csv"
        if full_suite_csv.exists():
            full_suite_stats = aggregate_deltas(full_suite_csv)

        if args.run_online_controller:
            full_suite_online_csv = full_suite_online_out_dir / "m5_run_deltas.csv"
            if full_suite_online_csv.exists():
                full_suite_online_stats = aggregate_deltas(full_suite_online_csv)

    lopo_rows: List[Dict[str, Any]] = []
    if args.run_lopo_replay and len(included) > 1:
        lopo_root = calibration_out_dir / "lopo_replay"
        lopo_root.mkdir(parents=True, exist_ok=True)
        for holdout_provider in included:
            train_provider_dirs = [p for p in included if p != holdout_provider]
            if not train_provider_dirs:
                continue
            lopo_dir = lopo_root / f"holdout_{slugify(holdout_provider)}"
            lopo_calibration_dir = lopo_dir / "calibration"
            lopo_calibrate_cmd = [
                sys.executable,
                "scripts/calibrate_m5_provider_thresholds.py",
                "--providers-root",
                str(providers_root),
                "--provider-dirs",
                ",".join(train_provider_dirs),
                "--base-policy-json",
                str(base_policy_json),
                "--selector",
                str(args.selector),
                "--default-split",
                str(args.default_split),
                "--threshold-min",
                str(args.threshold_min),
                "--threshold-max",
                str(args.threshold_max),
                "--threshold-step",
                str(args.threshold_step),
                "--out-dir",
                str(lopo_calibration_dir),
            ]
            run_command(lopo_calibrate_cmd)
            lopo_map_path = (
                lopo_calibration_dir / "provider_thresholds_crossfit_v1.json"
            )
            if not lopo_map_path.exists():
                continue

            lopo_map_payload = json.loads(lopo_map_path.read_text(encoding="utf-8"))
            if provider_threshold_offsets:
                lopo_map_payload = apply_provider_threshold_offsets(
                    provider_map=lopo_map_payload,
                    offsets=provider_threshold_offsets,
                    threshold_min=0.0,
                    threshold_max=1.5,
                )
                lopo_map_path = (
                    lopo_calibration_dir
                    / "provider_thresholds_crossfit_v1_perturbed.json"
                )
                lopo_map_path.write_text(
                    json.dumps(lopo_map_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            holdout_repair_dir = providers_root / holdout_provider
            out_static = lopo_dir / "static"
            out_online = lopo_dir / "online"
            lopo_reference_path = (
                lopo_calibration_dir / "provider_reference_dists_crossfit_v1.json"
            )
            run_transfer_for_repair_dir(
                repair_dir=holdout_repair_dir,
                base_policy_json=base_policy_json,
                provider_map_path=lopo_map_path,
                default_split=str(args.default_split),
                out_dir_static=out_static,
                run_online_controller=bool(args.run_online_controller),
                out_dir_online=out_online,
                args=args,
                provider_reference_path=lopo_reference_path,
            )

            static_csv = out_static / "m5_run_deltas.csv"
            online_csv = out_online / "m5_run_deltas.csv"
            if not args.run_online_controller:
                stats = aggregate_deltas(static_csv) if static_csv.exists() else None
                if stats is not None:
                    lopo_rows.append(
                        {
                            "holdout_provider": holdout_provider,
                            "train_provider_dirs": ",".join(train_provider_dirs),
                            **stats,
                            "mean_online_minus_static": 0.0,
                        }
                    )
            else:
                paired_stats = summarize_paired_transfer(
                    static_csv=static_csv,
                    online_csv=online_csv,
                    split_filter="heldout",
                )
                if paired_stats is not None:
                    lopo_rows.append(
                        {
                            "holdout_provider": holdout_provider,
                            "train_provider_dirs": ",".join(train_provider_dirs),
                            **paired_stats,
                        }
                    )

    temporal_rows: List[Dict[str, Any]] = []
    if args.run_temporal_backtest:
        temporal_root = calibration_out_dir / "temporal_backtest"
        temporal_root.mkdir(parents=True, exist_ok=True)
        split_map_root = temporal_root / "split_maps"
        split_map_root.mkdir(parents=True, exist_ok=True)

        for provider_dir in included:
            repair_dir = providers_root / provider_dir
            run_ids = list_repair_run_ids(repair_dir)
            cuts = build_temporal_splits(
                run_ids=run_ids,
                min_train_runs=int(args.temporal_min_train_runs),
                min_test_runs=int(args.temporal_min_test_runs),
                max_cuts=int(args.temporal_max_cuts),
            )
            for cut_index, split_rows in cuts:
                split_map_path = (
                    split_map_root
                    / f"{slugify(provider_dir)}_cut_{int(cut_index):02d}_split.csv"
                )
                write_csv(
                    split_map_path,
                    split_rows,
                    fieldnames=["run_id", "split"],
                )

                out_base = (
                    temporal_root / slugify(provider_dir) / f"cut_{int(cut_index):02d}"
                )
                out_static = out_base / "static"
                out_online = out_base / "online"
                run_transfer_for_repair_dir(
                    repair_dir=repair_dir,
                    base_policy_json=base_policy_json,
                    provider_map_path=provider_map_path,
                    default_split="heldout",
                    out_dir_static=out_static,
                    run_online_controller=bool(args.run_online_controller),
                    out_dir_online=out_online,
                    args=args,
                    provider_reference_path=provider_reference_path,
                    split_map_csv=split_map_path,
                    online_update_splits_override=str(args.temporal_update_splits),
                )

                static_csv = out_static / "m5_run_deltas.csv"
                online_csv = out_online / "m5_run_deltas.csv"
                if not args.run_online_controller:
                    heldout_rows = [
                        row
                        for row in load_csv_rows(static_csv)
                        if str(row.get("split") or "") == "heldout"
                    ]
                    if not heldout_rows:
                        continue
                    stats = aggregate_deltas_from_rows(heldout_rows)
                    temporal_rows.append(
                        {
                            "provider_dir": provider_dir,
                            "cut_index": float(cut_index),
                            "n_train": float(cut_index + 1),
                            "n_test": float(len(run_ids) - (cut_index + 1)),
                            **stats,
                            "mean_online_minus_static": 0.0,
                        }
                    )
                else:
                    paired_stats = summarize_paired_transfer(
                        static_csv=static_csv,
                        online_csv=online_csv,
                        split_filter="heldout",
                    )
                    if paired_stats is None:
                        continue
                    temporal_rows.append(
                        {
                            "provider_dir": provider_dir,
                            "cut_index": float(cut_index),
                            "n_train": float(cut_index + 1),
                            "n_test": float(len(run_ids) - (cut_index + 1)),
                            **paired_stats,
                        }
                    )

    comparisons = {
        "m5_static_crossfit": (args.transfer_out_tag, "m5_run_deltas.csv"),
        "m5_provider_v1": ("m5_instance_transfer_provider_v1", "m5_run_deltas.csv"),
        "m5_global_t012": ("m5_instance_transfer_t0p12_cli", "m5_run_deltas.csv"),
        "m4_ms2": ("m4_selective_ms2_transfer", "m4_run_deltas.csv"),
        "m4_ms4": ("m4_selective_ms4", "m4_run_deltas.csv"),
    }
    if args.run_online_controller:
        comparisons = {
            "m6_online_crossfit": (online_transfer_out_tag, "m5_run_deltas.csv"),
            **comparisons,
        }
    comparison_rows: List[Dict[str, Any]] = []
    for variant, (folder, filename) in comparisons.items():
        stats = aggregate_provider_set(
            providers_root=providers_root,
            provider_dirs=included,
            folder_name=folder,
            file_name=filename,
        )
        if stats is None:
            continue
        row: Dict[str, Any] = {"variant": variant}
        row.update(stats)
        comparison_rows.append(row)

    comparison_rows.sort(
        key=lambda row: as_float(row.get("mean_diff"), 0.0), reverse=True
    )

    cycle_dir = calibration_out_dir / "cycle"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        cycle_dir / "m5_cycle_comparison.csv",
        comparison_rows,
        fieldnames=[
            "variant",
            "n",
            "mean_baseline",
            "mean_policy",
            "mean_diff",
            "mean_harm_loss",
            "alarm_rate",
            "mean_shift_score",
            "mean_update_applied",
            "positives",
            "negatives",
            "zeros",
            "mean_flip_policy",
        ],
    )

    lopo_rows.sort(key=lambda row: as_float(row.get("mean_online_minus_static"), 0.0))
    write_csv(
        cycle_dir / "m6_lopo_replay.csv",
        lopo_rows,
        fieldnames=[
            "holdout_provider",
            "train_provider_dirs",
            "n",
            "mean_baseline",
            "mean_policy",
            "mean_diff",
            "mean_online_minus_static",
            "mean_harm_loss",
            "alarm_rate",
            "mean_shift_score",
            "mean_update_applied",
            "positives",
            "negatives",
            "zeros",
            "mean_flip_policy",
        ],
    )

    temporal_rows.sort(
        key=lambda row: (
            str(row.get("provider_dir") or ""),
            as_float(row.get("cut_index"), 0.0),
        )
    )
    write_csv(
        cycle_dir / "m6_temporal_backtest_replay.csv",
        temporal_rows,
        fieldnames=[
            "provider_dir",
            "cut_index",
            "n_train",
            "n_test",
            "n",
            "mean_baseline",
            "mean_policy",
            "mean_diff",
            "mean_online_minus_static",
            "mean_harm_loss",
            "alarm_rate",
            "mean_shift_score",
            "mean_update_applied",
            "positives",
            "negatives",
            "zeros",
            "mean_flip_policy",
        ],
    )

    build_report(
        out_path=cycle_dir / "M5_CYCLE_RESULTS.md",
        included=included,
        skipped=skipped,
        map_payload=provider_map,
        comparison_rows=comparison_rows,
        full_suite_stats=full_suite_stats,
        full_suite_online_stats=full_suite_online_stats,
        lopo_rows=lopo_rows,
        temporal_rows=temporal_rows,
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "providers_root": str(providers_root),
        "candidate_provider_dirs": provider_dirs,
        "included_provider_dirs": included,
        "skipped_provider_dirs": skipped,
        "base_policy_json": str(base_policy_json),
        "calibration_out_dir": str(calibration_out_dir),
        "provider_threshold_map": str(provider_map_path),
        "provider_threshold_offsets": {
            str(key): float(value)
            for key, value in sorted(provider_threshold_offsets.items())
        },
        "provider_reference_map": str(provider_reference_path)
        if provider_reference_path.exists()
        else None,
        "stable_provider_threshold_map": str(STABLE_PROVIDER_MAP_PATH),
        "stable_provider_threshold_map_synced": bool(args.sync_stable_config),
        "run_online_controller": bool(args.run_online_controller),
        "online_sweep_radius": int(args.online_sweep_radius),
        "online_sweep_mix": float(args.online_sweep_mix),
        "online_sweep_min_improvement": float(args.online_sweep_min_improvement),
        "online_label_lag_runs": int(args.online_label_lag_runs),
        "online_risk_budget_b0": float(args.online_risk_budget_b0),
        "online_shift_kappa": float(args.online_shift_kappa),
        "online_alarm_threshold": float(args.online_alarm_threshold),
        "online_provider_step_mults": str(args.online_provider_step_mults),
        "online_provider_step_default": float(args.online_provider_step_default),
        "online_non_degrade_margin": float(args.online_non_degrade_margin),
        "online_non_degrade_rollback_step": float(
            args.online_non_degrade_rollback_step
        ),
        "online_disable_non_degrade_guard": bool(args.online_disable_non_degrade_guard),
        "transfer_out_tag": args.transfer_out_tag,
        "online_transfer_out_tag": online_transfer_out_tag,
        "comparison_variants": [row["variant"] for row in comparison_rows],
        "full_suite_repair_dir": str(full_suite_repair_dir),
        "full_suite_out_dir": str(full_suite_out_dir),
        "full_suite_online_out_dir": str(full_suite_online_out_dir)
        if full_suite_online_out_dir is not None
        else None,
        "full_suite_ran": bool(
            (not args.skip_full_suite) and full_suite_repair_dir.exists()
        ),
        "run_lopo_replay": bool(args.run_lopo_replay),
        "run_temporal_backtest": bool(args.run_temporal_backtest),
        "temporal_update_splits": str(args.temporal_update_splits),
        "temporal_min_train_runs": int(args.temporal_min_train_runs),
        "temporal_min_test_runs": int(args.temporal_min_test_runs),
        "temporal_max_cuts": int(args.temporal_max_cuts),
        "lopo_replay_rows": len(lopo_rows),
        "temporal_backtest_rows": len(temporal_rows),
    }
    (cycle_dir / "m5_cycle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("M5 cross-fit cycle complete")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
