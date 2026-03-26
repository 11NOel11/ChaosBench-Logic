#!/usr/bin/env python3
"""Cross-fit calibration for M5 provider-specific thresholds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.repair.engine import load_selector_index
from chaosbench.repair.instance_policy import apply_instance_policy
from chaosbench.repair.online_controller import candidate_distribution
from scripts.run_m4_selective_guardrail import (
    as_float,
    load_repair_config,
    load_runs,
    load_split_map,
    metrics_from_records,
    write_csv,
)


def parse_provider_dirs(raw: str) -> List[str]:
    out = [item.strip() for item in raw.split(",") if item.strip()]
    if not out:
        raise ValueError("--provider-dirs must contain at least one entry")
    return out


def threshold_grid(min_value: float, max_value: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("threshold step must be > 0")
    if min_value > max_value:
        raise ValueError("threshold min must be <= max")
    n_steps = int(round((max_value - min_value) / step))
    values = [round(min_value + i * step, 8) for i in range(n_steps + 1)]
    values[-1] = float(max_value)
    return sorted(set(values))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "n": 0.0,
            "mean_baseline": 0.0,
            "mean_policy": 0.0,
            "mean_diff": 0.0,
            "mean_flip_policy": 0.0,
            "positives": 0.0,
            "negatives": 0.0,
            "zeros": 0.0,
        }
    diffs = [as_float(row.get("policy_minus_baseline_mcc"), 0.0) for row in rows]
    baseline = [as_float(row.get("delta_mcc_baseline"), 0.0) for row in rows]
    policy = [as_float(row.get("delta_mcc_policy"), 0.0) for row in rows]
    flip_rates = [as_float(row.get("row_flip_rate_policy"), 0.0) for row in rows]
    positives = sum(1 for value in diffs if value > 0)
    negatives = sum(1 for value in diffs if value < 0)
    zeros = len(diffs) - positives - negatives
    return {
        "n": float(len(rows)),
        "mean_baseline": mean(baseline),
        "mean_policy": mean(policy),
        "mean_diff": mean(diffs),
        "mean_flip_policy": mean(flip_rates),
        "positives": float(positives),
        "negatives": float(negatives),
        "zeros": float(zeros),
    }


def choose_threshold(
    threshold_stats: Sequence[Tuple[float, Dict[str, float]]],
    default_threshold: float,
) -> float:
    safe = [item for item in threshold_stats if int(item[1]["negatives"]) == 0]
    if safe:
        return float(
            max(
                safe,
                key=lambda item: (
                    as_float(item[1]["mean_diff"], 0.0),
                    as_float(item[1]["mean_policy"], 0.0),
                    -as_float(item[1]["mean_flip_policy"], 0.0),
                    -abs(float(item[0]) - float(default_threshold)),
                ),
            )[0]
        )

    return float(
        max(
            threshold_stats,
            key=lambda item: (
                -as_float(item[1]["negatives"], 0.0),
                as_float(item[1]["mean_diff"], 0.0),
                as_float(item[1]["mean_policy"], 0.0),
                -as_float(item[1]["mean_flip_policy"], 0.0),
                -abs(float(item[0]) - float(default_threshold)),
            ),
        )[0]
    )


def provider_key_from_runs(provider_dir: str, run_rows: Sequence[Any]) -> str:
    keys = {
        str(run.provider).strip().lower().split("/", 1)[0]
        for run in run_rows
        if str(run.provider).strip()
    }
    keys = {key for key in keys if key}
    if len(keys) == 1:
        return next(iter(keys))
    return provider_dir.strip().lower()


def evaluate_threshold_row(
    run: Any, policy_with_threshold: Dict[str, Any]
) -> Dict[str, Any]:
    pre_metrics = metrics_from_records(run.records, "parsed_label")
    baseline_metrics = metrics_from_records(run.baseline_records, "repaired_label")

    policy_records, _ = apply_instance_policy(
        repaired_records=run.baseline_records,
        candidates=run.candidates,
        policy=policy_with_threshold,
    )
    policy_metrics = metrics_from_records(policy_records, "repaired_label")

    delta_mcc_baseline = baseline_metrics["mcc"] - pre_metrics["mcc"]
    delta_mcc_policy = policy_metrics["mcc"] - pre_metrics["mcc"]

    policy_row_flips = sum(
        1 for row in policy_records if row.get("was_flipped") is True
    )
    policy_row_flip_rate = (
        policy_row_flips / policy_metrics["valid"] if policy_metrics["valid"] else 0.0
    )

    return {
        "run_id": run.run_id,
        "provider": run.provider,
        "split": run.split,
        "delta_mcc_baseline": float(delta_mcc_baseline),
        "delta_mcc_policy": float(delta_mcc_policy),
        "policy_minus_baseline_mcc": float(delta_mcc_policy - delta_mcc_baseline),
        "row_flip_rate_baseline": float(run.baseline_row_flip_rate),
        "row_flip_rate_policy": float(policy_row_flip_rate),
    }


def threshold_mode(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    counts: Dict[float, int] = {}
    for value in values:
        key = float(value)
        counts[key] = counts.get(key, 0) + 1
    return float(max(counts.items(), key=lambda item: (item[1], item[0]))[0])


def build_report(
    out_path: Path,
    fold_rows: List[Dict[str, Any]],
    provider_rows: List[Dict[str, Any]],
    deployment_map: Dict[str, float],
) -> None:
    overall = aggregate_rows(fold_rows)
    lines = [
        "# M5 Cross-Fit Provider Threshold Calibration",
        "",
        "## Out-of-fold aggregate",
        "",
        f"- Runs: {int(overall['n'])}",
        f"- Mean baseline delta MCC: {overall['mean_baseline']:+.4f}",
        f"- Mean policy delta MCC: {overall['mean_policy']:+.4f}",
        f"- Mean (policy - baseline) delta MCC: {overall['mean_diff']:+.4f}",
        f"- Positives / negatives: {int(overall['positives'])}/{int(overall['negatives'])}",
        "",
        "## Deployment map",
        "",
    ]
    for key in sorted(deployment_map.keys()):
        lines.append(f"- {key}: {deployment_map[key]:.4f}")

    lines.extend(
        [
            "",
            "## Provider summary",
            "",
            "| Provider dir | Provider key | Runs | Mean OOF diff | OOF + | OOF - | Median selected | Mean selected | Mode selected | Std selected |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in provider_rows:
        lines.append(
            f"| {row['provider_dir']} | {row['provider_key']} | {int(row['n_runs'])} | "
            f"{row['oof_mean_diff']:+.4f} | {int(row['oof_positives'])} | "
            f"{int(row['oof_negatives'])} | {row['selected_threshold_median']:.4f} | "
            f"{row['selected_threshold_mean']:.4f} | {row['selected_threshold_mode']:.4f} | "
            f"{row['selected_threshold_std']:.4f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-fit calibrate M5 provider-specific thresholds"
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
        default="openai,deepseek,gemini_v2",
        help="Comma-separated provider subdirectories under providers-root",
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
    parser.add_argument("--threshold-max", type=float, default=0.60)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "workspace"
            / "deep_survey_2026-03-01"
            / "repair_v3"
            / "m5_instance"
            / "crossfit_calibration"
        ),
    )
    args = parser.parse_args()

    providers_root = Path(args.providers_root)
    provider_dirs = parse_provider_dirs(args.provider_dirs)
    base_policy = json.loads(Path(args.base_policy_json).read_text(encoding="utf-8"))
    default_threshold = as_float(base_policy.get("threshold"), 0.0)
    margin_step = as_float(base_policy.get("margin_step"), 0.05)
    support_cap = int(as_float(base_policy.get("support_cap"), 8.0))
    thresholds = threshold_grid(
        args.threshold_min, args.threshold_max, args.threshold_step
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_meta = load_selector_index(
        selector_path=Path(args.selector),
        project_root=PROJECT_ROOT,
    )

    fold_rows: List[Dict[str, Any]] = []
    provider_rows: List[Dict[str, Any]] = []
    deployment_map: Dict[str, float] = {}
    provider_reference_map: Dict[str, Dict[str, float]] = {}

    for provider_dir in provider_dirs:
        repair_dir = providers_root / provider_dir
        config = load_repair_config(repair_dir)
        split_map = load_split_map(repair_dir / "tables" / "repair_deltas.csv")
        runs = load_runs(
            repair_dir=repair_dir,
            split_map=split_map,
            id_to_meta=id_to_meta,
            config=config,
            default_split=str(args.default_split),
        )
        if not runs:
            raise RuntimeError(f"No runs found under {repair_dir}")

        provider_key = provider_key_from_runs(provider_dir, runs)
        if provider_key in deployment_map:
            raise RuntimeError(
                f"Duplicate provider key '{provider_key}' from provider dirs; "
                "use unique provider sets per calibration run"
            )
        provider_reference_map[provider_key] = candidate_distribution(
            candidates=[candidate for run in runs for candidate in run.candidates],
            margin_step=margin_step,
            support_cap=support_cap,
        )

        eval_by_threshold: Dict[float, Dict[str, Dict[str, Any]]] = {}
        for threshold in thresholds:
            policy_t = dict(base_policy)
            policy_t["threshold"] = float(threshold)
            rows_for_threshold: Dict[str, Dict[str, Any]] = {}
            for run in runs:
                rows_for_threshold[run.run_id] = evaluate_threshold_row(run, policy_t)
            eval_by_threshold[float(threshold)] = rows_for_threshold

        selected_thresholds: List[float] = []
        oof_provider_rows: List[Dict[str, Any]] = []
        for run in runs:
            threshold_stats: List[Tuple[float, Dict[str, float]]] = []
            for threshold in thresholds:
                train_rows = [
                    row
                    for run_id, row in eval_by_threshold[float(threshold)].items()
                    if run_id != run.run_id
                ]
                threshold_stats.append((float(threshold), aggregate_rows(train_rows)))

            selected = choose_threshold(threshold_stats, default_threshold)
            selected_thresholds.append(float(selected))

            train_stats_for_selected = dict(
                next(
                    stats
                    for threshold, stats in threshold_stats
                    if threshold == selected
                )
            )
            test_row = dict(eval_by_threshold[float(selected)][run.run_id])
            test_row.update(
                {
                    "provider_dir": provider_dir,
                    "provider_key": provider_key,
                    "selected_threshold": float(selected),
                    "train_n": train_stats_for_selected["n"],
                    "train_mean_diff": train_stats_for_selected["mean_diff"],
                    "train_negatives": train_stats_for_selected["negatives"],
                    "train_positives": train_stats_for_selected["positives"],
                }
            )
            fold_rows.append(test_row)
            oof_provider_rows.append(test_row)

        selected_median = float(statistics.median(selected_thresholds))
        selected_mean = mean(selected_thresholds)
        selected_mode = threshold_mode(selected_thresholds)
        selected_std = (
            float(statistics.pstdev(selected_thresholds))
            if len(selected_thresholds) > 1
            else 0.0
        )

        deployment_map[provider_key] = selected_median

        oof_stats = aggregate_rows(oof_provider_rows)
        provider_rows.append(
            {
                "provider_dir": provider_dir,
                "provider_key": provider_key,
                "n_runs": float(len(runs)),
                "selected_threshold_median": float(selected_median),
                "selected_threshold_mean": float(selected_mean),
                "selected_threshold_mode": float(selected_mode),
                "selected_threshold_std": float(selected_std),
                "oof_mean_baseline": oof_stats["mean_baseline"],
                "oof_mean_policy": oof_stats["mean_policy"],
                "oof_mean_diff": oof_stats["mean_diff"],
                "oof_positives": oof_stats["positives"],
                "oof_negatives": oof_stats["negatives"],
                "oof_zeros": oof_stats["zeros"],
            }
        )

    write_csv(
        out_dir / "m5_crossfit_fold_results.csv",
        fold_rows,
        fieldnames=[
            "provider_dir",
            "provider_key",
            "run_id",
            "provider",
            "split",
            "selected_threshold",
            "train_n",
            "train_mean_diff",
            "train_negatives",
            "train_positives",
            "delta_mcc_baseline",
            "delta_mcc_policy",
            "policy_minus_baseline_mcc",
            "row_flip_rate_baseline",
            "row_flip_rate_policy",
        ],
    )
    write_csv(
        out_dir / "m5_crossfit_provider_summary.csv",
        provider_rows,
        fieldnames=[
            "provider_dir",
            "provider_key",
            "n_runs",
            "selected_threshold_median",
            "selected_threshold_mean",
            "selected_threshold_mode",
            "selected_threshold_std",
            "oof_mean_baseline",
            "oof_mean_policy",
            "oof_mean_diff",
            "oof_positives",
            "oof_negatives",
            "oof_zeros",
        ],
    )

    map_path = out_dir / "provider_thresholds_crossfit_v1.json"
    map_path.write_text(
        json.dumps(deployment_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    prior_payload: Dict[str, Dict[str, float]] = {}
    for row in provider_rows:
        key = str(row["provider_key"])
        prior_payload[key] = {
            "threshold": float(row["selected_threshold_median"]),
            "threshold_mean": float(row["selected_threshold_mean"]),
            "threshold_mode": float(row["selected_threshold_mode"]),
            "threshold_std": float(row["selected_threshold_std"]),
            "n_runs": float(row["n_runs"]),
            "oof_mean_diff": float(row["oof_mean_diff"]),
            "oof_negatives": float(row["oof_negatives"]),
            "oof_positives": float(row["oof_positives"]),
        }

    prior_path = out_dir / "provider_threshold_priors_crossfit_v1.json"
    prior_path.write_text(
        json.dumps(prior_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    provider_refs_path = out_dir / "provider_reference_dists_crossfit_v1.json"
    provider_refs_path.write_text(
        json.dumps(provider_reference_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "providers_root": str(providers_root),
        "provider_dirs": provider_dirs,
        "base_policy_json": str(args.base_policy_json),
        "default_split": str(args.default_split),
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_step": args.threshold_step,
        "n_thresholds": len(thresholds),
        "n_fold_rows": len(fold_rows),
        "deployment_map": deployment_map,
        "deployment_prior_path": str(prior_path),
        "provider_reference_map_path": str(provider_refs_path),
    }
    (out_dir / "m5_crossfit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    build_report(
        out_path=out_dir / "M5_CROSSFIT_CALIBRATION.md",
        fold_rows=fold_rows,
        provider_rows=provider_rows,
        deployment_map=deployment_map,
    )

    print("M5 cross-fit calibration complete")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
