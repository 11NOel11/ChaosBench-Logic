#!/usr/bin/env python3
"""Run stricter M6-vs-static evaluation on a sweep grid.

This script compares online transfer against static transfer per run, then reports:
- paired run-level bootstrap CI for mean online-minus-static delta MCC
- provider-cluster bootstrap CI for mean online-minus-static delta MCC
- exact two-sided sign test on paired run differences
- worst-provider (slice) mean difference and non-degradation check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float((1.0 - frac) * ordered[lo] + frac * ordered[hi])


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int,
    rng: random.Random,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    draws: List[float] = []
    for _ in range(max(1, n_bootstrap)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(mean(sample))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def bootstrap_cluster_mean_ci(
    values_by_cluster: Mapping[str, Sequence[float]],
    n_bootstrap: int,
    rng: random.Random,
) -> Tuple[float, float]:
    cluster_keys = sorted(str(key) for key in values_by_cluster.keys())
    if not cluster_keys:
        return 0.0, 0.0
    if len(cluster_keys) == 1:
        vals = list(values_by_cluster[cluster_keys[0]])
        baseline = mean(vals)
        return baseline, baseline

    k = len(cluster_keys)
    draws: List[float] = []
    for _ in range(max(1, n_bootstrap)):
        sampled_keys = [cluster_keys[rng.randrange(k)] for _ in range(k)]
        sampled_values: List[float] = []
        for key in sampled_keys:
            sampled_values.extend(float(v) for v in values_by_cluster.get(key, []))
        if sampled_values:
            draws.append(mean(sampled_values))
    if not draws:
        return 0.0, 0.0
    return quantile(draws, 0.025), quantile(draws, 0.975)


def sign_test_two_sided(values: Sequence[float]) -> Dict[str, float]:
    positives = sum(1 for value in values if value > 0.0)
    negatives = sum(1 for value in values if value < 0.0)
    zeros = len(values) - positives - negatives
    n = positives + negatives
    if n == 0:
        p_value = 1.0
    else:
        tail = min(positives, negatives)
        cumulative = sum(math.comb(n, i) for i in range(0, tail + 1)) / (2**n)
        p_value = min(1.0, 2.0 * cumulative)
    return {
        "n": float(n),
        "positives": float(positives),
        "negatives": float(negatives),
        "zeros": float(zeros),
        "p_value": float(p_value),
    }


def load_paired_rows_for_config(config_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = config_dir / "cycle" / "m5_cycle_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    providers_root = Path(str(manifest.get("providers_root") or "")).resolve()
    provider_dirs = [str(item) for item in manifest.get("included_provider_dirs", [])]
    static_tag = str(manifest.get("transfer_out_tag") or "")
    online_tag = str(manifest.get("online_transfer_out_tag") or "")
    if not providers_root.exists() or not static_tag or not online_tag:
        return []

    paired_rows: List[Dict[str, Any]] = []
    for provider_dir in provider_dirs:
        static_path = providers_root / provider_dir / static_tag / "m5_run_deltas.csv"
        online_path = providers_root / provider_dir / online_tag / "m5_run_deltas.csv"
        if not static_path.exists() or not online_path.exists():
            continue

        static_rows_list = read_csv_rows(static_path)
        online_rows_list = read_csv_rows(online_path)
        static_rows = {row["run_id"]: row for row in static_rows_list}
        online_rows = {row["run_id"]: row for row in online_rows_list}
        static_order = {
            row.get("run_id", ""): float(index)
            for index, row in enumerate(static_rows_list)
            if row.get("run_id")
        }
        online_order = {
            row.get("run_id", ""): float(index)
            for index, row in enumerate(online_rows_list)
            if row.get("run_id")
        }
        common = sorted(set(static_rows.keys()) & set(online_rows.keys()))

        for run_id in common:
            s_row = static_rows[run_id]
            o_row = online_rows[run_id]
            provider_text = str(
                o_row.get("provider") or s_row.get("provider") or provider_dir
            ).strip()
            provider_key = (
                provider_text.lower().split("/", 1)[0] if provider_text else "unknown"
            )

            static_value = as_float(s_row.get("policy_minus_baseline_mcc"), 0.0)
            online_value = as_float(o_row.get("policy_minus_baseline_mcc"), 0.0)

            paired_rows.append(
                {
                    "config": config_dir.name,
                    "provider": provider_key,
                    "provider_full": provider_text,
                    "run_id": run_id,
                    "split": str(o_row.get("split") or s_row.get("split") or ""),
                    "provider_run_index": as_float(
                        online_order.get(run_id, static_order.get(run_id, -1.0)),
                        -1.0,
                    ),
                    "static_policy_minus_baseline_mcc": static_value,
                    "online_policy_minus_baseline_mcc": online_value,
                    "online_minus_static_mcc": online_value - static_value,
                    "static_harm_loss": as_float(s_row.get("harm_loss"), 0.0),
                    "online_harm_loss": as_float(o_row.get("harm_loss"), 0.0),
                    "static_alarm": as_float(s_row.get("alarm_triggered"), 0.0),
                    "online_alarm": as_float(o_row.get("alarm_triggered"), 0.0),
                }
            )

    return paired_rows


def summarize_config(
    config_name: str,
    paired_rows: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    rng: random.Random,
) -> Dict[str, Any]:
    diffs = [as_float(row.get("online_minus_static_mcc"), 0.0) for row in paired_rows]
    static_values = [
        as_float(row.get("static_policy_minus_baseline_mcc"), 0.0)
        for row in paired_rows
    ]
    online_values = [
        as_float(row.get("online_policy_minus_baseline_mcc"), 0.0)
        for row in paired_rows
    ]
    static_harm = [as_float(row.get("static_harm_loss"), 0.0) for row in paired_rows]
    online_harm = [as_float(row.get("online_harm_loss"), 0.0) for row in paired_rows]
    static_alarm = [as_float(row.get("static_alarm"), 0.0) for row in paired_rows]
    online_alarm = [as_float(row.get("online_alarm"), 0.0) for row in paired_rows]

    run_ci_low, run_ci_high = bootstrap_mean_ci(diffs, n_bootstrap=n_bootstrap, rng=rng)

    by_provider: Dict[str, List[float]] = {}
    for row in paired_rows:
        key = str(row.get("provider") or "unknown")
        by_provider.setdefault(key, []).append(
            as_float(row.get("online_minus_static_mcc"), 0.0)
        )
    provider_ci_low, provider_ci_high = bootstrap_cluster_mean_ci(
        by_provider,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    provider_means = {key: mean(values) for key, values in by_provider.items()}
    worst_provider = ""
    worst_provider_mean = 0.0
    if provider_means:
        worst_provider, worst_provider_mean = min(
            provider_means.items(), key=lambda item: item[1]
        )

    sign_stats = sign_test_two_sided(diffs)

    return {
        "config": config_name,
        "n_pairs": float(len(paired_rows)),
        "n_providers": float(len(by_provider)),
        "mean_static": mean(static_values),
        "mean_online": mean(online_values),
        "mean_online_minus_static": mean(diffs),
        "run_bootstrap_ci_low": run_ci_low,
        "run_bootstrap_ci_high": run_ci_high,
        "provider_bootstrap_ci_low": provider_ci_low,
        "provider_bootstrap_ci_high": provider_ci_high,
        "sign_n": sign_stats["n"],
        "sign_positives": sign_stats["positives"],
        "sign_negatives": sign_stats["negatives"],
        "sign_zeros": sign_stats["zeros"],
        "sign_p_value": sign_stats["p_value"],
        "worst_provider": worst_provider,
        "worst_provider_mean_diff": worst_provider_mean,
        "provider_non_degrade_pass": 1.0 if worst_provider_mean >= 0.0 else 0.0,
        "mean_static_harm": mean(static_harm),
        "mean_online_harm": mean(online_harm),
        "mean_harm_delta": mean(online_harm) - mean(static_harm),
        "mean_static_alarm": mean(static_alarm),
        "mean_online_alarm": mean(online_alarm),
        "mean_alarm_delta": mean(online_alarm) - mean(static_alarm),
        "safety_weighted_score": mean(diffs)
        - (0.5 * (mean(online_harm) - mean(static_harm)))
        - (0.25 * (mean(online_alarm) - mean(static_alarm))),
        "strict_pass": 1.0
        if (provider_ci_low >= 0.0 and worst_provider_mean >= 0.0)
        else 0.0,
    }


def summarize_lopo(
    config_name: str,
    paired_rows: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    providers = sorted({str(row.get("provider") or "unknown") for row in paired_rows})
    rows: List[Dict[str, Any]] = []
    for holdout_provider in providers:
        kept = [
            row
            for row in paired_rows
            if str(row.get("provider") or "unknown") != holdout_provider
        ]
        if not kept:
            continue
        summary = summarize_config(
            config_name=config_name,
            paired_rows=kept,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        summary["holdout_provider"] = holdout_provider
        rows.append(summary)
    rows.sort(
        key=lambda row: (
            row["config"],
            str(row.get("holdout_provider") or ""),
        )
    )
    return rows


def summarize_temporal_prefix_suffix(
    config_name: str,
    paired_rows: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    indexed = [
        row
        for row in paired_rows
        if as_float(row.get("provider_run_index"), -1.0) >= 0.0
    ]
    if not indexed:
        return []
    max_index = int(
        max(as_float(row.get("provider_run_index"), 0.0) for row in indexed)
    )
    out: List[Dict[str, Any]] = []
    for cut in range(0, max_index + 1):
        prefix_rows = [
            row
            for row in indexed
            if as_float(row.get("provider_run_index"), -1.0) <= float(cut)
        ]
        suffix_rows = [
            row
            for row in indexed
            if as_float(row.get("provider_run_index"), -1.0) > float(cut)
        ]

        if prefix_rows:
            prefix_summary = summarize_config(
                config_name=config_name,
                paired_rows=prefix_rows,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            prefix_summary["window"] = f"prefix_le_{cut}"
            prefix_summary["cut_index"] = float(cut)
            out.append(prefix_summary)

        if suffix_rows:
            suffix_summary = summarize_config(
                config_name=config_name,
                paired_rows=suffix_rows,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            suffix_summary["window"] = f"suffix_gt_{cut}"
            suffix_summary["cut_index"] = float(cut)
            out.append(suffix_summary)

    out.sort(
        key=lambda row: (
            row["config"],
            as_float(row.get("cut_index"), 0.0),
            str(row.get("window") or ""),
        )
    )
    return out


def write_csv(
    path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(
    out_path: Path,
    rows: Sequence[Dict[str, Any]],
    lopo_rows: Sequence[Dict[str, Any]],
    temporal_rows: Sequence[Dict[str, Any]],
    seed: int,
    n_bootstrap: int,
) -> None:
    by_mean = sorted(
        rows,
        key=lambda row: (
            as_float(row.get("mean_online_minus_static"), 0.0),
            -as_float(row.get("mean_harm_delta"), 0.0),
            -as_float(row.get("mean_alarm_delta"), 0.0),
        ),
        reverse=True,
    )
    by_safety = sorted(
        rows,
        key=lambda row: as_float(row.get("safety_weighted_score"), 0.0),
        reverse=True,
    )
    strict_pass = [row for row in rows if as_float(row.get("strict_pass"), 0.0) > 0.0]

    lines = [
        "# M6 Strict Evaluation",
        "",
        "## Setup",
        "",
        f"- Configs analyzed: {len(rows)}",
        f"- Paired bootstrap draws: {n_bootstrap}",
        f"- Random seed: {seed}",
        "- Primary effect: online - static for policy-minus-baseline delta MCC",
        "- Slice check: worst provider mean difference (non-degrade if >= 0)",
        "",
        "## Top by mean effect",
        "",
        "| Config | Mean diff | Run CI | Provider CI | Worst provider mean | Alarm delta | Harm delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_mean[:5]:
        lines.append(
            f"| {row['config']} | {row['mean_online_minus_static']:+.6f} | "
            f"[{row['run_bootstrap_ci_low']:+.6f}, {row['run_bootstrap_ci_high']:+.6f}] | "
            f"[{row['provider_bootstrap_ci_low']:+.6f}, {row['provider_bootstrap_ci_high']:+.6f}] | "
            f"{row['worst_provider_mean_diff']:+.6f} | "
            f"{row['mean_alarm_delta']:+.6f} | {row['mean_harm_delta']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Top by safety-weighted score",
            "",
            "| Config | Safety score | Mean diff | Alarm delta | Harm delta | Provider non-degrade |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_safety[:5]:
        lines.append(
            f"| {row['config']} | {row['safety_weighted_score']:+.6f} | "
            f"{row['mean_online_minus_static']:+.6f} | "
            f"{row['mean_alarm_delta']:+.6f} | "
            f"{row['mean_harm_delta']:+.6f} | "
            f"{int(as_float(row['provider_non_degrade_pass']))} |"
        )

    lines.extend(
        [
            "",
            "## Strict pass status",
            "",
            "- Criterion: provider-bootstrap CI low >= 0 and worst provider mean >= 0.",
            f"- Passing configs: {len(strict_pass)}",
        ]
    )
    if strict_pass:
        for row in strict_pass:
            lines.append(f"- {row['config']}")
    else:
        lines.append("- (none)")

    if lopo_rows:
        lopo_worst = min(
            lopo_rows,
            key=lambda row: as_float(row.get("mean_online_minus_static"), 0.0),
        )
        lines.extend(
            [
                "",
                "## Leave-One-Provider-Out",
                "",
                "- Summary: each row re-evaluates a config after removing one provider.",
                f"- Rows: {len(lopo_rows)}",
                "- Worst case: "
                f"{lopo_worst['config']} (holdout={lopo_worst['holdout_provider']}) "
                f"mean diff {as_float(lopo_worst['mean_online_minus_static'], 0.0):+.6f}",
            ]
        )

    if temporal_rows:
        temporal_worst = min(
            temporal_rows,
            key=lambda row: as_float(row.get("mean_online_minus_static"), 0.0),
        )
        lines.extend(
            [
                "",
                "## Temporal Slices",
                "",
                "- Summary: per-provider prefix/suffix windows by run order index.",
                f"- Rows: {len(temporal_rows)}",
                "- Worst slice: "
                f"{temporal_worst['config']} {temporal_worst['window']} "
                f"mean diff {as_float(temporal_worst['mean_online_minus_static'], 0.0):+.6f}",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_config_dirs(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "cycle" / "m5_cycle_manifest.json").exists():
            yield entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze M6 strict evaluation metrics")
    parser.add_argument(
        "--grid-root",
        default="workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-lopo-csv", default=None)
    parser.add_argument("--out-temporal-csv", default=None)
    args = parser.parse_args()

    root = Path(args.grid_root)
    if not root.exists():
        raise RuntimeError(f"missing grid root: {root}")

    out_csv = Path(args.out_csv) if args.out_csv else root / "m6_online_strict_eval.csv"
    out_md = Path(args.out_md) if args.out_md else root / "M6_STRICT_EVAL.md"
    out_lopo_csv = (
        Path(args.out_lopo_csv)
        if args.out_lopo_csv
        else root / "m6_online_strict_eval_lopo.csv"
    )
    out_temporal_csv = (
        Path(args.out_temporal_csv)
        if args.out_temporal_csv
        else root / "m6_online_temporal_eval.csv"
    )

    rng = random.Random(int(args.seed))
    rows: List[Dict[str, Any]] = []
    lopo_rows: List[Dict[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []
    for config_dir in iter_config_dirs(root):
        paired_rows = load_paired_rows_for_config(config_dir)
        if not paired_rows:
            continue
        lopo_rows.extend(
            summarize_lopo(
                config_name=config_dir.name,
                paired_rows=paired_rows,
                n_bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )
        temporal_rows.extend(
            summarize_temporal_prefix_suffix(
                config_name=config_dir.name,
                paired_rows=paired_rows,
                n_bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )
        rows.append(
            summarize_config(
                config_name=config_dir.name,
                paired_rows=paired_rows,
                n_bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )

    rows.sort(
        key=lambda row: (
            as_float(row.get("mean_online_minus_static"), 0.0),
            as_float(row.get("provider_bootstrap_ci_low"), 0.0),
            -as_float(row.get("mean_alarm_delta"), 0.0),
            -as_float(row.get("mean_harm_delta"), 0.0),
        ),
        reverse=True,
    )

    fieldnames = [
        "config",
        "n_pairs",
        "n_providers",
        "mean_static",
        "mean_online",
        "mean_online_minus_static",
        "run_bootstrap_ci_low",
        "run_bootstrap_ci_high",
        "provider_bootstrap_ci_low",
        "provider_bootstrap_ci_high",
        "sign_n",
        "sign_positives",
        "sign_negatives",
        "sign_zeros",
        "sign_p_value",
        "worst_provider",
        "worst_provider_mean_diff",
        "provider_non_degrade_pass",
        "mean_static_harm",
        "mean_online_harm",
        "mean_harm_delta",
        "mean_static_alarm",
        "mean_online_alarm",
        "mean_alarm_delta",
        "safety_weighted_score",
        "strict_pass",
    ]
    write_csv(out_csv, rows, fieldnames=fieldnames)

    lopo_fieldnames = ["holdout_provider", *fieldnames]
    if lopo_rows:
        write_csv(out_lopo_csv, lopo_rows, fieldnames=lopo_fieldnames)
    else:
        write_csv(out_lopo_csv, [], fieldnames=lopo_fieldnames)

    temporal_fieldnames = ["window", "cut_index", *fieldnames]
    if temporal_rows:
        write_csv(out_temporal_csv, temporal_rows, fieldnames=temporal_fieldnames)
    else:
        write_csv(out_temporal_csv, [], fieldnames=temporal_fieldnames)

    build_report(
        out_path=out_md,
        rows=rows,
        lopo_rows=lopo_rows,
        temporal_rows=temporal_rows,
        seed=int(args.seed),
        n_bootstrap=int(args.bootstrap),
    )

    print("M6 strict evaluation complete")
    print(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "grid_root": str(root),
                "configs_analyzed": len(rows),
                "bootstrap": int(args.bootstrap),
                "seed": int(args.seed),
                "out_csv": str(out_csv),
                "out_md": str(out_md),
                "out_lopo_csv": str(out_lopo_csv),
                "out_temporal_csv": str(out_temporal_csv),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
