#!/usr/bin/env python3
"""Run M4 selective margin-guardrail experiments on CARE-v3 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chaosbench.repair.engine import (
    compute_axiom_violation_rate,
    load_selector_index,
    read_jsonl,
    repair_records,
)
from chaosbench.repair.selective import (
    apply_margin_policy,
    collect_flip_candidates,
    fit_margin_policy,
    policy_map,
)
from chaosbench.repair.types import RepairConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_BINARY_LABELS = {"TRUE", "FALSE"}


@dataclass
class RunPack:
    run_id: str
    provider: str
    split: str
    records: List[Dict[str, Any]]
    baseline_records: List[Dict[str, Any]]
    baseline_row_flip_rate: float
    candidates: List[Any]


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metrics_from_records(
    records: Sequence[Dict[str, Any]], label_key: str
) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    total = len(records)
    valid = 0
    pred_true = 0
    for row in records:
        gt = row.get("ground_truth")
        pred = row.get(label_key)
        if gt not in VALID_BINARY_LABELS:
            continue
        if pred not in VALID_BINARY_LABELS:
            continue
        valid += 1
        if pred == "TRUE":
            pred_true += 1
        if gt == "TRUE" and pred == "TRUE":
            tp += 1
        elif gt == "FALSE" and pred == "FALSE":
            tn += 1
        elif gt == "FALSE" and pred == "TRUE":
            fp += 1
        elif gt == "TRUE" and pred == "FALSE":
            fn += 1

    coverage = (valid / total) if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    ba = (tpr + tnr) / 2.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "total": float(total),
        "valid": float(valid),
        "coverage": float(coverage),
        "mcc": float(mcc),
        "balanced_accuracy": float(ba),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "pred_true_pct": (float(pred_true / valid) if valid else 0.0),
    }


def sign_test_two_sided(values: Sequence[float]) -> Dict[str, float]:
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    n = positives + negatives
    if n == 0:
        return {"n": 0.0, "positives": 0.0, "negatives": 0.0, "p_value": 1.0}
    tail = min(positives, negatives)
    cumulative = 0.0
    for i in range(tail + 1):
        cumulative += math.comb(n, i) * (0.5**n)
    p_value = min(1.0, 2.0 * cumulative)
    return {
        "n": float(n),
        "positives": float(positives),
        "negatives": float(negatives),
        "p_value": float(p_value),
    }


def load_frozen_config(path: Path) -> RepairConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    return config_from_dict(cfg)


def config_from_dict(cfg: Dict[str, Any]) -> RepairConfig:
    return RepairConfig(
        name=str(cfg["name"]),
        gate_families=tuple(cfg["gate_families"]) if cfg.get("gate_families") else None,
        extractor_strategy=str(cfg["extractor_strategy"]),
        polarity_mode=str(cfg["polarity_mode"]),
        leave_invalid_unchanged=bool(cfg["leave_invalid_unchanged"]),
        enable_group_consistency=bool(cfg["enable_group_consistency"]),
        seed=int(cfg.get("seed", 42)),
    )


def load_repair_config(repair_dir: Path) -> RepairConfig:
    frozen_path = repair_dir / "frozen_config.json"
    if frozen_path.exists():
        return load_frozen_config(frozen_path)

    for repaired_runs_dir in (repair_dir / "repair_runs", repair_dir / "runs"):
        if not repaired_runs_dir.exists():
            continue
        for run_dir in sorted(repaired_runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "repair_manifest.json"
            if not manifest_path.exists():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            cfg = payload.get("config")
            if isinstance(cfg, dict):
                return config_from_dict(cfg)

    raise FileNotFoundError(
        f"Unable to locate repair config under {repair_dir} "
        "(expected frozen_config.json or per-run repair_manifest.json)"
    )


def load_split_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_id = str(row.get("run_id") or "")
            split = str(row.get("split") or "unknown")
            if run_id:
                out[run_id] = split
    return out


def load_runs(
    repair_dir: Path,
    split_map: Dict[str, str],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
    default_split: str = "unknown",
) -> List[RunPack]:
    runs: List[RunPack] = []
    repaired_runs_dir = repair_dir / "repair_runs"
    if not repaired_runs_dir.exists():
        repaired_runs_dir = repair_dir / "runs"

    raw_run_roots = [repair_dir / "runs", PROJECT_ROOT / "runs"]
    if not repaired_runs_dir.exists():
        return runs

    for repaired_run_dir in sorted(repaired_runs_dir.iterdir()):
        if not repaired_run_dir.is_dir():
            continue

        repair_manifest_path = repaired_run_dir / "repair_manifest.json"
        repaired_predictions_path = repaired_run_dir / "repaired_predictions.jsonl"
        if not repair_manifest_path.exists() or not repaired_predictions_path.exists():
            continue

        repair_manifest = json.loads(repair_manifest_path.read_text(encoding="utf-8"))
        run_id = str(repair_manifest.get("run_id") or repaired_run_dir.name)
        raw_run_dir: Optional[Path] = None
        for root in raw_run_roots:
            candidate = root / run_id
            if candidate.exists():
                raw_run_dir = candidate
                break

        predictions_path = (raw_run_dir / "predictions.jsonl") if raw_run_dir else None
        run_manifest_path = (raw_run_dir / "run_manifest.json") if raw_run_dir else None

        provider = str(repair_manifest.get("provider", "unknown"))
        if run_manifest_path and run_manifest_path.exists():
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            provider = str(run_manifest.get("provider", provider))
        split = split_map.get(run_id, default_split)

        if predictions_path and predictions_path.exists():
            records = read_jsonl(predictions_path)
        else:
            records = read_jsonl(repaired_predictions_path)
        baseline_records = read_jsonl(repaired_predictions_path)
        if len(baseline_records) != len(records):
            baseline = repair_records(
                records=records, id_to_meta=id_to_meta, config=config
            )
            baseline_records = baseline.records

        candidates = collect_flip_candidates(
            records=records, id_to_meta=id_to_meta, config=config
        )
        baseline_valid = sum(
            1
            for row in baseline_records
            if row.get("parsed_label") in VALID_BINARY_LABELS
        )
        baseline_flips = sum(
            1 for row in baseline_records if row.get("was_flipped") is True
        )
        baseline_row_flip_rate = (
            (baseline_flips / baseline_valid) if baseline_valid else 0.0
        )

        runs.append(
            RunPack(
                run_id=run_id,
                provider=provider,
                split=split,
                records=records,
                baseline_records=baseline_records,
                baseline_row_flip_rate=float(baseline_row_flip_rate),
                candidates=candidates,
            )
        )

    return runs


def load_policy_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, Any] = {"family": str(row.get("family") or "")}
            for key, value in row.items():
                if key == "family":
                    continue
                parsed[key] = as_float(value, 0.0)
            if parsed["family"]:
                rows.append(parsed)
    return rows


def build_report(
    out_path: Path,
    policy_rows: List[Dict[str, Any]],
    run_rows: List[Dict[str, Any]],
    heldout_sign: Dict[str, float],
) -> None:
    heldout = [row for row in run_rows if row["split"] == "heldout"]
    dev = [row for row in run_rows if row["split"] == "dev"]

    def mean_col(rows: List[Dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return statistics.mean(as_float(row.get(key), 0.0) for row in rows)

    lines = [
        "# M4 Selective Guardrail Results",
        "",
        "## Policy fit (dev)",
        "",
        "| Family | Enabled | Threshold | Candidates | Accepted | Accepted net | Total net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in policy_rows:
        lines.append(
            f"| {row['family']} | {int(row['enabled'])} | {row['threshold']:.3f} | "
            f"{int(row['n_candidates'])} | {int(row['n_accepted'])} | "
            f"{row['accepted_net']:+.0f} | {row['total_net']:+.0f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Dev mean delta MCC (baseline): {mean_col(dev, 'delta_mcc_baseline'):+.4f}",
            f"- Dev mean delta MCC (policy): {mean_col(dev, 'delta_mcc_policy'):+.4f}",
            f"- Held-out mean delta MCC (baseline): {mean_col(heldout, 'delta_mcc_baseline'):+.4f}",
            f"- Held-out mean delta MCC (policy): {mean_col(heldout, 'delta_mcc_policy'):+.4f}",
            f"- Held-out mean (policy - baseline) delta MCC: {mean_col(heldout, 'policy_minus_baseline_mcc'):+.4f}",
            f"- Held-out sign test (policy - baseline): positives={int(heldout_sign['positives'])}/{int(heldout_sign['n'])}, p={heldout_sign['p_value']:.6f}",
            "",
            "## Per-run",
            "",
            "| Run ID | Provider | Split | Delta MCC baseline | Delta MCC policy | Policy - baseline | Flip rate baseline | Flip rate policy |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for row in run_rows:
        lines.append(
            f"| {row['run_id']} | {row['provider']} | {row['split']} | "
            f"{row['delta_mcc_baseline']:+.4f} | {row['delta_mcc_policy']:+.4f} | "
            f"{row['policy_minus_baseline_mcc']:+.4f} | "
            f"{row['row_flip_rate_baseline']:.4f} | {row['row_flip_rate_policy']:.4f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run M4 selective margin-guardrail suite"
    )
    parser.add_argument(
        "--repair-dir",
        default=str(
            PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "repair_v3"
        ),
    )
    parser.add_argument(
        "--split-map-csv",
        default=None,
        help="Optional CSV mapping run_id->split (defaults to <repair-dir>/tables/repair_deltas.csv)",
    )
    parser.add_argument(
        "--default-split",
        default="unknown",
        help="Split label used when run_id missing from split map",
    )
    parser.add_argument(
        "--policy-csv",
        default=None,
        help="Optional pre-fit family policy CSV; if provided, skip dev policy fitting",
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "workspace"
            / "deep_survey_2026-03-01"
            / "repair_v3"
            / "m4_selective"
        ),
    )
    parser.add_argument("--selector", default="data/canonical_v2_files.json")
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--min-family-samples", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument(
        "--degrade-penalty",
        type=float,
        default=1.0,
        help="Penalty multiplier on degraded flips during policy fitting",
    )
    args = parser.parse_args()

    repair_dir = Path(args.repair_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_repair_config(repair_dir)
    id_to_meta = load_selector_index(
        selector_path=Path(args.selector),
        project_root=PROJECT_ROOT,
    )
    split_map_path = (
        Path(args.split_map_csv)
        if args.split_map_csv
        else repair_dir / "tables" / "repair_deltas.csv"
    )
    split_map = load_split_map(split_map_path)

    runs = load_runs(
        repair_dir=repair_dir,
        split_map=split_map,
        id_to_meta=id_to_meta,
        config=config,
        default_split=str(args.default_split),
    )
    if not runs:
        raise RuntimeError("No runs discovered under repair_dir/{repair_runs,runs}")

    if args.policy_csv:
        policy_rows = load_policy_rows(Path(args.policy_csv))
        if not policy_rows:
            raise RuntimeError(f"No policy rows found in {args.policy_csv}")
    else:
        dev_candidates = [
            candidate
            for run in runs
            if run.split == "dev"
            for candidate in run.candidates
        ]
        if not dev_candidates:
            raise RuntimeError(
                "No dev candidates available to fit policy. "
                "Provide --policy-csv or a split map with dev runs."
            )

        policy_rows = fit_margin_policy(
            candidates=dev_candidates,
            threshold_step=args.threshold_step,
            min_family_samples=args.min_family_samples,
            min_support=args.min_support,
            degrade_penalty=args.degrade_penalty,
        )

    policy_min_support_values = sorted(
        {
            int(round(as_float(row.get("min_support"), float(args.min_support))))
            for row in policy_rows
        }
    )
    per_family_policy = policy_map(policy_rows)

    run_rows: List[Dict[str, Any]] = []
    for run in runs:
        pre_metrics = metrics_from_records(run.records, "parsed_label")
        baseline_metrics = metrics_from_records(run.baseline_records, "repaired_label")

        policy_records, policy_stats = apply_margin_policy(
            repaired_records=run.baseline_records,
            candidates=run.candidates,
            per_family_policy=per_family_policy,
        )
        policy_metrics = metrics_from_records(policy_records, "repaired_label")

        pre_v_count, pre_v_rate = compute_axiom_violation_rate(
            records=run.records,
            label_key="parsed_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )
        baseline_v_count, baseline_v_rate = compute_axiom_violation_rate(
            records=run.baseline_records,
            label_key="repaired_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )
        policy_v_count, policy_v_rate = compute_axiom_violation_rate(
            records=policy_records,
            label_key="repaired_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )

        policy_row_flips = sum(
            1 for row in policy_records if row.get("was_flipped") is True
        )
        policy_row_flip_rate = (
            policy_row_flips / policy_metrics["valid"]
            if policy_metrics["valid"]
            else 0.0
        )

        delta_mcc_baseline = baseline_metrics["mcc"] - pre_metrics["mcc"]
        delta_mcc_policy = policy_metrics["mcc"] - pre_metrics["mcc"]

        run_rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "n_items": pre_metrics["total"],
                "delta_mcc_baseline": delta_mcc_baseline,
                "delta_mcc_policy": delta_mcc_policy,
                "policy_minus_baseline_mcc": delta_mcc_policy - delta_mcc_baseline,
                "delta_ba_baseline": baseline_metrics["balanced_accuracy"]
                - pre_metrics["balanced_accuracy"],
                "delta_ba_policy": policy_metrics["balanced_accuracy"]
                - pre_metrics["balanced_accuracy"],
                "row_flip_rate_baseline": run.baseline_row_flip_rate,
                "row_flip_rate_policy": policy_row_flip_rate,
                "veto_rate": policy_stats["veto_rate"],
                "kept_flips": policy_stats["kept_flips"],
                "vetoed_flips": policy_stats["vetoed_flips"],
                "pre_axiom_violation_rate": pre_v_rate,
                "baseline_axiom_violation_rate": baseline_v_rate,
                "policy_axiom_violation_rate": policy_v_rate,
                "baseline_axiom_violations": float(pre_v_count - baseline_v_count),
                "policy_axiom_violations": float(pre_v_count - policy_v_count),
            }
        )

    heldout_improvements = [
        as_float(row["policy_minus_baseline_mcc"])
        for row in run_rows
        if row["split"] == "heldout"
    ]
    heldout_sign = sign_test_two_sided(heldout_improvements)

    write_csv(
        out_dir / "m4_family_policy.csv",
        policy_rows,
        fieldnames=[
            "family",
            "enabled",
            "threshold",
            "min_support",
            "n_candidates",
            "n_accepted",
            "accept_rate",
            "accepted_improved",
            "accepted_degraded",
            "accepted_net",
            "total_improved",
            "total_degraded",
            "total_net",
        ],
    )
    write_csv(
        out_dir / "m4_run_deltas.csv",
        run_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "n_items",
            "delta_mcc_baseline",
            "delta_mcc_policy",
            "policy_minus_baseline_mcc",
            "delta_ba_baseline",
            "delta_ba_policy",
            "row_flip_rate_baseline",
            "row_flip_rate_policy",
            "veto_rate",
            "kept_flips",
            "vetoed_flips",
            "pre_axiom_violation_rate",
            "baseline_axiom_violation_rate",
            "policy_axiom_violation_rate",
            "baseline_axiom_violations",
            "policy_axiom_violations",
        ],
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repair_dir": str(repair_dir),
        "split_map_csv": str(split_map_path),
        "default_split": str(args.default_split),
        "policy_csv": str(args.policy_csv) if args.policy_csv else None,
        "policy_min_support_values": policy_min_support_values,
        "n_runs": len(runs),
        "n_dev_runs": sum(1 for run in runs if run.split == "dev"),
        "n_heldout_runs": sum(1 for run in runs if run.split == "heldout"),
        "threshold_step": args.threshold_step,
        "min_family_samples": args.min_family_samples,
        "min_support": args.min_support,
        "degrade_penalty": args.degrade_penalty,
        "heldout_sign_test_policy_minus_baseline": heldout_sign,
        "config": config.to_dict(),
    }
    (out_dir / "m4_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    build_report(
        out_path=out_dir / "M4_RESULTS.md",
        policy_rows=policy_rows,
        run_rows=run_rows,
        heldout_sign=heldout_sign,
    )

    print("M4 selective guardrail complete")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
