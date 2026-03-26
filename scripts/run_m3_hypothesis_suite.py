#!/usr/bin/env python3
"""Run M3 hypothesis suite for CARE-v3 using existing canonical runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.repair.engine import (
    compute_axiom_violation_rate,
    load_selector_index,
    read_jsonl,
    repair_records,
    write_jsonl,
)
from chaosbench.repair.types import RepairConfig


@dataclass
class RunMeta:
    run_id: str
    provider: str
    run_dir: Path
    total_items: int
    prompt_hash: str
    dataset_sha: str


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sign_test_two_sided(deltas: Sequence[float]) -> Dict[str, float]:
    positives = sum(1 for value in deltas if value > 0)
    negatives = sum(1 for value in deltas if value < 0)
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


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def discover_canonical_runs(
    runs_dir: Path,
    selector_rel: str,
    min_items: int,
) -> List[RunMeta]:
    runs: List[RunMeta] = []
    for manifest_path in sorted(runs_dir.glob("**/run_manifest.json")):
        if "_archive_excluded" in str(manifest_path):
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total = int(manifest.get("total_items_evaluated", 0) or 0)
        selector = str(manifest.get("canonical_selector", ""))
        if total < min_items or selector != selector_rel:
            continue

        run_dir = manifest_path.parent
        if not (run_dir / "predictions.jsonl").exists():
            continue

        runs.append(
            RunMeta(
                run_id=str(manifest.get("run_id", run_dir.name)),
                provider=str(manifest.get("provider", "")),
                run_dir=run_dir,
                total_items=total,
                prompt_hash=str(manifest.get("prompt_hash", "")),
                dataset_sha=str(manifest.get("dataset_global_sha256", "")),
            )
        )
    return runs


def select_provider_max_runs(runs: Sequence[RunMeta]) -> List[RunMeta]:
    best: Dict[str, RunMeta] = {}
    for run in runs:
        current = best.get(run.provider)
        if current is None:
            best[run.provider] = run
            continue
        if run.total_items > current.total_items:
            best[run.provider] = run
            continue
        if run.total_items == current.total_items and run.run_id > current.run_id:
            best[run.provider] = run
    return sorted(best.values(), key=lambda item: item.provider)


def load_ids_from_jsonl(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id") or row.get("item_id")
            if row_id:
                ids.append(str(row_id))
    return ids


def build_panels(subsets_dir: Path) -> Dict[str, set[str]]:
    panels: Dict[str, set[str]] = {}

    split1 = set(load_ids_from_jsonl(subsets_dir / "canonical_split1.jsonl"))
    split2 = set(load_ids_from_jsonl(subsets_dir / "canonical_split2.jsonl"))
    panels["split1"] = split1
    panels["split2"] = split2

    fam_dir = subsets_dir / "subset_family_suites"
    fam_ids = {
        "adversarial": set(load_ids_from_jsonl(fam_dir / "adversarial.jsonl")),
        "atomic": set(load_ids_from_jsonl(fam_dir / "atomic.jsonl")),
        "consistency_paraphrase": set(
            load_ids_from_jsonl(fam_dir / "consistency_paraphrase.jsonl")
        ),
        "cross_indicator": set(load_ids_from_jsonl(fam_dir / "cross_indicator.jsonl")),
        "extended_systems": set(
            load_ids_from_jsonl(fam_dir / "extended_systems.jsonl")
        ),
        "fol_inference": set(load_ids_from_jsonl(fam_dir / "fol_inference.jsonl")),
        "indicator_diagnostics": set(
            load_ids_from_jsonl(fam_dir / "indicator_diagnostics.jsonl")
        ),
        "multi_hop": set(load_ids_from_jsonl(fam_dir / "multi_hop.jsonl")),
        "perturbation_robustness": set(
            load_ids_from_jsonl(fam_dir / "perturbation_robustness.jsonl")
        ),
        "regime_transition": set(
            load_ids_from_jsonl(fam_dir / "regime_transition.jsonl")
        ),
    }

    panels["stress_adv_perturb"] = (
        fam_ids["adversarial"] | fam_ids["perturbation_robustness"]
    )
    panels["stress_long_tail"] = (
        fam_ids["cross_indicator"]
        | fam_ids["extended_systems"]
        | fam_ids["regime_transition"]
        | fam_ids["indicator_diagnostics"]
    )
    panels["stress_compositional"] = (
        fam_ids["multi_hop"]
        | fam_ids["fol_inference"]
        | fam_ids["consistency_paraphrase"]
    )
    panels["stress_atomic"] = fam_ids["atomic"]
    panels["stress_adversarial_only"] = fam_ids["adversarial"]
    panels["stress_perturb_only"] = fam_ids["perturbation_robustness"]

    return panels


def metrics_from_records(
    records: Sequence[Dict[str, Any]], label_key: str
) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    valid = 0
    total = len(records)

    for row in records:
        gt = row.get("ground_truth")
        pred = row.get(label_key)
        if gt not in {"TRUE", "FALSE"}:
            continue
        if pred not in {"TRUE", "FALSE"}:
            continue
        valid += 1
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
    }


def subset_records_by_ids(
    records: Sequence[Dict[str, Any]], id_set: set[str]
) -> List[Dict[str, Any]]:
    return [
        row for row in records if str(row.get("id", row.get("item_id", ""))) in id_set
    ]


def load_frozen_config(path: Path) -> RepairConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    return RepairConfig(
        name=str(cfg["name"]),
        gate_families=tuple(cfg["gate_families"]) if cfg.get("gate_families") else None,
        extractor_strategy=str(cfg["extractor_strategy"]),
        polarity_mode=str(cfg["polarity_mode"]),
        leave_invalid_unchanged=bool(cfg["leave_invalid_unchanged"]),
        enable_group_consistency=bool(cfg["enable_group_consistency"]),
        seed=int(cfg.get("seed", 42)),
    )


def evaluate_run_with_cache(
    run: RunMeta,
    repair_dir: Path,
    out_runs_dir: Path,
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
) -> List[Dict[str, Any]]:
    cached = repair_dir / "runs" / run.run_id / "repaired_predictions.jsonl"
    if cached.exists():
        return read_jsonl(cached)

    local_cached = out_runs_dir / run.run_id / "repaired_predictions.jsonl"
    if local_cached.exists():
        return read_jsonl(local_cached)

    records = read_jsonl(run.run_dir / "predictions.jsonl")
    repaired = repair_records(records=records, id_to_meta=id_to_meta, config=config)

    out_path = out_runs_dir / run.run_id / "repaired_predictions.jsonl"
    write_jsonl(out_path, repaired.records)
    return repaired.records


def pass_fail(
    positives: int,
    total: int,
    p_value: float,
    min_positive_rate: float,
    alpha: float,
) -> str:
    if total <= 0:
        return "INCONCLUSIVE"
    if (positives / total) >= min_positive_rate and p_value <= alpha:
        return "PASS"
    return "FAIL"


def neutral_pass(deltas: Sequence[float], mean_abs_tol: float) -> str:
    if not deltas:
        return "INCONCLUSIVE"
    mean_abs = statistics.mean(abs(value) for value in deltas)
    if mean_abs <= mean_abs_tol:
        return "PASS"
    return "FAIL"


def build_report(
    out_path: Path,
    global_rows_provider: List[Dict[str, Any]],
    panel_provider_rows: List[Dict[str, Any]],
    tests: Dict[str, Dict[str, Any]],
) -> None:
    lines: List[str] = [
        "# M3 Hypothesis Suite Results",
        "",
        "## Hypothesis tests",
        "",
        "| Hypothesis | Panel | Positives | Total | p-value | Status |",
        "|---|---|---:|---:|---:|---|",
    ]

    for key in sorted(tests.keys()):
        row = tests[key]
        lines.append(
            f"| {row['hypothesis']} | {row['panel']} | {int(row['positives'])} | {int(row['total'])} | {row['p_value']:.6f} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Provider-max global deltas",
            "",
            "| Provider | Total items | MCC pre | MCC post | Delta MCC | Violation reduction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(global_rows_provider, key=lambda item: item["provider"]):
        lines.append(
            f"| {row['provider']} | {int(row['total_items'])} | {row['mcc_pre']:.4f} | {row['mcc_post']:.4f} | {row['delta_mcc']:+.4f} | {row['axiom_violation_reduction']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Stress panel means (provider-max)",
            "",
            "| Panel | Mean delta MCC | Positive providers |",
            "|---|---:|---:|",
        ]
    )
    by_panel: Dict[str, List[float]] = {}
    by_panel_pos: Dict[str, int] = {}
    for row in panel_provider_rows:
        panel = str(row["panel"])
        by_panel.setdefault(panel, []).append(float(row["delta_mcc"]))
        by_panel_pos[panel] = by_panel_pos.get(panel, 0) + (
            1 if float(row["delta_mcc"]) > 0 else 0
        )

    for panel in sorted(by_panel.keys()):
        vals = by_panel[panel]
        lines.append(
            f"| {panel} | {statistics.mean(vals):+.4f} | {by_panel_pos[panel]}/{len(vals)} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M3 hypothesis suite")
    parser.add_argument(
        "--repair-dir",
        default=str(
            PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "repair_v3"
        ),
    )
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--selector", default="data/canonical_v2_files.json")
    parser.add_argument("--subsets-dir", default=str(PROJECT_ROOT / "data" / "subsets"))
    parser.add_argument("--min-items", type=int, default=5000)
    parser.add_argument("--out-subdir", default="m3")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-positive-rate", type=float, default=0.75)
    parser.add_argument("--neutral-mean-abs-tol", type=float, default=0.005)
    args = parser.parse_args()

    repair_dir = Path(args.repair_dir)
    out_dir = repair_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_runs_dir = out_dir / "runs"
    out_runs_dir.mkdir(parents=True, exist_ok=True)

    config = load_frozen_config(repair_dir / "frozen_config.json")
    id_to_meta = load_selector_index(
        selector_path=Path(args.selector), project_root=PROJECT_ROOT
    )
    runs = discover_canonical_runs(
        runs_dir=Path(args.runs_dir),
        selector_rel=str(args.selector),
        min_items=args.min_items,
    )
    provider_max = select_provider_max_runs(runs)
    panels = build_panels(Path(args.subsets_dir))

    run_global_rows: List[Dict[str, Any]] = []
    run_panel_rows: List[Dict[str, Any]] = []

    for run in runs:
        records = evaluate_run_with_cache(
            run=run,
            repair_dir=repair_dir,
            out_runs_dir=out_runs_dir,
            id_to_meta=id_to_meta,
            config=config,
        )
        pre = metrics_from_records(records, "parsed_label")
        post = metrics_from_records(records, "repaired_label")
        delta_mcc = post["mcc"] - pre["mcc"]
        delta_ba = post["balanced_accuracy"] - pre["balanced_accuracy"]

        pre_v_count, pre_v_rate = compute_axiom_violation_rate(
            records=records,
            label_key="parsed_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )
        post_v_count, post_v_rate = compute_axiom_violation_rate(
            records=records,
            label_key="repaired_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )

        run_global_rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "total_items": float(run.total_items),
                "prompt_hash": run.prompt_hash,
                "mcc_pre": pre["mcc"],
                "mcc_post": post["mcc"],
                "delta_mcc": delta_mcc,
                "delta_balanced_accuracy": delta_ba,
                "axiom_violation_rate_pre": pre_v_rate,
                "axiom_violation_rate_post": post_v_rate,
                "axiom_violation_reduction": pre_v_rate - post_v_rate,
                "axiom_violations_pre": float(pre_v_count),
                "axiom_violations_post": float(post_v_count),
            }
        )

        for panel_name, id_set in panels.items():
            panel_records = subset_records_by_ids(records, id_set)
            if not panel_records:
                continue
            panel_pre = metrics_from_records(panel_records, "parsed_label")
            panel_post = metrics_from_records(panel_records, "repaired_label")
            run_panel_rows.append(
                {
                    "run_id": run.run_id,
                    "provider": run.provider,
                    "total_items": float(run.total_items),
                    "panel": panel_name,
                    "n_items": panel_pre["total"],
                    "mcc_pre": panel_pre["mcc"],
                    "mcc_post": panel_post["mcc"],
                    "delta_mcc": panel_post["mcc"] - panel_pre["mcc"],
                    "ba_pre": panel_pre["balanced_accuracy"],
                    "ba_post": panel_post["balanced_accuracy"],
                    "delta_ba": panel_post["balanced_accuracy"]
                    - panel_pre["balanced_accuracy"],
                }
            )

    write_csv(
        out_dir / "m3_run_global.csv",
        run_global_rows,
        fieldnames=[
            "run_id",
            "provider",
            "total_items",
            "prompt_hash",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "delta_balanced_accuracy",
            "axiom_violation_rate_pre",
            "axiom_violation_rate_post",
            "axiom_violation_reduction",
            "axiom_violations_pre",
            "axiom_violations_post",
        ],
    )
    write_csv(
        out_dir / "m3_run_panel_deltas.csv",
        run_panel_rows,
        fieldnames=[
            "run_id",
            "provider",
            "total_items",
            "panel",
            "n_items",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "ba_pre",
            "ba_post",
            "delta_ba",
        ],
    )

    provider_max_ids = {run.run_id for run in provider_max}
    provider_global_rows = [
        row for row in run_global_rows if row["run_id"] in provider_max_ids
    ]
    provider_panel_rows = [
        row for row in run_panel_rows if row["run_id"] in provider_max_ids
    ]

    write_csv(
        out_dir / "m3_provider_global.csv",
        provider_global_rows,
        fieldnames=[
            "run_id",
            "provider",
            "total_items",
            "prompt_hash",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "delta_balanced_accuracy",
            "axiom_violation_rate_pre",
            "axiom_violation_rate_post",
            "axiom_violation_reduction",
            "axiom_violations_pre",
            "axiom_violations_post",
        ],
    )
    write_csv(
        out_dir / "m3_provider_panel_deltas.csv",
        provider_panel_rows,
        fieldnames=[
            "run_id",
            "provider",
            "total_items",
            "panel",
            "n_items",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "ba_pre",
            "ba_post",
            "delta_ba",
        ],
    )

    tests: Dict[str, Dict[str, Any]] = {}

    global_deltas = [row["delta_mcc"] for row in provider_global_rows]
    global_sign = sign_test_two_sided(global_deltas)
    tests["H1"] = {
        "hypothesis": "H1: Global provider-max gains remain positive",
        "panel": "provider_max_global",
        "positives": global_sign["positives"],
        "total": global_sign["n"],
        "p_value": global_sign["p_value"],
        "status": pass_fail(
            int(global_sign["positives"]),
            int(global_sign["n"]),
            float(global_sign["p_value"]),
            min_positive_rate=args.min_positive_rate,
            alpha=args.alpha,
        ),
    }

    targeted_panels = ["split2", "stress_compositional"]
    for panel_name in targeted_panels:
        panel_rows = [row for row in provider_panel_rows if row["panel"] == panel_name]
        panel_sign = sign_test_two_sided([row["delta_mcc"] for row in panel_rows])
        key = f"H_target_{panel_name}"
        tests[key] = {
            "hypothesis": f"H2: Positive deltas on targeted panel {panel_name}",
            "panel": panel_name,
            "positives": panel_sign["positives"],
            "total": panel_sign["n"],
            "p_value": panel_sign["p_value"],
            "status": pass_fail(
                int(panel_sign["positives"]),
                int(panel_sign["n"]),
                float(panel_sign["p_value"]),
                min_positive_rate=0.60,
                alpha=args.alpha,
            ),
        }

    non_target_panels = ["split1", "stress_adv_perturb", "stress_long_tail"]
    for panel_name in non_target_panels:
        panel_rows = [row for row in provider_panel_rows if row["panel"] == panel_name]
        deltas = [float(row["delta_mcc"]) for row in panel_rows]
        mean_abs = statistics.mean(abs(value) for value in deltas) if deltas else 0.0
        tests[f"H_neutral_{panel_name}"] = {
            "hypothesis": f"H3: Near-neutral deltas on non-target panel {panel_name}",
            "panel": panel_name,
            "positives": float(sum(1 for value in deltas if value > 0.0)),
            "total": float(len(deltas)),
            "p_value": 1.0,
            "status": neutral_pass(deltas, mean_abs_tol=args.neutral_mean_abs_tol),
            "mean_abs_delta": mean_abs,
        }

    split1_map = {
        row["provider"]: row["delta_mcc"]
        for row in provider_panel_rows
        if row["panel"] == "split1"
    }
    split2_map = {
        row["provider"]: row["delta_mcc"]
        for row in provider_panel_rows
        if row["panel"] == "split2"
    }
    common_providers = sorted(set(split1_map.keys()) & set(split2_map.keys()))
    split_corr = correlation(
        [split1_map[p] for p in common_providers],
        [split2_map[p] for p in common_providers],
    )
    if not math.isfinite(split_corr):
        split_corr = 0.0

    baseline_corr = correlation(
        [row["mcc_pre"] for row in provider_global_rows],
        [row["delta_mcc"] for row in provider_global_rows],
    )
    if not math.isfinite(baseline_corr):
        baseline_corr = 0.0
    violation_corr = correlation(
        [row["axiom_violation_reduction"] for row in provider_global_rows],
        [row["delta_mcc"] for row in provider_global_rows],
    )
    if not math.isfinite(violation_corr):
        violation_corr = 0.0

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repair_dir": str(repair_dir),
        "out_dir": str(out_dir),
        "selector": str(args.selector),
        "n_runs_min_items": len(runs),
        "n_provider_max": len(provider_max),
        "global_sign_test_provider_max": global_sign,
        "split_delta_correlation": split_corr,
        "baseline_delta_correlation": baseline_corr,
        "violation_reduction_delta_correlation": violation_corr,
        "neutral_mean_abs_tol": args.neutral_mean_abs_tol,
        "tests": tests,
    }
    (out_dir / "m3_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    build_report(
        out_path=out_dir / "M3_RESULTS.md",
        global_rows_provider=provider_global_rows,
        panel_provider_rows=provider_panel_rows,
        tests=tests,
    )

    print("M3 hypothesis suite complete")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
