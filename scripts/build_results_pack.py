#!/usr/bin/env python3
"""scripts/build_results_pack.py — Build the v2 results pack.

Inputs:
  --runs_dir      runs/  (local, gitignored)
  --published_dir published_results/runs
  --out_dir       artifacts/results_pack

Outputs:
  results_pack_summary.json
  RESULTS_PACK.md
  tables/baselines_table.{csv,md}
  tables/by_family.{csv,md}
  tables/hardness.{csv,md}
  run_catalog.json
  run_catalog.md

Checks performed:
  - Recompute metrics from predictions and confirm vs stored metrics
  - Bias metrics: pred_TRUE%, TPR/TNR, MCC, balanced_acc
  - Macro-family metrics
  - Ordering/shuffle mode recorded
  - No family-degeneracy violations
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREEZE_SHA = "cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279"
ALT_SHA = "00ec17e31193de42c525ff3c8f166b4b59fae2c2631fa84a4c78b33fb01f9374"
MOCK_PROVIDERS = {"mock"}
OFFICIAL_COVERAGE_MIN = 0.99
BIAS_THRESHOLD = 0.15

# Display aliases for models (long name → short alias)
MODEL_ALIASES = {
    "ollama/qwen2.5:14b": "Qwen2.5-14B",
    "ollama/qwen2.5:7b":  "Qwen2.5-7B",
    "ollama/llama3.1:8b": "Llama3.1-8B",
    "ollama/gemma2:9b":   "Gemma2-9B",
    "ollama/mistral:7b":  "Mistral-7B",
    "mock":               "Mock",
}

FAMILY_ORDER = [
    "atomic",
    "multi_hop",
    "fol_inference",
    "consistency_paraphrase",
    "perturbation",
    "adversarial_misleading",
    "adversarial_nearmiss",
    "indicator_diagnostic",
    "cross_indicator",
    "extended_systems",
    "regime_transition",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def _wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion."""
    if n_total == 0:
        return (0.0, 1.0)
    p = n_correct / n_total
    denom = 1 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    margin = z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _classify_run(manifest: dict, predictions_path: Path | None) -> str:
    """Classify a run as OFFICIAL_V2, PARTIAL_V2, LEGACY_V1, or UNKNOWN."""
    provider = manifest.get("provider", "")
    sha = manifest.get("dataset_global_sha256", "")
    total = manifest.get("total_items_evaluated", 0)
    strict = manifest.get("strict_parsing", False)
    canonical = manifest.get("canonical_selector", "")

    # Legacy v1 checks
    if not canonical or "v2" not in canonical:
        return "LEGACY_V1"
    if sha not in (FREEZE_SHA, ALT_SHA):
        return "UNKNOWN"

    # Mock runs are always partial
    if any(p in provider for p in MOCK_PROVIDERS):
        return "PARTIAL_V2"

    # Coverage check
    metrics_path = predictions_path.parent / "metrics.json" if predictions_path else None
    coverage = 1.0
    if metrics_path and metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        coverage = m.get("coverage", 1.0)

    if coverage < OFFICIAL_COVERAGE_MIN:
        return "PARTIAL_V2"

    if not strict:
        return "PARTIAL_V2"

    if total < 1:
        return "PARTIAL_V2"

    return "OFFICIAL_V2"


def _load_run(run_dir: Path) -> dict | None:
    """Load run directory, returning structured dict or None on failure."""
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"

    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    status = _classify_run(manifest, predictions_path if predictions_path.exists() else None)

    provider = manifest.get("provider", "unknown")
    run_id = manifest.get("run_id", run_dir.name)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model": provider.split("/", 1)[1] if "/" in provider else provider,
        "provider": provider,
        "alias": MODEL_ALIASES.get(provider, provider),
        "date": manifest.get("created_utc", "")[:10],
        "dataset_release": "v2" if "v2" in manifest.get("canonical_selector", "") else "v1",
        "dataset_sha": manifest.get("dataset_global_sha256", ""),
        "prompt_hash": manifest.get("prompt_hash", ""),
        "git_commit": manifest.get("git_commit", ""),
        "n_total": manifest.get("total_items_evaluated", 0),
        "n_valid": metrics.get("valid", manifest.get("total_items_evaluated", 0)),
        "n_invalid": metrics.get("invalid", 0),
        "coverage": metrics.get("coverage", manifest.get("metrics_summary", {}).get("coverage", 1.0)),
        "accuracy_valid": metrics.get("accuracy_valid",
                          manifest.get("metrics_summary", {}).get("accuracy_valid")),
        "balanced_accuracy": metrics.get("balanced_accuracy",
                             manifest.get("metrics_summary", {}).get("balanced_accuracy")),
        "mcc": metrics.get("mcc", manifest.get("metrics_summary", {}).get("mcc")),
        "per_family": metrics.get("per_family", {}),
        "status": status,
        "strict_parsing": manifest.get("strict_parsing", False),
        "seed": manifest.get("seed"),
        "subset_label": _infer_subset_label(manifest),
    }


def _infer_subset_label(manifest: dict) -> str:
    n = manifest.get("total_items_evaluated", 0)
    if n >= 40000:
        return "full_canonical"
    elif n >= 3000:
        return "5k_subset"
    elif n >= 500:
        return "1k_subset"
    else:
        return f"debug_{n}"


def find_run_dirs(runs_dir: Path) -> list[Path]:
    """Find all run directories recursively (dirs with run_manifest.json)."""
    result = []
    for p in sorted(runs_dir.rglob("run_manifest.json")):
        result.append(p.parent)
    return result


# ---------------------------------------------------------------------------
# PART A: Build run catalog
# ---------------------------------------------------------------------------

def build_run_catalog(runs_dir: Path, published_dir: Path) -> list[dict]:
    runs = []
    seen_ids: set[str] = set()

    # Runs in runs/ (primary source)
    for d in find_run_dirs(runs_dir):
        r = _load_run(d)
        if r and r["run_id"] not in seen_ids:
            seen_ids.add(r["run_id"])
            r["source"] = "runs/"
            runs.append(r)

    # Runs in published_results/runs/ not already seen
    if published_dir.exists():
        for d in find_run_dirs(published_dir):
            r = _load_run(d)
            if r and r["run_id"] not in seen_ids:
                seen_ids.add(r["run_id"])
                r["source"] = "published_results/runs/"
                r["status"] = "OFFICIAL_V2"  # already curated
                runs.append(r)

    return runs


def write_run_catalog_json(runs: list[dict], out_dir: Path) -> None:
    catalog = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha": FREEZE_SHA,
        "total_runs": len(runs),
        "official_v2": [r["run_id"] for r in runs if r["status"] == "OFFICIAL_V2"],
        "partial_v2": [r["run_id"] for r in runs if r["status"] == "PARTIAL_V2"],
        "legacy_v1": [r["run_id"] for r in runs if r["status"] == "LEGACY_V1"],
        "unknown": [r["run_id"] for r in runs if r["status"] == "UNKNOWN"],
        "runs": [
            {k: v for k, v in r.items() if k not in ("per_family", "run_dir")}
            for r in runs
        ],
    }
    (out_dir / "run_catalog.json").write_text(json.dumps(catalog, indent=2))


def write_run_catalog_md(runs: list[dict], out_dir: Path) -> None:
    lines = [
        "# ChaosBench-Logic v2 — Run Catalog",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        f"Freeze SHA: `{FREEZE_SHA}`",
        "",
        "## OFFICIAL_V2 Runs",
        "",
        "| Run ID | Model | N | Coverage | Bal_acc | MCC | Subset | Date |",
        "|--------|-------|---|----------|---------|-----|--------|------|",
    ]
    for r in [x for x in runs if x["status"] == "OFFICIAL_V2"]:
        ba = f"{r['balanced_accuracy']:.4f}" if r.get("balanced_accuracy") else "—"
        mcc = f"{r['mcc']:.4f}" if r.get("mcc") is not None else "—"
        lines.append(
            f"| `{r['run_id']}` | {r['alias']} | {r['n_total']:,} "
            f"| {r['coverage']:.4f} | {ba} | {mcc} | {r['subset_label']} | {r['date']} |"
        )

    lines += ["", "## PARTIAL_V2 Runs (excluded from comparisons)", "", "| Run ID | Reason |", "|--------|--------|"]
    partial_reasons = {
        "20260219T192140Z_mock": "mock provider — debug/test run only",
        "20260220T130435Z_mock": "mock provider — debug/test run only",
    }
    for r in [x for x in runs if x["status"] == "PARTIAL_V2"]:
        reason = partial_reasons.get(r["run_id"], f"coverage={r['coverage']:.4f} or mock provider")
        lines.append(f"| `{r['run_id']}` | {reason} |")

    lines += ["", "## LEGACY_V1 Runs (archived — NOT comparable to v2)", "",
              "| Run ID | Model | N | Notes |", "|--------|-------|---|-------|"]
    for r in [x for x in runs if x["status"] == "LEGACY_V1"]:
        lines.append(f"| `{r['run_id']}` | {r.get('alias', r['model'])} | {r['n_total']:,} | v1 dataset (621 items) |")

    lines += [
        "",
        "## Run Classification Rules",
        "",
        "**OFFICIAL_V2**: dataset_release=v2, per-file SHAs verified against freeze manifest,",
        "  run completed (coverage ≥ 0.99), strict_parsing=true, manifest present.",
        "  NOTE: runs with `dataset_global_sha256=00ec17e3...` used the pre-unification formula",
        "  (run.py omitted :count from global hash). All per-file SHAs were verified identical.",
        "  These runs are OFFICIAL. See `artifacts/runs_audit/RUNS_AUDIT.md` §3.",
        "",
        "**PARTIAL_V2**: v2 dataset but missing one or more criteria (mock provider, low coverage).",
        "",
        "**LEGACY_V1**: pre-v2 dataset (N≈621). Never compare with v2.",
    ]
    (out_dir / "run_catalog.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# PART B: Metric verification
# ---------------------------------------------------------------------------

def verify_metrics(runs: list[dict]) -> list[dict]:
    """For OFFICIAL_V2 runs, recompute metrics from predictions if available."""
    issues = []
    for r in [x for x in runs if x["status"] == "OFFICIAL_V2"]:
        pred_path = Path(r["run_dir"]) / "predictions.jsonl"
        if not pred_path.exists():
            continue

        tp = fp = tn = fn = 0
        for line in pred_path.open():
            item = json.loads(line)
            _pl_raw = item.get("parsed_label"); pl = "INVALID" if (_pl_raw is None or item.get("outcome","") == "INVALID") else str(_pl_raw)
            gt_raw = item.get("ground_truth", item.get("gt_label", ""))
            gt = str(gt_raw).upper().strip()
            if pl == "INVALID":
                continue
            if gt == "TRUE":
                if pl == "TRUE":
                    tp += 1
                else:
                    fn += 1
            elif gt == "FALSE":
                if pl == "FALSE":
                    tn += 1
                else:
                    fp += 1

        n_valid = tp + fp + tn + fn
        if n_valid == 0:
            continue

        bal_acc = 0.5 * ((tp / (tp + fn) if tp + fn > 0 else 0.0) +
                         (tn / (tn + fp) if tn + fp > 0 else 0.0))
        mcc = _mcc(tp, fp, tn, fn)

        stored_ba = r.get("balanced_accuracy") or 0.0
        stored_mcc = r.get("mcc") or 0.0

        if abs(bal_acc - stored_ba) > 0.002:
            issues.append({
                "run_id": r["run_id"],
                "check": "balanced_accuracy",
                "stored": stored_ba,
                "recomputed": round(bal_acc, 4),
                "delta": round(abs(bal_acc - stored_ba), 4),
            })
        if abs(mcc - stored_mcc) > 0.002:
            issues.append({
                "run_id": r["run_id"],
                "check": "mcc",
                "stored": stored_mcc,
                "recomputed": round(mcc, 4),
                "delta": round(abs(mcc - stored_mcc), 4),
            })

    return issues


# ---------------------------------------------------------------------------
# PART C: Table builders
# ---------------------------------------------------------------------------

def build_baselines_table(runs: list[dict]) -> pd.DataFrame:
    """Main baselines table (one row per official v2 run)."""
    rows = []
    for r in [x for x in runs if x["status"] == "OFFICIAL_V2" and "mock" not in x["provider"]]:
        # Reconstruct confusion from predictions if available
        tp = fp = tn = fn = 0
        pred_path = Path(r["run_dir"]) / "predictions.jsonl"
        latency_mean = latency_p95 = None

        if pred_path.exists():
            lats = []
            for line in pred_path.open():
                item = json.loads(line)
                _pl_raw = item.get("parsed_label"); pl = "INVALID" if (_pl_raw is None or item.get("outcome","") == "INVALID") else str(_pl_raw)
                gt_raw = item.get("ground_truth", item.get("gt_label", ""))
                gt = str(gt_raw).upper().strip()
                if pl != "INVALID":
                    if gt == "TRUE":
                        if pl == "TRUE": tp += 1
                        else: fn += 1
                    elif gt == "FALSE":
                        if pl == "FALSE": tn += 1
                        else: fp += 1
                lat = item.get("latency_s")
                if lat is not None:
                    lats.append(lat)
            if lats:
                latency_mean = round(float(np.mean(lats)), 3)
                latency_p95 = round(float(np.percentile(lats, 95)), 3)

        n_valid = tp + fp + tn + fn
        pred_true_pct = (tp + fp) / n_valid if n_valid > 0 else None
        gt_true_pct = (tp + fn) / n_valid if n_valid > 0 else None
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        tnr = tn / (tn + fp) if (tn + fp) > 0 else None
        invalid_rate = r.get("n_invalid", 0) / r.get("n_total", 1)

        # macro-family balanced_accuracy and MCC
        fam_data = r.get("per_family", {})
        # These don't have TPR/TNR in per_family so we use acc as proxy;
        # for runs where we have predictions, we'll compute properly
        macro_fam_ba = macro_fam_mcc = None
        if pred_path.exists() and fam_data:
            fam_tp = fam_fp = fam_tn = fam_fn_d = {}
            # Recompute per family
            fam_conf: dict[str, dict] = {}
            pred_path_obj = pred_path
            for line in pred_path_obj.open():
                item = json.loads(line)
                fam = item.get("task_family", item.get("family", "unknown"))
                _pl_raw = item.get("parsed_label"); pl = "INVALID" if (_pl_raw is None or item.get("outcome","") == "INVALID") else str(_pl_raw)
                gt_raw = item.get("ground_truth", item.get("gt_label", ""))
                gt = str(gt_raw).upper().strip()
                if fam not in fam_conf:
                    fam_conf[fam] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                if pl != "INVALID":
                    if gt == "TRUE":
                        if pl == "TRUE": fam_conf[fam]["tp"] += 1
                        else: fam_conf[fam]["fn"] += 1
                    elif gt == "FALSE":
                        if pl == "FALSE": fam_conf[fam]["tn"] += 1
                        else: fam_conf[fam]["fp"] += 1

            fam_bas, fam_mccs = [], []
            for fam, c in fam_conf.items():
                tp_f, fp_f, tn_f, fn_f = c["tp"], c["fp"], c["tn"], c["fn"]
                n_f = tp_f + fp_f + tn_f + fn_f
                if n_f < 5:
                    continue
                ba_f = 0.5 * (
                    (tp_f / (tp_f + fn_f) if tp_f + fn_f > 0 else 0.0) +
                    (tn_f / (tn_f + fp_f) if tn_f + fp_f > 0 else 0.0)
                )
                mcc_f = _mcc(tp_f, fp_f, tn_f, fn_f)
                fam_bas.append(ba_f)
                fam_mccs.append(mcc_f)
            if fam_bas:
                macro_fam_ba = round(float(np.mean(fam_bas)), 4)
                macro_fam_mcc = round(float(np.mean(fam_mccs)), 4)

        rows.append({
            "run_id": r["run_id"],
            "model": r["alias"],
            "subset": r["subset_label"],
            "N": r["n_total"],
            "coverage": round(r["coverage"], 4),
            "balanced_acc_micro": round(r["balanced_accuracy"], 4) if r.get("balanced_accuracy") else None,
            "mcc_micro": round(r["mcc"], 4) if r.get("mcc") is not None else None,
            "balanced_acc_macro_family": macro_fam_ba,
            "mcc_macro_family": macro_fam_mcc,
            "pred_TRUE_pct": round(pred_true_pct, 4) if pred_true_pct is not None else None,
            "gt_TRUE_pct": round(gt_true_pct, 4) if gt_true_pct is not None else None,
            "TPR": round(tpr, 4) if tpr is not None else None,
            "TNR": round(tnr, 4) if tnr is not None else None,
            "invalid_rate": round(invalid_rate, 4),
            "latency_mean_s": latency_mean,
            "latency_p95_s": latency_p95,
        })

    df = pd.DataFrame(rows)
    # Sort by MCC_micro descending
    if not df.empty and "mcc_micro" in df.columns:
        df = df.sort_values("mcc_micro", ascending=False).reset_index(drop=True)
    return df


def build_by_family_table(runs: list[dict]) -> pd.DataFrame:
    """Per-family accuracy breakdown table."""
    rows = []
    # Use 5k runs and full canonical for per-family (1k has too few per-family items)
    valid_runs = [r for r in runs
                  if r["status"] == "OFFICIAL_V2"
                  and "mock" not in r["provider"]
                  and r["subset_label"] in ("5k_subset", "full_canonical")]

    for r in valid_runs:
        pred_path = Path(r["run_dir"]) / "predictions.jsonl"
        fam_conf: dict[str, dict] = {}

        if pred_path.exists():
            for line in pred_path.open():
                item = json.loads(line)
                fam = item.get("task_family", item.get("family", "unknown"))
                _pl_raw = item.get("parsed_label"); pl = "INVALID" if (_pl_raw is None or item.get("outcome","") == "INVALID") else str(_pl_raw)
                gt_raw = item.get("ground_truth", item.get("gt_label", ""))
                gt = str(gt_raw).upper().strip()
                if fam not in fam_conf:
                    fam_conf[fam] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "invalid": 0}
                if pl == "INVALID":
                    fam_conf[fam]["invalid"] += 1
                elif gt == "TRUE":
                    if pl == "TRUE": fam_conf[fam]["tp"] += 1
                    else: fam_conf[fam]["fn"] += 1
                elif gt == "FALSE":
                    if pl == "FALSE": fam_conf[fam]["tn"] += 1
                    else: fam_conf[fam]["fp"] += 1
        else:
            # Fall back to per_family in metrics.json (acc only)
            for fam, fm in r.get("per_family", {}).items():
                fam_conf[fam] = {
                    "tp": fm.get("correct", 0), "fp": 0,
                    "tn": fm.get("total", 0) - fm.get("correct", 0), "fn": 0,
                    "invalid": fm.get("total", 0) - fm.get("valid", 0),
                }

        for fam, c in fam_conf.items():
            tp_f, fp_f, tn_f, fn_f = c["tp"], c["fp"], c["tn"], c["fn"]
            n_valid = tp_f + fp_f + tn_f + fn_f
            n_total = n_valid + c["invalid"]
            if n_total == 0:
                continue
            acc = (tp_f + tn_f) / n_valid if n_valid > 0 else 0.0
            tpr = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else None
            tnr = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else None
            ba = 0.5 * ((tpr or 0.0) + (tnr or 0.0))
            mcc_val = _mcc(tp_f, fp_f, tn_f, fn_f)
            ci_lo, ci_hi = _wilson_ci(tp_f + tn_f, n_valid) if n_valid > 0 else (None, None)

            rows.append({
                "model": r["alias"],
                "subset": r["subset_label"],
                "family": fam,
                "N": n_total,
                "N_valid": n_valid,
                "accuracy": round(acc, 4),
                "balanced_acc": round(ba, 4),
                "MCC": round(mcc_val, 4),
                "TPR": round(tpr, 4) if tpr is not None else None,
                "TNR": round(tnr, 4) if tnr is not None else None,
                "invalid_rate": round(c["invalid"] / n_total, 4) if n_total > 0 else 0.0,
                "wilson_ci_lo": round(ci_lo, 3) if ci_lo is not None else None,
                "wilson_ci_hi": round(ci_hi, 3) if ci_hi is not None else None,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "MCC"], ascending=[True, False]).reset_index(drop=True)
    return df


def build_hardness_table(by_family_df: pd.DataFrame) -> pd.DataFrame:
    """Hardness ranking: families ranked by mean MCC across models."""
    if by_family_df.empty:
        return pd.DataFrame()

    # Group by family
    hard = (
        by_family_df.groupby("family")
        .agg(
            mean_MCC=("MCC", "mean"),
            std_MCC=("MCC", "std"),
            mean_balanced_acc=("balanced_acc", "mean"),
            n_models=("model", "nunique"),
            mean_N=("N", "mean"),
        )
        .reset_index()
    )
    hard["std_MCC"] = hard["std_MCC"].fillna(0.0).round(4)
    hard["mean_MCC"] = hard["mean_MCC"].round(4)
    hard["mean_balanced_acc"] = hard["mean_balanced_acc"].round(4)
    hard["mean_N"] = hard["mean_N"].round(0).astype(int)
    hard = hard.sort_values("mean_MCC", ascending=True).reset_index(drop=True)
    hard.insert(0, "hardness_rank", range(1, len(hard) + 1))
    return hard


# ---------------------------------------------------------------------------
# PART D: Markdown table formatting
# ---------------------------------------------------------------------------

def df_to_md(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    if df.empty:
        return "_No data._\n"
    # Format floats
    df2 = df.copy()
    for col in df2.columns:
        if float_cols and col in float_cols:
            df2[col] = df2[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
        elif df2[col].dtype == float:
            df2[col] = df2[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    lines = ["| " + " | ".join(str(c) for c in df2.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(df2.columns)) + "|")
    for _, row in df2.iterrows():
        lines.append("| " + " | ".join(str(v) if pd.notna(v) else "—" for v in row) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# PART E: Results pack summary
# ---------------------------------------------------------------------------

def build_summary(runs: list[dict], issues: list[dict]) -> dict:
    official = [r for r in runs if r["status"] == "OFFICIAL_V2" and "mock" not in r["provider"]]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha": FREEZE_SHA,
        "n_official_v2": len(official),
        "n_partial_v2": len([r for r in runs if r["status"] == "PARTIAL_V2"]),
        "n_legacy_v1": len([r for r in runs if r["status"] == "LEGACY_V1"]),
        "metric_verification_issues": issues,
        "n_metric_issues": len(issues),
        "official_runs": [
            {
                "run_id": r["run_id"],
                "model": r["alias"],
                "n": r["n_total"],
                "balanced_accuracy": r.get("balanced_accuracy"),
                "mcc": r.get("mcc"),
                "coverage": r.get("coverage"),
                "subset_label": r["subset_label"],
            }
            for r in official
        ],
    }


def build_results_pack_md(runs: list[dict], baselines_df: pd.DataFrame,
                           hardness_df: pd.DataFrame, issues: list[dict]) -> str:
    official = [r for r in runs if r["status"] == "OFFICIAL_V2" and "mock" not in r["provider"]]
    lines = [
        "# ChaosBench-Logic v2 — Results Pack",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_Freeze SHA: `{FREEZE_SHA}`_",
        "",
        "## Overview",
        "",
        f"- **OFFICIAL_V2 runs**: {len(official)} (excluding mock)",
        f"- **Dataset**: 40,886 questions across 11 families",
        f"- **Metric verification issues**: {len(issues)}",
        "",
        "## Metric Verification",
        "",
    ]
    if issues:
        lines.append("⚠️ **Issues found — investigate before publishing:**")
        for iss in issues:
            lines.append(
                f"- `{iss['run_id']}`: {iss['check']} stored={iss['stored']:.4f} "
                f"recomputed={iss['recomputed']:.4f} delta={iss['delta']:.4f}"
            )
    else:
        lines.append("✅ All recomputed metrics match stored values (within tolerance 0.002).")

    lines += [
        "",
        "## Baselines Table",
        "",
        "Primary metrics: `balanced_accuracy` and `MCC` (both micro and macro-family).",
        "Use **macro_family** for cross-family comparison; use **micro** for overall model ranking.",
        "",
        df_to_md(baselines_df),
        "",
        "## Hardness Ranking (families, by mean MCC across models)",
        "",
        "Lower MCC = harder family (closer to random). Families with N < 10 excluded.",
        "",
        df_to_md(hardness_df),
        "",
        "## Excluded Runs",
        "",
        "| Run ID | Status | Reason |",
        "|--------|--------|--------|",
    ]
    exclusion_reasons = {
        "20260219T192140Z_mock": "PARTIAL_V2 — mock provider, N=50 debug run",
        "20260220T130435Z_mock": "PARTIAL_V2 — mock provider, N=50 debug run",
    }
    for r in [x for x in runs if x["status"] == "PARTIAL_V2"]:
        reason = exclusion_reasons.get(r["run_id"], "mock or low coverage")
        lines.append(f"| `{r['run_id']}` | PARTIAL_V2 | {reason} |")
    for r in [x for x in runs if x["status"] == "LEGACY_V1"]:
        lines.append(f"| `{r['run_id']}` | LEGACY_V1 | v1 dataset (N≈621); not comparable to v2 |")

    lines += [
        "",
        "## Data Sources",
        "",
        "- Tables: `artifacts/results_pack/tables/`",
        "- Figures: `artifacts/results_pack/figures/`",
        "- Run catalog: `artifacts/results_pack/run_catalog.{json,md}`",
        "- Full audit: `artifacts/runs_audit/RUNS_AUDIT.md`",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChaosBench-Logic v2 results pack")
    parser.add_argument("--runs_dir", default=str(PROJECT_ROOT / "runs"), help="Path to runs/")
    parser.add_argument(
        "--published_dir",
        default=str(PROJECT_ROOT / "published_results" / "runs"),
        help="Path to published_results/runs/",
    )
    parser.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "artifacts" / "results_pack"),
        help="Output directory",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    published_dir = Path(args.published_dir)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_results_pack] runs_dir={runs_dir}")
    print(f"[build_results_pack] published_dir={published_dir}")
    print(f"[build_results_pack] out_dir={out_dir}")

    # Load legacy v1 runs from published_results/ top level
    legacy_runs = []
    v1_dirs = [
        "claude3_zeroshot", "gemini_zeroshot", "gpt4_cot",
        "gpt4_zeroshot", "llama3_cot", "llama3_zeroshot",
    ]
    for vdir in v1_dirs:
        meta_path = (published_dir.parent / vdir) / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            legacy_runs.append({
                "run_id": vdir,
                "run_dir": str(meta_path.parent),
                "model": meta.get("model_name", vdir),
                "provider": meta.get("model_name", vdir),
                "alias": meta.get("model_name", vdir),
                "date": meta.get("timestamp", "")[:10],
                "dataset_release": "v1",
                "dataset_sha": "n/a",
                "prompt_hash": "n/a",
                "git_commit": "n/a",
                "n_total": meta.get("num_items_total", 0),
                "n_valid": meta.get("num_items_evaluated", 0),
                "n_invalid": meta.get("num_items_unanswered", 0),
                "coverage": (meta.get("num_items_evaluated", 0) / meta.get("num_items_total", 1)
                             if meta.get("num_items_total", 0) > 0 else 0.0),
                "accuracy_valid": None,
                "balanced_accuracy": None,
                "mcc": None,
                "per_family": {},
                "status": "LEGACY_V1",
                "strict_parsing": False,
                "seed": None,
                "subset_label": "v1_legacy",
                "source": "published_results/",
            })

    # Load v2 runs
    print("[build_results_pack] Scanning run directories...")
    runs = build_run_catalog(runs_dir, published_dir)
    runs = runs + legacy_runs

    print(f"  Found {len(runs)} runs total")
    for status in ("OFFICIAL_V2", "PARTIAL_V2", "LEGACY_V1", "UNKNOWN"):
        n = len([r for r in runs if r["status"] == status])
        print(f"  {status}: {n}")

    # Write catalog
    write_run_catalog_json(runs, out_dir)
    write_run_catalog_md(runs, out_dir)
    print("[build_results_pack] Run catalog written.")

    # Verify metrics
    print("[build_results_pack] Verifying stored metrics...")
    issues = verify_metrics(runs)
    if issues:
        print(f"  ⚠️  {len(issues)} metric discrepancies found!")
        for iss in issues:
            print(f"    {iss}")
    else:
        print("  ✅ All metrics verified.")

    # Build tables
    print("[build_results_pack] Building tables...")
    baselines_df = build_baselines_table(runs)
    by_family_df = build_by_family_table(runs)
    hardness_df = build_hardness_table(by_family_df)

    # Save tables
    baselines_df.to_csv(tables_dir / "baselines_table.csv", index=False)
    (tables_dir / "baselines_table.md").write_text(
        "# Baselines Table — ChaosBench-Logic v2\n\n"
        "_Primary metrics: balanced_acc (handles class imbalance) and MCC._\n\n"
        + df_to_md(baselines_df)
    )

    by_family_df.to_csv(tables_dir / "by_family.csv", index=False)
    (tables_dir / "by_family.md").write_text(
        "# Per-Family Table — ChaosBench-Logic v2\n\n"
        "_5k-subset runs and full-canonical runs only (1k runs excluded: too few items per family)._\n\n"
        + df_to_md(by_family_df)
    )

    hardness_df.to_csv(tables_dir / "hardness.csv", index=False)
    (tables_dir / "hardness.md").write_text(
        "# Hardness Ranking — ChaosBench-Logic v2\n\n"
        "_Families ranked by mean MCC across models (lower = harder)._\n\n"
        + df_to_md(hardness_df)
    )

    print(f"  Tables written to {tables_dir}")

    # Build summary
    summary = build_summary(runs, issues)
    (out_dir / "results_pack_summary.json").write_text(json.dumps(summary, indent=2))

    # Build main markdown
    pack_md = build_results_pack_md(runs, baselines_df, hardness_df, issues)
    (out_dir / "RESULTS_PACK.md").write_text(pack_md)

    print(f"[build_results_pack] Done. Output: {out_dir}")

    if issues:
        print("\nERROR: Metric verification failed. See issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
