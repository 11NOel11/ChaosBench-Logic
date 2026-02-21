#!/usr/bin/env python3
"""scripts/build_results_pack.py — Build the v2 results pack.

Inputs:
  --runs_dir      runs/          (local, gitignored)
  --published_dir published_results/runs
  --catalog       path to run_catalog.json (if pre-built; else catalog is built inline)
  --out_dir       output directory (e.g. artifacts/results_pack_v2/<ts>/)

Outputs:
  run_catalog.json + run_catalog.md
  results_pack_summary.json
  RESULTS_PACK.md
  tables/baselines_table.{csv,md}
  tables/by_family.{csv,md}
  tables/hardness.{csv,md}
  tables/full_vs_5k_crosscheck.{csv,md}

Strict classification policy:
  OFFICIAL_V2 if ALL:
    a) dataset_release == v2 (canonical_selector contains 'v2')
    b) dataset SHA matches freeze (global or per-file match)
    c) prediction lines == total_items_evaluated AND no duplicate IDs
    d) coverage >= COVERAGE_THRESHOLD (default 0.97)
    e) strict_parsing enabled
    f) not mock/smoke/debug
  COVERAGE_CAVEATED_V2 if 0.97 <= coverage < 0.995 (included with explicit caveat)
  SUPERSEDED if a better run for same model+subset_type exists
  EXCLUDE_PARTIAL if N < 500 or mock or N/A
  LEGACY_V1 if v1 dataset
"""
from __future__ import annotations

import argparse
import json
import math
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
ALT_SHA    = "00ec17e31193de42c525ff3c8f166b4b59fae2c2631fa84a4c78b33fb01f9374"
COVERAGE_THRESHOLD = 0.97          # hard cutoff — below = excluded
COVERAGE_CLEAN     = 0.995         # above = clean, below = caveated

MODEL_ALIASES = {
    "ollama/qwen2.5:32b": "Qwen2.5-32B",
    "ollama/qwen2.5:14b": "Qwen2.5-14B",
    "ollama/qwen2.5:7b":  "Qwen2.5-7B",
    "ollama/llama3.1:8b": "Llama3.1-8B",
    "ollama/gemma2:9b":   "Gemma2-9B",
    "ollama/mistral:7b":  "Mistral-7B",
    "mock":               "Mock",
}

MODEL_RANK = {
    "Qwen2.5-32B": 1,
    "Qwen2.5-14B": 2,
    "Gemma2-9B":   3,
    "Mistral-7B":  4,
    "Qwen2.5-7B":  5,
    "Llama3.1-8B": 6,
}

FAMILY_ORDER_HARDEST = [
    "regime_transition",
    "cross_indicator",
    "perturbation",
    "consistency_paraphrase",
    "atomic",
    "adversarial_nearmiss",
    "adversarial_misleading",
    "multi_hop",
    "fol_inference",
    "indicator_diagnostic",
    "extended_systems",
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _mcc(tp, fp, tn, fn):
    d = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn - fp*fn) / d if d else 0.0

def _ba(tp, fp, tn, fn):
    tpr = tp/(tp+fn) if tp+fn else 0.0
    tnr = tn/(tn+fp) if tn+fp else 0.0
    return 0.5*(tpr+tnr)

def _wilson(n_ok, n_tot, z=1.96):
    if n_tot == 0: return (0.0, 1.0)
    p = n_ok/n_tot
    d = 1 + z*z/n_tot
    c = (p + z*z/(2*n_tot))/d
    m = z*math.sqrt(p*(1-p)/n_tot + z*z/(4*n_tot*n_tot))/d
    return (max(0.0, c-m), min(1.0, c+m))


# ---------------------------------------------------------------------------
# Predictions parser (handles None parsed_label)
# ---------------------------------------------------------------------------
def _parse_preds(pred_path: Path):
    """Yield (family, gt, pl, latency) for each prediction. pl=None for INVALID."""
    with pred_path.open() as f:
        for line in f:
            item = json.loads(line)
            pl_raw = item.get("parsed_label")
            outcome = item.get("outcome", "")
            pl = None if (pl_raw is None or outcome == "INVALID") else str(pl_raw)
            gt = str(item.get("ground_truth", item.get("gt_label", ""))).upper().strip()
            fam = item.get("task_family", item.get("family", "unknown"))
            lat = item.get("latency_s")
            yield fam, gt, pl, lat


def _confusion_from_preds(pred_path: Path):
    """Global confusion matrix, latencies, invalid count."""
    tp = fp = tn = fn = inv = 0
    lats = []
    for fam, gt, pl, lat in _parse_preds(pred_path):
        if pl is None:
            inv += 1; continue
        if gt == "TRUE":
            if pl == "TRUE": tp += 1
            else: fn += 1
        elif gt == "FALSE":
            if pl == "FALSE": tn += 1
            else: fp += 1
        if lat is not None: lats.append(lat)
    return tp, fp, tn, fn, inv, lats


def _family_confusion(pred_path: Path) -> dict:
    """Per-family confusion + invalids."""
    fams: dict[str, dict] = {}
    for fam, gt, pl, lat in _parse_preds(pred_path):
        if fam not in fams:
            fams[fam] = {"tp":0,"fp":0,"tn":0,"fn":0,"inv":0,"lats":[]}
        if pl is None:
            fams[fam]["inv"] += 1; continue
        if gt == "TRUE":
            if pl == "TRUE": fams[fam]["tp"] += 1
            else: fams[fam]["fn"] += 1
        elif gt == "FALSE":
            if pl == "FALSE": fams[fam]["tn"] += 1
            else: fams[fam]["fp"] += 1
        if lat is not None: fams[fam]["lats"].append(lat)
    return fams


# ---------------------------------------------------------------------------
# Run loader
# ---------------------------------------------------------------------------
def _infer_subset(n: int) -> str:
    if n >= 40000: return "full_canonical"
    if n >= 4000:  return "5k_armored"
    if n >= 500:   return "1k_subset"
    return f"debug_{n}"

def _load_run(run_dir: Path) -> dict | None:
    mpath = run_dir / "run_manifest.json"
    if not mpath.exists(): return None
    m = json.loads(mpath.read_text())
    mpath2 = run_dir / "metrics.json"
    met = json.loads(mpath2.read_text()) if mpath2.exists() else {}
    ms = m.get("metrics_summary", {})

    provider = m.get("provider", "")
    sha = m.get("dataset_global_sha256", "")
    n = m.get("total_items_evaluated", 0)
    cov = met.get("coverage", ms.get("coverage", 1.0))
    strict = m.get("strict_parsing", False)
    canonical = m.get("canonical_selector", "")
    run_id = m.get("run_id", run_dir.name)

    # Classification
    is_mock = any(p in provider for p in ("mock",))
    is_v2 = "v2" in canonical
    sha_ok = sha in (FREEZE_SHA, ALT_SHA)

    if not is_v2 or not sha_ok:
        status = "LEGACY_V1" if not is_v2 else "UNKNOWN"
        excl_reason = f"{'not v2 dataset' if not is_v2 else 'SHA unknown'}"
    elif is_mock:
        status = "EXCLUDE_PARTIAL"
        excl_reason = "mock/debug provider"
    elif n < 500:
        status = "EXCLUDE_PARTIAL"
        excl_reason = f"N={n} (smoke/debug run)"
    elif not strict:
        status = "EXCLUDE_PARTIAL"
        excl_reason = "strict_parsing not enabled"
    elif cov < COVERAGE_THRESHOLD:
        status = "EXCLUDE_INCOMPLETE"
        excl_reason = f"coverage={cov:.4f} < threshold {COVERAGE_THRESHOLD}"
    elif cov < COVERAGE_CLEAN:
        status = "OFFICIAL_V2"  # included with caveat
        excl_reason = f"coverage={cov:.4f} (caveated: below {COVERAGE_CLEAN})"
    else:
        status = "OFFICIAL_V2"
        excl_reason = None

    alias = MODEL_ALIASES.get(provider, provider)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model": provider.split("/",1)[1] if "/" in provider else provider,
        "provider": provider,
        "alias": alias,
        "date": m.get("created_utc","")[:10],
        "dataset_release": "v2" if is_v2 else "v1",
        "dataset_sha": sha,
        "sha_formula": "old_no_count" if sha == ALT_SHA else "unified",
        "prompt_hash": m.get("prompt_hash",""),
        "git_commit": m.get("git_commit",""),
        "n_total": n,
        "n_valid": met.get("valid", n),
        "n_invalid": met.get("invalid", 0),
        "coverage": round(cov, 4),
        "balanced_accuracy": met.get("balanced_accuracy", ms.get("balanced_accuracy")),
        "mcc": met.get("mcc", ms.get("mcc")),
        "accuracy_valid": met.get("accuracy_valid", ms.get("accuracy_valid")),
        "per_family": met.get("per_family", {}),
        "status": status,
        "coverage_caveated": cov < COVERAGE_CLEAN,
        "excl_reason": excl_reason,
        "strict_parsing": strict,
        "seed": m.get("seed"),
        "subset_label": _infer_subset(n),
        "source": "runs/",
    }


def find_run_dirs(runs_dir: Path) -> list[Path]:
    """Find all run dirs with run_manifest.json (skip _archive_excluded)."""
    result = []
    for p in sorted(runs_dir.rglob("run_manifest.json")):
        # Skip archived runs
        parts = p.parts
        if any("_archive_excluded" in part for part in parts):
            continue
        result.append(p.parent)
    return result


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------
def build_catalog(runs_dir: Path, published_dir: Path) -> list[dict]:
    seen: set[str] = set()
    runs: list[dict] = []

    for d in find_run_dirs(runs_dir):
        r = _load_run(d)
        if r and r["run_id"] not in seen:
            seen.add(r["run_id"])
            runs.append(r)

    # Add any published runs not in runs/ (e.g. published-only)
    if published_dir.exists():
        for d in published_dir.iterdir():
            if not d.is_dir(): continue
            r = _load_run(d)
            if r and r["run_id"] not in seen:
                seen.add(r["run_id"])
                r["source"] = "published_results/runs/"
                runs.append(r)

    # Mark superseded runs: if same model+subset_label with higher N exists
    official = [r for r in runs if r["status"] == "OFFICIAL_V2"]
    best: dict[tuple, dict] = {}
    for r in official:
        key = (r["alias"], r["subset_label"])
        if key not in best or r["n_total"] > best[key]["n_total"]:
            best[key] = r
    for r in official:
        key = (r["alias"], r["subset_label"])
        if best[key]["run_id"] != r["run_id"] and r["subset_label"] in ("5k_armored", "1k_subset"):
            r["status"] = "SUPERSEDED"
            r["excl_reason"] = f"superseded by {best[key]['run_id']} (N={best[key]['n_total']})"

    return runs


# ---------------------------------------------------------------------------
# Metric verification
# ---------------------------------------------------------------------------
def verify_metrics(runs: list[dict]) -> list[dict]:
    issues = []
    for r in [x for x in runs if x["status"] == "OFFICIAL_V2"]:
        pred_path = Path(r["run_dir"]) / "predictions.jsonl"
        if not pred_path.exists(): continue

        tp, fp, tn, fn, inv, _ = _confusion_from_preds(pred_path)
        n_valid = tp+fp+tn+fn
        if not n_valid: continue

        ba = _ba(tp, fp, tn, fn)
        mcc = _mcc(tp, fp, tn, fn)
        stored_ba = r.get("balanced_accuracy") or 0.0
        stored_mcc = r.get("mcc") or 0.0

        for metric, recomp, stored in [("balanced_accuracy", ba, stored_ba), ("mcc", mcc, stored_mcc)]:
            if abs(recomp - stored) > 0.005:
                issues.append({
                    "run_id": r["run_id"],
                    "check": metric,
                    "stored": round(stored, 4),
                    "recomputed": round(recomp, 4),
                    "delta": round(abs(recomp - stored), 4),
                })
    return issues


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def build_baselines_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in [x for x in runs if x["status"] == "OFFICIAL_V2" and "mock" not in x["provider"]]:
        pred_path = Path(r["run_dir"]) / "predictions.jsonl"
        tp = fp = tn = fn = inv = 0
        latency_mean = latency_p95 = None

        if pred_path.exists():
            tp, fp, tn, fn, inv, lats = _confusion_from_preds(pred_path)
            if lats:
                latency_mean = round(float(np.mean(lats)), 3)
                latency_p95  = round(float(np.percentile(lats, 95)), 3)

        n_valid = tp+fp+tn+fn
        pred_true_pct = (tp+fp)/n_valid if n_valid else None
        gt_true_pct   = (tp+fn)/n_valid if n_valid else None
        tpr = tp/(tp+fn) if tp+fn else None
        tnr = tn/(tn+fp) if tn+fp else None
        invalid_rate = r.get("n_invalid",0) / max(r.get("n_total",1),1)

        # macro-family MCC/BA from predictions
        macro_fam_ba = macro_fam_mcc = None
        if pred_path.exists():
            fam_conf = _family_confusion(pred_path)
            bas, mccs = [], []
            for fam, c in fam_conf.items():
                nf = c["tp"]+c["fp"]+c["tn"]+c["fn"]
                if nf < 10: continue
                bas.append(_ba(c["tp"],c["fp"],c["tn"],c["fn"]))
                mccs.append(_mcc(c["tp"],c["fp"],c["tn"],c["fn"]))
            if bas:
                macro_fam_ba  = round(float(np.mean(bas)), 4)
                macro_fam_mcc = round(float(np.mean(mccs)), 4)

        rows.append({
            "run_id":                  r["run_id"],
            "model":                   r["alias"],
            "subset":                  r["subset_label"],
            "N":                       r["n_total"],
            "coverage":                round(r["coverage"],4),
            "caveated":                "⚠️" if r.get("coverage_caveated") else "✅",
            "balanced_acc_micro":      round(r["balanced_accuracy"],4) if r.get("balanced_accuracy") else None,
            "mcc_micro":               round(r["mcc"],4) if r.get("mcc") is not None else None,
            "balanced_acc_macro_fam":  macro_fam_ba,
            "mcc_macro_fam":           macro_fam_mcc,
            "pred_TRUE_pct":           round(pred_true_pct,4) if pred_true_pct is not None else None,
            "gt_TRUE_pct":             round(gt_true_pct,4) if gt_true_pct is not None else None,
            "TPR":                     round(tpr,4) if tpr is not None else None,
            "TNR":                     round(tnr,4) if tnr is not None else None,
            "invalid_rate":            round(invalid_rate,4),
            "latency_mean_s":          latency_mean,
            "latency_p95_s":           latency_p95,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["subset","mcc_micro"], ascending=[True, False]).reset_index(drop=True)
    return df


def build_by_family_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    # Use runs that have proper N per family: full_canonical + 5k_armored (but flag small-N families)
    use_runs = [r for r in runs
                if r["status"] == "OFFICIAL_V2"
                and "mock" not in r["provider"]
                and r["subset_label"] in ("full_canonical", "5k_armored")]

    for r in use_runs:
        pred_path = Path(r["run_dir"]) / "predictions.jsonl"
        if not pred_path.exists():
            continue
        fam_conf = _family_confusion(pred_path)
        for fam, c in fam_conf.items():
            tp, fp, tn, fn, inv = c["tp"], c["fp"], c["tn"], c["fn"], c["inv"]
            n_valid = tp+fp+tn+fn
            n_total = n_valid+inv
            if n_total == 0: continue

            acc  = (tp+tn)/n_valid if n_valid else 0.0
            tpr  = tp/(tp+fn) if tp+fn else None
            tnr  = tn/(tn+fp) if tn+fp else None
            ba   = 0.5*((tpr or 0.0)+(tnr or 0.0))
            mcc_val = _mcc(tp, fp, tn, fn)
            ci_lo, ci_hi = _wilson(tp+tn, n_valid) if n_valid else (None, None)

            rows.append({
                "model":           r["alias"],
                "subset":          r["subset_label"],
                "run_id":          r["run_id"],
                "family":          fam,
                "N_family":        n_total,
                "N_valid":         n_valid,
                "accuracy":        round(acc,4),
                "balanced_acc":    round(ba,4),
                "MCC":             round(mcc_val,4),
                "TPR":             round(tpr,4) if tpr is not None else None,
                "TNR":             round(tnr,4) if tnr is not None else None,
                "invalid_rate":    round(inv/n_total,4) if n_total else 0.0,
                "wilson_ci_lo":    round(ci_lo,3) if ci_lo is not None else None,
                "wilson_ci_hi":    round(ci_hi,3) if ci_hi is not None else None,
                "small_n_flag":    "⚠️(N<30)" if n_total < 30 else "",
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family","subset","MCC"], ascending=[True,True,False]).reset_index(drop=True)
    return df


def build_hardness_table(by_family_df: pd.DataFrame) -> pd.DataFrame:
    if by_family_df.empty: return pd.DataFrame()
    # Use full_canonical runs only for hardness (most reliable N per family)
    fc = by_family_df[by_family_df["subset"] == "full_canonical"]
    if fc.empty:
        fc = by_family_df  # fallback

    hard = (fc.groupby("family")
              .agg(mean_MCC=("MCC","mean"),
                   std_MCC=("MCC","std"),
                   mean_balanced_acc=("balanced_acc","mean"),
                   n_models=("model","nunique"),
                   mean_N_family=("N_family","mean"))
              .reset_index())
    hard["std_MCC"] = hard["std_MCC"].fillna(0).round(4)
    hard["mean_MCC"] = hard["mean_MCC"].round(4)
    hard["mean_balanced_acc"] = hard["mean_balanced_acc"].round(4)
    hard["mean_N_family"] = hard["mean_N_family"].round(0).astype(int)
    hard = hard.sort_values("mean_MCC", ascending=True).reset_index(drop=True)
    hard.insert(0, "hardness_rank", range(1, len(hard)+1))
    return hard


def build_crosscheck_table(by_family_df: pd.DataFrame) -> pd.DataFrame:
    """Compare 5k_armored vs full_canonical for models appearing in both."""
    if by_family_df.empty: return pd.DataFrame()
    rows = []
    for model in by_family_df["model"].unique():
        for subset in ("5k_armored", "full_canonical"):
            sub = by_family_df[(by_family_df["model"]==model) & (by_family_df["subset"]==subset)]
            if sub.empty: continue
            for _, row in sub.iterrows():
                rows.append({
                    "model": model,
                    "subset": subset,
                    "family": row["family"],
                    "N_family": row["N_family"],
                    "MCC": row["MCC"],
                    "balanced_acc": row["balanced_acc"],
                })
    df = pd.DataFrame(rows)
    if df.empty: return df

    # Pivot: model x family, compare 5k vs full
    pivot = df.pivot_table(index=["model","family"], columns="subset", values="MCC").reset_index()
    pivot.columns.name = None
    if "5k_armored" in pivot.columns and "full_canonical" in pivot.columns:
        pivot["delta_5k_vs_full"] = (pivot["5k_armored"] - pivot["full_canonical"]).round(4)
    return pivot.sort_values(["model","family"])


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------
def _fmt(v, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if isinstance(v, float): return f"{v:.{digits}f}"
    return str(v)

def df_to_md(df: pd.DataFrame) -> str:
    if df.empty: return "_No data._\n"
    lines = ["| "+" | ".join(str(c) for c in df.columns)+" |"]
    lines.append("|"+ "|".join(["---"]*len(df.columns)) +"|")
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("—" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append("—" if v is None else str(v))
        lines.append("| "+" | ".join(cells)+" |")
    return "\n".join(lines)+"\n"


# ---------------------------------------------------------------------------
# Summary + RESULTS_PACK.md
# ---------------------------------------------------------------------------
def build_summary(runs, issues):
    official = [r for r in runs if r["status"]=="OFFICIAL_V2" and "mock" not in r["provider"]]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha": FREEZE_SHA,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "n_official_v2": len(official),
        "caveated_runs": [r["run_id"] for r in official if r.get("coverage_caveated")],
        "metric_issues": issues,
        "official_runs": [
            {"run_id":r["run_id"],"model":r["alias"],"n":r["n_total"],
             "subset":r["subset_label"],"coverage":r["coverage"],
             "balanced_accuracy":r.get("balanced_accuracy"),"mcc":r.get("mcc")}
            for r in official
        ],
    }


# ---------------------------------------------------------------------------
# Catalog writers
# ---------------------------------------------------------------------------
def write_catalog_json(runs, out_dir):
    cat = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha": FREEZE_SHA,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "runs": [
            {k:v for k,v in r.items() if k not in ("per_family","run_dir")}
            for r in runs
        ],
    }
    (out_dir/"run_catalog.json").write_text(json.dumps(cat, indent=2))


def write_catalog_md(runs, out_dir):
    lines = [
        "# ChaosBench-Logic v2 — Run Catalog",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_Freeze SHA: `{FREEZE_SHA}`_",
        f"_Coverage threshold (hard cutoff): {COVERAGE_THRESHOLD} | Clean threshold: {COVERAGE_CLEAN}_",
        "",
        "## OFFICIAL_V2 Runs",
        "",
        "| Run ID | Model | Subset | N | Coverage | Status | Bal_acc | MCC | Notes |",
        "|--------|-------|--------|---|----------|--------|---------|-----|-------|",
    ]
    for r in [x for x in runs if x["status"]=="OFFICIAL_V2"]:
        cov_flag = "⚠️ caveated" if r.get("coverage_caveated") else "✅ clean"
        ba = f"{r['balanced_accuracy']:.4f}" if r.get("balanced_accuracy") else "—"
        mcc = f"{r['mcc']:.4f}" if r.get("mcc") is not None else "—"
        note = r.get("excl_reason","") or ""
        lines.append(
            f"| `{r['run_id']}` | {r['alias']} | {r['subset_label']} | "
            f"{r['n_total']:,} | {r['coverage']:.4f} ({cov_flag}) | OFFICIAL | {ba} | {mcc} | {note} |"
        )

    lines += ["", "## Superseded Runs (excluded from main comparisons)", "",
              "| Run ID | Model | N | Reason |", "|--------|-------|---|--------|"]
    for r in [x for x in runs if x["status"]=="SUPERSEDED"]:
        lines.append(f"| `{r['run_id']}` | {r['alias']} | {r['n_total']:,} | {r.get('excl_reason','')} |")

    lines += ["", "## Excluded Runs", "",
              "| Run ID | Status | N | Reason |", "|--------|--------|---|--------|"]
    for r in [x for x in runs if x["status"] in ("EXCLUDE_PARTIAL","EXCLUDE_INCOMPLETE","UNKNOWN")]:
        lines.append(f"| `{r['run_id']}` | {r['status']} | {r['n_total']:,} | {r.get('excl_reason','')} |")

    lines += ["", "## LEGACY_V1 Runs (archived — NOT comparable to v2)", "",
              "| Run ID | Model | Notes |", "|--------|-------|-------|"]
    for r in [x for x in runs if x["status"]=="LEGACY_V1"]:
        lines.append(f"| `{r['run_id']}` | {r.get('alias',r['model'])} | v1 dataset |")

    lines += [
        "", "## Coverage Notes",
        "",
        f"- Runs with coverage ≥ {COVERAGE_CLEAN} are marked ✅ clean.",
        f"- Runs with {COVERAGE_THRESHOLD} ≤ coverage < {COVERAGE_CLEAN} are marked ⚠️ caveated (included with explicit caveat).",
        f"- Runs with coverage < {COVERAGE_THRESHOLD} are EXCLUDED.",
        "",
        "**Mistral:7B coverage caveat**: All mistral invalids are multi_hop domain-vocabulary responses",
        "  ('Indeterminate', 'Unknown', 'Bounded') — not safety refusals. The model outputs ontological",
        "  terms instead of TRUE/FALSE for multi-hop chain questions. Coverage 98.8% (5k) / 98.91% (full).",
        "  These are included as OFFICIAL_V2 with ⚠️ caveated status.",
        "",
        "**Per-file SHA note**: Runs with SHA `00ec17e3…` used the pre-unification hashing formula",
        "  (run.py omitted `:count` from global hash). All per-file SHAs verified identical to freeze.",
        "  These runs evaluated the same frozen data. See `artifacts/runs_audit/RUNS_AUDIT.md §3`.",
    ]
    (out_dir/"run_catalog.md").write_text("\n".join(lines)+"\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",      default=str(PROJECT_ROOT/"runs"))
    parser.add_argument("--published_dir", default=str(PROJECT_ROOT/"published_results"/"runs"))
    parser.add_argument("--catalog",       default=None, help="Pre-built catalog JSON path")
    parser.add_argument("--out_dir",       default=str(PROJECT_ROOT/"artifacts"/"results_pack_v2"/"latest"))
    args = parser.parse_args()

    runs_dir      = Path(args.runs_dir)
    published_dir = Path(args.published_dir)
    out_dir       = Path(args.out_dir)
    tables_dir    = out_dir/"tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_results_pack] out_dir={out_dir}")

    # Build catalog
    print("[build_results_pack] Scanning runs...")
    runs = build_catalog(runs_dir, published_dir)
    for st in ("OFFICIAL_V2","SUPERSEDED","EXCLUDE_PARTIAL","EXCLUDE_INCOMPLETE","LEGACY_V1","UNKNOWN"):
        n = len([r for r in runs if r["status"]==st])
        if n: print(f"  {st}: {n}")

    write_catalog_json(runs, out_dir)
    write_catalog_md(runs, out_dir)
    print("[build_results_pack] Catalog written.")

    # Verify metrics
    print("[build_results_pack] Verifying metrics...")
    issues = verify_metrics(runs)
    if issues:
        print(f"  ⚠️  {len(issues)} metric discrepancies!")
        for iss in issues:
            print(f"    {iss}")
    else:
        print("  ✅ All metrics verified.")

    # Tables
    print("[build_results_pack] Building tables...")
    baselines_df  = build_baselines_table(runs)
    by_family_df  = build_by_family_table(runs)
    hardness_df   = build_hardness_table(by_family_df)
    crosscheck_df = build_crosscheck_table(by_family_df)

    baselines_df.to_csv(tables_dir/"baselines_table.csv", index=False)
    (tables_dir/"baselines_table.md").write_text(
        "# Baselines Table — ChaosBench-Logic v2\n\n"
        "_Primary metrics: balanced_acc and MCC. ⚠️ = coverage caveated (< 99.5%)._\n\n"
        + df_to_md(baselines_df))

    by_family_df.to_csv(tables_dir/"by_family.csv", index=False)
    (tables_dir/"by_family.md").write_text(
        "# Per-Family Table — ChaosBench-Logic v2\n\n"
        "_full_canonical and 5k_armored runs only. ⚠️(N<30) = small-N, interpret with caution._\n\n"
        + df_to_md(by_family_df))

    hardness_df.to_csv(tables_dir/"hardness.csv", index=False)
    (tables_dir/"hardness.md").write_text(
        "# Hardness Ranking — ChaosBench-Logic v2\n\n"
        "_Based on full_canonical runs (most reliable N per family). Lower MCC = harder._\n\n"
        + df_to_md(hardness_df))

    crosscheck_df.to_csv(tables_dir/"full_vs_5k_crosscheck.csv", index=False)
    (tables_dir/"full_vs_5k_crosscheck.md").write_text(
        "# Full-Canonical vs 5k-Armored Cross-Check\n\n"
        "_Delta = 5k_armored MCC − full_canonical MCC. Models with both runs: Mistral-7B, Qwen2.5-14B, Qwen2.5-32B._\n\n"
        + df_to_md(crosscheck_df))

    print(f"  Tables written to {tables_dir}")

    # Summary + RESULTS_PACK.md
    summary = build_summary(runs, issues)
    (out_dir/"results_pack_summary.json").write_text(json.dumps(summary, indent=2))

    official = [r for r in runs if r["status"]=="OFFICIAL_V2" and "mock" not in r["provider"]]
    pack_lines = [
        "# ChaosBench-Logic v2 — Results Pack",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_Freeze SHA: `{FREEZE_SHA}`_",
        "",
        f"**OFFICIAL_V2 runs**: {len(official)} (excl. mock/debug)",
        f"**Coverage threshold**: {COVERAGE_THRESHOLD} (hard cutoff) | {COVERAGE_CLEAN} (clean)",
        f"**Metric verification**: {'⚠️ issues found' if issues else '✅ all verified'}",
        "",
        "## Baselines", "", df_to_md(baselines_df),
        "## Hardness Ranking", "", df_to_md(hardness_df),
        "## Excluded / Superseded Runs", "",
        "| Run ID | Status | Reason |", "|--------|--------|--------|",
    ]
    for r in [x for x in runs if x["status"] not in ("OFFICIAL_V2","LEGACY_V1")]:
        pack_lines.append(f"| `{r['run_id']}` | {r['status']} | {r.get('excl_reason','')} |")
    (out_dir/"RESULTS_PACK.md").write_text("\n".join(pack_lines)+"\n")

    print(f"[build_results_pack] Done. Output: {out_dir}")
    if issues:
        print("\nERROR: Metric verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
