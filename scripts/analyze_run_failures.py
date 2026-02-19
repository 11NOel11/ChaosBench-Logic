#!/usr/bin/env python3
"""Analyze evaluation run failures and produce paper-ready diagnostic tables.

Usage:
    python scripts/analyze_run_failures.py --run-dir runs/<run_id>
    python scripts/analyze_run_failures.py --run-dir runs/<run_id> --out-dir artifacts/paper_assets/failure_analysis

Outputs (all in --out-dir/<run_id>/):
    top_failures.md          — top 50 most frequent failure templates
    per_family_breakdown.md  — per-family accuracy, coverage, invalid rate
    per_split_breakdown.md   — per-split accuracy
    hard_systems.md          — hardest systems/families
    group_flip_analysis.md   — flip rate for paraphrase/perturbation groups
    summary.json             — machine-readable numbers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_predictions(run_dir: Path) -> List[Dict]:
    preds_path = run_dir / "predictions.jsonl"
    if not preds_path.exists():
        print(f"ERROR: predictions.jsonl not found in {run_dir}", file=sys.stderr)
        sys.exit(2)
    records = []
    with open(preds_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_manifest(run_dir: Path) -> Dict:
    p = run_dir / "run_manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _label_from_rec(rec: Dict) -> Optional[str]:
    return rec.get("parsed_label")


def _outcome(rec: Dict) -> str:
    return rec.get("outcome", "INVALID")


def _is_invalid(rec: Dict) -> bool:
    return _outcome(rec) == "INVALID"


def _is_correct(rec: Dict) -> bool:
    return rec.get("correct", False) is True


def _truncate_question(q: str, max_len: int = 80) -> str:
    q = re.sub(r"\s+", " ", q.strip())
    return q[:max_len] + "…" if len(q) > max_len else q


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def top_failures(records: List[Dict], top_n: int = 50) -> List[Dict]:
    """Collect failure records (INVALID + incorrect) and find frequent templates."""
    failures = [r for r in records if _is_invalid(r) or not _is_correct(r)]

    # Bucket by question prefix (first 60 chars stripped of numbers)
    template_re = re.compile(r"\d+(\.\d+)?")
    template_counts: Counter = Counter()
    template_examples: Dict[str, Dict] = {}
    for r in failures:
        q = r.get("question", "")
        template = template_re.sub("N", _truncate_question(q, 60))
        template_counts[template] += 1
        if template not in template_examples:
            template_examples[template] = r

    results = []
    for template, count in template_counts.most_common(top_n):
        ex = template_examples[template]
        results.append({
            "template": template,
            "count": count,
            "example_id": ex.get("id"),
            "outcome": _outcome(ex),
            "ground_truth": ex.get("ground_truth"),
            "pred_text_snippet": (ex.get("pred_text") or "")[:80],
        })
    return results


def per_family_breakdown(records: List[Dict]) -> Dict[str, Dict]:
    by_family: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        fam = r.get("task_family") or "unknown"
        by_family[fam].append(r)

    result = {}
    for fam, recs in sorted(by_family.items()):
        total = len(recs)
        valid = [r for r in recs if not _is_invalid(r)]
        correct = [r for r in valid if _is_correct(r)]
        invalid = [r for r in recs if _is_invalid(r)]
        result[fam] = {
            "total": total,
            "valid": len(valid),
            "invalid": len(invalid),
            "correct": len(correct),
            "coverage": len(valid) / total if total else 0.0,
            "accuracy_valid": len(correct) / len(valid) if valid else 0.0,
            "invalid_rate": len(invalid) / total if total else 0.0,
        }
    return result


def per_split_breakdown(records: List[Dict]) -> Dict[str, Dict]:
    by_split: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        sp = r.get("split") or "unknown"
        by_split[sp].append(r)

    result = {}
    for sp, recs in sorted(by_split.items()):
        total = len(recs)
        valid = [r for r in recs if not _is_invalid(r)]
        correct = [r for r in valid if _is_correct(r)]
        result[sp] = {
            "total": total,
            "valid": len(valid),
            "correct": len(correct),
            "coverage": len(valid) / total if total else 0.0,
            "accuracy_valid": len(correct) / len(valid) if valid else 0.0,
        }
    return result


def hard_systems(records: List[Dict], top_n: int = 20) -> List[Tuple[str, Dict]]:
    """Find systems/families with lowest accuracy_valid."""
    by_fam = per_family_breakdown(records)
    ranked = sorted(
        [(fam, stats) for fam, stats in by_fam.items() if stats["valid"] >= 5],
        key=lambda x: x[1]["accuracy_valid"],
    )
    return ranked[:top_n]


def group_flip_analysis(records: List[Dict]) -> Dict[str, Any]:
    """Compute flip rate for paraphrase/perturbation groups."""
    # Group by question text similarity prefix (paraphrase groups share same base question)
    # Perturbation groups can be identified by item IDs that share a prefix

    # Strategy: group by (task_family, ground_truth, first_60_chars_of_question)
    groups: Dict[str, List[Dict]] = defaultdict(list)
    paraphrase_families = {"consistency_paraphrase", "perturbation", "perturbation_robustness"}

    for r in records:
        fam = r.get("task_family") or ""
        if fam in paraphrase_families or "paraphrase" in fam or "perturbation" in fam:
            q = r.get("question", "")
            key = f"{fam}::{r.get('ground_truth')}::{q[:50]}"
            groups[key].append(r)

    n_groups_with_flip = 0
    n_groups_total = 0
    group_stats = []

    for key, group_recs in groups.items():
        if len(group_recs) < 2:
            continue
        n_groups_total += 1
        valid_recs = [r for r in group_recs if not _is_invalid(r)]
        if not valid_recs:
            continue
        labels = set(r.get("parsed_label") for r in valid_recs)
        has_flip = len(labels) > 1
        if has_flip:
            n_groups_with_flip += 1
        group_stats.append({
            "key": key[:80],
            "group_size": len(group_recs),
            "valid": len(valid_recs),
            "has_flip": has_flip,
        })

    flip_rate = n_groups_with_flip / n_groups_total if n_groups_total else 0.0
    return {
        "n_groups_total": n_groups_total,
        "n_groups_with_flip": n_groups_with_flip,
        "flip_rate": flip_rate,
        "group_stats": group_stats[:50],
    }


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------


def write_top_failures_md(path: Path, failures: List[Dict], run_id: str) -> None:
    lines = [
        f"# Top Failure Templates — {run_id}",
        "",
        "Failures = INVALID + incorrect predictions.",
        "",
        "| Rank | Count | Outcome | GT | Question template |",
        "|------|-------|---------|----|--------------------|",
    ]
    for i, f in enumerate(failures, 1):
        lines.append(
            f"| {i} | {f['count']} | {f['outcome']} | {f['ground_truth']} | `{f['template']}` |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_per_family_md(path: Path, breakdown: Dict[str, Dict], run_id: str) -> None:
    lines = [
        f"# Per-Family Breakdown — {run_id}",
        "",
        "| Family | Total | Valid | Invalid | Coverage | Acc (valid) | Invalid rate |",
        "|--------|-------|-------|---------|----------|-------------|--------------|",
    ]
    for fam, s in breakdown.items():
        lines.append(
            f"| {fam} | {s['total']} | {s['valid']} | {s['invalid']} "
            f"| {s['coverage']:.3f} | {s['accuracy_valid']:.3f} | {s['invalid_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_per_split_md(path: Path, breakdown: Dict[str, Dict], run_id: str) -> None:
    lines = [
        f"# Per-Split Breakdown — {run_id}",
        "",
        "| Split | Total | Valid | Coverage | Acc (valid) |",
        "|-------|-------|-------|----------|-------------|",
    ]
    for sp, s in breakdown.items():
        lines.append(
            f"| {sp} | {s['total']} | {s['valid']} | {s['coverage']:.3f} | {s['accuracy_valid']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_hard_systems_md(path: Path, ranked: List[Tuple[str, Dict]], run_id: str) -> None:
    lines = [
        f"# Hardest Task Families — {run_id}",
        "",
        "Ranked by accuracy_valid (ascending); minimum 5 valid predictions.",
        "",
        "| Rank | Family | Total | Acc (valid) | Coverage |",
        "|------|--------|-------|-------------|----------|",
    ]
    for i, (fam, s) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {fam} | {s['total']} | {s['accuracy_valid']:.3f} | {s['coverage']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_flip_analysis_md(path: Path, flip_data: Dict, run_id: str) -> None:
    lines = [
        f"# Group Flip Analysis — {run_id}",
        "",
        f"**Flip rate**: {flip_data['flip_rate']:.4f} "
        f"({flip_data['n_groups_with_flip']} / {flip_data['n_groups_total']} groups)",
        "",
        "A 'flip' occurs when a model gives different answers to paraphrase/perturbation "
        "variants of the same question.",
        "",
        "| Group | Size | Valid | Has Flip |",
        "|-------|------|-------|----------|",
    ]
    for g in flip_data["group_stats"][:30]:
        lines.append(
            f"| `{g['key'][:60]}` | {g['group_size']} | {g['valid']} | {'✓' if g['has_flip'] else '–'} |"
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze(run_dir: str, out_dir: str) -> Dict:
    run_path = Path(run_dir)
    manifest = load_manifest(run_path)
    run_id = manifest.get("run_id", run_path.name)
    records = load_predictions(run_path)

    out_path = Path(out_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {len(records)} predictions for run: {run_id}")
    print(f"Output dir: {out_path}")

    failures = top_failures(records)
    fam_breakdown = per_family_breakdown(records)
    split_breakdown = per_split_breakdown(records)
    hard = hard_systems(records)
    flip_data = group_flip_analysis(records)

    write_top_failures_md(out_path / "top_failures.md", failures, run_id)
    write_per_family_md(out_path / "per_family_breakdown.md", fam_breakdown, run_id)
    write_per_split_md(out_path / "per_split_breakdown.md", split_breakdown, run_id)
    write_hard_systems_md(out_path / "hard_systems.md", hard, run_id)
    write_flip_analysis_md(out_path / "group_flip_analysis.md", flip_data, run_id)

    # Load metrics.json
    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    summary = {
        "run_id": run_id,
        "total": len(records),
        "n_invalid": sum(1 for r in records if _is_invalid(r)),
        "n_correct": sum(1 for r in records if _is_correct(r)),
        "coverage": metrics.get("coverage", 0.0),
        "accuracy_valid": metrics.get("accuracy_valid", 0.0),
        "effective_accuracy": metrics.get("effective_accuracy", 0.0),
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        "mcc": metrics.get("mcc", 0.0),
        "flip_rate": flip_data["flip_rate"],
        "n_failure_templates": len(failures),
        "per_family": fam_breakdown,
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nResults:")
    print(f"  Coverage       : {summary['coverage']:.4f}")
    print(f"  Accuracy(valid): {summary['accuracy_valid']:.4f}")
    print(f"  Eff. accuracy  : {summary['effective_accuracy']:.4f}")
    print(f"  Balanced acc   : {summary['balanced_accuracy']:.4f}")
    print(f"  MCC            : {summary['mcc']:.4f}")
    print(f"  Flip rate      : {summary['flip_rate']:.4f}")
    print(f"  Artifacts in   : {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation run failures.")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument(
        "--out-dir",
        default="artifacts/paper_assets/failure_analysis",
        help="Output directory",
    )
    args = parser.parse_args()
    import os
    os.chdir(PROJECT_ROOT)
    analyze(run_dir=args.run_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
