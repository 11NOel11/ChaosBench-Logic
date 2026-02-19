#!/usr/bin/env python3
"""Analyze label balance (TRUE/FALSE distribution) in ChaosBench-Logic dataset.

Usage:
    python scripts/analyze_label_balance.py --data_dir data/ --out_dir reports/
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_items(data_dir: str) -> List[Dict[str, Any]]:
    """Load all JSONL items from data directory."""
    items = []
    data_path = Path(data_dir)

    for fname in sorted(data_path.glob("batch*.jsonl")):
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item["_batch_file"] = fname.name
                    items.append(item)

    return items


def analyze_overall_balance(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall label balance."""
    labels = [item.get("ground_truth", "") for item in items]
    n_true = sum(1 for l in labels if l in ("TRUE", "YES"))
    n_false = sum(1 for l in labels if l in ("FALSE", "NO"))
    n_total = len(labels)

    return {
        "total": n_total,
        "true_count": n_true,
        "false_count": n_false,
        "true_ratio": n_true / n_total if n_total > 0 else 0.0,
        "false_ratio": n_false / n_total if n_total > 0 else 0.0,
    }


def analyze_per_family_balance(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute label balance per family."""
    by_family: Dict[str, List[str]] = defaultdict(list)
    for item in items:
        family = item.get("type", "unknown")
        label = item.get("ground_truth", "")
        by_family[family].append(label)

    family_stats = {}
    for family, labels in sorted(by_family.items()):
        n_true = sum(1 for l in labels if l in ("TRUE", "YES"))
        n_false = sum(1 for l in labels if l in ("FALSE", "NO"))
        n_total = len(labels)

        family_stats[family] = {
            "total": n_total,
            "true_count": n_true,
            "false_count": n_false,
            "true_ratio": n_true / n_total if n_total > 0 else 0.0,
            "false_ratio": n_false / n_total if n_total > 0 else 0.0,
        }

    return family_stats


def analyze_per_split_balance(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute label balance per split (inferred from batch file)."""
    from chaosbench.data.splits import SPLIT_ASSIGNMENTS

    by_split: Dict[str, List[str]] = defaultdict(list)
    for item in items:
        batch_name = item.get("_batch_file", "").replace(".jsonl", "")
        split = SPLIT_ASSIGNMENTS.get(batch_name, "unknown")
        label = item.get("ground_truth", "")
        by_split[split].append(label)

    split_stats = {}
    for split, labels in sorted(by_split.items()):
        n_true = sum(1 for l in labels if l in ("TRUE", "YES"))
        n_false = sum(1 for l in labels if l in ("FALSE", "NO"))
        n_total = len(labels)

        split_stats[split] = {
            "total": n_total,
            "true_count": n_true,
            "false_count": n_false,
            "true_ratio": n_true / n_total if n_total > 0 else 0.0,
            "false_ratio": n_false / n_total if n_total > 0 else 0.0,
        }

    return split_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze label balance in dataset")
    parser.add_argument("--data_dir", default="data/", help="Data directory")
    parser.add_argument("--out_dir", default="reports/", help="Output directory")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    print(f"Loading items from {args.data_dir}...")
    items = load_all_items(args.data_dir)
    print(f"  Loaded {len(items)} items")

    print("\nAnalyzing overall balance...")
    overall = analyze_overall_balance(items)
    print(f"  TRUE: {overall['true_ratio']*100:.1f}% ({overall['true_count']}/{overall['total']})")
    print(f"  FALSE: {overall['false_ratio']*100:.1f}% ({overall['false_count']}/{overall['total']})")

    print("\nAnalyzing per-family balance...")
    family_stats = analyze_per_family_balance(items)

    print("\nAnalyzing per-split balance...")
    split_stats = analyze_per_split_balance(items)

    # Create output directory
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write JSON report
    report = {
        "overall": overall,
        "per_family": family_stats,
        "per_split": split_stats,
    }
    json_path = out_path / "label_balance.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote JSON report to {json_path}")

    # Write CSV report
    csv_path = out_path / "label_balance.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "value", "total", "true_count", "false_count", "true_ratio", "false_ratio"])

        # Overall
        writer.writerow(["overall", "all", overall["total"], overall["true_count"], overall["false_count"], f"{overall['true_ratio']:.3f}", f"{overall['false_ratio']:.3f}"])

        # Per family
        for family, stats in sorted(family_stats.items()):
            writer.writerow(["family", family, stats["total"], stats["true_count"], stats["false_count"], f"{stats['true_ratio']:.3f}", f"{stats['false_ratio']:.3f}"])

        # Per split
        for split, stats in sorted(split_stats.items()):
            writer.writerow(["split", split, stats["total"], stats["true_count"], stats["false_count"], f"{stats['true_ratio']:.3f}", f"{stats['false_ratio']:.3f}"])

    print(f"Wrote CSV report to {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("LABEL BALANCE SUMMARY")
    print("=" * 80)
    print(f"Overall: {overall['true_ratio']*100:.1f}% TRUE | {overall['false_ratio']*100:.1f}% FALSE")

    print("\nFamilies with imbalance (outside 35%-65% range):")
    for family, stats in sorted(family_stats.items(), key=lambda x: abs(0.5 - x[1]["true_ratio"]), reverse=True):
        if stats["true_ratio"] < 0.35 or stats["true_ratio"] > 0.65:
            print(f"  {family:30s}: {stats['true_ratio']*100:5.1f}% TRUE ({stats['true_count']:4d}/{stats['total']:4d})")

    print(f"\nFull reports saved to {args.out_dir}")


if __name__ == "__main__":
    main()
