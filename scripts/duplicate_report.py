#!/usr/bin/env python3
"""Generate duplicate detection report for ChaosBench-Logic dataset.

Distinguishes intentional perturbation groups from accidental duplicates.

Usage:
    python scripts/duplicate_report.py --data_dir data/ --out_dir reports/duplicates/
    python scripts/duplicate_report.py --data_dir data/ --near_threshold 0.92
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.data.grouping import compute_group_id, is_accidental_duplicate, _normalize_text
from chaosbench.quality.gates import _text_hash, _jaccard_similarity


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


def detect_exact_duplicates(items: List[Dict[str, Any]]) -> Tuple[List[List[str]], List[List[str]]]:
    """Detect exact duplicates and classify as intentional groups vs accidental.

    Args:
        items: List of question dicts.

    Returns:
        Tuple of (intentional_groups, accidental_groups).
    """
    # Phase 1: Hash-based grouping
    seen_hashes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        text = item.get("question", "")
        h = _text_hash(text)
        seen_hashes[h].append(item)

    # Phase 2: Classify each duplicate group
    intentional_groups = []
    accidental_groups = []

    for h, group in seen_hashes.items():
        if len(group) <= 1:
            continue  # Not a duplicate

        # Check if all items in group have same system_id, ground_truth, type
        # If yes: accidental duplicate
        # If no: may be intentional (different systems or ground truths)
        first = group[0]
        all_same = all(
            item.get("system_id") == first.get("system_id")
            and item.get("ground_truth") == first.get("ground_truth")
            and item.get("type") == first.get("type")
            for item in group
        )

        ids = [item["id"] for item in group]

        if all_same:
            accidental_groups.append(ids)
        else:
            intentional_groups.append(ids)

    return intentional_groups, accidental_groups


def detect_near_duplicates(
    items: List[Dict[str, Any]],
    threshold: float = 0.92,
) -> List[Tuple[str, str, float]]:
    """Detect near-duplicates using Jaccard similarity.

    Args:
        items: List of question dicts.
        threshold: Similarity threshold (default: 0.92).

    Returns:
        List of (id_a, id_b, similarity) tuples.
    """
    # Exclude exact duplicates (already covered)
    unique_texts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        h = _text_hash(item.get("question", ""))
        unique_texts[h].append(item)

    unique_items = [group[0] for group in unique_texts.values()]

    near_duplicates = []
    n = len(unique_items)

    for i in range(n):
        for j in range(i + 1, min(i + 100, n)):  # Window-based check
            text_a = unique_items[i].get("question", "")
            text_b = unique_items[j].get("question", "")
            sim = _jaccard_similarity(text_a, text_b)

            if sim >= threshold:
                near_duplicates.append((
                    unique_items[i]["id"],
                    unique_items[j]["id"],
                    sim,
                ))

    return near_duplicates


def analyze_per_family(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute duplicate statistics per family.

    Args:
        items: List of question dicts.

    Returns:
        Dict mapping family name to stats dict.
    """
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        family = item.get("type", "unknown")
        by_family[family].append(item)

    family_stats = {}

    for family, family_items in by_family.items():
        # Count exact duplicates within family
        seen_hashes: Dict[str, List[str]] = defaultdict(list)
        for item in family_items:
            h = _text_hash(item.get("question", ""))
            seen_hashes[h].append(item["id"])

        exact_dups = sum(len(ids) - 1 for ids in seen_hashes.values() if len(ids) > 1)

        family_stats[family] = {
            "total_items": len(family_items),
            "exact_duplicates": exact_dups,
            "duplicate_rate": exact_dups / len(family_items) if family_items else 0.0,
        }

    return family_stats


def analyze_per_batch(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute duplicate statistics per batch.

    Args:
        items: List of question dicts.

    Returns:
        Dict mapping batch filename to stats dict.
    """
    by_batch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        batch = item.get("_batch_file", "unknown")
        by_batch[batch].append(item)

    batch_stats = {}

    for batch, batch_items in by_batch.items():
        # Count exact duplicates within batch
        seen_hashes: Dict[str, List[str]] = defaultdict(list)
        for item in batch_items:
            h = _text_hash(item.get("question", ""))
            seen_hashes[h].append(item["id"])

        exact_dups = sum(len(ids) - 1 for ids in seen_hashes.values() if len(ids) > 1)

        batch_stats[batch] = {
            "total_items": len(batch_items),
            "exact_duplicates": exact_dups,
            "duplicate_rate": exact_dups / len(batch_items) if batch_items else 0.0,
        }

    return batch_stats


def main():
    parser = argparse.ArgumentParser(description="Generate duplicate detection report")
    parser.add_argument("--data_dir", default="data/", help="Data directory")
    parser.add_argument("--out_dir", default="reports/duplicates/", help="Output directory")
    parser.add_argument("--near_threshold", type=float, default=0.92, help="Near-duplicate threshold")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    print(f"Loading items from {args.data_dir}...")
    items = load_all_items(args.data_dir)
    print(f"  Loaded {len(items)} items")

    print("\nDetecting exact duplicates...")
    intentional_groups, accidental_groups = detect_exact_duplicates(items)
    n_intentional = sum(len(g) - 1 for g in intentional_groups)
    n_accidental = sum(len(g) - 1 for g in accidental_groups)
    print(f"  Intentional groups: {len(intentional_groups)} groups ({n_intentional} duplicate items)")
    print(f"  Accidental duplicates: {len(accidental_groups)} groups ({n_accidental} duplicate items)")

    print("\nDetecting near-duplicates...")
    near_duplicates = detect_near_duplicates(items, threshold=args.near_threshold)
    print(f"  Found {len(near_duplicates)} near-duplicate pairs (threshold={args.near_threshold})")

    print("\nAnalyzing per-family statistics...")
    family_stats = analyze_per_family(items)

    print("\nAnalyzing per-batch statistics...")
    batch_stats = analyze_per_batch(items)

    # Create output directory
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write exact duplicates CSV
    exact_dups_path = out_path / "exact_duplicates.csv"
    with open(exact_dups_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "group_size", "sample_ids"])
        for group in intentional_groups:
            writer.writerow(["intentional", len(group), "; ".join(group[:5])])
        for group in accidental_groups:
            writer.writerow(["accidental", len(group), "; ".join(group[:5])])
    print(f"\nWrote exact duplicates to {exact_dups_path}")

    # Write near-duplicates CSV
    near_dups_path = out_path / "near_duplicates.csv"
    with open(near_dups_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_a", "id_b", "similarity"])
        for id_a, id_b, sim in near_duplicates:
            writer.writerow([id_a, id_b, f"{sim:.4f}"])
    print(f"Wrote near-duplicates to {near_dups_path}")

    # Write per-family summary JSON
    family_summary_path = out_path / "per_family_summary.json"
    with open(family_summary_path, "w") as f:
        json.dump(family_stats, f, indent=2)
    print(f"Wrote per-family summary to {family_summary_path}")

    # Write per-batch summary JSON
    batch_summary_path = out_path / "per_batch_summary.json"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_stats, f, indent=2)
    print(f"Wrote per-batch summary to {batch_summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DUPLICATE REPORT SUMMARY")
    print("=" * 70)
    print(f"Total items: {len(items)}")
    print(f"Exact duplicates (accidental): {n_accidental} ({n_accidental/len(items)*100:.1f}%)")
    print(f"Exact duplicates (intentional groups): {n_intentional} ({n_intentional/len(items)*100:.1f}%)")
    print(f"Near-duplicates (pairs): {len(near_duplicates)}")
    print("\nTop families by duplicate rate:")
    sorted_families = sorted(
        family_stats.items(),
        key=lambda x: x[1]["duplicate_rate"],
        reverse=True
    )
    for family, stats in sorted_families[:5]:
        print(f"  {family}: {stats['duplicate_rate']*100:.1f}% ({stats['exact_duplicates']}/{stats['total_items']})")

    print(f"\nFull reports saved to {args.out_dir}")


if __name__ == "__main__":
    main()
