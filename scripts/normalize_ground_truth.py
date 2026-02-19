#!/usr/bin/env python3
"""Normalize ground_truth field from YES/NO to TRUE/FALSE in JSONL files.

Rewrites files in-place to ensure uniform TRUE/FALSE format across all data.

Usage:
    python scripts/normalize_ground_truth.py --data_dir data/archive/v1/ --dry_run
    python scripts/normalize_ground_truth.py --data_dir data/archive/v1/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def normalize_ground_truth(value: str) -> str:
    """Normalize ground truth to canonical TRUE/FALSE format.

    Args:
        value: Ground truth value.

    Returns:
        Normalized value: "TRUE" or "FALSE", or unchanged if non-binary.
    """
    normalized = value.upper().strip()
    if normalized in {"TRUE", "YES", "Y", "T"}:
        return "TRUE"
    elif normalized in {"FALSE", "NO", "N", "F"}:
        return "FALSE"
    else:
        return value  # Pass through non-binary values


def process_jsonl_file(file_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """Process a JSONL file and normalize ground_truth fields.

    Args:
        file_path: Path to JSONL file.
        dry_run: If True, only count changes without writing.

    Returns:
        Dict with counts: total, changed, unchanged.
    """
    items = []
    changed_count = 0
    unchanged_count = 0

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            original = item.get("ground_truth", "")
            normalized = normalize_ground_truth(original)

            if original != normalized:
                changed_count += 1
                if not dry_run:
                    item["ground_truth"] = normalized
            else:
                unchanged_count += 1

            items.append(item)

    # Write file (if not dry run)
    if not dry_run and changed_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    return {
        "total": len(items),
        "changed": changed_count,
        "unchanged": unchanged_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize ground_truth fields to TRUE/FALSE")
    parser.add_argument("--data_dir", required=True, help="Data directory to process")
    parser.add_argument("--dry_run", action="store_true", help="Dry run (count changes without writing)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Directory {args.data_dir} does not exist")
        sys.exit(1)

    print(f"Processing JSONL files in {args.data_dir}...")
    if args.dry_run:
        print("  [DRY RUN] No files will be modified")

    jsonl_files = sorted(data_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  No JSONL files found in {args.data_dir}")
        sys.exit(0)

    total_changed = 0
    total_unchanged = 0
    total_items = 0

    for file_path in jsonl_files:
        stats = process_jsonl_file(file_path, dry_run=args.dry_run)
        total_items += stats["total"]
        total_changed += stats["changed"]
        total_unchanged += stats["unchanged"]

        if stats["changed"] > 0:
            print(f"  {file_path.name}: {stats['changed']} changed, {stats['unchanged']} unchanged")
        else:
            print(f"  {file_path.name}: All {stats['total']} already normalized")

    print("\n" + "=" * 70)
    print("NORMALIZATION SUMMARY")
    print("=" * 70)
    print(f"Total items: {total_items}")
    print(f"Changed (YES/NO → TRUE/FALSE): {total_changed}")
    print(f"Unchanged (already TRUE/FALSE): {total_unchanged}")

    if args.dry_run:
        print("\n[DRY RUN] Re-run without --dry_run to apply changes")
    elif total_changed > 0:
        print(f"\n✓ Successfully normalized {total_changed} items")


if __name__ == "__main__":
    main()
