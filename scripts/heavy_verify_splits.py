#!/usr/bin/env python3
"""Phase 3 heavy split integrity verification.

Recomputes splits deterministically for all records, checks leakage, and
validates that each record is assigned to exactly one split.

Usage:
    python scripts/heavy_verify_splits.py [--data-dir data/] [--output-dir artifacts/heavy_verify/]

Exit Codes:
    0  All checks passed
    1  One or more hard-fail checks failed
    2  Script error

Outputs:
    artifacts/heavy_verify/splits_report.md
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.data.splits import (
    assign_split_v22,
    validate_splits,
    compute_split_stats,
    HELDOUT_SYSTEM_IDS,
    VALID_SPLITS,
)

def _load_canonical_files(data_dir: Path) -> List[str]:
    """Load canonical file names from data/canonical_v2_files.json."""
    selector_path = data_dir.parent / "data" / "canonical_v2_files.json"
    if not selector_path.exists():
        selector_path = data_dir / "canonical_v2_files.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    return [Path(f).name for f in selector["files"]]


def load_all_records(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all canonical records."""
    records = []
    for fname in _load_canonical_files(data_dir):
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        for line in fpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["_file"] = fname
                records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


def recompute_splits(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Recompute split assignment for all records deterministically."""
    splits: Dict[str, List[Dict[str, Any]]] = {s: [] for s in VALID_SPLITS}
    for rec in records:
        split = assign_split_v22(rec)
        splits[split].append(rec)
    return splits


def check_single_split_assignment(
    records: List[Dict[str, Any]],
    splits: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Verify each record is assigned to exactly one split."""
    errors = []
    all_split_ids: Set[str] = set()
    total_in_splits = sum(len(v) for v in splits.values())
    if total_in_splits != len(records):
        errors.append(
            f"Split total ({total_in_splits}) != total records ({len(records)}). "
            "Some records may be assigned to multiple splits."
        )
    return errors


def check_heldout_system_leakage(
    splits: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Check that heldout system IDs don't appear in non-heldout splits."""
    errors = []

    for split_name in ["core", "robustness", "hard", "heldout_templates"]:
        items = splits.get(split_name, [])
        leaked = [
            rec for rec in items
            if rec.get("system_id", "") in HELDOUT_SYSTEM_IDS
        ]
        if leaked:
            leaked_ids = sorted({rec.get("system_id", "") for rec in leaked})
            errors.append(
                f"Heldout system leakage into '{split_name}': "
                f"{len(leaked)} records from {leaked_ids[:5]}"
            )

    return errors


def check_id_overlap(splits: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Check no item ID appears in more than one split."""
    errors = []
    id_to_split: Dict[str, str] = {}

    for split_name, items in splits.items():
        for rec in items:
            rec_id = rec.get("id", "")
            if not rec_id:
                continue
            if rec_id in id_to_split:
                errors.append(
                    f"ID '{rec_id}' appears in both '{id_to_split[rec_id]}' and '{split_name}'"
                )
            else:
                id_to_split[rec_id] = split_name

    return errors


def compute_near_dup_leakage(
    splits: Dict[str, List[Dict[str, Any]]],
    threshold: float = 0.85,
    sample_size: int = 500,
) -> List[str]:
    """Optional: check for near-duplicate questions between heldout and core splits.

    Uses a lightweight Jaccard similarity check on word sets.
    Only checks a sample for performance.
    """
    warnings = []

    heldout_items = splits.get("heldout_systems", [])[:sample_size]
    core_items = splits.get("core", [])[:sample_size]

    if not heldout_items or not core_items:
        return warnings

    def word_set(text: str) -> Set[str]:
        return set(text.lower().split())

    core_sets = [(rec.get("id", ""), word_set(rec.get("question", "")))
                 for rec in core_items]

    near_dups = 0
    for held_rec in heldout_items:
        held_ws = word_set(held_rec.get("question", ""))
        if not held_ws:
            continue
        for core_id, core_ws in core_sets:
            if not core_ws:
                continue
            jaccard = len(held_ws & core_ws) / len(held_ws | core_ws)
            if jaccard >= threshold:
                near_dups += 1
                if near_dups <= 3:
                    warnings.append(
                        f"Near-dup: heldout '{held_rec.get('id')}' ≈ core '{core_id}' "
                        f"(Jaccard={jaccard:.2f})"
                    )

    if near_dups > 0:
        warnings.append(f"Total near-dups (sample): {near_dups}")

    return warnings


def run_verification(data_dir: Path, output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "hard_fails": [],
        "warnings": [],
        "split_stats": {},
        "summary": {},
    }

    print("=" * 60)
    print("Phase 3 — Split Integrity Verification")
    print("=" * 60)

    # Load records
    print("\nLoading records...")
    records = load_all_records(data_dir)
    print(f"  Loaded {len(records):,} records")

    if not records:
        report["hard_fails"].append("No records loaded — check data_dir")
        report["summary"]["passed"] = False
        return False, report

    # Recompute splits deterministically
    print("\nRecomputing splits...")
    splits = recompute_splits(records)
    for split_name in VALID_SPLITS:
        count = len(splits.get(split_name, []))
        pct = 100 * count / len(records) if records else 0
        print(f"  {split_name}: {count:,} ({pct:.1f}%)")

    # Compute stats
    stats = compute_split_stats(splits)
    report["split_stats"] = {
        k: {
            "item_count": v["item_count"] if isinstance(v, dict) else v,
            "system_count": v.get("system_count", 0) if isinstance(v, dict) else 0,
            "content_hash": v.get("content_hash", "") if isinstance(v, dict) else "",
        }
        for k, v in stats.items()
    }

    # Run existing validation
    print("\nRunning built-in split validation...")
    validation_errors = validate_splits(splits)
    if validation_errors:
        for err in validation_errors:
            print(f"  FAIL: {err}")
        report["hard_fails"].extend(validation_errors)
    else:
        print("  OK: Built-in validation passed")

    # Single split assignment
    print("\nChecking single-split assignment...")
    assignment_errors = check_single_split_assignment(records, splits)
    if assignment_errors:
        for err in assignment_errors:
            print(f"  FAIL: {err}")
        report["hard_fails"].extend(assignment_errors)
    else:
        print("  OK: Each record assigned to exactly one split")

    # Heldout system leakage
    print("\nChecking heldout system leakage...")
    leakage_errors = check_heldout_system_leakage(splits)
    if leakage_errors:
        for err in leakage_errors:
            print(f"  FAIL: {err}")
        report["hard_fails"].extend(leakage_errors)
    else:
        print("  OK: No heldout system leakage detected")

    # ID overlap
    print("\nChecking ID overlap across splits...")
    overlap_errors = check_id_overlap(splits)
    if overlap_errors:
        for err in overlap_errors[:10]:
            print(f"  FAIL: {err}")
        if len(overlap_errors) > 10:
            print(f"  ... and {len(overlap_errors) - 10} more")
        report["hard_fails"].extend(overlap_errors)
    else:
        print("  OK: No ID overlap between splits")

    # Near-duplicate leakage (optional/soft)
    print("\nChecking near-duplicate leakage (sample)...")
    nd_warnings = compute_near_dup_leakage(splits)
    if nd_warnings:
        for w in nd_warnings:
            print(f"  WARN: {w}")
        report["warnings"].extend(nd_warnings)
    else:
        print("  OK: No near-duplicates found in sample")

    # Summary
    passed = len(report["hard_fails"]) == 0
    report["summary"] = {
        "passed": passed,
        "hard_fail_count": len(report["hard_fails"]),
        "warning_count": len(report["warnings"]),
        "total_records": len(records),
        "split_counts": {s: len(splits.get(s, [])) for s in VALID_SPLITS},
        "heldout_system_ids": sorted(HELDOUT_SYSTEM_IDS),
    }

    return passed, report


def write_report(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "splits_report.md"

    summary = report["summary"]
    lines = [
        "# Split Integrity Report — Phase 3",
        "",
        f"Generated: {report['timestamp']}",
        "",
        "## Summary",
        "",
        f"**Status**: {'✓ PASSED' if summary['passed'] else '✗ FAILED'}",
        f"- Total records: {summary['total_records']:,}",
        f"- Hard failures: {summary['hard_fail_count']}",
        f"- Warnings: {summary['warning_count']}",
        "",
        "## Split Counts",
        "",
        "| Split | Count | Pct |",
        "|-------|-------|-----|",
    ]

    total = summary["total_records"]
    for split_name, count in sorted(summary.get("split_counts", {}).items()):
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"| {split_name} | {count:,} | {pct:.1f}% |")

    lines += [""]

    if report["hard_fails"]:
        lines += ["## Hard Failures", ""]
        for hf in report["hard_fails"]:
            lines.append(f"- **FAIL**: {hf}")
        lines.append("")

    if report["warnings"]:
        lines += ["## Warnings", ""]
        for w in report["warnings"]:
            lines.append(f"- WARN: {w}")
        lines.append("")

    lines += [
        "## Heldout System IDs",
        "",
        ", ".join(sorted(summary.get("heldout_system_ids", []))),
        "",
        "## Split Content Hashes",
        "",
        "| Split | Items | Content Hash |",
        "|-------|-------|-------------|",
    ]
    for split_name, stats in sorted(report.get("split_stats", {}).items()):
        if isinstance(stats, dict):
            lines.append(
                f"| {split_name} | {stats.get('item_count', 0):,} | `{stats.get('content_hash', '')}` |"
            )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Wrote: {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "heavy_verify",
    )
    args = parser.parse_args()

    try:
        passed, report = run_verification(args.data_dir, args.output_dir)
        write_report(report, args.output_dir)
    except Exception as e:
        print(f"\nSCRIPT ERROR: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 2

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: ALL CHECKS PASSED ✓")
    else:
        print("RESULT: HARD FAILURES DETECTED ✗")
        for hf in report["hard_fails"]:
            print(f"  - {hf}")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
