#!/usr/bin/env python3
"""Phase 2 heavy dataset verification.

Validates schema, ground_truth canonicality, uniqueness, duplicate detection,
and distribution sanity across all canonical v2 dataset files.

Usage:
    python scripts/heavy_verify_dataset.py [--data-dir data/] [--output-dir artifacts/heavy_verify/]

Exit Codes:
    0  All checks passed
    1  One or more hard-fail checks failed
    2  Script error

Outputs:
    artifacts/heavy_verify/dataset_report.json
    artifacts/heavy_verify/dataset_report.md
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _load_canonical_files(data_dir: Path) -> List[str]:
    """Load canonical file names from data/canonical_v2_files.json."""
    selector_path = data_dir.parent / "data" / "canonical_v2_files.json"
    if not selector_path.exists():
        # Fallback: check same dir
        selector_path = data_dir / "canonical_v2_files.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    # Return just the file basenames (strip the "data/" prefix)
    return [Path(f).name for f in selector["files"]]

REQUIRED_FIELDS = {
    "id": str,
    "question": str,
    "ground_truth": str,
    "type": str,
}

OPTIONAL_FIELDS = {
    "system_id": str,
    "template": str,
    "difficulty": (int, float),
    "group_id": str,
    "split": str,
}

VALID_GROUND_TRUTHS = {"TRUE", "FALSE"}

REGISTERED_FAMILIES = {
    # atomic
    "atomic",
    # consistency
    "consistency_paraphrase",
    # perturbation
    "perturbation",
    # multi-hop
    "multi_hop",
    # FOL
    "fol_inference",
    # adversarial
    "adversarial_misleading",
    "adversarial_nearmiss",
    "adversarial_confusion",
    "adversarial",
    # indicator diagnostics (both singular and plural forms are valid)
    "indicator_diagnostics",
    "indicator_diagnostic",
    # cross-indicator
    "cross_indicator",
    # regime transition
    "regime_transition",
    # extended systems
    "extended_systems",
}

# Cross-family duplicate pairs that are INTENTIONAL by design.
# - atomic ↔ consistency_paraphrase: base atomic question reused as paraphrase seed
# - perturbation ↔ consistency_paraphrase: both use paraphrase variants of atomic Qs
# - extended_systems ↔ atomic: extended_systems reuses atomic question templates
INTENTIONAL_CROSS_FAMILY_DUPLICATE_PAIRS = {
    frozenset({"atomic", "consistency_paraphrase"}),
    frozenset({"perturbation", "consistency_paraphrase"}),
    frozenset({"extended_systems", "atomic"}),
}

MIN_QUESTION_LEN = 20
MAX_QUESTION_LEN = 2000


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_record(record: Dict[str, Any], file_name: str, line_no: int) -> List[str]:
    """Validate a single record against the schema. Returns list of errors."""
    errors = []

    # Required fields
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"{file_name}:{line_no} missing required field '{field}'")
            continue
        val = record[field]
        if not isinstance(val, expected_type):
            errors.append(
                f"{file_name}:{line_no} field '{field}' has wrong type "
                f"(got {type(val).__name__}, expected {expected_type.__name__})"
            )

    # ground_truth must be TRUE or FALSE
    gt = record.get("ground_truth", "")
    if isinstance(gt, str) and gt not in VALID_GROUND_TRUTHS:
        errors.append(
            f"{file_name}:{line_no} ground_truth='{gt}' is not in {{TRUE,FALSE}}"
        )

    # question length
    q = record.get("question", "")
    if isinstance(q, str):
        if len(q) < MIN_QUESTION_LEN:
            errors.append(
                f"{file_name}:{line_no} question too short ({len(q)} chars, min={MIN_QUESTION_LEN})"
            )
        if len(q) > MAX_QUESTION_LEN:
            errors.append(
                f"{file_name}:{line_no} question too long ({len(q)} chars, max={MAX_QUESTION_LEN})"
            )

    # type must be in registered families
    item_type = record.get("type", "")
    if isinstance(item_type, str) and item_type and item_type not in REGISTERED_FAMILIES:
        errors.append(
            f"{file_name}:{line_no} unknown type '{item_type}' "
            f"(not in registered families)"
        )

    # Optional field type checks
    for field, expected_type in OPTIONAL_FIELDS.items():
        if field not in record:
            continue
        val = record[field]
        if val is None:
            continue
        if isinstance(expected_type, tuple):
            if not isinstance(val, expected_type):
                errors.append(
                    f"{file_name}:{line_no} optional field '{field}' has wrong type "
                    f"(got {type(val).__name__})"
                )
        else:
            if not isinstance(val, expected_type):
                errors.append(
                    f"{file_name}:{line_no} optional field '{field}' has wrong type "
                    f"(got {type(val).__name__}, expected {expected_type.__name__})"
                )

    return errors


# ---------------------------------------------------------------------------
# Normalization for duplicate detection
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def run_verification(data_dir: Path, output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Run all Phase 2 verifications. Returns (passed, report_dict)."""

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "files_checked": [],
        "schema_errors": [],
        "ground_truth_errors": [],
        "id_errors": [],
        "duplicate_errors": [],
        "distribution": {},
        "label_balance": {},
        "summary": {},
    }

    all_records: List[Dict[str, Any]] = []
    all_ids: Dict[str, str] = {}  # id -> file
    schema_errors: List[str] = []
    missing_files: List[str] = []
    file_stats: List[Dict[str, Any]] = []

    print("=" * 60)
    print("Phase 2 — Heavy Dataset Verification")
    print("=" * 60)

    canonical_files = _load_canonical_files(data_dir)

    # --- A. Load and validate schema ---
    print("\n[A] Schema validation...")
    for fname in canonical_files:
        fpath = data_dir / fname
        if not fpath.exists():
            missing_files.append(fname)
            print(f"  MISSING: {fname}")
            continue

        count = 0
        file_errors = 0
        for line_no, line in enumerate(fpath.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                schema_errors.append(f"{fname}:{line_no} JSON parse error: {e}")
                file_errors += 1
                continue

            errs = validate_record(record, fname, line_no)
            schema_errors.extend(errs)
            file_errors += len(errs)
            record["_file"] = fname
            all_records.append(record)
            count += 1

        file_stats.append({
            "file": fname,
            "count": count,
            "schema_errors": file_errors,
        })
        status = "OK" if file_errors == 0 else f"FAIL ({file_errors} errors)"
        print(f"  {fname}: {count} records — {status}")

    report["files_checked"] = file_stats
    report["schema_errors"] = schema_errors

    if missing_files:
        print(f"\nFAIL: Missing canonical files: {missing_files}")
        report["summary"]["missing_files"] = missing_files

    print(f"\n  Total records loaded: {len(all_records)}")
    print(f"  Schema errors: {len(schema_errors)}")

    # --- B. Uniqueness: check for duplicate IDs ---
    print("\n[B] ID uniqueness check...")
    id_errors: List[str] = []
    for rec in all_records:
        rec_id = rec.get("id", "")
        if not rec_id:
            id_errors.append(f"Record in {rec['_file']} has empty/missing id")
            continue
        if rec_id in all_ids:
            id_errors.append(
                f"Duplicate id '{rec_id}' in {rec['_file']} (first seen in {all_ids[rec_id]})"
            )
        else:
            all_ids[rec_id] = rec["_file"]

    report["id_errors"] = id_errors
    print(f"  Unique IDs: {len(all_ids)}")
    print(f"  Duplicate ID violations: {len(id_errors)}")

    # --- C. Duplicate detection (exact and normalized) ---
    print("\n[C] Duplicate question detection...")

    # Build mapping: question -> (id, file, type) for first occurrence
    seen_raw: Dict[str, tuple] = {}  # q -> (id, file, type)
    seen_norm: Dict[str, tuple] = {}
    dup_exact_genuine: List[str] = []
    dup_exact_intentional: List[str] = []
    dup_norm_genuine: List[str] = []

    # Build type lookup: id -> type
    id_to_type: Dict[str, str] = {rec.get("id", ""): rec.get("type", "") for rec in all_records}

    for rec in all_records:
        q = rec.get("question", "")
        rec_id = rec.get("id", "?")
        file_name = rec.get("_file", "?")
        rec_type = rec.get("type", "")

        if q in seen_raw:
            first_id, first_file, first_type = seen_raw[q]
            pair = frozenset({rec_type, first_type})
            if pair in INTENTIONAL_CROSS_FAMILY_DUPLICATE_PAIRS:
                dup_exact_intentional.append(
                    f"Intentional dup ({rec_type}↔{first_type}): "
                    f"'{rec_id}' ≈ '{first_id}'"
                )
            else:
                dup_exact_genuine.append(
                    f"Exact dup: '{rec_id}' in {file_name} ({rec_type}) "
                    f"== '{first_id}' in {first_file} ({first_type})"
                )
        else:
            seen_raw[q] = (rec_id, file_name, rec_type)

        norm_q = _normalize_text(q)
        if norm_q and norm_q in seen_norm:
            first_id, first_file, first_type = seen_norm[norm_q]
            pair = frozenset({rec_type, first_type})
            if pair not in INTENTIONAL_CROSS_FAMILY_DUPLICATE_PAIRS:
                dup_norm_genuine.append(
                    f"Norm dup: '{rec_id}' in {file_name} ≈ '{first_id}' in {first_file}"
                )
        else:
            seen_norm[norm_q] = (rec_id, file_name, rec_type)

    report["duplicate_errors"] = {
        "exact_duplicates_genuine": dup_exact_genuine,
        "exact_duplicates_intentional_count": len(dup_exact_intentional),
        "normalized_duplicates": dup_norm_genuine,
    }
    print(f"  Exact duplicates (genuine): {len(dup_exact_genuine)}")
    print(
        f"  Exact duplicates (intentional by design, atomic↔consistency): "
        f"{len(dup_exact_intentional)}"
    )
    print(f"  Normalized duplicates (genuine): {len(dup_norm_genuine)}")

    # --- D. Distribution sanity ---
    print("\n[D] Distribution sanity...")
    family_counts: Dict[str, int] = defaultdict(int)
    family_true: Dict[str, int] = defaultdict(int)
    system_counts: Dict[str, int] = defaultdict(int)
    system_family: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    gt_counts: Dict[str, int] = defaultdict(int)

    for rec in all_records:
        item_type = rec.get("type", "unknown")
        system_id = rec.get("system_id", "unknown")
        gt = rec.get("ground_truth", "?")

        family_counts[item_type] += 1
        gt_counts[gt] += 1
        if gt == "TRUE":
            family_true[item_type] += 1
        system_counts[system_id] += 1
        system_family[system_id][item_type] += 1

    total = len(all_records)
    true_count = gt_counts.get("TRUE", 0)
    false_count = gt_counts.get("FALSE", 0)
    other_gt = {k: v for k, v in gt_counts.items() if k not in ("TRUE", "FALSE")}

    overall_balance = true_count / total if total > 0 else 0.0

    label_balance: Dict[str, Any] = {
        "total": total,
        "TRUE": true_count,
        "FALSE": false_count,
        "other": other_gt,
        "TRUE_pct": round(overall_balance * 100, 2),
        "per_family": {},
    }

    balance_issues = []
    for fam, count in sorted(family_counts.items()):
        true_n = family_true.get(fam, 0)
        false_n = count - true_n
        bal = true_n / count if count > 0 else 0.5
        label_balance["per_family"][fam] = {
            "count": count,
            "TRUE": true_n,
            "FALSE": false_n,
            "TRUE_pct": round(bal * 100, 2),
        }
        if count >= 20 and (bal < 0.30 or bal > 0.70):
            balance_issues.append(f"{fam}: {bal*100:.1f}% TRUE (count={count})")
        print(f"  {fam}: {count} items, {bal*100:.1f}% TRUE")

    report["distribution"] = {
        "family_counts": dict(family_counts),
        "system_count": len(system_counts),
        "top_systems": dict(
            sorted(system_counts.items(), key=lambda x: -x[1])[:10]
        ),
    }
    report["label_balance"] = label_balance

    print(f"\n  Overall: {total} items, {overall_balance*100:.1f}% TRUE")
    print(f"  Ground truth values: {dict(gt_counts)}")

    if other_gt:
        print(f"  WARNING: Non-canonical ground truth values: {other_gt}")

    # --- Summary ---
    hard_fails = []

    if missing_files:
        hard_fails.append(f"Missing canonical files: {missing_files}")
    if schema_errors:
        hard_fails.append(f"{len(schema_errors)} schema validation errors")
    if other_gt:
        hard_fails.append(
            f"Non-canonical ground_truth values found: {other_gt}. "
            "Only TRUE/FALSE are allowed."
        )
    if id_errors:
        hard_fails.append(f"{len(id_errors)} duplicate ID errors")
    dup_exact_genuine = report.get("duplicate_errors", {}).get("exact_duplicates_genuine", [])
    dup_norm_genuine = report.get("duplicate_errors", {}).get("normalized_duplicates", [])
    if dup_exact_genuine:
        hard_fails.append(f"{len(dup_exact_genuine)} genuine exact duplicate questions")
    if dup_norm_genuine:
        hard_fails.append(f"{len(dup_norm_genuine)} genuine normalized duplicate questions")
    if balance_issues:
        # Soft fail: report but don't hard-fail (some small families are exempt)
        print(f"\n  WARN: Label balance issues (>20 items, outside [30%,70%]):")
        for issue in balance_issues:
            print(f"    {issue}")

    report["summary"] = {
        "passed": len(hard_fails) == 0,
        "hard_fails": hard_fails,
        "balance_issues": balance_issues,
        "total_records": total,
        "schema_error_count": len(schema_errors),
        "id_error_count": len(id_errors),
        "exact_dup_count_genuine": len(dup_exact_genuine),
        "exact_dup_count_intentional": report.get("duplicate_errors", {}).get("exact_duplicates_intentional_count", 0),
        "norm_dup_count_genuine": len(dup_norm_genuine),
        "other_ground_truth": other_gt,
    }

    return len(hard_fails) == 0, report


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_reports(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "dataset_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  Wrote: {json_path}")

    # Markdown
    md_path = output_dir / "dataset_report.md"
    summary = report["summary"]
    balance = report.get("label_balance", {})
    dist = report.get("distribution", {})

    lines = [
        "# Dataset Verification Report — Phase 2",
        "",
        f"Generated: {report['timestamp']}",
        "",
        "## Summary",
        "",
        f"**Status**: {'✓ PASSED' if summary['passed'] else '✗ FAILED'}",
        f"- Total records: {summary['total_records']:,}",
        f"- Schema errors: {summary['schema_error_count']}",
        f"- Duplicate IDs: {summary['id_error_count']}",
        f"- Exact duplicate questions (genuine): {summary.get('exact_dup_count_genuine', 0)}",
        f"- Exact duplicate questions (intentional atomic↔consistency): {summary.get('exact_dup_count_intentional', 0)}",
        f"- Normalized duplicate questions (genuine): {summary.get('norm_dup_count_genuine', 0)}",
        f"- Non-canonical ground_truth values: {summary['other_ground_truth']}",
        "",
    ]

    if summary["hard_fails"]:
        lines += ["## Hard Failures", ""]
        for hf in summary["hard_fails"]:
            lines.append(f"- **FAIL**: {hf}")
        lines.append("")

    if summary["balance_issues"]:
        lines += ["## Label Balance Warnings", ""]
        for bi in summary["balance_issues"]:
            lines.append(f"- WARN: {bi}")
        lines.append("")

    lines += [
        "## Overall Label Distribution",
        "",
        f"- TRUE: {balance.get('TRUE', 0):,} ({balance.get('TRUE_pct', 0):.1f}%)",
        f"- FALSE: {balance.get('FALSE', 0):,}",
        f"- Total: {balance.get('total', 0):,}",
        "",
        "## Per-Family Distribution",
        "",
        "| Family | Count | TRUE | FALSE | TRUE% |",
        "|--------|-------|------|-------|-------|",
    ]

    for fam, stats in sorted(balance.get("per_family", {}).items()):
        lines.append(
            f"| {fam} | {stats['count']:,} | {stats['TRUE']:,} | {stats['FALSE']:,} | {stats['TRUE_pct']:.1f}% |"
        )

    lines += [
        "",
        "## Files Checked",
        "",
        "| File | Records | Schema Errors |",
        "|------|---------|---------------|",
    ]
    for fs in report.get("files_checked", []):
        lines.append(
            f"| {fs['file']} | {fs['count']:,} | {fs['schema_errors']} |"
        )

    if report.get("schema_errors"):
        lines += [
            "",
            "## Schema Errors (first 20)",
            "",
        ]
        for err in report["schema_errors"][:20]:
            lines.append(f"- {err}")

    if report.get("id_errors"):
        lines += [
            "",
            "## ID Errors (first 20)",
            "",
        ]
        for err in report["id_errors"][:20]:
            lines.append(f"- {err}")

    dup_info = report.get("duplicate_errors", {})
    if dup_info.get("exact_duplicates"):
        lines += [
            "",
            "## Exact Duplicate Questions (first 10)",
            "",
        ]
        for d in dup_info["exact_duplicates"][:10]:
            lines.append(f"- {d}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {md_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing canonical v22 JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "heavy_verify",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    try:
        passed, report = run_verification(args.data_dir, args.output_dir)
        write_reports(report, args.output_dir)
    except Exception as e:
        print(f"\nSCRIPT ERROR: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 2

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: ALL CHECKS PASSED ✓")
    else:
        print("RESULT: HARD FAILURES DETECTED ✗")
        for hf in report["summary"]["hard_fails"]:
            print(f"  - {hf}")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
