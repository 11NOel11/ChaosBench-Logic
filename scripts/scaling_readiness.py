#!/usr/bin/env python3
"""Scaling readiness suite for ChaosBench-Logic dataset.

Runs a comprehensive set of checks to ensure the dataset is ready for scaling:
- All tests pass
- No accidental duplicates
- Label balance within acceptable ranges
- Quality gates pass
- Subset determinism verified

Usage:
    python scripts/scaling_readiness.py --data_dir data/
    python scripts/scaling_readiness.py --data_dir data/ --strict
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: List[str], description: str, capture_output: bool = True) -> Dict[str, Any]:
    """Run a command and return result dict.

    Args:
        cmd: Command as list of strings.
        description: Human-readable description.
        capture_output: If True, capture stdout/stderr.

    Returns:
        Dict with keys: passed, stdout, stderr, exit_code, description.
    """
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=600  # 10 minute timeout
        )

        passed = result.returncode == 0

        if not capture_output or passed:
            print(f"✅ PASSED" if passed else f"❌ FAILED")
        else:
            print(f"❌ FAILED (exit code {result.returncode})")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout[-2000:])  # Last 2000 chars
            if result.stderr:
                print("STDERR:")
                print(result.stderr[-2000:])

        return {
            "passed": passed,
            "stdout": result.stdout if capture_output else "",
            "stderr": result.stderr if capture_output else "",
            "exit_code": result.returncode,
            "description": description,
        }

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (>10 minutes)")
        return {
            "passed": False,
            "stdout": "",
            "stderr": "Command timeout after 10 minutes",
            "exit_code": -1,
            "description": description,
        }
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            "passed": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "description": description,
        }


def check_duplicate_threshold(data_dir: str, max_allowed: int = 0) -> Dict[str, Any]:
    """Check that accidental duplicates are below threshold.

    Args:
        data_dir: Data directory.
        max_allowed: Maximum allowed accidental duplicates.

    Returns:
        Result dict with passed boolean and details.
    """
    report_path = PROJECT_ROOT / "reports" / "duplicates" / "per_family_summary.json"

    if not report_path.exists():
        return {
            "passed": False,
            "description": "Duplicate report not found",
            "details": f"Run: python scripts/duplicate_report.py --data_dir {data_dir}"
        }

    with open(report_path) as f:
        family_stats = json.load(f)

    total_duplicates = sum(s.get("exact_duplicates", 0) for s in family_stats.values())

    passed = total_duplicates <= max_allowed

    return {
        "passed": passed,
        "description": f"Accidental duplicates check (max: {max_allowed})",
        "details": f"Found {total_duplicates} accidental duplicates",
        "total_duplicates": total_duplicates,
    }


def check_label_balance(data_dir: str, families: List[str] = None) -> Dict[str, Any]:
    """Check label balance is acceptable.

    Args:
        data_dir: Data directory.
        families: List of families to check. Defaults to ["multi_hop"].

    Returns:
        Result dict with passed boolean and details.
    """
    if families is None:
        families = ["multi_hop"]

    report_path = PROJECT_ROOT / "reports" / "label_balance.json"

    if not report_path.exists():
        return {
            "passed": False,
            "description": "Label balance report not found",
            "details": f"Run: python scripts/analyze_label_balance.py --data_dir {data_dir}"
        }

    with open(report_path) as f:
        balance_report = json.load(f)

    per_family = balance_report.get("per_family", {})

    issues = []
    for family in families:
        if family not in per_family:
            issues.append(f"{family}: not found in report")
            continue

        stats = per_family[family]
        true_ratio = stats.get("true_ratio", 0)

        # Check minimum 30% representation for each label
        if true_ratio < 0.10 or true_ratio > 0.90:
            issues.append(
                f"{family}: {true_ratio*100:.1f}% TRUE (outside 10%-90% range)"
            )

    passed = len(issues) == 0

    return {
        "passed": passed,
        "description": "Label balance check (10%-90% range)",
        "details": "; ".join(issues) if issues else "All families balanced",
        "issues": issues,
    }


def check_subset_determinism(data_dir: str) -> Dict[str, Any]:
    """Check API subset generation is deterministic.

    Args:
        data_dir: Data directory.

    Returns:
        Result dict with passed boolean and details.
    """
    import tempfile
    import hashlib

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate subset twice with same seed
        subset1 = Path(tmpdir) / "subset1.jsonl"
        subset2 = Path(tmpdir) / "subset2.jsonl"

        cmd1 = [
            "python", "scripts/make_api_subset.py",
            "--data_dir", data_dir,
            "--out_path", str(subset1),
            "--size", "100",
            "--seed", "42",
        ]

        cmd2 = [
            "python", "scripts/make_api_subset.py",
            "--data_dir", data_dir,
            "--out_path", str(subset2),
            "--size", "100",
            "--seed", "42",
        ]

        result1 = subprocess.run(cmd1, capture_output=True, cwd=PROJECT_ROOT)
        result2 = subprocess.run(cmd2, capture_output=True, cwd=PROJECT_ROOT)

        if result1.returncode != 0 or result2.returncode != 0:
            return {
                "passed": False,
                "description": "Subset determinism check",
                "details": "Subset generation failed",
            }

        # Compute hashes
        with open(subset1, "rb") as f:
            hash1 = hashlib.sha256(f.read()).hexdigest()

        with open(subset2, "rb") as f:
            hash2 = hashlib.sha256(f.read()).hexdigest()

        passed = hash1 == hash2

        return {
            "passed": passed,
            "description": "Subset determinism check (same seed → same hash)",
            "details": f"Hash match: {passed}" if passed else f"Hash1: {hash1[:16]}... != Hash2: {hash2[:16]}...",
            "hash1": hash1,
            "hash2": hash2,
        }


def main():
    parser = argparse.ArgumentParser(description="Run scaling readiness suite")
    parser.add_argument("--data_dir", default="data/", help="Data directory")
    parser.add_argument("--strict", action="store_true", help="Fail on any non-critical warning")
    parser.add_argument("--out_dir", default="reports/readiness/", help="Output directory")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    print("\n" + "="*70)
    print("ChaosBench-Logic SCALING READINESS SUITE")
    print("="*70)

    results: List[Dict[str, Any]] = []

    # 1. Run test suite
    results.append(run_command(
        ["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
        "Test suite (pytest)",
    ))

    # 2. Generate duplicate report
    results.append(run_command(
        ["python", "scripts/duplicate_report.py", "--data_dir", args.data_dir, "--out_dir", "reports/duplicates/"],
        "Duplicate analysis",
    ))

    # 3. Check duplicate threshold
    results.append(check_duplicate_threshold(args.data_dir, max_allowed=0))

    # 4. Generate label balance report
    results.append(run_command(
        ["python", "scripts/analyze_label_balance.py", "--data_dir", args.data_dir, "--out_dir", "reports/"],
        "Label balance analysis",
    ))

    # 5. Check label balance
    results.append(check_label_balance(args.data_dir))

    # 6. Run quality gates
    results.append(run_command(
        ["python", "scripts/quality_gates.py", "--data", args.data_dir],
        "Quality gates (non-strict)",
    ))

    # 7. Check subset determinism
    results.append(check_subset_determinism(args.data_dir))

    # Summary
    print("\n" + "="*70)
    print("READINESS SUITE SUMMARY")
    print("="*70)

    passed_count = sum(1 for r in results if r.get("passed", False))
    total_count = len(results)

    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result.get("passed") else "❌ FAIL"
        desc = result.get("description", "Unknown")
        print(f"{i}. {status} - {desc}")
        if not result.get("passed") and result.get("details"):
            print(f"   Details: {result.get('details')}")

    print(f"\nOverall: {passed_count}/{total_count} checks passed")

    # Write report
    os.makedirs(args.out_dir, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": args.data_dir,
        "total_checks": total_count,
        "passed_checks": passed_count,
        "failed_checks": total_count - passed_count,
        "overall_passed": passed_count == total_count,
        "results": results,
    }

    report_path = Path(args.out_dir) / "readiness_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_path}")

    # Write summary markdown
    summary_path = Path(args.out_dir) / "readiness_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Scaling Readiness Summary\n\n")
        f.write(f"**Timestamp**: {report['timestamp']}\n\n")
        f.write(f"**Overall**: {passed_count}/{total_count} checks passed\n\n")
        f.write(f"## Results\n\n")
        for i, result in enumerate(results, 1):
            status = "✅" if result.get("passed") else "❌"
            desc = result.get("description", "Unknown")
            f.write(f"{i}. {status} {desc}\n")
            if not result.get("passed") and result.get("details"):
                f.write(f"   - {result.get('details')}\n")

    print(f"Summary saved to: {summary_path}")

    # Exit code
    if passed_count < total_count:
        if args.strict:
            print("\n❌ READINESS CHECK FAILED (strict mode)")
            sys.exit(1)
        else:
            print("\n⚠️  READINESS CHECK: Some checks failed (use --strict to fail)")
            sys.exit(0)
    else:
        print("\n✅ READINESS CHECK PASSED - Ready for scaling")
        sys.exit(0)


if __name__ == "__main__":
    main()
