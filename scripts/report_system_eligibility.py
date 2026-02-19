#!/usr/bin/env python3
"""Generate a transparency report of system eligibility for all question families.

Usage:
    python scripts/report_system_eligibility.py
    python scripts/report_system_eligibility.py --json  # Machine-readable output
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.build_v2_dataset import load_all_systems, load_all_indicators
from chaosbench.data.eligibility import generate_eligibility_report, FAMILY_REQUIREMENTS


def main():
    parser = argparse.ArgumentParser(description="Report system eligibility for all families")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--systems-dir", default="systems", help="Path to systems directory")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    all_systems = load_all_systems(args.systems_dir)
    all_indicators = load_all_indicators(args.systems_dir)

    report = generate_eligibility_report(all_systems, all_indicators)

    if args.json:
        # Remove full system lists for readability
        for family_info in report["per_family"].values():
            del family_info["eligible_systems"]
        for sys_info in report["per_system"].values():
            del sys_info["reasons"]
        print(json.dumps(report, indent=2))
        return

    # Human-readable output
    print("=" * 70)
    print("  System Eligibility Report")
    print("=" * 70)
    print(f"  Total systems: {report['total_systems']}")
    print(f"  Total families: {report['total_families']}")

    print("\n  Per-Family Eligibility:")
    print(f"  {'Family':<30} {'Eligible':>8} {'Total':>6} {'%':>6}")
    print("  " + "-" * 52)
    for family in sorted(report["per_family"].keys()):
        info = report["per_family"][family]
        pct = info["eligible_count"] / info["total_systems"] * 100
        print(f"  {family:<30} {info['eligible_count']:>8} {info['total_systems']:>6} {pct:>5.1f}%")

    # Summary of systems with limited eligibility
    limited = []
    for sid, info in report["per_system"].items():
        if info["eligible_families"] < report["total_families"]:
            limited.append((sid, info["eligible_families"]))

    if limited:
        limited.sort(key=lambda x: x[1])
        print(f"\n  Systems with Limited Eligibility ({len(limited)} systems):")
        for sid, n_families in limited[:10]:
            reasons = report["per_system"][sid]["reasons"]
            ineligible = [f for f, ok in report["per_system"][sid]["eligibility"].items() if not ok]
            print(f"    {sid}: {n_families}/{report['total_families']} families")
            for f in ineligible[:3]:
                print(f"      - {f}: {reasons[f]}")
        if len(limited) > 10:
            print(f"    ... and {len(limited) - 10} more")


if __name__ == "__main__":
    main()
