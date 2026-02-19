#!/usr/bin/env python3
"""Run quality gates on the dataset.

Usage:
    python scripts/quality_gates.py --data data/
    python scripts/quality_gates.py --data data/ --strict
    python scripts/quality_gates.py --data data/ --skip-quality-gates  # Skip for dev
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from chaosbench.quality.gates import run_all_gates


def load_all_items(data_dir: str) -> list:
    """Load all JSONL items from a data directory."""
    items = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="Run quality gates on ChaosBench-Logic dataset")
    parser.add_argument("--data", required=True, help="Data directory containing JSONL files")
    parser.add_argument("--strict", action="store_true", help="Exit with error if any gate fails")
    parser.add_argument("--skip-quality-gates", action="store_true", help="Skip all gates (for dev)")
    parser.add_argument("--config", type=str, default=None, help="Quality gate config YAML")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.skip_quality_gates:
        print("  [SKIP] Quality gates skipped (--skip-quality-gates)")
        sys.exit(0)

    items = load_all_items(args.data)
    print(f"  Loaded {len(items)} items from {args.data}")

    config = {}
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            full_cfg = yaml.safe_load(f)
        config = full_cfg.get("quality_gates", {})

    results = run_all_gates(items, config)

    print("\n" + "=" * 70)
    print("  QUALITY GATES")
    print("=" * 70)

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.gate_name}: {result.details}")
        if result.violations:
            for v in result.violations[:5]:
                print(f"         - {v}")
        if not result.passed:
            all_passed = False

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    if args.strict and not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
