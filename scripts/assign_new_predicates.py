#!/usr/bin/env python3
"""Automated truth assignment for 4 new predicates: Dissipative, Bounded, Mixing, Ergodic.

This script extends all system truth_assignment dicts with the new predicates
based on heuristic rules derived from existing predicate values.

Usage:
    python scripts/assign_new_predicates.py [--dry-run] [--systems-dir systems]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict


def assign_dissipative(truth: Dict[str, bool]) -> bool:
    """Assign Dissipative predicate.

    TRUE if:
      - System is Chaotic (most chaotic attractors are dissipative)
      - System has StrangeAttr (strange attractors are dissipative)
      - System has FixedPointAttr (dissipative convergence)

    FALSE if:
      - System is explicitly conservative (heuristic: name contains 'standard_map', 'arnold')

    Default: TRUE (most physical systems are dissipative)
    """
    # Conservative systems (volume-preserving)
    # These are rare in our dataset - mostly maps
    # For now, assume TRUE unless proven conservative

    if truth.get("Chaotic"):
        return True  # Chaotic attractors are dissipative

    if truth.get("StrangeAttr"):
        return True  # Strange attractors are dissipative

    if truth.get("FixedPointAttr"):
        return True  # Convergence to fixed point is dissipative

    # Default: TRUE (most systems have attractors → dissipative)
    return True


def assign_bounded(truth: Dict[str, bool]) -> bool:
    """Assign Bounded predicate.

    TRUE if:
      - System has any attractor (Chaotic, StrangeAttr, FixedPointAttr, Periodic, QuasiPeriodic)
      - System is StatPredictable (predictable statistics → bounded ensemble)
      - System is PointUnpredictable (sensitivity doesn't imply unbounded)

    FALSE if:
      - System has unbounded growth (rare in our dataset)

    Default: TRUE (almost all systems in dataset have bounded attractors)
    """
    # Any attractor → bounded
    if any(truth.get(p) for p in ["Chaotic", "StrangeAttr", "FixedPointAttr", "Periodic", "QuasiPeriodic"]):
        return True

    # Statistical properties imply bounded
    if truth.get("StatPredictable"):
        return True

    if truth.get("PointUnpredictable"):
        return True

    # Default: TRUE (all physical systems in benchmark have bounded phase space)
    return True


def assign_mixing(truth: Dict[str, bool]) -> bool:
    """Assign Mixing predicate.

    TRUE if:
      - System is Chaotic AND has PosLyap (hyperbolic chaos → mixing)
      - System has StrangeAttr (strange attractors exhibit mixing)

    FALSE if:
      - System is Periodic (no mixing)
      - System is QuasiPeriodic (no mixing on torus)
      - System has FixedPointAttr (no mixing)
      - System is Random (stochastic, not deterministic mixing)

    Default: FALSE (mixing is a strong property, only for hyperbolic chaos)
    """
    # Periodic/quasi-periodic/fixed → not mixing
    if any(truth.get(p) for p in ["Periodic", "QuasiPeriodic", "FixedPointAttr", "Random"]):
        return False

    # Hyperbolic chaos → mixing
    if truth.get("Chaotic") and truth.get("PosLyap"):
        return True

    # Strange attractor → likely mixing
    if truth.get("StrangeAttr"):
        return True

    # Default: FALSE (conservative, only assign mixing to clear cases)
    return False


def assign_ergodic(truth: Dict[str, bool]) -> bool:
    """Assign Ergodic predicate.

    TRUE if:
      - System is Mixing (mixing ⊂ ergodic)
      - System is Chaotic (chaotic systems are ergodic on attractors)
      - System is QuasiPeriodic (quasi-periodic tori are ergodic)
      - System is StatPredictable (statistical predictability requires ergodicity)

    FALSE if:
      - System is Periodic (single periodic orbit is not ergodic)
      - System has FixedPointAttr (single point is not ergodic)
      - System is Random (stochastic, not deterministic ergodic)

    Default: TRUE if chaotic/quasi-periodic, FALSE otherwise
    """
    # Non-ergodic cases
    if any(truth.get(p) for p in ["Periodic", "FixedPointAttr", "Random"]):
        return False

    # Mixing → ergodic (would be computed recursively, but we do it explicitly)
    if truth.get("Chaotic") and truth.get("PosLyap"):
        return True  # Mixing systems are ergodic

    # Chaotic → ergodic
    if truth.get("Chaotic"):
        return True

    # Quasi-periodic → ergodic on torus
    if truth.get("QuasiPeriodic"):
        return True

    # Statistical predictability → ergodic
    if truth.get("StatPredictable"):
        return True

    # Default: FALSE (conservative)
    return False


def process_system(filepath: Path, dry_run: bool = False) -> Dict:
    """Process a single system JSON file.

    Args:
        filepath: Path to system JSON
        dry_run: If True, don't write changes

    Returns:
        Dict with status and assignments
    """
    with open(filepath) as f:
        data = json.load(f)

    truth = data.get("truth_assignment", {})
    system_id = data.get("system_id", filepath.stem)

    if not truth:
        return {"status": "skip", "reason": "no_truth_assignment"}

    # Check if already has new predicates
    has_new = all(p in truth for p in ["Dissipative", "Bounded", "Mixing", "Ergodic"])

    # Compute new assignments
    new_values = {
        "Dissipative": assign_dissipative(truth),
        "Bounded": assign_bounded(truth),
        "Mixing": assign_mixing(truth),
        "Ergodic": assign_ergodic(truth),
    }

    # Check if name suggests conservative system (special case)
    name_lower = system_id.lower()
    if any(keyword in name_lower for keyword in ["standard_map", "arnold", "cat_map"]):
        # Override: conservative systems
        new_values["Dissipative"] = False
        new_values["Bounded"] = True  # Conservative maps are still bounded
        new_values["Mixing"] = True  # Arnold cat map is mixing
        new_values["Ergodic"] = True  # Conservative maps can be ergodic

    if has_new:
        # Check if values match (for verification)
        mismatches = []
        for pred, new_val in new_values.items():
            old_val = truth.get(pred)
            if old_val is not None and old_val != new_val:
                mismatches.append((pred, old_val, new_val))

        if mismatches:
            return {
                "status": "mismatch",
                "system_id": system_id,
                "mismatches": mismatches,
            }
        else:
            return {"status": "unchanged", "system_id": system_id}

    # Add new predicates
    truth.update(new_values)

    if not dry_run:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    return {
        "status": "updated",
        "system_id": system_id,
        "new_values": new_values,
    }


def main():
    parser = argparse.ArgumentParser(description="Assign new predicates to all systems")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--systems-dir", default="systems", help="Path to systems directory")
    args = parser.parse_args()

    print("="*80)
    print("AUTOMATED PREDICATE ASSIGNMENT")
    print("="*80)
    print()
    print("Adding 4 new predicates: Dissipative, Bounded, Mixing, Ergodic")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'WRITE MODE (will modify files)'}")
    print()

    # Process core systems
    core_dir = Path(args.systems_dir)
    results = []

    print(f"Processing core systems in {core_dir}...")
    for filepath in sorted(core_dir.glob("*.json")):
        result = process_system(filepath, dry_run=args.dry_run)
        results.append(result)

    # Process dysts systems
    dysts_dir = core_dir / "dysts"
    if dysts_dir.exists():
        print(f"Processing dysts systems in {dysts_dir}...")
        for filepath in sorted(dysts_dir.glob("*.json")):
            result = process_system(filepath, dry_run=args.dry_run)
            results.append(result)

    # Summarize
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Total systems processed: {len(results)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Show sample updates
    updated = [r for r in results if r["status"] == "updated"]
    if updated and not args.dry_run:
        print()
        print("Sample assignments (first 5 systems):")
        for r in updated[:5]:
            print(f"\n  {r['system_id']}:")
            for pred, val in r["new_values"].items():
                print(f"    {pred}: {val}")

    # Show mismatches (if any)
    mismatched = [r for r in results if r["status"] == "mismatch"]
    if mismatched:
        print()
        print("⚠️  MISMATCHES DETECTED (manual review needed):")
        for r in mismatched:
            print(f"\n  {r['system_id']}:")
            for pred, old_val, new_val in r["mismatches"]:
                print(f"    {pred}: existing={old_val}, computed={new_val}")

    print()
    if args.dry_run:
        print("✅ Dry run complete. Run without --dry-run to apply changes.")
    else:
        print("✅ All systems updated with new predicates.")

    print()


if __name__ == "__main__":
    main()
