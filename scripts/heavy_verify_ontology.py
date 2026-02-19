#!/usr/bin/env python3
"""Phase 4 heavy ontology & FOL consistency verification.

Validates the ontology graph for cycles, contradictions, and computes
chain-space statistics. Also verifies predicate assignments for curated
core systems.

Usage:
    python scripts/heavy_verify_ontology.py [--output-dir artifacts/heavy_verify/]

Exit Codes:
    0  All checks passed
    1  One or more hard-fail checks (contradictions or cycles)
    2  Script error

Outputs:
    artifacts/heavy_verify/ontology_report.md
"""

import argparse
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.logic.axioms import get_fol_rules
from chaosbench.logic.ontology import PREDICATES


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(rules: Dict[str, Dict[str, List[str]]]) -> Tuple[
    Dict[str, Set[str]],  # requires edges
    Dict[str, Set[str]],  # excludes edges
]:
    """Build directed requires and excludes graphs from FOL rules."""
    requires: Dict[str, Set[str]] = defaultdict(set)
    excludes: Dict[str, Set[str]] = defaultdict(set)

    for pred, rel in rules.items():
        for req in rel.get("requires", []):
            requires[pred].add(req)
        for exc in rel.get("excludes", []):
            excludes[pred].add(exc)
            excludes[exc].add(pred)  # excludes is symmetric

    return dict(requires), dict(excludes)


# ---------------------------------------------------------------------------
# Cycle detection (DFS on requires graph)
# ---------------------------------------------------------------------------

def detect_cycles(requires: Dict[str, Set[str]], predicates: List[str]) -> List[List[str]]:
    """Detect cycles in the requires graph using DFS.

    Returns list of cycles (each cycle is a list of predicate names).
    """
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    cycles: List[List[str]] = []
    path: List[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in requires.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found cycle — extract it
                idx = path.index(neighbor)
                cycle = path[idx:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        rec_stack.discard(node)

    for pred in predicates:
        if pred not in visited:
            dfs(pred)

    return cycles


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def detect_contradictions(
    requires: Dict[str, Set[str]],
    excludes: Dict[str, Set[str]],
    predicates: List[str],
) -> List[str]:
    """Detect contradictions: A requires B but also excludes B."""
    contradictions = []

    for pred in predicates:
        req_set = requires.get(pred, set())
        exc_set = excludes.get(pred, set())
        conflicts = req_set & exc_set
        if conflicts:
            contradictions.append(
                f"{pred} both requires and excludes: {sorted(conflicts)}"
            )

    return contradictions


# ---------------------------------------------------------------------------
# Chain-space statistics (k-hop reachability in requires graph)
# ---------------------------------------------------------------------------

def compute_chain_stats(
    requires: Dict[str, Set[str]],
    predicates: List[str],
    max_hops: int = 6,
) -> Dict[int, int]:
    """Compute number of reachable predicate pairs at each hop distance."""
    hop_counts: Dict[int, int] = {k: 0 for k in range(2, max_hops + 1)}

    for source in predicates:
        # BFS from source
        frontier: Set[str] = {source}
        visited: Set[str] = {source}
        depth = 0

        while frontier and depth < max_hops:
            depth += 1
            next_frontier: Set[str] = set()
            for node in frontier:
                for neighbor in requires.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        if depth >= 2:
                            hop_counts[depth] = hop_counts.get(depth, 0) + 1
            frontier = next_frontier

    return hop_counts


def compute_longest_chain(
    requires: Dict[str, Set[str]],
    predicates: List[str],
) -> Tuple[int, List[str]]:
    """Find the longest chain in the requires graph."""
    best_length = 0
    best_chain: List[str] = []

    def dfs(node: str, path: List[str], visited: Set[str]) -> None:
        nonlocal best_length, best_chain
        if len(path) > best_length:
            best_length = len(path)
            best_chain = list(path)
        for neighbor in requires.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.discard(neighbor)

    for pred in predicates:
        dfs(pred, [pred], {pred})

    return best_length, best_chain


# ---------------------------------------------------------------------------
# System predicate satisfiability
# ---------------------------------------------------------------------------

CURATED_SYSTEM_PREDICATES: Dict[str, Dict[str, bool]] = {
    # Lorenz63: chaotic, dissipative, bounded strange attractor
    "lorenz63": {
        "Chaotic": True,
        "Deterministic": True,
        "PosLyap": True,
        "Sensitive": True,
        "StrangeAttr": True,
        "PointUnpredictable": True,
        "StatPredictable": True,
        "Dissipative": True,
        "Bounded": True,
        "Mixing": True,
        "Ergodic": True,
        "Periodic": False,
        "QuasiPeriodic": False,
        "FixedPointAttr": False,
        "Random": False,
        "Conservative": False,
        "ContinuousTime": True,
        "DiscreteTime": False,
    },
    # Arnold cat map: discrete-time, strongly mixing, conservative
    "arnold_cat_map": {
        "Chaotic": True,
        "Deterministic": True,
        "PosLyap": True,
        "Sensitive": True,
        "DiscreteTime": True,
        "ContinuousTime": False,
        "Periodic": False,
        "QuasiPeriodic": False,
        "Conservative": True,
        "Dissipative": False,
        "Ergodic": True,
        "Mixing": True,
        "Bounded": True,
    },
    # Damped oscillator: fixed point attractor
    "damped_oscillator": {
        "FixedPointAttr": True,
        "Deterministic": True,
        "Chaotic": False,
        "Periodic": False,
        "Random": False,
        "Dissipative": True,
        "Bounded": True,
        "ContinuousTime": True,
    },
    # Stochastic OU: random, not deterministic
    "stochastic_ou": {
        "Random": True,
        "Deterministic": False,
        "Chaotic": False,
        "ContinuousTime": True,
    },
    # Logistic r=4: chaotic map
    "logistic_r4": {
        "Chaotic": True,
        "Deterministic": True,
        "PosLyap": True,
        "Sensitive": True,
        "DiscreteTime": True,
        "Periodic": False,
        "Bounded": True,
        "Mixing": True,
        "Ergodic": True,
    },
}


def verify_system_satisfiability(
    rules: Dict[str, Dict[str, List[str]]],
    system_predicates: Dict[str, Dict[str, bool]],
) -> List[str]:
    """Verify that predicate assignments for curated systems don't violate FOL rules."""
    violations = []

    for system_id, assignments in system_predicates.items():
        true_preds = {p for p, v in assignments.items() if v is True}
        false_preds = {p for p, v in assignments.items() if v is False}

        for pred in true_preds:
            if pred not in rules:
                continue
            # Check requires: if pred is True, all required predicates must be True
            for req in rules[pred].get("requires", []):
                if req in false_preds:
                    violations.append(
                        f"{system_id}: {pred}=TRUE but requires {req}=TRUE "
                        f"(got FALSE)"
                    )
                # If req is not in our known assignments, we can't check
            # Check excludes: if pred is True, all excluded predicates must be False
            for exc in rules[pred].get("excludes", []):
                if exc in true_preds:
                    violations.append(
                        f"{system_id}: {pred}=TRUE but excludes {exc} "
                        f"(got TRUE)"
                    )

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_verification(output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hard_fails": [],
        "warnings": [],
        "chain_stats": {},
        "summary": {},
    }

    print("=" * 60)
    print("Phase 4 — Ontology & FOL Consistency Verification")
    print("=" * 60)

    rules = get_fol_rules()
    predicates = PREDICATES

    print(f"\n  Predicates: {len(predicates)}")
    print(f"  Rules defined for: {len(rules)} predicates")

    # Check for predicates in PREDICATES but not in rules
    missing_rules = set(predicates) - set(rules.keys())
    if missing_rules:
        print(f"  INFO: Predicates without rules: {sorted(missing_rules)}")

    # Build graph
    requires, excludes = build_graph(rules)

    # --- Cycle detection ---
    print("\n[A] Cycle detection in requires graph...")
    cycles = detect_cycles(requires, predicates)
    if cycles:
        for cycle in cycles:
            msg = f"Cycle detected: {' -> '.join(cycle)}"
            print(f"  FAIL: {msg}")
            report["hard_fails"].append(msg)
    else:
        print("  OK: No cycles in requires graph")

    # --- Contradiction detection ---
    print("\n[B] Contradiction detection...")
    contradictions = detect_contradictions(requires, excludes, predicates)
    if contradictions:
        for c in contradictions:
            print(f"  FAIL: {c}")
            report["hard_fails"].append(c)
    else:
        print("  OK: No contradictions found")

    # --- Chain-space statistics ---
    print("\n[C] Chain-space statistics (requires graph)...")
    chain_stats = compute_chain_stats(requires, predicates, max_hops=6)
    longest_len, longest_chain = compute_longest_chain(requires, predicates)

    print(f"  Longest chain: {longest_len} hops: {' -> '.join(longest_chain)}")
    for k in range(2, 7):
        count = chain_stats.get(k, 0)
        print(f"  {k}-hop pairs: {count}")

    report["chain_stats"] = {
        "longest_chain_length": longest_len,
        "longest_chain": longest_chain,
        "hop_counts": {str(k): chain_stats.get(k, 0) for k in range(2, 7)},
    }

    # Warn if max hops < 4 (benchmark claims 4-5 hop chains)
    if longest_len < 4:
        warn = f"Longest requires-chain is only {longest_len} hops (expected >= 4)"
        print(f"  WARN: {warn}")
        report["warnings"].append(warn)

    # --- System satisfiability ---
    print("\n[D] System predicate satisfiability (curated systems)...")
    sat_violations = verify_system_satisfiability(rules, CURATED_SYSTEM_PREDICATES)
    if sat_violations:
        for v in sat_violations:
            print(f"  FAIL: {v}")
            report["hard_fails"].append(v)
    else:
        print(f"  OK: All {len(CURATED_SYSTEM_PREDICATES)} curated systems satisfy FOL rules")

    # --- Symmetric excludes check ---
    print("\n[E] Symmetric excludes check...")
    asymmetric = []
    for pred, rel in rules.items():
        for exc in rel.get("excludes", []):
            if exc in rules:
                exc_of_exc = rules[exc].get("excludes", [])
                if pred not in exc_of_exc:
                    asymmetric.append(
                        f"{pred} excludes {exc}, but {exc} does not explicitly exclude {pred}"
                    )
    if asymmetric:
        print(f"  INFO: {len(asymmetric)} asymmetric excludes (implicit symmetry assumed)")
        # This is informational only — the build_graph function enforces symmetry
    else:
        print("  OK: All excludes are explicitly symmetric")

    # Summary
    passed = len(report["hard_fails"]) == 0
    report["summary"] = {
        "passed": passed,
        "predicates": len(predicates),
        "rules_defined": len(rules),
        "cycle_count": len(cycles),
        "contradiction_count": len(contradictions),
        "sat_violations": len(sat_violations),
        "hard_fail_count": len(report["hard_fails"]),
        "warning_count": len(report["warnings"]),
    }

    return passed, report


def write_report(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "ontology_report.md"

    summary = report["summary"]
    chain = report.get("chain_stats", {})

    lines = [
        "# Ontology & FOL Consistency Report — Phase 4",
        "",
        f"Generated: {report['timestamp']}",
        "",
        "## Summary",
        "",
        f"**Status**: {'✓ PASSED' if summary['passed'] else '✗ FAILED'}",
        f"- Predicates: {summary['predicates']}",
        f"- Rules defined: {summary['rules_defined']}",
        f"- Cycles detected: {summary['cycle_count']}",
        f"- Contradictions: {summary['contradiction_count']}",
        f"- System satisfiability violations: {summary['sat_violations']}",
        f"- Hard failures: {summary['hard_fail_count']}",
        f"- Warnings: {summary['warning_count']}",
        "",
        "## Chain-Space Statistics",
        "",
        f"- Longest chain: {chain.get('longest_chain_length', 0)} hops",
        f"  `{' -> '.join(chain.get('longest_chain', []))}`",
        "",
        "| Hop Depth | Reachable Pairs |",
        "|-----------|----------------|",
    ]
    for k in range(2, 7):
        count = chain.get("hop_counts", {}).get(str(k), 0)
        lines.append(f"| {k}-hop | {count} |")

    lines.append("")

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
        "## Curated System Checks",
        "",
        "Systems verified: " + ", ".join(sorted(CURATED_SYSTEM_PREDICATES.keys())),
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Wrote: {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "heavy_verify",
    )
    args = parser.parse_args()

    try:
        passed, report = run_verification(args.output_dir)
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
