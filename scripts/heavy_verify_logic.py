#!/usr/bin/env python3
"""Heavy logic correctness verification for ChaosBench-Logic v2.

Checks:
  A. Ontology constraint validation per system (requires/excludes)
  B. Multi-hop ground_truth correctness (full scan)
  C. Paraphrase group label consistency
  D. Indicator diagnostic sanity

Usage:
    python scripts/heavy_verify_logic.py [--data-dir data/] [--systems-dir systems/] [--output-dir artifacts/heavy_verify/]

Exit Codes:
    0  All checks passed
    1  One or more hard-fail checks failed
    2  Script error

Outputs:
    artifacts/heavy_verify/logic_report.md
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.logic.axioms import get_fol_rules
from chaosbench.logic.ontology import KEYWORD_MAP, PREDICATES


# ---------------------------------------------------------------------------
# Canonical file loader
# ---------------------------------------------------------------------------

def _load_canonical_files(data_dir: Path) -> List[str]:
    """Load canonical file names from data/canonical_v2_files.json."""
    selector_path = data_dir.parent / "data" / "canonical_v2_files.json"
    if not selector_path.exists():
        selector_path = data_dir / "canonical_v2_files.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    return [Path(f).name for f in selector["files"]]


def load_family(data_dir: Path, family_fname: str) -> List[Dict[str, Any]]:
    """Load records from a single family file."""
    fpath = data_dir / family_fname
    if not fpath.exists():
        return []
    records = []
    for line in fpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# System truth loader
# ---------------------------------------------------------------------------

def load_system_truths(systems_dir: Path) -> Dict[str, Dict[str, bool]]:
    """Load truth_assignment for all systems from systems/ and systems/dysts/."""
    truths: Dict[str, Dict[str, bool]] = {}

    # Core systems
    for json_file in systems_dir.glob("*.json"):
        if json_file.name.endswith("_indicators.json"):
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            system_id = data.get("system_id", "")
            ta = data.get("truth_assignment", {})
            if system_id and ta:
                truths[system_id] = {k: bool(v) for k, v in ta.items()}
        except Exception:
            pass

    # dysts systems
    dysts_dir = systems_dir / "dysts"
    if dysts_dir.exists():
        for json_file in dysts_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                system_id = data.get("system_id", "")
                ta = data.get("truth_assignment", {})
                if system_id and ta:
                    truths[system_id] = {k: bool(v) for k, v in ta.items()}
            except Exception:
                pass

    return truths


# ---------------------------------------------------------------------------
# Predicate extraction from text
# ---------------------------------------------------------------------------

def extract_predicate_from_text(text: str) -> Optional[str]:
    """Find the LAST matching predicate from KEYWORD_MAP in text.

    Multi-hop questions embed premises before the conclusion. The conclusion
    predicate (what's being asked) appears last in the sentence, so we use
    rfind to locate the rightmost predicate keyword.
    """
    text_lower = text.lower()
    best_pos = -1
    best_pred = None
    for keywords, predicate in KEYWORD_MAP:
        for kw in keywords:
            idx = text_lower.rfind(kw)
            if idx > best_pos:
                best_pos = idx
                best_pred = predicate
    return best_pred


def _has_negative_logic(text: str, predicate_keyword: str) -> bool:
    """Check if the question uses negative logic about the predicate.

    Detects patterns like "does NOT X tell us about Y?" or "cannot be".
    """
    text_lower = text.lower()
    pred_lower = predicate_keyword.lower()

    # Look for "not" within 5 words before the predicate mention
    idx = text_lower.find(pred_lower)
    if idx == -1:
        return False
    prefix = text_lower[max(0, idx - 40):idx]
    negative_markers = ["not ", "cannot ", "can't ", "never ", "no longer ", "doesn't ", "does not "]
    return any(m in prefix for m in negative_markers)


# ---------------------------------------------------------------------------
# Check A: Ontology constraint validation
# ---------------------------------------------------------------------------

def check_a_ontology_constraints(
    system_truths: Dict[str, Dict[str, bool]],
    fol_rules: Dict[str, Dict[str, List[str]]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """For each system, check requires/excludes constraints hold.

    Returns:
        (systems_checked, violations)
    """
    violations = []
    systems_checked = 0

    for system_id, ta in system_truths.items():
        systems_checked += 1
        for pred, is_true in ta.items():
            if not is_true:
                continue
            rules = fol_rules.get(pred, {})
            # Check requires
            for req in rules.get("requires", []):
                if req in ta and not ta[req]:
                    violations.append({
                        "system_id": system_id,
                        "predicate": pred,
                        "constraint": "requires",
                        "violated_pred": req,
                        "detail": f"{pred}=True requires {req}=True, but {req}=False",
                    })
            # Check excludes
            for exc in rules.get("excludes", []):
                if exc in ta and ta[exc]:
                    violations.append({
                        "system_id": system_id,
                        "predicate": pred,
                        "constraint": "excludes",
                        "violated_pred": exc,
                        "detail": f"{pred}=True excludes {exc}=True, but {exc}=True",
                    })

    return systems_checked, violations


# ---------------------------------------------------------------------------
# Check B: Multi-hop ground_truth correctness
# ---------------------------------------------------------------------------

def check_b_multi_hop_logic(
    multi_hop_records: List[Dict[str, Any]],
    system_truths: Dict[str, Dict[str, bool]],
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Check multi-hop ground_truth vs system truth assignments.

    Returns:
        (total, checked, skipped, violations)
    """
    violations = []
    checked = 0
    skipped = 0
    total = len(multi_hop_records)

    for item in multi_hop_records:
        system_id = item.get("system_id", "")
        ground_truth = item.get("ground_truth", "")
        question = item.get("question", "")

        # Only check items with known system truth
        if system_id not in system_truths:
            skipped += 1
            continue

        ta = system_truths[system_id]

        # Extract the conclusion sentence (last sentence ending with ?)
        sentences = re.split(r'(?<=[.!?])\s+', question.strip())
        conclusion = ""
        for sent in reversed(sentences):
            if "?" in sent:
                conclusion = sent
                break
        if not conclusion:
            skipped += 1
            continue

        # Skip conditional/counterfactual questions — they require full FOL inference
        conclusion_lower = conclusion.lower()
        if any(conclusion_lower.startswith(m) for m in ["if ", "given that ", "suppose "]):
            skipped += 1
            continue

        # Mask system name from conclusion to prevent false keyword matches
        # e.g., "CaTwoPlusQuasiperiodic" contains "periodic" → false match
        conclusion_masked = re.sub(re.escape(system_id.replace("dysts_", "").replace("_", "")),
                                   "<SYSTEM>", conclusion, flags=re.IGNORECASE)
        # Also strip the display name form (CamelCase, with spaces)
        conclusion_masked = re.sub(r'\b\w*(?:quasi|periodic|chaotic|random|mixing|ergodic|lorenz|brusselator|rossler)\w*\b',
                                   lambda m: "<SYS>" if len(m.group(0)) > 8 else m.group(0),
                                   conclusion_masked, flags=re.IGNORECASE)

        # Find predicate in conclusion (last/rightmost keyword = what's being asked)
        pred = extract_predicate_from_text(conclusion_masked)
        if pred is None:
            skipped += 1
            continue

        # Skip negative-logic questions
        # Find the keyword that matched (last occurrence)
        matched_keyword = None
        best_pos = -1
        for keywords, p in KEYWORD_MAP:
            if p == pred:
                for kw in keywords:
                    idx = conclusion_lower.rfind(kw)
                    if idx > best_pos:
                        best_pos = idx
                        matched_keyword = kw
                break

        if matched_keyword and _has_negative_logic(conclusion, matched_keyword):
            skipped += 1
            continue

        # Look up expected answer
        if pred not in ta:
            skipped += 1
            continue

        expected = "TRUE" if ta[pred] else "FALSE"
        checked += 1

        if ground_truth != expected:
            violations.append({
                "id": item.get("id", ""),
                "system_id": system_id,
                "predicate": pred,
                "expected": expected,
                "actual": ground_truth,
                "conclusion": conclusion[:120],
            })

    return total, checked, skipped, violations


# ---------------------------------------------------------------------------
# Check C: Paraphrase group label consistency
# ---------------------------------------------------------------------------

def check_c_paraphrase_consistency(
    para_records: List[Dict[str, Any]],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Check that all items in a paraphrase group share the same ground_truth.

    Returns:
        (groups_checked, groups_with_flips, violations)
    """
    # Group by base ID: "atomic_{N}_para_{M}" -> group key "atomic_{N}"
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in para_records:
        item_id = item.get("id", "")
        # Try para ID pattern first
        m = re.match(r"^(atomic_\d+)_para_\d+$", item_id)
        if m:
            groups[m.group(1)].append(item)
        else:
            # Use group_id if available
            group_id = item.get("group_id")
            if group_id:
                groups[group_id].append(item)
            else:
                # Single item, skip
                pass

    groups_checked = len(groups)
    violations = []
    groups_with_flips = 0

    for group_key, items in groups.items():
        if len(items) < 2:
            continue
        labels = set(item.get("ground_truth", "") for item in items)
        if len(labels) > 1:
            groups_with_flips += 1
            violations.append({
                "group_key": group_key,
                "labels_found": list(labels),
                "items": [{"id": i.get("id", ""), "gt": i.get("ground_truth", "")} for i in items],
            })

    return groups_checked, groups_with_flips, violations


# ---------------------------------------------------------------------------
# Check D: Indicator diagnostic sanity
# ---------------------------------------------------------------------------

def check_d_indicator_sanity(
    indicator_records: List[Dict[str, Any]],
    system_truths: Dict[str, Dict[str, bool]],
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Check indicator diagnostic ground_truth vs known system properties.

    Returns:
        (total, checked, skipped, violations)
    """
    violations = []
    checked = 0
    skipped = 0
    total = len(indicator_records)

    for item in indicator_records:
        system_id = item.get("system_id", "")
        ground_truth = item.get("ground_truth", "")
        question = item.get("question", "")

        if system_id not in system_truths:
            skipped += 1
            continue

        ta = system_truths[system_id]

        # Extract conclusion sentence
        sentences = re.split(r'(?<=[.!?])\s+', question.strip())
        conclusion = ""
        for sent in reversed(sentences):
            if "?" in sent:
                conclusion = sent
                break
        if not conclusion:
            skipped += 1
            continue

        # Find predicate
        pred = extract_predicate_from_text(conclusion)
        if pred is None:
            skipped += 1
            continue

        if pred not in ta:
            skipped += 1
            continue

        # Special case: "regular regime" means NOT chaotic
        conclusion_lower = conclusion.lower()
        if "regular regime" in conclusion_lower or "regular behavior" in conclusion_lower:
            expected_chaotic = False
            expected = "TRUE" if (ta.get("Chaotic", False) == expected_chaotic) else "FALSE"
            # Actually: "Is this system in a regular regime?" → TRUE iff NOT Chaotic
            expected = "TRUE" if not ta.get("Chaotic", True) else "FALSE"
        else:
            expected = "TRUE" if ta[pred] else "FALSE"

        checked += 1

        if ground_truth != expected:
            violations.append({
                "id": item.get("id", ""),
                "system_id": system_id,
                "predicate": pred,
                "expected": expected,
                "actual": ground_truth,
                "conclusion": conclusion[:120],
            })

    return total, checked, skipped, violations


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    check_a: Tuple,
    check_b: Tuple,
    check_c: Tuple,
    check_d: Tuple,
    passed: bool,
) -> None:
    """Write logic verification report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "logic_report.md"

    a_systems, a_violations = check_a
    b_total, b_checked, b_skipped, b_violations = check_b
    c_groups, c_flips, c_violations = check_c
    d_total, d_checked, d_skipped, d_violations = check_d

    lines = [
        "# Logic Correctness Verification Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Overall status:** {'✅ PASSED' if passed else '❌ FAILED'}",
        "",
        "---",
        "",
        "## Check A: Ontology Constraint Validation",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Systems checked | {a_systems} |",
        f"| Violations found | {len(a_violations)} |",
        f"| Status | {'✅ PASS' if len(a_violations) == 0 else '❌ FAIL'} |",
        "",
    ]

    if a_violations:
        lines += [
            "### Violations (up to 10):",
            "",
            "| System | Predicate | Constraint | Detail |",
            "|--------|-----------|------------|--------|",
        ]
        for v in a_violations[:10]:
            lines.append(f"| {v['system_id']} | {v['predicate']} | {v['constraint']} | {v['detail']} |")
        lines.append("")

    lines += [
        "## Check B: Multi-Hop Ground Truth Correctness",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total multi-hop items | {b_total} |",
        f"| Checkable (predicate + system known) | {b_checked} |",
        f"| Skipped (negative logic / unknown system) | {b_skipped} |",
        f"| Coverage | {b_checked/b_total*100:.1f}% |" if b_total > 0 else f"| Coverage | N/A |",
        f"| Violations found | {len(b_violations)} |",
        f"| Status | {'✅ PASS' if len(b_violations) == 0 else '❌ FAIL'} |",
        "",
    ]

    if b_violations:
        lines += [
            "### Violations (up to 10):",
            "",
            "| ID | System | Predicate | Expected | Actual | Conclusion |",
            "|----|--------|-----------|----------|--------|------------|",
        ]
        for v in b_violations[:10]:
            lines.append(
                f"| {v['id']} | {v['system_id']} | {v['predicate']} | "
                f"{v['expected']} | {v['actual']} | {v['conclusion'][:60]}... |"
            )
        lines.append("")

    lines += [
        "## Check C: Paraphrase Group Label Consistency",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Groups checked | {c_groups} |",
        f"| Groups with label flips | {c_flips} |",
        f"| Status | {'✅ PASS' if c_flips == 0 else '❌ FAIL'} |",
        "",
    ]

    if c_violations:
        lines += [
            "### Groups with Flips (up to 5):",
            "",
        ]
        for v in c_violations[:5]:
            lines.append(f"- **{v['group_key']}**: labels {v['labels_found']}")
            for item_info in v['items']:
                lines.append(f"  - `{item_info['id']}`: {item_info['gt']}")
        lines.append("")

    lines += [
        "## Check D: Indicator Diagnostic Sanity",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total indicator items | {d_total} |",
        f"| Checkable | {d_checked} |",
        f"| Skipped | {d_skipped} |",
        f"| Violations found | {len(d_violations)} |",
        f"| Status | {'✅ PASS' if len(d_violations) == 0 else '❌ FAIL'} |",
        "",
    ]

    if d_violations:
        lines += [
            "### Violations (up to 10):",
            "",
            "| ID | System | Predicate | Expected | Actual |",
            "|----|--------|-----------|----------|--------|",
        ]
        for v in d_violations[:10]:
            lines.append(f"| {v['id']} | {v['system_id']} | {v['predicate']} | {v['expected']} | {v['actual']} |")
        lines.append("")

    lines += [
        "---",
        "",
        f"## Summary",
        "",
        f"| Check | Violations | Status |",
        f"|-------|------------|--------|",
        f"| A: Ontology constraints | {len(a_violations)} | {'✅' if len(a_violations) == 0 else '❌'} |",
        f"| B: Multi-hop logic | {len(b_violations)} | {'✅' if len(b_violations) == 0 else '❌'} |",
        f"| C: Paraphrase consistency | {c_flips} | {'✅' if c_flips == 0 else '❌'} |",
        f"| D: Indicator sanity | {len(d_violations)} | {'✅' if len(d_violations) == 0 else '❌'} |",
        "",
        f"**Final result: {'PASS ✅' if passed else 'FAIL ❌'}**",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Logic correctness verification for v2 dataset")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--systems-dir", default="systems/", help="Systems directory")
    parser.add_argument("--output-dir", default="artifacts/heavy_verify/", help="Output directory")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    systems_dir = PROJECT_ROOT / args.systems_dir
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 60)
    print("Heavy Logic Verification")
    print("=" * 60)

    try:
        # Load system truths
        print("\n[1] Loading system truth assignments...")
        system_truths = load_system_truths(systems_dir)
        print(f"  Loaded {len(system_truths)} system truth assignments")

        # Load FOL rules
        print("\n[2] Loading FOL rules...")
        fol_rules = get_fol_rules()
        print(f"  Loaded {len(fol_rules)} predicate rules")

        # Check A
        print("\n[A] Ontology constraint validation...")
        a_systems, a_violations = check_a_ontology_constraints(system_truths, fol_rules)
        print(f"  Systems: {a_systems}, Violations: {len(a_violations)}")

        # Check B
        print("\n[B] Multi-hop ground_truth correctness...")
        multi_hop = load_family(data_dir, "v22_multi_hop.jsonl")
        b_total, b_checked, b_skipped, b_violations = check_b_multi_hop_logic(multi_hop, system_truths)
        print(f"  Total: {b_total}, Checked: {b_checked}, Skipped: {b_skipped}, Violations: {len(b_violations)}")

        # Check C
        print("\n[C] Paraphrase group label consistency...")
        para = load_family(data_dir, "v22_consistency_paraphrase.jsonl")
        c_groups, c_flips, c_violations = check_c_paraphrase_consistency(para)
        print(f"  Groups: {c_groups}, Flips: {c_flips}")

        # Check D
        print("\n[D] Indicator diagnostic sanity...")
        indicators = load_family(data_dir, "v22_indicator_diagnostics.jsonl")
        d_total, d_checked, d_skipped, d_violations = check_d_indicator_sanity(indicators, system_truths)
        print(f"  Total: {d_total}, Checked: {d_checked}, Skipped: {d_skipped}, Violations: {len(d_violations)}")

        # Determine pass/fail
        passed = (
            len(a_violations) == 0 and
            len(b_violations) == 0 and
            c_flips == 0 and
            len(d_violations) == 0
        )

        # Write report
        print("\n[5] Writing report...")
        write_report(
            output_dir,
            (a_systems, a_violations),
            (b_total, b_checked, b_skipped, b_violations),
            (c_groups, c_flips, c_violations),
            (d_total, d_checked, d_skipped, d_violations),
            passed,
        )

        if not passed:
            total_violations = len(a_violations) + len(b_violations) + c_flips + len(d_violations)
            print(f"\n❌ HARD FAIL: {total_violations} logic violation(s) found")
            return 1

        print("\n✅ PASSED — all logic checks clean")
        return 0

    except Exception as e:
        print(f"\nScript error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
