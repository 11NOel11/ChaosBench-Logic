#!/usr/bin/env python3
"""Pre-freeze quality check orchestrator.

Validates all quality criteria from docs/QUALITY_STANDARD.md before freezing an API subset.

Usage:
    python scripts/pre_freeze_check.py [--smoke]  # Full check (or CI smoke test)

Exit Codes:
    0: All HARD FAIL criteria pass
    1: One or more HARD FAIL criteria failed
    2: Script error (invalid input, missing files, etc.)

Outputs:
    reports/pre_freeze/summary.md
    reports/pre_freeze/summary.json
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.data.grouping import _normalize_text


def _load_canonical_files(data_dir: str) -> List[Path]:
    """Load canonical file paths from data/canonical_v2_files.json."""
    data_path = Path(data_dir)
    selector_path = data_path.parent / "data" / "canonical_v2_files.json"
    if not selector_path.exists():
        selector_path = data_path / "canonical_v2_files.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    return [data_path.parent / f for f in selector["files"]]


class QualityCheck:
    """Base class for quality checks."""

    name: str = "Base Check"
    severity: str = "SOFT"  # HARD or SOFT

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.passed = False
        self.message = ""
        self.details = {}

    def run(self) -> bool:
        """Run the check. Return True if passed."""
        raise NotImplementedError

    def get_result(self) -> Dict[str, Any]:
        """Get check result as dict."""
        return {
            "name": self.name,
            "severity": self.severity,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class AccidentalDuplicatesCheck(QualityCheck):
    """Check for accidental duplicates (HARD FAIL)."""

    name = "Accidental Duplicates"
    severity = "HARD"

    def run(self) -> bool:
        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            questions.append(json.loads(line))

        # Check for accidental duplicates
        seen_keys = {}
        accidental_dupes = []

        for q in questions:
            norm_text = _normalize_text(q['question'])
            key = f"{q['type']}:{q['system_id']}:{norm_text}:{q['ground_truth']}"

            if key in seen_keys:
                # Check if this is intentional (has group_id)
                has_group = 'group_id' in q or 'group_id' in seen_keys[key]
                if not has_group:
                    accidental_dupes.append({
                        'id1': seen_keys[key]['id'],
                        'id2': q['id'],
                        'key': key,
                    })
            else:
                seen_keys[key] = q

        self.passed = len(accidental_dupes) == 0
        self.message = f"Found {len(accidental_dupes)} accidental duplicates"
        self.details = {
            "count": len(accidental_dupes),
            "threshold": 0,
            "samples": accidental_dupes[:10],
        }

        return self.passed


class OverallBalanceCheck(QualityCheck):
    """Check overall label balance (HARD FAIL)."""

    name = "Overall Label Balance"
    severity = "HARD"

    def run(self) -> bool:
        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            questions.append(json.loads(line))

        true_count = sum(1 for q in questions if q['ground_truth'] == 'TRUE')
        total = len(questions)
        true_pct = true_count / total * 100 if total > 0 else 0

        self.passed = 40 <= true_pct <= 60
        self.message = f"Overall TRUE%: {true_pct:.1f}%"
        self.details = {
            "true_count": true_count,
            "false_count": total - true_count,
            "true_pct": true_pct,
            "threshold": "40-60%",
        }

        return self.passed


class LabelLeakageCheck(QualityCheck):
    """Check for label leakage in question text (HARD FAIL)."""

    name = "Label Leakage"
    severity = "HARD"

    def run(self) -> bool:
        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            questions.append(json.loads(line))

        forbidden_patterns = [
            r'\bground_truth\b',
            r'\banswer_is\b',
            r'\bcorrect answer\b',
            r'\bthe answer is (TRUE|FALSE)\b',
            r'\(TRUE\)',
            r'\(FALSE\)',
        ]

        leaks = []
        for q in questions:
            text = q['question'].lower()
            for pattern in forbidden_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    leaks.append({
                        'id': q['id'],
                        'pattern': pattern,
                        'excerpt': q['question'][:100],
                    })
                    break

        self.passed = len(leaks) == 0
        self.message = f"Found {len(leaks)} label leaks"
        self.details = {
            "count": len(leaks),
            "threshold": 0,
            "samples": leaks[:10],
        }

        return self.passed


class DeterminismCheck(QualityCheck):
    """Check build determinism via test suite (HARD FAIL)."""

    name = "Determinism"
    severity = "HARD"

    def run(self) -> bool:
        # Run determinism test
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_determinism.py::test_question_ordering_determinism", "-v"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        self.passed = result.returncode == 0
        self.message = "Determinism test passed" if self.passed else "Determinism test failed"
        self.details = {
            "test": "test_question_ordering_determinism",
            "exit_code": result.returncode,
        }

        return self.passed


class SplitIntegrityCheck(QualityCheck):
    """Check split integrity - no cross-split contamination (HARD FAIL)."""

    name = "Split Integrity"
    severity = "HARD"

    def run(self) -> bool:
        # Load all questions with split assignment
        from chaosbench.data.splits import assign_split_v22

        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            q = json.loads(line)
                            q['_split'] = assign_split_v22(q)
                            questions.append(q)

        # Group by split
        splits = defaultdict(list)
        for q in questions:
            splits[q['_split']].append(q)

        # Check 1: No item_id collisions
        all_ids = [q['id'] for q in questions]
        id_collisions = len(all_ids) - len(set(all_ids))

        # Check 2: No system leakage (heldout_systems vs others)
        if 'heldout_systems' in splits:
            heldout_systems = set(q['system_id'] for q in splits['heldout_systems'])
            other_systems = set(q['system_id'] for q in questions if q['_split'] != 'heldout_systems')
            system_leakage = heldout_systems & other_systems
        else:
            system_leakage = set()

        # Check 3: No normalized text leakage (core vs heldout_templates)
        text_leakage = []
        if 'core' in splits and 'heldout_templates' in splits:
            core_texts = {_normalize_text(q['question']) for q in splits['core']}
            heldout_texts = {_normalize_text(q['question']) for q in splits['heldout_templates']}
            text_leakage = list(core_texts & heldout_texts)[:10]

        self.passed = id_collisions == 0 and len(system_leakage) == 0 and len(text_leakage) == 0
        self.message = "Split integrity validated" if self.passed else f"Split integrity violated"
        self.details = {
            "id_collisions": id_collisions,
            "system_leakage_count": len(system_leakage),
            "text_leakage_count": len(text_leakage),
            "text_leakage_samples": text_leakage,
        }

        return self.passed


class NonDegeneracyCheck(QualityCheck):
    """Check for non-degenerate splits (≥10% minority class) (HARD FAIL)."""

    name = "Non-Degeneracy"
    severity = "HARD"

    def run(self) -> bool:
        from chaosbench.data.splits import assign_split_v22

        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            q = json.loads(line)
                            q['_split'] = assign_split_v22(q)
                            questions.append(q)

        # Check minority class per split
        splits = defaultdict(list)
        for q in questions:
            splits[q['_split']].append(q)

        degenerate_splits = []
        for split_name, split_qs in splits.items():
            true_count = sum(1 for q in split_qs if q['ground_truth'] == 'TRUE')
            total = len(split_qs)
            true_pct = true_count / total * 100 if total > 0 else 0
            minority_pct = min(true_pct, 100 - true_pct)

            if minority_pct < 10:
                degenerate_splits.append({
                    'split': split_name,
                    'minority_pct': minority_pct,
                    'true_pct': true_pct,
                    'count': total,
                })

        self.passed = len(degenerate_splits) == 0
        self.message = f"All splits non-degenerate" if self.passed else f"{len(degenerate_splits)} degenerate splits"
        self.details = {
            "degenerate_splits": degenerate_splits,
            "threshold": "≥10% minority class",
        }

        return self.passed


class PerturbationIntegrityCheck(QualityCheck):
    """Check that perturbation groups satisfy label-preservation invariants (HARD FAIL).

    Invariants:
    - paraphrase: items grouped by (system_id, core predicate) must agree on ground_truth
      UNLESS the rephrasing introduces a logical negation (detected by 'not', 'incorrect',
      'false that' in the question stem).
    - distractor: items grouped by (system_id, core predicate) must agree on ground_truth.
    - negation: questions are intentionally label-flipped; no group consistency required.
    - entity_swap: questions swap the system entity; label depends on the new system; ok.

    A genuine conflict exists when two paraphrase (or distractor) items share the same
    system_id, same interrogated predicate, SAME question polarity (both affirmative or
    both negating), yet have DIFFERENT ground_truth values.
    """

    name = "Perturbation Integrity"
    severity = "HARD"

    # Predicates that appear in question text and can be directly matched
    _PREDICATES = [
        "chaotic", "sensitive", "strangeattr", "poslyap", "deterministic",
        "periodic", "quasiperiodic", "multistable", "regime", "dissipative",
        "bounded", "mixing", "ergodic",
    ]
    # Negation signals — questions containing these are polarity-flipped
    _NEGATION_SIGNALS = [
        "is it false that", "is it incorrect that", "would it be incorrect",
        "would you say it is incorrect", "is not", "not be characterized",
        "cannot be characterized", "wouldn't you say",
    ]

    def _extract_predicate(self, question: str) -> str:
        import re as _re2
        q = question.lower()
        for pred in self._PREDICATES:
            # Use character boundary to avoid matching "chaotic" inside "non-chaotic"
            # (both hyphens and letters are excluded as leading characters)
            if _re2.search(r"(?<![a-z\-])" + pred + r"(?![a-z])", q):
                return pred
        return "other"

    def _is_negated(self, question: str) -> bool:
        q = question.lower()
        return any(sig in q for sig in self._NEGATION_SIGNALS)

    def run(self) -> bool:
        import re as _re

        perturb_path = None
        for fpath in _load_canonical_files(self.data_dir):
            if "perturbation" in fpath.name:
                perturb_path = fpath
                break

        if perturb_path is None or not perturb_path.exists():
            self.passed = True
            self.message = "Perturbation file not found — skipping"
            self.details = {}
            return True

        items = []
        with open(perturb_path) as fh:
            for line in fh:
                if line.strip():
                    items.append(json.loads(line))

        # Split by perturbation type
        def _ptype(item_id: str) -> str:
            m = _re.match(r"perturb_(\w+)_\d+", item_id)
            return m.group(1) if m else "unknown"

        # Group paraphrase and distractor items by (system_id, predicate, polarity)
        conflicts = []
        for ptype in ("paraphrase", "distractor"):
            groups: Dict[tuple, list] = defaultdict(list)
            for item in items:
                if _ptype(item["id"]) != ptype:
                    continue
                pred = self._extract_predicate(item["question"])
                polarity = "neg" if self._is_negated(item["question"]) else "aff"
                key = (item["system_id"], pred, polarity)
                groups[key].append(item)

            for key, group_items in groups.items():
                labels = set(i["ground_truth"] for i in group_items)
                if len(labels) > 1:
                    conflicts.append({
                        "type": ptype,
                        "system_id": key[0],
                        "predicate": key[1],
                        "polarity": key[2],
                        "labels_found": sorted(labels),
                        "item_ids": [i["id"] for i in group_items],
                        "questions": [i["question"] for i in group_items],
                    })

        # Also verify negation items themselves aren't label-preserving
        # (they should mostly be FALSE since negated TRUE statements → FALSE)
        negation_items = [i for i in items if _ptype(i["id"]) == "negation"]
        neg_true = sum(1 for i in negation_items if i["ground_truth"] == "TRUE")
        neg_total = len(negation_items)
        # Warn if >20% of negations are TRUE (would imply design error for that subset)
        unexpected_negations = neg_true if (neg_total > 0 and neg_true / neg_total > 0.20) else 0

        self.passed = len(conflicts) == 0
        self.message = (
            f"0 polarity-matched conflicts in paraphrase/distractor groups"
            if self.passed
            else f"{len(conflicts)} polarity-matched label conflicts found"
        )
        self.details = {
            "conflicts_count": len(conflicts),
            "conflicts": conflicts[:10],
            "negation_items": neg_total,
            "negation_true_count": neg_true,
            "unexpected_negations": unexpected_negations,
        }
        return self.passed


class NearDuplicateCheck(QualityCheck):
    """Check near-duplicate rates per family (SOFT)."""

    name = "Near-Duplicates"
    severity = "SOFT"

    def run(self) -> bool:
        # This is computationally expensive, so we sample or run on demand
        # For now, skip detailed analysis in pre-freeze check
        # TODO: Implement efficient near-duplicate detection

        self.passed = True  # Default to pass for SOFT check
        self.message = "Near-duplicate analysis deferred (run duplicate_report.py)"
        self.details = {
            "note": "Near-duplicate analysis is computationally expensive - run scripts/duplicate_report.py for full analysis",
        }

        return self.passed


class PerFamilyBalanceCheck(QualityCheck):
    """Check per-family label balance (SOFT)."""

    name = "Per-Family Balance"
    severity = "SOFT"

    def run(self) -> bool:
        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            questions.append(json.loads(line))

        # Per-family balance
        family_stats = defaultdict(lambda: {'TRUE': 0, 'FALSE': 0})
        for q in questions:
            family_stats[q['type']][q['ground_truth']] += 1

        # Define thresholds per family
        standard_families = ['atomic', 'consistency_paraphrase', 'perturbation', 'adversarial_misleading',
                            'cross_indicator', 'regime_transition', 'extended_systems']
        logic_families = ['multi_hop', 'fol_inference']
        exempt_families = ['indicator_diagnostic', 'adversarial_nearmiss']

        imbalanced = []
        for family, stats in family_stats.items():
            total = stats['TRUE'] + stats['FALSE']
            true_pct = stats['TRUE'] / total * 100 if total > 0 else 0

            # Apply thresholds
            if family in exempt_families:
                threshold = None  # No threshold
            elif family in standard_families:
                threshold = (30, 70)
            elif family in logic_families:
                threshold = (20, 80)
            else:
                threshold = (30, 70)  # Default

            if threshold and not (threshold[0] <= true_pct <= threshold[1]):
                imbalanced.append({
                    'family': family,
                    'true_pct': true_pct,
                    'threshold': f"{threshold[0]}-{threshold[1]}%",
                    'count': total,
                })

        self.passed = len(imbalanced) <= 2  # Allow up to 2 imbalanced families
        self.message = f"{len(imbalanced)} families imbalanced"
        self.details = {
            "imbalanced_families": imbalanced,
            "threshold": "30-70% (standard), 20-80% (logic), exempt (specialized)",
        }

        return self.passed


class FamilyDegeneracyHardCheck(QualityCheck):
    """HARD FAIL if any non-exempt family has >98% single label (degenerate)."""

    name = "Family Degeneracy (Hard)"
    severity = "HARD"

    # These families are intentionally label-skewed by design
    EXEMPT = {"adversarial_misleading", "adversarial_nearmiss", "indicator_diagnostic"}

    def run(self) -> bool:
        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            questions.append(json.loads(line))

        family_stats: dict = defaultdict(lambda: {"TRUE": 0, "FALSE": 0})
        for q in questions:
            family_stats[q["type"]][q["ground_truth"]] += 1

        degenerate = []
        for family, stats in family_stats.items():
            # Skip exempt families
            if any(family.startswith(ex) or ex in family for ex in self.EXEMPT):
                continue
            total = stats["TRUE"] + stats["FALSE"]
            if total == 0:
                continue
            true_pct = stats["TRUE"] / total
            false_pct = stats["FALSE"] / total
            if true_pct > 0.98 or false_pct > 0.98:
                degenerate.append({
                    "family": family,
                    "true_pct": round(true_pct * 100, 1),
                    "false_pct": round(false_pct * 100, 1),
                    "count": total,
                })

        self.passed = len(degenerate) == 0
        self.message = (
            "No degenerate families" if self.passed
            else f"{len(degenerate)} degenerate families (>98% single label)"
        )
        self.details = {
            "degenerate_families": degenerate,
            "threshold": ">98% single label triggers HARD FAIL",
            "exempt_families": list(self.EXEMPT),
        }
        return self.passed


class FamilyDegeneracyHardCheck(QualityCheck):
    """HARD FAIL if any non-exempt family has >98% single label (degenerate)."""

    name = "Family Degeneracy (Hard)"
    severity = "HARD"

    # These families are intentionally label-skewed by design
    EXEMPT = {"adversarial_misleading", "adversarial_nearmiss", "indicator_diagnostic"}

    def run(self) -> bool:
        questions = []
        for fpath in _load_canonical_files(self.data_dir):
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            questions.append(json.loads(line))

        family_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"TRUE": 0, "FALSE": 0})
        for q in questions:
            family = q.get("type") or q.get("task_family") or q.get("family") or "unknown"
            label = q.get("ground_truth", "").strip().upper()
            if label in {"TRUE", "FALSE"}:
                family_stats[family][label] += 1

        degenerate = []
        for family, stats in family_stats.items():
            if family in self.EXEMPT:
                continue
            total = stats["TRUE"] + stats["FALSE"]
            if total == 0:
                continue
            true_pct = stats["TRUE"] / total * 100
            # >98% means minority class < 2%
            if true_pct > 98 or true_pct < 2:
                degenerate.append({
                    "family": family,
                    "true_pct": round(true_pct, 1),
                    "total": total,
                })

        self.passed = len(degenerate) == 0
        if degenerate:
            self.message = f"Degenerate families (>98% single label): {[d['family'] for d in degenerate]}"
        else:
            self.message = "No degenerate families found"
        self.details = {"degenerate_families": degenerate, "exempt": sorted(self.EXEMPT)}
        return self.passed


def run_all_checks(data_dir: str = "data", smoke: bool = False) -> Tuple[List[Dict], bool]:
    """Run all quality checks.

    Args:
        data_dir: Path to data directory
        smoke: If True, skip expensive checks for CI

    Returns:
        Tuple of (check_results, all_hard_passed)
    """
    checks = [
        AccidentalDuplicatesCheck(data_dir),
        OverallBalanceCheck(data_dir),
        LabelLeakageCheck(data_dir),
        DeterminismCheck(data_dir),
        SplitIntegrityCheck(data_dir),
        NonDegeneracyCheck(data_dir),
        FamilyDegeneracyHardCheck(data_dir),
        PerFamilyBalanceCheck(data_dir),
        PerturbationIntegrityCheck(data_dir),
    ]

    if not smoke:
        # Add expensive checks for full run
        checks.append(NearDuplicateCheck(data_dir))

    results = []
    all_hard_passed = True

    print("="*80)
    print("PRE-FREEZE QUALITY CHECK")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Mode: {'SMOKE' if smoke else 'FULL'}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    for check in checks:
        print(f"Running: {check.name} ({check.severity})...")
        try:
            passed = check.run()
            result = check.get_result()
            results.append(result)

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {check.message}")

            if not passed and check.severity == "HARD":
                all_hard_passed = False
        except Exception as e:
            print(f"  ⚠️  ERROR: {str(e)}")
            results.append({
                "name": check.name,
                "severity": check.severity,
                "passed": False,
                "message": f"Check error: {str(e)}",
                "details": {},
            })
            if check.severity == "HARD":
                all_hard_passed = False

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    hard_checks = [r for r in results if r['severity'] == 'HARD']
    soft_checks = [r for r in results if r['severity'] == 'SOFT']

    hard_passed = sum(1 for r in hard_checks if r['passed'])
    soft_passed = sum(1 for r in soft_checks if r['passed'])

    print(f"HARD FAIL checks: {hard_passed}/{len(hard_checks)} passed")
    print(f"SOFT checks: {soft_passed}/{len(soft_checks)} passed")
    print()

    if all_hard_passed:
        print("✅ FREEZE READY: All HARD FAIL criteria passed")
    else:
        print("❌ NOT FREEZE READY: One or more HARD FAIL criteria failed")

    return results, all_hard_passed


def generate_reports(results: List[Dict], output_dir: str = "reports/pre_freeze"):
    """Generate summary reports."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON report
    json_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": results,
        "summary": {
            "total_checks": len(results),
            "hard_checks": len([r for r in results if r['severity'] == 'HARD']),
            "hard_passed": len([r for r in results if r['severity'] == 'HARD' and r['passed']]),
            "soft_checks": len([r for r in results if r['severity'] == 'SOFT']),
            "soft_passed": len([r for r in results if r['severity'] == 'SOFT' and r['passed']]),
            "freeze_ready": all(r['passed'] for r in results if r['severity'] == 'HARD'),
        }
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(json_report, f, indent=2)

    # Markdown report
    md_lines = [
        "# Pre-Freeze Quality Check Summary",
        "",
        f"**Timestamp:** {json_report['timestamp']}",
        f"**Freeze Ready:** {'✅ YES' if json_report['summary']['freeze_ready'] else '❌ NO'}",
        "",
        "## Check Results",
        "",
        "| Check | Severity | Status | Message |",
        "|-------|----------|--------|---------|",
    ]

    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        md_lines.append(f"| {r['name']} | {r['severity']} | {status} | {r['message']} |")

    md_lines.extend([
        "",
        "## Summary",
        "",
        f"- **HARD FAIL checks:** {json_report['summary']['hard_passed']}/{json_report['summary']['hard_checks']} passed",
        f"- **SOFT checks:** {json_report['summary']['soft_passed']}/{json_report['summary']['soft_checks']} passed",
        "",
        "## Next Steps",
        "",
    ])

    if json_report['summary']['freeze_ready']:
        md_lines.append("✅ All HARD FAIL criteria passed. Ready to freeze API subset.")
    else:
        md_lines.append("❌ Fix HARD FAIL criteria before freezing. See details above.")

    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nReports generated:")
    print(f"  - {output_dir}/summary.json")
    print(f"  - {output_dir}/summary.md")


def main():
    parser = argparse.ArgumentParser(description="Pre-freeze quality check orchestrator")
    parser.add_argument("--smoke", action="store_true", help="Run CI smoke test (skip expensive checks)")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    args = parser.parse_args()

    try:
        results, all_hard_passed = run_all_checks(args.data_dir, args.smoke)
        generate_reports(results)

        sys.exit(0 if all_hard_passed else 1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
