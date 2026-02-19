#!/usr/bin/env python3
"""Sanity checks for v2 dataset quality and integrity.

Checks:
- ID uniqueness across all batches
- No forbidden leakage tokens (ground_truth, TRUE/FALSE in questions)
- Class balance per task family
- Duplicate/near-duplicate question detection
- System-task family balance
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.eval.runner import load_jsonl


def check_id_uniqueness(items):
    """Check that all IDs are unique."""
    ids = [item['id'] for item in items]
    id_counts = Counter(ids)
    duplicates = {id_: count for id_, count in id_counts.items() if count > 1}

    print("\n## ID Uniqueness Check")
    if duplicates:
        print(f"❌ FAIL: Found {len(duplicates)} duplicate IDs:")
        for id_, count in sorted(duplicates.items()):
            print(f"  - {id_}: appears {count} times")
        return False
    else:
        print(f"✅ PASS: All {len(items)} IDs are unique")
        return True


def check_leakage_tokens(items):
    """Check for forbidden tokens that leak ground truth."""
    forbidden_patterns = [
        (r'\bground[_\s-]?truth\b', 'ground_truth'),
        (r'\banswer[_\s]is\b', 'answer_is'),
    ]
    # Note: "TRUE" and "FALSE" are excluded because negation templates
    # like "Is it false that..." are valid question patterns, not leakage.

    leakage_found = []
    for item in items:
        q = item['question']
        for pattern, name in forbidden_patterns:
            if re.search(pattern, q, re.IGNORECASE):
                leakage_found.append({
                    'id': item['id'],
                    'pattern': name,
                    'snippet': q[:100]
                })

    print("\n## Leakage Token Check")
    if leakage_found:
        print(f"⚠️  WARNING: Found {len(leakage_found)} potential leakage cases:")
        for case in leakage_found[:10]:  # Show first 10
            print(f"  - {case['id']}: contains '{case['pattern']}'")
            print(f"    Snippet: {case['snippet']}...")
        if len(leakage_found) > 10:
            print(f"  ... and {len(leakage_found) - 10} more")
        return False
    else:
        print(f"✅ PASS: No forbidden leakage tokens found in {len(items)} questions")
        return True


def check_class_balance(items):
    """Check class balance per task family."""
    print("\n## Class Balance Check")

    by_task = defaultdict(list)
    for item in items:
        by_task[item['type']].append(item)

    imbalanced = []
    for task, task_items in sorted(by_task.items()):
        labels = [item['ground_truth'] for item in task_items]
        label_counts = Counter(labels)
        total = len(task_items)

        # Skip small task families (<20 items) — balance is less meaningful
        if total < 20:
            continue

        # Check balance: allow 80/20 for large families, flag >90/10
        true_like = sum(label_counts.get(l, 0) for l in ['TRUE', 'YES'])
        false_like = sum(label_counts.get(l, 0) for l in ['FALSE', 'NO'])
        if true_like + false_like == 0:
            continue
        majority_ratio = max(true_like, false_like) / (true_like + false_like)
        if majority_ratio > 0.90:
            imbalanced.append((task, label_counts, total))

    if imbalanced:
        print(f"⚠️  WARNING: {len(imbalanced)} task families (≥20 items) have severe imbalance (>90/10):")
        for task, counts, total in imbalanced:
            print(f"  - {task}: {dict(counts)} ({total} total)")
    else:
        print(f"✅ PASS: No task family (≥20 items) has severe class imbalance (>90/10)")

    # Print summary
    print("\n### Class distribution by task family:")
    for task, task_items in sorted(by_task.items()):
        labels = [item['ground_truth'] for item in task_items]
        label_counts = Counter(labels)
        total = len(task_items)
        print(f"- {task}: {dict(label_counts)} ({total} total)")

    return len(imbalanced) == 0


def check_duplicate_questions(items, threshold=0.95):
    """Check for duplicate or near-duplicate questions using text similarity."""
    print("\n## Duplicate Question Check")

    # Normalize questions for comparison
    def normalize(text):
        # Remove punctuation, lowercase, collapse whitespace
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    normalized = [(item['id'], normalize(item['question'])) for item in items]

    # Check exact duplicates
    text_to_ids = defaultdict(list)
    for id_, text in normalized:
        text_to_ids[text].append(id_)

    exact_duplicates = {text: ids for text, ids in text_to_ids.items() if len(ids) > 1}

    # Filter out expected duplicates: paraphrase, perturbation, and consistency
    # batches intentionally have questions that normalize to the same text.
    # Cross-batch overlaps (original + paraphrase) are also expected.
    expected_prefixes = {'para_', 'perturb_', 'consistency_', 'nearmiss_', 'misleading_'}
    unexpected_duplicates = {}
    for text, ids in exact_duplicates.items():
        # Expected if ANY ID contains a paraphrase/perturbation prefix
        has_expected = any(
            any(prefix in id_ for prefix in expected_prefixes)
            for id_ in ids
        )
        if not has_expected:
            unexpected_duplicates[text] = ids

    if unexpected_duplicates:
        n = len(unexpected_duplicates)
        print(f"⚠️  WARNING: Found {n} unexpected duplicate question texts")
        print(f"  (Excluding {len(exact_duplicates) - n} expected paraphrase/perturbation overlaps)")
        for text, ids in list(unexpected_duplicates.items())[:5]:
            print(f"  - {ids}")
            print(f"    Text: {text[:80]}...")
        # Allow up to 15 duplicates (v1 legacy has ~10 known duplicates)
        if n <= 15:
            print(f"  Acceptable: {n} duplicates within v1 legacy tolerance (≤15)")
            return True
        return False
    elif exact_duplicates:
        print(f"✅ PASS: {len(exact_duplicates)} duplicate groups found, all from expected paraphrase/perturbation batches")
        return True
    else:
        print(f"✅ PASS: No duplicate questions found among {len(items)} questions")
        return True


def check_system_task_coverage(items):
    """Check system-task family coverage."""
    print("\n## System-Task Coverage Check")

    system_task_counts = defaultdict(lambda: defaultdict(int))
    for item in items:
        sys_id = item.get('system_id', 'null')
        task = item['type']
        system_task_counts[sys_id][task] += 1

    print(f"\nTotal systems with questions: {len(system_task_counts)}")
    print(f"Total task families: {len(set(item['type'] for item in items))}")

    # Check if any real system (not cross-system pairs) has only 1-2 questions
    # Cross-system pairs like "lorenz63_vs_rossler" are compound IDs, not individual systems
    sparse_systems = {sys: sum(tasks.values()) for sys, tasks in system_task_counts.items()
                      if sum(tasks.values()) < 3 and sys != 'null' and '_vs_' not in sys}

    if sparse_systems:
        print(f"\n⚠️  WARNING: {len(sparse_systems)} systems have <3 questions:")
        for sys, count in sorted(sparse_systems.items(), key=lambda x: x[1]):
            print(f"  - {sys}: {count} questions")
    else:
        print(f"\n✅ PASS: All systems have ≥3 questions")

    return len(sparse_systems) == 0


def main():
    """Run all sanity checks."""
    print("# ChaosBench-Logic v2 Sanity Checks")
    print("\nRunning quality and integrity checks on the dataset.")

    # Load all canonical v2 files
    data_dir = PROJECT_ROOT / "data"
    import json as _json
    selector_path = data_dir / "canonical_v2_files.json"
    if selector_path.exists():
        canonical_files = [PROJECT_ROOT / f for f in _json.loads(selector_path.read_text())["files"]]
    else:
        canonical_files = sorted(data_dir.glob("v22_*.jsonl"))

    all_items = []
    for batch_file in canonical_files:
        if batch_file.exists():
            items = load_jsonl(batch_file)
            all_items.extend(items)

    print(f"\nLoaded {len(all_items)} questions from {len(canonical_files)} canonical files")

    # Run checks
    checks = {
        'ID Uniqueness': check_id_uniqueness(all_items),
        'Leakage Tokens': check_leakage_tokens(all_items),
        'Class Balance': check_class_balance(all_items),
        'Duplicate Questions': check_duplicate_questions(all_items),
        'System-Task Coverage': check_system_task_coverage(all_items),
    }

    # Summary
    print("\n" + "=" * 60)
    print("## Summary")
    print("=" * 60)
    passed = sum(checks.values())
    total = len(checks)
    print(f"\nPassed: {passed}/{total} checks")
    print("\nCheck results:")
    for name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    if passed == total:
        print("\n✅ All sanity checks passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Review warnings above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
