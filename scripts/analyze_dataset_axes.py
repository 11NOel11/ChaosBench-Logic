#!/usr/bin/env python3
"""Analyze v2 dataset along scientific contribution axes.

This script quantifies the distinctive characteristics of v2:
- Task family distribution and balance
- Indicator question coverage
- System diversity (core vs extended)
- Paraphrase multiplicity
- Difficulty stratification
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.eval.runner import load_jsonl


def analyze_task_family_distribution(items):
    """Analyze task family distribution."""
    task_counts = Counter(item['type'] for item in items)
    print("\n## Task Family Distribution")
    print("\n| Task Family | Count | Percentage |")
    print("|-------------|------:|------------|")
    total = len(items)
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"| {task} | {count} | {pct:.1f}% |")
    print(f"| **TOTAL** | **{total}** | **100.0%** |")
    return task_counts


def analyze_system_coverage(items):
    """Analyze system coverage (core vs extended)."""
    core_systems = set()
    extended_systems = set()

    core_dir = PROJECT_ROOT / "systems"
    for f in core_dir.glob("*.json"):
        if f.stem != "dysts":
            core_systems.add(f.stem)

    dysts_dir = core_dir / "dysts"
    if dysts_dir.exists():
        for f in dysts_dir.glob("*.json"):
            extended_systems.add(f"dysts/{f.stem}")

    items_by_system_type = defaultdict(list)
    for item in items:
        sys_id = item.get('system_id')
        if sys_id is None:
            items_by_system_type['ontology'].append(item)
        elif sys_id in core_systems:
            items_by_system_type['core'].append(item)
        else:
            items_by_system_type['extended'].append(item)

    print("\n## System Coverage")
    print(f"\nCore systems defined: {len(core_systems)}")
    print(f"Extended systems (dysts): {len(extended_systems)}")
    print(f"\nQuestions by system type:")
    print(f"- Core system questions: {len(items_by_system_type['core'])}")
    print(f"- Extended system questions: {len(items_by_system_type['extended'])}")
    print(f"- Ontology questions (no system): {len(items_by_system_type['ontology'])}")

    return items_by_system_type


def analyze_indicator_coverage(items):
    """Analyze chaos indicator question coverage."""
    indicator_items = [item for item in items if item['type'] == 'indicator_diagnostic']

    # Parse indicator types from question text
    indicator_counts = Counter()
    for item in indicator_items:
        q = item['question'].lower()
        if '0-1 test' in q or 'zero-one' in q or 'k=' in q:
            indicator_counts['zero_one_test'] += 1
        if 'permutation entropy' in q or 'pe=' in q:
            indicator_counts['permutation_entropy'] += 1
        if 'megno' in q or 'mean exponential' in q:
            indicator_counts['megno'] += 1

    print("\n## Indicator Diagnostic Coverage")
    print(f"\nTotal indicator questions: {len(indicator_items)}")
    print("\nBy indicator type:")
    for indicator, count in sorted(indicator_counts.items()):
        print(f"- {indicator}: {count}")

    return indicator_counts


def analyze_paraphrase_multiplicity(items):
    """Analyze paraphrase consistency testing."""
    paraphrase_items = [item for item in items if item['type'] == 'consistency_paraphrase']

    # Group by base question ID (assuming pattern like "q0001_p1", "q0001_p2")
    base_groups = defaultdict(list)
    for item in paraphrase_items:
        base_id = item['id'].rsplit('_', 1)[0] if '_p' in item['id'] else item['id']
        base_groups[base_id].append(item)

    print("\n## Paraphrase Consistency Testing")
    print(f"\nTotal paraphrase variants: {len(paraphrase_items)}")
    print(f"Base questions paraphrased: {len(base_groups)}")
    if base_groups:
        variant_counts = [len(variants) for variants in base_groups.values()]
        avg_variants = sum(variant_counts) / len(variant_counts)
        print(f"Average variants per base question: {avg_variants:.1f}")
        print(f"Variant count range: {min(variant_counts)}-{max(variant_counts)}")

    return base_groups


def analyze_v2_novelty(items):
    """Identify v2-specific novelty."""
    v1_tasks = {'atomic', 'multi_hop', 'cross_system', 'bias', 'counterfactual', 'multi_turn', 'hard'}
    v2_tasks = {'indicator_diagnostic', 'regime_transition', 'fol_inference', 'cross_indicator',
                'extended_systems', 'adversarial', 'consistency_paraphrase', 'perturbation_robustness'}

    v1_items = [item for item in items if item['type'] in v1_tasks]
    v2_items = [item for item in items if item['type'] in v2_tasks]

    print("\n## v2 Novelty Analysis")
    print(f"\nv1 task families: {len(v1_tasks)}")
    print(f"v2-new task families: {len(v2_tasks)}")
    print(f"\nv1 questions: {len(v1_items)} ({100*len(v1_items)/len(items):.1f}%)")
    print(f"v2-new questions: {len(v2_items)} ({100*len(v2_items)/len(items):.1f}%)")

    print("\n### v2-new task families:")
    for task in sorted(v2_tasks):
        count = sum(1 for item in items if item['type'] == task)
        if count > 0:
            print(f"- {task}: {count} questions")

    return {'v1': v1_items, 'v2': v2_items}


def main():
    """Run all analyses."""
    print("# ChaosBench-Logic v2 Dataset Analysis")
    print("\nAnalyzing distinctive v2 contributions and dataset composition.")

    # Load all batch files
    data_dir = PROJECT_ROOT / "data"
    all_items = []
    for batch_file in sorted(data_dir.glob("batch*.jsonl")):
        items = load_jsonl(batch_file)
        all_items.extend(items)
        print(f"\nLoaded {batch_file.name}: {len(items)} questions")

    print(f"\n**Total questions loaded: {len(all_items)}**")

    # Run analyses
    analyze_task_family_distribution(all_items)
    analyze_system_coverage(all_items)
    analyze_indicator_coverage(all_items)
    analyze_paraphrase_multiplicity(all_items)
    novelty_split = analyze_v2_novelty(all_items)

    print("\n## Summary")
    print(f"\n- Total questions: {len(all_items)}")
    print(f"- Task families: {len(set(item['type'] for item in all_items))}")
    print(f"- v1 questions: {len(novelty_split['v1'])}")
    print(f"- v2-new questions: {len(novelty_split['v2'])}")
    print("\nv2 adds:")
    print("- Indicator diagnostics (numerical threshold reasoning)")
    print("- Regime transitions (bifurcation understanding)")
    print("- FOL inference (formal logic chains)")
    print("- Cross-indicator reasoning (multi-metric validation)")
    print("- Adversarial robustness testing")
    print("- Consistency under paraphrase")
    print("- Extended system generalization")


if __name__ == "__main__":
    main()
