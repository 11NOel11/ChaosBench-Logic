#!/usr/bin/env python3
"""Generate v2 novelty report summarizing unique contributions.

This script produces a concise report suitable for paper intro or README,
highlighting measurable v2 contributions beyond scale.
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.eval.runner import load_jsonl


def main():
    """Generate novelty report."""
    print("# ChaosBench-Logic v2 Novelty Report")
    print("\nGenerated:", Path(__file__).parent.parent.name)

    # Load dataset
    data_dir = PROJECT_ROOT / "data"
    all_items = []
    for batch_file in sorted(data_dir.glob("batch*.jsonl")):
        all_items.extend(load_jsonl(batch_file))

    # Count task families
    task_counts = Counter(item['type'] for item in all_items)

    # Identify v2-new families
    v2_new = {
        'indicator_diagnostic': 'Numerical chaos indicator interpretation',
        'regime_transition': 'Bifurcation and parameter-dependent behavior',
        'fol_inference': 'First-order logic premise-conclusion chains',
        'cross_indicator': 'Multi-indicator cross-validation reasoning',
        'adversarial': 'Common misconception and edge case testing',
        'consistency_paraphrase': 'Linguistic variation robustness',
        'extended_systems': 'Generalization to underrepresented systems',
    }

    print("\n## v2 Unique Contributions (Beyond Scale)")
    print("\nChaosBench-Logic v2.1 introduces **7 new task families** that test capabilities not assessed in v1:\n")

    for i, (task, description) in enumerate(v2_new.items(), 1):
        count = task_counts.get(task, 0)
        print(f"{i}. **{task.replace('_', ' ').title()}** ({count} questions)")
        print(f"   - {description}")

    # Calculate v1 vs v2 split
    v1_tasks = {'atomic', 'multi_hop', 'cross_system', 'bias', 'counterfactual', 'multi_turn', 'hard'}
    v1_count = sum(task_counts[t] for t in v1_tasks if t in task_counts)
    v2_count = sum(task_counts[t] for t in v2_new if t in task_counts)

    print("\n## Dataset Composition")
    print(f"\n- **v1 core tasks** (batches 1-7): {v1_count} questions")
    print(f"- **v2 extensions** (batches 8-14): {v2_count} questions")
    print(f"- **Total**: {len(all_items)} questions")

    # System coverage
    systems_with_null = set(item.get('system_id') for item in all_items)
    systems = systems_with_null - {None}
    core_systems = len([f for f in (PROJECT_ROOT / "systems").glob("*.json")
                        if f.stem != "dysts"])
    dysts_systems = len(list((PROJECT_ROOT / "systems" / "dysts").glob("*.json")))

    print("\n## System Coverage")
    print(f"\n- **Core systems**: {core_systems} (manually curated)")
    print(f"- **Extended systems (dysts)**: {dysts_systems} (imported)")
    print(f"- **Total**: {core_systems + dysts_systems} systems")

    # Key metrics for paper
    print("\n## Key Metrics for Publication")
    print("\n### Measurable v2 Contributions:\n")

    print("1. **Indicator Diagnostics (550 questions)**")
    print("   - Tests numerical threshold reasoning with real chaos indicators")
    print("   - Expected challenge: Models may struggle with K≈0.5 boundary cases")
    print("   - Metric: Accuracy on indicator_diagnostic vs baseline atomic tasks\n")

    print("2. **Regime Transition Understanding (68 questions)**")
    print("   - Tests bifurcation and qualitative behavior change reasoning")
    print("   - Expected challenge: Parameter-dependent reasoning")
    print("   - Metric: Accuracy on regime_transition task family\n")

    print("3. **FOL Consistency (121 questions)**")
    print("   - Tests formal logic chains and axiom adherence")
    print("   - Expected challenge: Maintaining logical consistency under multi-premise reasoning")
    print("   - Metric: FOL violation rate, accuracy on fol_inference\n")

    print("4. **Robustness (404 questions)**")
    print("   - Tests stability under adversarial + paraphrase perturbations")
    print("   - Expected challenge: Consistency across linguistic variations")
    print("   - Metric: Accuracy delta between original and paraphrased questions\n")

    print("5. **Cross-Indicator Validation (70 questions)**")
    print("   - Tests reasoning across multiple chaos indicators simultaneously")
    print("   - Expected challenge: Multi-metric integration")
    print("   - Metric: Accuracy on cross_indicator vs single indicator baseline\n")

    print("6. **System Generalization (45 questions)**")
    print("   - Tests performance on rare/underrepresented dysts systems")
    print("   - Expected challenge: Zero-shot transfer to unseen systems")
    print("   - Metric: Accuracy gap between core and extended systems\n")

    print("\n## Recommended Evaluation Protocol")
    print("\n1. Run baseline on v1 core tasks (batches 1-7) for comparison")
    print("2. Evaluate full v2.1 (batches 1-14)")
    print("3. Report per-task-family accuracy to assess contribution-specific performance")
    print("4. Compute FOL violation rate to measure logical consistency")
    print("5. Measure paraphrase stability (batch 11)")
    print("6. Compare indicator diagnostic accuracy to atomic baseline")

    print("\n## Summary for Paper Intro")
    print("\nChaosBench-Logic v2.1 extends v1 with **1,258 new questions** across **7 novel")
    print("task families**, testing indicator interpretation, regime transitions, FOL reasoning,")
    print("adversarial robustness, and system generalization. The v2 dataset enables fine-grained")
    print("evaluation of reasoning capabilities not assessed in v1, providing evidence for:")
    print("- Numerical threshold reasoning (indicators)")
    print("- Dynamical qualitative change understanding (bifurcations)")
    print("- Formal logic consistency (FOL)")
    print("- Linguistic robustness (paraphrases + adversarial)")
    print("- Zero-shot generalization (extended systems)")


if __name__ == "__main__":
    main()
