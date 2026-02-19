#!/usr/bin/env python3
"""Comprehensive system utilization audit for ChaosBench-Logic.

Analyzes which systems are actually being used, by how much, and why some may be underutilized.

Usage:
    python scripts/system_utilization_audit.py [--data-dir data] [--output-dir reports/system_utilization]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def load_all_questions(data_dir: str):
    """Load all v2.2 questions from JSONL files."""
    questions = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith('v22_') and fname.endswith('.jsonl'):
            with open(os.path.join(data_dir, fname)) as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
    return questions


def load_generation_stats():
    """Load generation statistics if available."""
    stats_path = "reports/scale_diag/generation_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return None


def load_system_metadata(systems_dir: str = "systems"):
    """Load system metadata to check eligibility."""
    systems = {}

    # Load core systems
    for fname in sorted(os.listdir(systems_dir)):
        if fname.endswith('.json'):
            with open(os.path.join(systems_dir, fname)) as f:
                data = json.load(f)
                if 'system_id' in data:
                    systems[data['system_id']] = {
                        'source': 'core',
                        'has_truth_assignment': 'truth_assignment' in data,
                        'predicates': list(data.get('truth_assignment', {}).keys()),
                    }

    # Load dysts systems
    dysts_dir = os.path.join(systems_dir, 'dysts')
    if os.path.isdir(dysts_dir):
        for fname in sorted(os.listdir(dysts_dir)):
            if fname.endswith('.json'):
                with open(os.path.join(dysts_dir, fname)) as f:
                    data = json.load(f)
                    if 'system_id' in data:
                        systems[data['system_id']] = {
                            'source': 'dysts',
                            'has_truth_assignment': 'truth_assignment' in data,
                            'predicates': list(data.get('truth_assignment', {}).keys()),
                        }

    return systems


def compute_quantiles(values, quantiles=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]):
    """Compute quantiles from a list of values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return {q: 0 for q in quantiles}

    result = {}
    for q in quantiles:
        idx = int(q * (n - 1))
        result[q] = sorted_vals[idx]
    return result


def main():
    parser = argparse.ArgumentParser(description="System utilization audit")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--output-dir", default="reports/system_utilization", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("SYSTEM UTILIZATION AUDIT")
    print("="*80)
    print()

    # Load data
    print("Loading questions...")
    questions = load_all_questions(args.data_dir)
    print(f"Loaded {len(questions)} questions")

    print("Loading system metadata...")
    systems_meta = load_system_metadata()
    print(f"Found {len(systems_meta)} systems in metadata")

    print("Loading generation stats...")
    gen_stats = load_generation_stats()

    print()

    # Filter real systems (not "vs" synthetic)
    real_questions = [q for q in questions if '_vs_' not in q['system_id']]
    print(f"Real system questions: {len(real_questions)}")
    print(f"Synthetic (vs) questions: {len(questions) - len(real_questions)}")
    print()

    # Per-system counts
    system_counts = defaultdict(lambda: {
        'total': 0,
        'by_family': defaultdict(int),
        'by_label': {'TRUE': 0, 'FALSE': 0},
    })

    for q in real_questions:
        sid = q['system_id']
        system_counts[sid]['total'] += 1
        system_counts[sid]['by_family'][q['type']] += 1
        system_counts[sid]['by_label'][q['ground_truth']] += 1

    # Separate core vs dysts
    core_systems = {}
    dysts_systems = {}

    for sid, counts in system_counts.items():
        if sid in systems_meta:
            if systems_meta[sid]['source'] == 'core':
                core_systems[sid] = counts
            else:
                dysts_systems[sid] = counts
        else:
            # Unknown, assume dysts
            dysts_systems[sid] = counts

    print(f"Systems Analysis:")
    print(f"  Core systems in data: {len(core_systems)}")
    print(f"  Dysts systems in data: {len(dysts_systems)}")
    print(f"  Total real systems: {len(system_counts)}")
    print()

    # Quantile analysis
    all_counts = [counts['total'] for counts in system_counts.values()]
    core_counts = [counts['total'] for counts in core_systems.values()]
    dysts_counts = [counts['total'] for counts in dysts_systems.values()]

    all_quantiles = compute_quantiles(all_counts)
    core_quantiles = compute_quantiles(core_counts) if core_counts else {}
    dysts_quantiles = compute_quantiles(dysts_counts) if dysts_counts else {}

    print("Quantile Analysis (questions per system):")
    print(f"  All systems:")
    print(f"    Min: {all_quantiles[0]}, P10: {all_quantiles[0.1]}, P25: {all_quantiles[0.25]}")
    print(f"    Median: {all_quantiles[0.5]}, P75: {all_quantiles[0.75]}, P90: {all_quantiles[0.9]}, Max: {all_quantiles[1.0]}")

    if core_quantiles:
        print(f"  Core systems:")
        print(f"    Median: {core_quantiles[0.5]}, Avg: {sum(core_counts)/len(core_counts):.1f}")

    if dysts_quantiles:
        print(f"  Dysts systems:")
        print(f"    Median: {dysts_quantiles[0.5]}, Avg: {sum(dysts_counts)/len(dysts_counts):.1f}")
    print()

    # Long tail analysis
    systems_lt_10 = [(sid, counts['total']) for sid, counts in system_counts.items() if counts['total'] < 10]
    systems_lt_50 = [(sid, counts['total']) for sid, counts in system_counts.items() if 10 <= counts['total'] < 50]
    systems_lt_100 = [(sid, counts['total']) for sid, counts in system_counts.items() if 50 <= counts['total'] < 100]
    systems_gte_100 = [(sid, counts['total']) for sid, counts in system_counts.items() if counts['total'] >= 100]

    print("Coverage Distribution:")
    print(f"  <10 questions: {len(systems_lt_10)} systems ({len(systems_lt_10)/len(system_counts)*100:.1f}%)")
    print(f"  10-49 questions: {len(systems_lt_50)} systems ({len(systems_lt_50)/len(system_counts)*100:.1f}%)")
    print(f"  50-99 questions: {len(systems_lt_100)} systems ({len(systems_lt_100)/len(system_counts)*100:.1f}%)")
    print(f"  >=100 questions: {len(systems_gte_100)} systems ({len(systems_gte_100)/len(system_counts)*100:.1f}%)")
    print()

    # Per-family coverage
    family_coverage = defaultdict(lambda: {'systems_touched': set(), 'total_questions': 0})
    for q in real_questions:
        family_coverage[q['type']]['systems_touched'].add(q['system_id'])
        family_coverage[q['type']]['total_questions'] += 1

    print("Per-Family System Coverage:")
    for family in sorted(family_coverage.keys()):
        touched = len(family_coverage[family]['systems_touched'])
        total_qs = family_coverage[family]['total_questions']
        pct = touched / len(system_counts) * 100 if system_counts else 0
        avg_per_sys = total_qs / touched if touched > 0 else 0
        print(f"  {family:30s}: {touched:3d}/{len(system_counts):3d} systems ({pct:5.1f}%), avg {avg_per_sys:.1f} q/sys")
    print()

    # Write CSV: usage by system
    csv_lines = ["system_id,source,total,atomic,consistency,multi_hop,perturbation,adversarial,other,true_pct"]
    for sid in sorted(system_counts.keys()):
        counts = system_counts[sid]
        source = systems_meta.get(sid, {}).get('source', 'unknown')
        atomic = counts['by_family'].get('atomic', 0)
        consistency = counts['by_family'].get('consistency_paraphrase', 0)
        multi_hop = counts['by_family'].get('multi_hop', 0)
        perturbation = counts['by_family'].get('perturbation', 0)
        adversarial = counts['by_family'].get('adversarial_misleading', 0) + counts['by_family'].get('adversarial_nearmiss', 0)
        other = counts['total'] - atomic - consistency - multi_hop - perturbation - adversarial
        true_pct = counts['by_label']['TRUE'] / counts['total'] * 100 if counts['total'] > 0 else 0

        csv_lines.append(f"{sid},{source},{counts['total']},{atomic},{consistency},{multi_hop},{perturbation},{adversarial},{other},{true_pct:.1f}")

    with open(os.path.join(args.output_dir, "usage_by_system.csv"), "w") as f:
        f.write("\n".join(csv_lines))

    print(f"✅ Generated: usage_by_system.csv")

    # Write CSV: usage by family
    csv_lines = ["family,total_questions,systems_touched,systems_available,coverage_pct,avg_per_system"]
    for family in sorted(family_coverage.keys()):
        touched = len(family_coverage[family]['systems_touched'])
        total_qs = family_coverage[family]['total_questions']
        pct = touched / len(system_counts) * 100 if system_counts else 0
        avg_per_sys = total_qs / touched if touched > 0 else 0
        csv_lines.append(f"{family},{total_qs},{touched},{len(system_counts)},{pct:.1f},{avg_per_sys:.1f}")

    with open(os.path.join(args.output_dir, "usage_by_family.csv"), "w") as f:
        f.write("\n".join(csv_lines))

    print(f"✅ Generated: usage_by_family.csv")

    # Write coverage summary markdown
    md_lines = [
        "# System Utilization Coverage Summary",
        "",
        f"**Total real systems:** {len(system_counts)}",
        f"**Total questions:** {len(real_questions)}",
        f"**Average questions per system:** {len(real_questions) / len(system_counts):.1f}",
        "",
        "## Quantile Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Min | {all_quantiles[0]} |",
        f"| P10 | {all_quantiles[0.1]} |",
        f"| P25 | {all_quantiles[0.25]} |",
        f"| Median | {all_quantiles[0.5]} |",
        f"| P75 | {all_quantiles[0.75]} |",
        f"| P90 | {all_quantiles[0.9]} |",
        f"| Max | {all_quantiles[1.0]} |",
        "",
        "## Coverage Distribution",
        "",
        f"- **<10 questions:** {len(systems_lt_10)} systems ({len(systems_lt_10)/len(system_counts)*100:.1f}%)",
        f"- **10-49 questions:** {len(systems_lt_50)} systems ({len(systems_lt_50)/len(system_counts)*100:.1f}%)",
        f"- **50-99 questions:** {len(systems_lt_100)} systems ({len(systems_lt_100)/len(system_counts)*100:.1f}%)",
        f"- **>=100 questions:** {len(systems_gte_100)} systems ({len(systems_gte_100)/len(system_counts)*100:.1f}%)",
        "",
        "## Core vs Dysts Comparison",
        "",
        f"- **Core systems:** {len(core_systems)} ({len(core_systems)/len(system_counts)*100:.1f}%)",
        f"  - Avg questions/system: {sum(core_counts)/len(core_counts):.1f}" if core_counts else "",
        f"  - Median: {core_quantiles.get(0.5, 0)}",
        f"- **Dysts systems:** {len(dysts_systems)} ({len(dysts_systems)/len(system_counts)*100:.1f}%)",
        f"  - Avg questions/system: {sum(dysts_counts)/len(dysts_counts):.1f}" if dysts_counts else "",
        f"  - Median: {dysts_quantiles.get(0.5, 0)}",
        "",
        "## Long Tail (<50 questions)",
        "",
    ]

    if systems_lt_10:
        md_lines.append("### Systems with <10 questions:")
        md_lines.append("")
        for sid, count in sorted(systems_lt_10, key=lambda x: x[1]):
            source = systems_meta.get(sid, {}).get('source', 'unknown')
            md_lines.append(f"- `{sid}` ({source}): {count} questions")
        md_lines.append("")

    if systems_lt_50:
        md_lines.append("### Systems with 10-49 questions:")
        md_lines.append("")
        for sid, count in sorted(systems_lt_50, key=lambda x: x[1])[:20]:  # Show top 20
            source = systems_meta.get(sid, {}).get('source', 'unknown')
            md_lines.append(f"- `{sid}` ({source}): {count} questions")
        if len(systems_lt_50) > 20:
            md_lines.append(f"- ... and {len(systems_lt_50) - 20} more")
        md_lines.append("")

    # Gap explanation
    md_lines.extend([
        "## Scaling Gap Attribution",
        "",
    ])

    if gen_stats:
        md_lines.append("### Per-Family Achievement vs Target")
        md_lines.append("")
        md_lines.append("| Family | Requested | Generated | Final | Achievement % | Gap |")
        md_lines.append("|--------|-----------|-----------|-------|---------------|-----|")

        for family, stats in gen_stats.get('per_family', {}).items():
            md_lines.append(
                f"| {family} | {stats['requested_target']} | {stats['generated_count']} | "
                f"{stats['final_count']} | {stats['achievement_pct']:.1f}% | {stats['gap_from_target']} |"
            )

        md_lines.append("")
        md_lines.append("### Top Underperforming Families (by gap)")
        md_lines.append("")

        # Sort by gap
        families_by_gap = sorted(
            gen_stats.get('per_family', {}).items(),
            key=lambda x: x[1]['gap_from_target'],
            reverse=True
        )[:5]

        for family, stats in families_by_gap:
            md_lines.append(f"**{family}:** Gap of {stats['gap_from_target']} ({stats['achievement_pct']:.1f}% achievement)")
            md_lines.append(f"- Eligible systems: {stats['eligible_systems']}")
            md_lines.append(f"- Generated: {stats['generated_count']}, Final: {stats['final_count']}, Dedupe removed: {stats['dedupe_removed']}")
            md_lines.append("")

    with open(os.path.join(args.output_dir, "coverage_summary.md"), "w") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Generated: coverage_summary.md")

    # Write coverage histogram JSON
    histogram = {
        "bins": {
            "<10": len(systems_lt_10),
            "10-49": len(systems_lt_50),
            "50-99": len(systems_lt_100),
            ">=100": len(systems_gte_100),
        },
        "quantiles": all_quantiles,
        "core_vs_dysts": {
            "core": {
                "count": len(core_systems),
                "avg": sum(core_counts)/len(core_counts) if core_counts else 0,
                "median": core_quantiles.get(0.5, 0),
            },
            "dysts": {
                "count": len(dysts_systems),
                "avg": sum(dysts_counts)/len(dysts_counts) if dysts_counts else 0,
                "median": dysts_quantiles.get(0.5, 0),
            },
        },
        "long_tail": {
            "<10": [sid for sid, _ in systems_lt_10],
            "10-49": [sid for sid, _ in systems_lt_50],
        }
    }

    with open(os.path.join(args.output_dir, "coverage_histogram.json"), "w") as f:
        json.dump(histogram, f, indent=2)

    print(f"✅ Generated: coverage_histogram.json")

    print()
    print("="*80)
    print("AUDIT COMPLETE")
    print("="*80)
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
