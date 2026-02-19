#!/usr/bin/env python3
"""Generate publication-ready tables and metrics for ChaosBench-Logic.

Outputs markdown and CSV tables suitable for papers, presentations, and documentation.

Usage:
    python scripts/generate_paper_tables.py [--data-dir data] [--output-dir reports/paper_assets]
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


def generate_family_table(questions, output_dir):
    """Generate per-family counts and balance table."""
    family_stats = defaultdict(lambda: {'count': 0, 'TRUE': 0, 'FALSE': 0})

    for q in questions:
        family = q['type']
        family_stats[family]['count'] += 1
        family_stats[family][q['ground_truth']] += 1

    # Markdown table
    md_lines = [
        "# Per-Family Statistics",
        "",
        "| Family | Count | TRUE | FALSE | TRUE % | Balance |",
        "|--------|-------|------|-------|--------|---------|",
    ]

    csv_lines = ["family,count,true,false,true_pct,balance"]

    for family in sorted(family_stats.keys()):
        stats = family_stats[family]
        count = stats['count']
        true_count = stats['TRUE']
        false_count = stats['FALSE']
        true_pct = true_count / count * 100 if count > 0 else 0
        balance = "Good" if 30 <= true_pct <= 70 else "Skewed"

        md_lines.append(
            f"| {family} | {count} | {true_count} | {false_count} | {true_pct:.1f}% | {balance} |"
        )
        csv_lines.append(f"{family},{count},{true_count},{false_count},{true_pct:.1f},{balance}")

    # Write files
    with open(os.path.join(output_dir, "family_stats.md"), "w") as f:
        f.write("\n".join(md_lines))

    with open(os.path.join(output_dir, "family_stats.csv"), "w") as f:
        f.write("\n".join(csv_lines))

    print(f"✅ Generated: family_stats.md, family_stats.csv")


def generate_split_table(questions, output_dir):
    """Generate per-split counts table."""
    from chaosbench.data.splits import assign_split_v22

    split_stats = defaultdict(lambda: {'count': 0, 'TRUE': 0, 'FALSE': 0})

    for q in questions:
        split = assign_split_v22(q)
        split_stats[split]['count'] += 1
        split_stats[split][q['ground_truth']] += 1

    # Markdown table
    md_lines = [
        "# Per-Split Statistics",
        "",
        "| Split | Count | TRUE | FALSE | TRUE % |",
        "|-------|-------|------|-------|--------|",
    ]

    csv_lines = ["split,count,true,false,true_pct"]

    for split in sorted(split_stats.keys()):
        stats = split_stats[split]
        count = stats['count']
        true_count = stats['TRUE']
        false_count = stats['FALSE']
        true_pct = true_count / count * 100 if count > 0 else 0

        md_lines.append(
            f"| {split} | {count} | {true_count} | {false_count} | {true_pct:.1f}% |"
        )
        csv_lines.append(f"{split},{count},{true_count},{false_count},{true_pct:.1f}")

    # Write files
    with open(os.path.join(output_dir, "split_stats.md"), "w") as f:
        f.write("\n".join(md_lines))

    with open(os.path.join(output_dir, "split_stats.csv"), "w") as f:
        f.write("\n".join(csv_lines))

    print(f"✅ Generated: split_stats.md, split_stats.csv")


def generate_system_coverage_table(questions, output_dir):
    """Generate system coverage summary table."""
    # Count per system
    system_counts = defaultdict(int)
    for q in questions:
        system_counts[q['system_id']] += 1

    # Filter real systems (not "vs" synthetic)
    real_systems = {sid: count for sid, count in system_counts.items() if '_vs_' not in sid}

    # Compute stats
    counts = list(real_systems.values())
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    median_count = sorted(counts)[len(counts) // 2] if counts else 0

    # Coverage bins
    bins = {
        '<10': sum(1 for c in counts if c < 10),
        '10-49': sum(1 for c in counts if 10 <= c < 50),
        '50-99': sum(1 for c in counts if 50 <= c < 100),
        '100+': sum(1 for c in counts if c >= 100),
    }

    # Markdown table
    md_lines = [
        "# System Coverage Summary",
        "",
        "## Overall Statistics",
        "",
        f"- **Total real systems:** {len(real_systems)}",
        f"- **Min questions per system:** {min_count}",
        f"- **Median questions per system:** {median_count}",
        f"- **Max questions per system:** {max_count}",
        "",
        "## Coverage Distribution",
        "",
        "| Range | Count | Percentage |",
        "|-------|-------|------------|",
    ]

    csv_lines = ["range,count,percentage"]

    for range_name, count in bins.items():
        pct = count / len(real_systems) * 100 if real_systems else 0
        md_lines.append(f"| {range_name} | {count} | {pct:.1f}% |")
        csv_lines.append(f"{range_name},{count},{pct:.1f}")

    # Write files
    with open(os.path.join(output_dir, "system_coverage.md"), "w") as f:
        f.write("\n".join(md_lines))

    with open(os.path.join(output_dir, "system_coverage.csv"), "w") as f:
        f.write("\n".join(csv_lines))

    print(f"✅ Generated: system_coverage.md, system_coverage.csv")


def generate_duplicate_summary_table(output_dir):
    """Generate duplicate summary table from pre-existing reports."""
    # Check if duplicate report exists
    dupes_path = "reports/pre_freeze_dupes/per_family_summary.json"
    if not os.path.exists(dupes_path):
        print(f"⚠️  Skipping duplicate summary (run duplicate_report.py first)")
        return

    with open(dupes_path) as f:
        dupe_data = json.load(f)

    # Markdown table
    md_lines = [
        "# Duplicate Summary",
        "",
        f"- **Exact duplicates (accidental):** {dupe_data.get('total_accidental_duplicates', 0)}",
        f"- **Near-duplicate pairs:** {dupe_data.get('total_near_duplicate_pairs', 'N/A')}",
        "",
    ]

    # Write file
    with open(os.path.join(output_dir, "duplicate_summary.md"), "w") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Generated: duplicate_summary.md")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready tables")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--output-dir", default="reports/paper_assets", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading questions from {args.data_dir}...")
    questions = load_all_questions(args.data_dir)
    print(f"Loaded {len(questions)} questions")
    print()

    print("Generating tables...")
    generate_family_table(questions, args.output_dir)
    generate_split_table(questions, args.output_dir)
    generate_system_coverage_table(questions, args.output_dir)
    generate_duplicate_summary_table(args.output_dir)

    print()
    print(f"✅ All tables generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
