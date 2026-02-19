"""Publication-quality figure generation for ChaosBench-Logic results.

Generates four main figures:
1. Model x task accuracy heatmap
2. 0-1 test vs permutation entropy scatter (colored by chaos label)
3. Logistic map bifurcation diagram with annotated transitions
4. Full vs hard split accuracy comparison

All figures are 300 DPI PDF, suitable for single-column papers.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from chaosbench.data.bifurcations import BIFURCATION_DATA


def plot_accuracy_heatmap(results_dir: str, output: str) -> None:
    """Plot model x task accuracy heatmap.

    Args:
        results_dir: Directory containing result subdirectories (e.g., gpt4_zeroshot/).
        output: Output file path (PDF).
    """
    results_path = Path(results_dir)

    model_dirs = [d for d in results_path.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"No model directories found in {results_dir}")
        return

    task_names = []
    model_names = []
    accuracy_matrix = []

    for model_dir in sorted(model_dirs):
        summary_file = model_dir / "summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file, "r") as f:
            summary = json.load(f)

        model_name = model_dir.name
        model_names.append(model_name)

        task_accuracy = summary.get("task_accuracy", {})

        if not task_names:
            task_names = sorted(task_accuracy.keys())

        row = [task_accuracy.get(task, 0.0) for task in task_names]
        accuracy_matrix.append(row)

    accuracy_matrix = np.array(accuracy_matrix)

    fig, ax = plt.subplots(figsize=(8, 5))

    if HAS_SEABORN:
        sns.heatmap(
            accuracy_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=task_names,
            yticklabels=model_names,
            cbar_kws={"label": "Accuracy"},
            ax=ax,
        )
    else:
        im = ax.imshow(accuracy_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(task_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.set_yticklabels(model_names)

        for i in range(len(model_names)):
            for j in range(len(task_names)):
                text = ax.text(
                    j, i, f"{accuracy_matrix[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=8
                )

        cbar = fig.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xlabel("Task Family")
    ax.set_ylabel("Model")
    ax.set_title("Task Accuracy by Model")

    plt.tight_layout()
    plt.savefig(output, dpi=300, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved accuracy heatmap to {output}")


def plot_indicator_scatter(indicators_dir: str, output: str) -> None:
    """Plot 0-1 test K vs permutation entropy scatter.

    Points are colored by ground truth Chaotic label from system JSON files.
    Looks for indicator data either embedded in system JSON files or in
    separate {system_id}_indicators.json files.

    Args:
        indicators_dir: Directory containing system JSON files with indicators.
        output: Output file path (PDF).
    """
    indicators_path = Path(indicators_dir)

    if not indicators_path.exists():
        print(f"Indicators directory not found: {indicators_dir}")
        return

    system_files = list(indicators_path.glob("*.json"))

    if not system_files:
        print(f"No system JSON files found in {indicators_dir}")
        return

    zero_one_K = []
    perm_entropy = []
    is_chaotic = []
    system_ids = []

    for sys_file in system_files:
        if sys_file.stem.endswith("_indicators"):
            continue

        with open(sys_file, "r") as f:
            sys_data = json.load(f)

        truth = sys_data.get("truth_assignment", {})
        chaotic = truth.get("Chaotic", False)
        system_id = sys_data.get("system_id", sys_file.stem)

        k_val = None
        pe_val = None

        if "zero_one_K" in sys_data and "permutation_entropy" in sys_data:
            k_val = sys_data["zero_one_K"]
            pe_val = sys_data["permutation_entropy"]
        else:
            indicator_file = indicators_path / f"{system_id}_indicators.json"
            if indicator_file.exists():
                with open(indicator_file, "r") as f:
                    ind_data = json.load(f)
                k_val = ind_data.get("zero_one_K")
                pe_val = ind_data.get("permutation_entropy")

        if k_val is not None and pe_val is not None:
            zero_one_K.append(k_val)
            perm_entropy.append(pe_val)
            is_chaotic.append(chaotic)
            system_ids.append(system_id)

    if not zero_one_K:
        print("No valid indicator data found.")
        print("To generate indicators, run:")
        print("  uv run python -c \"from chaosbench.data.indicators.populate import populate_all_systems; populate_all_systems('workspace/hf_release/systems', 'workspace/hf_release/systems/indicators')\"")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    chaotic_mask = np.array(is_chaotic)
    chaotic_points = chaotic_mask == True
    nonchaotic_points = chaotic_mask == False

    zero_one_K = np.array(zero_one_K)
    perm_entropy = np.array(perm_entropy)

    ax.scatter(
        zero_one_K[chaotic_points],
        perm_entropy[chaotic_points],
        c="red",
        label="Chaotic",
        alpha=0.7,
        s=50,
    )
    ax.scatter(
        zero_one_K[nonchaotic_points],
        perm_entropy[nonchaotic_points],
        c="blue",
        label="Non-chaotic",
        alpha=0.7,
        s=50,
    )

    ax.set_xlabel("0-1 Test (K)")
    ax.set_ylabel("Permutation Entropy")
    ax.set_title("Chaos Indicators: 0-1 Test vs Permutation Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=300, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved indicator scatter to {output}")


def plot_bifurcation_diagram(output: str) -> None:
    """Plot logistic map bifurcation diagram with annotated transitions.

    Uses hardcoded transition points from BIFURCATION_DATA.

    Args:
        output: Output file path (PDF).
    """
    logistic_info = BIFURCATION_DATA.get("logistic")

    if not logistic_info:
        print("Logistic map bifurcation data not found")
        return

    r_min = 2.5
    r_max = 4.0
    n_r = 2000
    n_iter = 1000
    n_plot = 300

    r_values = np.linspace(r_min, r_max, n_r)

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in r_values:
        x = 0.5

        for _ in range(n_iter - n_plot):
            x = r * x * (1 - x)

        x_vals = []
        for _ in range(n_plot):
            x = r * x * (1 - x)
            x_vals.append(x)

        ax.plot([r] * len(x_vals), x_vals, 'k,', markersize=0.5, alpha=0.5)

    transitions = logistic_info.transitions

    annotated_transitions = [
        transitions[2],
        transitions[5],
        transitions[6],
    ]

    y_positions = [0.85, 0.55, 0.25]

    for trans, y_pos in zip(annotated_transitions, y_positions):
        r_val = trans.param_value
        ax.axvline(r_val, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(
            r_val,
            y_pos,
            f"r={r_val:.3f}\n{trans.regime.replace('_', ' ')}",
            fontsize=8,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Parameter r")
    ax.set_ylabel("x")
    ax.set_title("Logistic Map Bifurcation Diagram")
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output, dpi=300, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved bifurcation diagram to {output}")


def plot_hard_split_comparison(results_dir: str, output: str) -> None:
    """Plot paired bar chart comparing full vs hard split accuracy.

    Args:
        results_dir: Directory containing result subdirectories.
        output: Output file path (PDF).
    """
    results_path = Path(results_dir)

    model_dirs = [d for d in results_path.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"No model directories found in {results_dir}")
        return

    model_names = []
    full_accuracy = []
    hard_accuracy = []

    for model_dir in sorted(model_dirs):
        summary_file = model_dir / "summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file, "r") as f:
            summary = json.load(f)

        model_name = model_dir.name
        model_names.append(model_name)

        overall_acc = summary.get("overall_accuracy", 0.0)
        full_accuracy.append(overall_acc)

        task_acc = summary.get("task_accuracy", {})
        hard_acc = task_acc.get("hard", 0.0)
        hard_accuracy.append(hard_acc)

    if not model_names:
        print("No valid model data found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, full_accuracy, width, label="Full Split", alpha=0.8)
    bars2 = ax.bar(x + width/2, hard_accuracy, width, label="Hard Split", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Full Split vs Hard Split Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output, dpi=300, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved hard split comparison to {output}")


def main():
    """Generate all figures from command line."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for ChaosBench-Logic"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures (default: figures/)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="published_results",
        help="Directory containing model result subdirectories (default: published_results/)",
    )
    parser.add_argument(
        "--indicators-dir",
        type=str,
        default="workspace/hf_release/systems",
        help="Directory containing system JSON files (default: workspace/hf_release/systems/)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")

    plot_accuracy_heatmap(
        args.results_dir,
        str(output_dir / "accuracy_heatmap.pdf")
    )

    plot_indicator_scatter(
        args.indicators_dir,
        str(output_dir / "indicator_scatter.pdf")
    )

    plot_bifurcation_diagram(
        str(output_dir / "bifurcation_diagram.pdf")
    )

    plot_hard_split_comparison(
        args.results_dir,
        str(output_dir / "hard_split_comparison.pdf")
    )

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
