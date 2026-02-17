"""Generate LaTeX tables from published results and system indicators.

Creates publication-ready booktabs tables for:
- Model x Mode accuracy matrix
- Model x TaskFamily breakdown
- System x Indicator values (0-1 test, PE, MEGNO)
- Full vs Hard split comparison
- Per-split and per-task-family accuracy using axis metrics
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from chaosbench.eval.metrics import EvalResult, compute_axis_metrics, format_axis_report


def _load_summary(results_dir: str, model: str, mode: str) -> Dict:
    """Load summary.json for a given model/mode configuration.

    Args:
        results_dir: Path to published_results directory.
        model: Model name (e.g., 'gpt4', 'claude3').
        mode: Inference mode ('zeroshot', 'cot').

    Returns:
        Dict containing summary data, or empty dict if file not found.
    """
    config_name = f"{model}_{mode}"
    summary_path = os.path.join(results_dir, config_name, "summary.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r") as fh:
        return json.load(fh)


def generate_accuracy_table(results_dir: str) -> str:
    """Generate Model x Mode overall accuracy table in booktabs LaTeX format.

    Args:
        results_dir: Path to published_results directory.

    Returns:
        LaTeX table string with booktabs formatting.
    """
    models = ["gpt4", "claude3", "gemini", "llama3"]
    modes = ["zeroshot", "cot"]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Zero-shot & Chain-of-Thought \\")
    lines.append(r"\midrule")

    for model in models:
        model_label = model.replace("gpt4", "GPT-4").replace("claude3", "Claude-3.5").replace("gemini", "Gemini-2.0").replace("llama3", "LLaMA-3-70B")
        row_vals = []
        for mode in modes:
            summary = _load_summary(results_dir, model, mode)
            if summary and "overall_accuracy" in summary:
                val = summary["overall_accuracy"]
                row_vals.append(f"{val:.2f}")
            else:
                row_vals.append("--")
        lines.append(f"{model_label} & {row_vals[0]} & {row_vals[1]} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Overall accuracy by model and inference mode.}")
    lines.append(r"\label{tab:accuracy}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_task_breakdown_table(results_dir: str) -> str:
    """Generate Model x TaskFamily accuracy breakdown table in booktabs LaTeX format.

    Args:
        results_dir: Path to published_results directory.

    Returns:
        LaTeX table string with booktabs formatting.
    """
    models = ["gpt4", "claude3", "gemini", "llama3"]
    modes = ["zeroshot", "cot"]

    task_families = [
        "atomic", "implication", "bias", "counterfactual", "multi_turn",
        "multi_hop", "adversarial", "compositional", "analogy", "trap",
        "structural", "fallacy", "cf", "cross_system", "cf_chain", "validity"
    ]

    all_summaries = {}
    for model in models:
        for mode in modes:
            key = f"{model}_{mode}"
            all_summaries[key] = _load_summary(results_dir, model, mode)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(all_summaries) + "}")
    lines.append(r"\toprule")

    header = "Task Family"
    for model in models:
        for mode in modes:
            model_label = model.replace("gpt4", "GPT-4").replace("claude3", "Claude").replace("gemini", "Gemini").replace("llama3", "LLaMA")
            mode_label = "ZS" if mode == "zeroshot" else "CoT"
            header += f" & {model_label}-{mode_label}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for task in task_families:
        task_label = task.replace("_", " ").title()
        row = task_label
        for model in models:
            for mode in modes:
                key = f"{model}_{mode}"
                summary = all_summaries.get(key, {})
                task_acc = summary.get("task_accuracy", {})
                if task in task_acc:
                    row += f" & {task_acc[task]:.2f}"
                else:
                    row += " & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Accuracy breakdown by task family and model configuration.}")
    lines.append(r"\label{tab:task_breakdown}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_indicator_table(indicators_dir: str) -> str:
    """Generate System x Indicator table in booktabs LaTeX format.

    Reads indicator JSON files from systems/ directory (not a subdirectory).
    Each system JSON contains indicator values.

    Args:
        indicators_dir: Path to systems directory containing system JSON files.

    Returns:
        LaTeX table string with booktabs formatting.
    """
    systems = [
        "lorenz63", "rossler", "chen_system", "chua_circuit", "double_pendulum",
        "duffing_chaotic", "fitzhugh_nagumo", "hindmarsh_rose", "lorenz84",
        "lorenz96", "lotka_volterra", "mackey_glass", "brusselator",
        "damped_oscillator", "damped_driven_pendulum_nonchaotic", "oregonator",
        "rikitake_dynamo", "shm", "vdp", "logistic_r4", "logistic_r2_8",
        "henon", "ikeda_map", "arnold_cat_map", "bakers_map",
        "circle_map_quasiperiodic", "standard_map", "kuramoto_sivashinsky",
        "stochastic_ou", "sine_gordon"
    ]

    indicator_data = {}
    for system_id in systems:
        json_path = os.path.join(indicators_dir, f"{system_id}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as fh:
                data = json.load(fh)
                indicator_data[system_id] = data

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"System & 0-1 Test (K) & Perm. Entropy & MEGNO \\")
    lines.append(r"\midrule")

    for system_id in systems:
        system_label = system_id.replace("_", " ").title()
        data = indicator_data.get(system_id, {})

        k_val = "--"
        pe_val = "--"
        megno_val = "--"

        if "zero_one_K" in data and data["zero_one_K"] is not None:
            k_val = f"{data['zero_one_K']:.2f}"
        if "permutation_entropy" in data and data["permutation_entropy"] is not None:
            pe_val = f"{data['permutation_entropy']:.2f}"
        if "megno" in data and data["megno"] is not None:
            megno_val = f"{data['megno']:.2f}"

        lines.append(f"{system_label} & {k_val} & {pe_val} & {megno_val} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Chaos indicators for benchmark systems (0-1 test K, permutation entropy, MEGNO).}")
    lines.append(r"\label{tab:indicators}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_hard_split_table(results_dir: str) -> str:
    """Generate Full vs Hard split accuracy comparison table in booktabs LaTeX format.

    Args:
        results_dir: Path to published_results directory.

    Returns:
        LaTeX table string with booktabs formatting.
    """
    models = ["gpt4", "claude3", "gemini", "llama3"]
    modes = ["zeroshot", "cot"]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Mode & Full & Hard \\")
    lines.append(r"\midrule")

    for model in models:
        model_label = model.replace("gpt4", "GPT-4").replace("claude3", "Claude-3.5").replace("gemini", "Gemini-2.0").replace("llama3", "LLaMA-3-70B")
        for mode in modes:
            mode_label = "Zero-shot" if mode == "zeroshot" else "CoT"
            summary = _load_summary(results_dir, model, mode)

            full_acc = "--"
            hard_acc = "--"

            if summary and "overall_accuracy" in summary:
                full_acc = f"{summary['overall_accuracy']:.2f}"

            if summary and "task_accuracy" in summary and "hard" in summary["task_accuracy"]:
                hard_acc = f"{summary['task_accuracy']['hard']:.2f}"

            lines.append(f"{model_label} & {mode_label} & {full_acc} & {hard_acc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Comparison of overall accuracy on full benchmark vs hard split.}")
    lines.append(r"\label{tab:hard_split}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _load_eval_results(results_dir: str, model: str, mode: str) -> List[EvalResult]:
    """Load EvalResult objects from a results.jsonl file.

    Args:
        results_dir: Path to published_results directory.
        model: Model name.
        mode: Inference mode.

    Returns:
        List of EvalResult objects, or empty list if file not found.
    """
    config_name = f"{model}_{mode}"
    results_path = os.path.join(results_dir, config_name, "results.jsonl")
    if not os.path.exists(results_path):
        return []

    results = []
    with open(results_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            results.append(EvalResult(**data))

    return results


def generate_axis_tables(results_dir: str, model: str, mode: str) -> Dict[str, str]:
    """Generate per-split and per-task-family tables using axis metrics.

    Args:
        results_dir: Path to published_results directory.
        model: Model name.
        mode: Inference mode.

    Returns:
        Dict mapping table name to LaTeX table string.
    """
    results = _load_eval_results(results_dir, model, mode)
    if not results:
        return {}

    axis_metrics = compute_axis_metrics(results, axes=["split", "task_family"])

    tables = {}

    if "split" in axis_metrics:
        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Split & Accuracy & Correct & Total \\")
        lines.append(r"\midrule")

        for metric in axis_metrics["split"]:
            split_label = metric.value.replace("_", " ").title()
            acc_val = f"{metric.accuracy:.2f}"
            lines.append(f"{split_label} & {acc_val} & {metric.n_correct} & {metric.n_total} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Accuracy by benchmark split.}")
        lines.append(r"\label{tab:split_accuracy}")
        lines.append(r"\end{table}")

        tables["split_accuracy"] = "\n".join(lines)

    if "task_family" in axis_metrics:
        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Task Family & Accuracy & Correct & Total \\")
        lines.append(r"\midrule")

        for metric in axis_metrics["task_family"]:
            task_label = metric.value.replace("_", " ").title()
            acc_val = f"{metric.accuracy:.2f}"
            lines.append(f"{task_label} & {acc_val} & {metric.n_correct} & {metric.n_total} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Accuracy by task family.}")
        lines.append(r"\label{tab:task_family_accuracy}")
        lines.append(r"\end{table}")

        tables["task_family_accuracy"] = "\n".join(lines)

    return tables


def main():
    """Generate LaTeX tables from published results and system indicators."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from ChaosBench results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="published_results",
        help="Path to published_results directory (default: published_results)",
    )
    parser.add_argument(
        "--indicators-dir",
        type=str,
        default="systems",
        help="Path to systems directory containing system JSON files (default: systems)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tables",
        help="Output directory for LaTeX table files (default: tables)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Optional: model name for axis tables (e.g., 'gpt4')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Optional: inference mode for axis tables (e.g., 'zeroshot')",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    accuracy_table = generate_accuracy_table(args.results_dir)
    with open(os.path.join(args.output_dir, "accuracy_table.tex"), "w") as fh:
        fh.write(accuracy_table)
    print(f"Wrote {os.path.join(args.output_dir, 'accuracy_table.tex')}")

    task_table = generate_task_breakdown_table(args.results_dir)
    with open(os.path.join(args.output_dir, "task_breakdown_table.tex"), "w") as fh:
        fh.write(task_table)
    print(f"Wrote {os.path.join(args.output_dir, 'task_breakdown_table.tex')}")

    indicator_table = generate_indicator_table(args.indicators_dir)
    with open(os.path.join(args.output_dir, "indicator_table.tex"), "w") as fh:
        fh.write(indicator_table)
    print(f"Wrote {os.path.join(args.output_dir, 'indicator_table.tex')}")

    hard_split_table = generate_hard_split_table(args.results_dir)
    with open(os.path.join(args.output_dir, "hard_split_table.tex"), "w") as fh:
        fh.write(hard_split_table)
    print(f"Wrote {os.path.join(args.output_dir, 'hard_split_table.tex')}")

    if args.model and args.mode:
        axis_tables = generate_axis_tables(args.results_dir, args.model, args.mode)
        for table_name, table_content in axis_tables.items():
            output_path = os.path.join(args.output_dir, f"{table_name}_{args.model}_{args.mode}.tex")
            with open(output_path, "w") as fh:
                fh.write(table_content)
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
