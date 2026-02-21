#!/usr/bin/env python3
"""scripts/generate_results_figures.py — Publication-ready figures for ChaosBench-Logic v2.

Reads from artifacts/results_pack/tables/ and runs/ predictions directly.
Writes to artifacts/results_pack/figures/ as both .pdf and .png.

Required figures:
  1. mcc_macro_family_bar.{pdf,png}   — Bar chart: MCC (micro + macro_family) by model
  2. family_heatmap.{pdf,png}         — Heatmap: model × family MCC, sorted by hardness
  3. bias_plot.{pdf,png}              — Bias: pred_TRUE% vs gt_TRUE% with TPR/TNR annotated
  4. latency_plot.{pdf,png}           — Latency mean/p95 by model
  5. family_acc_grouped_bar.{pdf,png} — Grouped bar per family, models compared

Quality constraints enforced:
  - constrained_layout / tight_layout to prevent overlap
  - legends outside plot when >3 entries
  - minimum font sizes: title=14, axis=12, ticks=10, legend=9
  - axis labels always present
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PACK = PROJECT_ROOT / "artifacts" / "results_pack"
TABLES_DIR = RESULTS_PACK / "tables"
FIGURES_DIR = RESULTS_PACK / "figures"
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Consistent style
# ---------------------------------------------------------------------------
FONT_TITLE = 14
FONT_AXIS = 12
FONT_TICK = 10
FONT_LEGEND = 9
FONT_ANNOT = 8
DPI_PNG = 200
FIGSIZE_BAR = (9, 5)
FIGSIZE_HEAT = (12, 6)
FIGSIZE_BIAS = (7, 6)
FIGSIZE_LAT = (8, 5)
FIGSIZE_GROUPED = (13, 5)

# Color palette — consistent per model across all figures
MODEL_COLORS = {
    "Qwen2.5-14B": "#2196F3",   # blue
    "Qwen2.5-7B":  "#64B5F6",   # light blue
    "Llama3.1-8B": "#FF7043",   # orange-red
    "Gemma2-9B":   "#66BB6A",   # green
    "Mistral-7B":  "#AB47BC",   # purple
}

FAMILY_DISPLAY = {
    "atomic":                  "Atomic",
    "multi_hop":               "Multi-Hop",
    "fol_inference":           "FOL Inference",
    "consistency_paraphrase":  "Consistency\nParaphrase",
    "perturbation":            "Perturbation",
    "adversarial_misleading":  "Adv.\nMisleading",
    "adversarial_nearmiss":    "Adv.\nNear-Miss",
    "indicator_diagnostic":    "Indicator\nDiagnostic",
    "cross_indicator":         "Cross\nIndicator",
    "extended_systems":        "Extended\nSystems",
    "regime_transition":       "Regime\nTransition",
}

# Family order: hardest first (lowest mean MCC from data)
FAMILY_HARDNESS_ORDER = [
    "regime_transition",
    "consistency_paraphrase",
    "perturbation",
    "cross_indicator",
    "adversarial_misleading",
    "adversarial_nearmiss",
    "atomic",
    "multi_hop",
    "extended_systems",
    "fol_inference",
    "indicator_diagnostic",
]


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as both PDF and PNG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=DPI_PNG)
    fig.savefig(png_path, bbox_inches="tight", dpi=DPI_PNG)
    print(f"  Saved: {pdf_path.name}, {png_path.name}")
    plt.close(fig)


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, "#999999")


# ---------------------------------------------------------------------------
# Figure 1: MCC bar chart (micro + macro_family)
# ---------------------------------------------------------------------------
def fig_mcc_bar(baselines_df: pd.DataFrame) -> None:
    df = baselines_df.copy()
    # Filter out mock/debug
    df = df[df["model"].isin(MODEL_COLORS)].copy()
    df = df.sort_values("mcc_micro", ascending=False)

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR, constrained_layout=True)
    colors = [_color(m) for m in models]

    bars1 = ax.bar(x - width / 2, df["mcc_micro"], width, label="MCC (micro)",
                   color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    # macro_family may be None for some runs
    macro_vals = df["mcc_macro_family"].fillna(0).tolist()
    bars2 = ax.bar(x + width / 2, macro_vals, width, label="MCC (macro-family)",
                   color=colors, alpha=0.55, edgecolor="white", linewidth=0.5, hatch="//")

    # Annotate values
    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    xticklabels = []
    for m, row in zip(models, df.itertuples()):
        n_k = f"{row.N:,}"
        xticklabels.append(f"{m}\n(N={n_k})")
    ax.set_xticklabels(xticklabels, fontsize=FONT_TICK)
    ax.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Model MCC (micro and macro-family)", fontsize=FONT_TITLE)
    ax.set_ylim(bottom=-0.05)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    _save(fig, "mcc_bar")


# ---------------------------------------------------------------------------
# Figure 2: Heatmap — model × family MCC, sorted by hardness
# ---------------------------------------------------------------------------
def fig_family_heatmap(by_family_df: pd.DataFrame) -> None:
    df = by_family_df.copy()
    df = df[df["model"].isin(MODEL_COLORS)]

    # Pivot: rows = family (hardness order), cols = model
    piv = df.pivot_table(index="family", columns="model", values="MCC", aggfunc="mean")

    # Sort families by hardness order, keep only known ones
    families = [f for f in FAMILY_HARDNESS_ORDER if f in piv.index]
    extra = [f for f in piv.index if f not in families]
    families = families + extra
    piv = piv.reindex(families)

    # Sort models by overall MCC descending
    model_order = sorted(piv.columns, key=lambda c: piv[c].mean(), reverse=True)
    piv = piv[model_order]

    fig, ax = plt.subplots(figsize=FIGSIZE_HEAT, constrained_layout=True)
    cmap = plt.cm.RdYlGn
    vals = piv.values.astype(float)
    vmin, vmax = -0.1, 0.7

    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Axis labels
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=FONT_TICK, rotation=15, ha="right")
    ax.set_yticks(range(len(families)))
    fam_labels = [FAMILY_DISPLAY.get(f, f) for f in families]
    ax.set_yticklabels(fam_labels, fontsize=FONT_TICK)

    # Annotate cells
    for i in range(len(families)):
        for j in range(len(piv.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                text_color = "black" if 0.1 < v < 0.6 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FONT_ANNOT, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("MCC", fontsize=FONT_AXIS)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    ax.set_xlabel("Model", fontsize=FONT_AXIS)
    ax.set_ylabel("Task Family (hardest → easiest, top → bottom)", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Per-Family MCC Heatmap", fontsize=FONT_TITLE)

    _save(fig, "family_heatmap")


# ---------------------------------------------------------------------------
# Figure 3: Bias plot — pred_TRUE% vs gt_TRUE% with TPR/TNR
# ---------------------------------------------------------------------------
def fig_bias_plot(baselines_df: pd.DataFrame) -> None:
    df = baselines_df.copy()
    df = df[df["model"].isin(MODEL_COLORS) & df["gt_TRUE_pct"].notna()].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE_BIAS, constrained_layout=True)

    # Diagonal reference: pred = gt
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Ideal (pred = gt)")
    ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.4, linestyle=":")
    ax.axvline(0.5, color="gray", linewidth=0.5, alpha=0.4, linestyle=":")

    jitter_x = np.linspace(-0.01, 0.01, len(df))
    for idx, (_, row) in enumerate(df.iterrows()):
        color = _color(row["model"])
        x = float(row["gt_TRUE_pct"]) + jitter_x[idx]
        y = float(row["pred_TRUE_pct"])
        ax.scatter(x, y, color=color, s=120, zorder=5, edgecolors="white", linewidths=1)
        # Annotation: model name + TPR/TNR
        tpr = row.get("TPR")
        tnr = row.get("TNR")
        label = f"{row['model']}\nTPR={tpr:.2f}, TNR={tnr:.2f}" if pd.notna(tpr) and pd.notna(tnr) else row["model"]
        offset_x = 0.02 if x < 0.5 else -0.02
        offset_y = 0.025 * (1 if idx % 2 == 0 else -1)
        ax.annotate(
            label,
            (x, y),
            xytext=(x + offset_x, y + offset_y + 0.04),
            fontsize=7,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
        )

    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.05, 0.85)
    ax.set_xlabel("Ground-truth TRUE rate (gt_TRUE%)", fontsize=FONT_AXIS)
    ax.set_ylabel("Predicted TRUE rate (pred_TRUE%)", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Label Bias Analysis", fontsize=FONT_TITLE)

    # Shade danger zone: |pred - gt| > 0.15
    ax.fill_between([0, 1], [0.15, 1.15], [0, 1], alpha=0.05, color="red", label="|bias| > 0.15")
    ax.fill_between([0, 1], [-0.15, 0.85], [0, 1], alpha=0.05, color="red")

    ax.legend(fontsize=FONT_LEGEND, loc="lower right")
    ax.tick_params(labelsize=FONT_TICK)

    _save(fig, "bias_plot")


# ---------------------------------------------------------------------------
# Figure 4: Latency plot — mean/p95 by model
# ---------------------------------------------------------------------------
def fig_latency_plot(baselines_df: pd.DataFrame) -> None:
    df = baselines_df.copy()
    df = df[df["model"].isin(MODEL_COLORS) & df["latency_mean_s"].notna()].copy()

    if df.empty:
        print("  [latency_plot] No latency data — skipping.")
        return

    df = df.sort_values("latency_mean_s")
    models = df["model"].tolist()
    means = df["latency_mean_s"].tolist()
    p95s = df["latency_p95_s"].tolist()
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=FIGSIZE_LAT, constrained_layout=True)
    colors = [_color(m) for m in models]

    bars = ax.bar(x, means, 0.5, label="Mean latency", color=colors, alpha=0.85,
                  edgecolor="white")
    ax.scatter(x, p95s, marker="D", s=60, color=colors, zorder=5, label="p95 latency",
               edgecolors="black", linewidths=0.5)

    for i, (m, p) in enumerate(zip(means, p95s)):
        ax.text(i, m + 0.005, f"{m:.2f}s", ha="center", va="bottom", fontsize=FONT_ANNOT)
        ax.text(i, p + 0.005, f"p95={p:.2f}s", ha="center", va="bottom",
                fontsize=FONT_ANNOT - 1, style="italic", color="gray")

    ax.set_xticks(x)
    xticklabels = []
    for m, row in zip(models, df.itertuples()):
        n_k = f"{row.N:,}"
        xticklabels.append(f"{m}\n(N={n_k})")
    ax.set_xticklabels(xticklabels, fontsize=FONT_TICK)
    ax.set_ylabel("Latency per question (seconds)", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Inference Latency (Ollama, local GPU)", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    _save(fig, "latency_plot")


# ---------------------------------------------------------------------------
# Figure 5: Per-family grouped bar (5k runs)
# ---------------------------------------------------------------------------
def fig_family_grouped_bar(by_family_df: pd.DataFrame) -> None:
    df = by_family_df.copy()
    df = df[df["model"].isin(MODEL_COLORS)]
    # 5k subset only for comparable N
    df = df[df["subset"] == "5k_subset"] if "5k_subset" in df["subset"].values else df

    models = [m for m in MODEL_COLORS if m in df["model"].unique()]
    families = [f for f in FAMILY_HARDNESS_ORDER if f in df["family"].unique()]

    if not families or not models:
        print("  [family_grouped_bar] No data — skipping.")
        return

    n_groups = len(families)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=FIGSIZE_GROUPED, constrained_layout=True)

    for i, model in enumerate(models):
        mccs = []
        for fam in families:
            row = df[(df["model"] == model) & (df["family"] == fam)]
            mccs.append(float(row["MCC"].iloc[0]) if not row.empty else 0.0)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, mccs, width, label=model,
                      color=_color(model), alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    fam_labels = [FAMILY_DISPLAY.get(f, f) for f in families]
    ax.set_xticklabels(fam_labels, fontsize=FONT_TICK - 1, rotation=0)
    ax.set_ylabel("MCC", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Per-Family MCC by Model (5k-subset runs)", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.15, 0.85)

    _save(fig, "family_grouped_bar")


# ---------------------------------------------------------------------------
# Model alias mapping table (saved as FIGURE_INDEX.md)
# ---------------------------------------------------------------------------
def write_figure_index(baselines_df: pd.DataFrame) -> None:
    aliases = sorted(MODEL_COLORS.items())
    lines = [
        "# Figure Index — ChaosBench-Logic v2 Results Pack",
        "",
        "## Model Alias Mapping",
        "",
        "| Alias (used in figures) | Full provider string |",
        "|-------------------------|----------------------|",
    ]
    provider_map = {
        "Qwen2.5-14B": "ollama/qwen2.5:14b",
        "Qwen2.5-7B":  "ollama/qwen2.5:7b",
        "Llama3.1-8B": "ollama/llama3.1:8b",
        "Gemma2-9B":   "ollama/gemma2:9b",
        "Mistral-7B":  "ollama/mistral:7b",
    }
    for alias, _ in aliases:
        lines.append(f"| {alias} | {provider_map.get(alias, '—')} |")

    lines += [
        "",
        "## Figures",
        "",
        "| File | Caption |",
        "|------|---------|",
        "| `mcc_bar.{pdf,png}` | Bar chart of MCC (micro + macro-family) per model. Primary headline figure. |",
        "| `family_heatmap.{pdf,png}` | Heatmap of per-family MCC by model, sorted hardest→easiest. Shows which families are universally difficult. |",
        "| `bias_plot.{pdf,png}` | Scatter of predicted TRUE% vs ground-truth TRUE% per model. Shows label-defaulting behavior (FALSE-lean vs TRUE-lean). TPR/TNR annotated per point. |",
        "| `latency_plot.{pdf,png}` | Bar chart of mean inference latency (s/question) with p95 markers. All runs on local Ollama. |",
        "| `family_grouped_bar.{pdf,png}` | Grouped bar chart comparing MCC per task family across models (5k-subset runs only). |",
        "",
        "## Notes",
        "",
        "- All figures use `constrained_layout` to prevent text overlap.",
        "- Legends with >3 entries are placed outside the plot axes.",
        "- PDFs are vector (suitable for paper submission); PNGs are raster at 200 DPI.",
        "- 5k-subset runs: N=3,828 (sampler drew 3,828 items from 40,886 canonical).",
        "- Full-canonical run (Llama3.1-8B): N=40,886.",
        "- 1k-subset runs excluded from per-family figures (too few items per family).",
    ]
    (FIGURES_DIR / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n")
    print("  Saved: FIGURE_INDEX.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[generate_results_figures] Loading tables...")
    baselines_path = TABLES_DIR / "baselines_table.csv"
    by_family_path = TABLES_DIR / "by_family.csv"

    if not baselines_path.exists():
        print("ERROR: baselines_table.csv not found. Run build_results_pack.py first.")
        sys.exit(1)

    baselines_df = pd.read_csv(baselines_path)
    by_family_df = pd.read_csv(by_family_path) if by_family_path.exists() else pd.DataFrame()

    # Load hardness order from data to dynamically update
    if not by_family_df.empty:
        hardness = (
            by_family_df[by_family_df["model"].isin(MODEL_COLORS)]
            .groupby("family")["MCC"].mean()
            .sort_values()
        )
        # Update global with data-driven order
        global FAMILY_HARDNESS_ORDER
        FAMILY_HARDNESS_ORDER = list(hardness.index)

    print("[generate_results_figures] Generating figures...")
    print("  Figure 1: MCC bar chart")
    fig_mcc_bar(baselines_df)

    if not by_family_df.empty:
        print("  Figure 2: Family heatmap")
        fig_family_heatmap(by_family_df)

        print("  Figure 3: Family grouped bar")
        fig_family_grouped_bar(by_family_df)

    print("  Figure 4: Bias plot")
    fig_bias_plot(baselines_df)

    print("  Figure 5: Latency plot")
    fig_latency_plot(baselines_df)

    print("  Writing FIGURE_INDEX.md")
    write_figure_index(baselines_df)

    print(f"[generate_results_figures] Done. Output: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
