#!/usr/bin/env python3
"""scripts/generate_results_figures.py — Publication-ready figures for ChaosBench-Logic v2.

Usage:
  python scripts/generate_results_figures.py \
      --tables_dir artifacts/results_pack_v2/<ts>/tables \
      --out_dir    artifacts/results_pack_v2/<ts>/figures

Figures produced (each as .pdf and .png):
  1. mcc_overview_bar       — grouped bar: MCC_micro by model+subset, color by model
  2. family_heatmap         — heatmap: model×family MCC (full_canonical runs), sorted hardest→easiest
  3. bias_plot              — scatter pred_TRUE% vs gt_TRUE%, annotated TPR/TNR
  4. latency_plot           — mean/p95 latency by model (5k_armored and full_canonical)
  5. family_grouped_bar     — per-family grouped bar, models on same subset compared
  6. subset_crosscheck_bar  — 5k_armored vs full_canonical MCC comparison (models with both runs)

Quality constraints: constrained_layout, min font sizes, legends outside when >3 entries.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

FONT_TITLE  = 13
FONT_AXIS   = 11
FONT_TICK   = 9
FONT_LEGEND = 8
FONT_ANNOT  = 7
DPI_PNG     = 200

MODEL_COLORS = {
    "Qwen2.5-32B": "#1565C0",   # dark blue
    "Qwen2.5-14B": "#42A5F5",   # light blue
    "Qwen2.5-7B":  "#90CAF9",   # very light blue
    "Llama3.1-8B": "#FF7043",   # orange-red
    "Gemma2-9B":   "#66BB6A",   # green
    "Mistral-7B":  "#AB47BC",   # purple
}
MODEL_ORDER = ["Qwen2.5-32B","Qwen2.5-14B","Gemma2-9B","Mistral-7B","Qwen2.5-7B","Llama3.1-8B"]

SUBSET_MARKERS = {"full_canonical": "●", "5k_armored": "▲", "1k_subset": "◆"}

FAMILY_DISPLAY = {
    "atomic":                  "Atomic",
    "multi_hop":               "Multi-Hop",
    "fol_inference":           "FOL\nInference",
    "consistency_paraphrase":  "Consistency\nParaphrase",
    "perturbation":            "Perturbation",
    "adversarial_misleading":  "Adv.\nMisleading",
    "adversarial_nearmiss":    "Adv.\nNear-Miss",
    "indicator_diagnostic":    "Indicator\nDiagnostic",
    "cross_indicator":         "Cross\nIndicator",
    "extended_systems":        "Extended\nSystems",
    "regime_transition":       "Regime\nTransition",
}


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, "#9E9E9E")

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"{stem}{ext}", bbox_inches="tight", dpi=DPI_PNG)
    print(f"  Saved: {stem}.pdf + {stem}.png")
    plt.close(fig)


# ── Figure 1: MCC overview bar (grouped by subset) ───────────────────────────
def fig_mcc_overview(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar: model×subset, y=MCC_micro. Three groups: 1k, 5k, full."""
    subsets = ["1k_subset", "5k_armored", "full_canonical"]
    subset_labels = {"1k_subset": "1k subset", "5k_armored": "5k armored", "full_canonical": "Full canonical (40k)"}
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    group_w = 0.9
    n_subsets = len(subsets)
    bar_w = group_w / n_subsets
    x_base = np.arange(len(MODEL_ORDER))

    for si, subset in enumerate(subsets):
        sub = df[df["subset"] == subset]
        mccs, models_present = [], []
        for m in MODEL_ORDER:
            row = sub[sub["model"] == m]
            mccs.append(float(row["mcc_micro"].iloc[0]) if not row.empty else np.nan)
            models_present.append(m)

        xs = x_base + (si - n_subsets/2 + 0.5) * bar_w
        colors = [_color(m) if not np.isnan(v) else "#EEEEEE" for m, v in zip(MODEL_ORDER, mccs)]
        alphas = [0.95 if subset == "full_canonical" else (0.75 if subset == "5k_armored" else 0.5)]

        bars = ax.bar(xs, mccs, bar_w * 0.92,
                      label=subset_labels[subset],
                      color=colors,
                      alpha=0.95 if subset == "full_canonical" else (0.80 if subset == "5k_armored" else 0.55),
                      edgecolor="white", linewidth=0.5,
                      hatch=("" if subset == "full_canonical" else ("" if subset == "5k_armored" else "//")))
        for bar, v in zip(bars, mccs):
            if not np.isnan(v) and v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.007, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6, color="#333")

    # Cosmetics
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.3)
    ax.set_xticks(x_base)
    ax.set_xticklabels(MODEL_ORDER, fontsize=FONT_TICK)
    ax.set_ylabel("MCC (micro)", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Model MCC by Evaluation Subset", fontsize=FONT_TITLE)
    ax.set_ylim(-0.05, 0.60)
    ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)

    # Legend for subsets
    legend_patches = [
        mpatches.Patch(facecolor="#CCCCCC", alpha=0.55, hatch="//", label="1k subset"),
        mpatches.Patch(facecolor="#CCCCCC", alpha=0.80, label="5k armored"),
        mpatches.Patch(facecolor="#CCCCCC", alpha=0.95, label="Full canonical (40k)"),
    ]
    ax.legend(handles=legend_patches, fontsize=FONT_LEGEND, loc="upper right")

    # Color legend for models
    model_patches = [mpatches.Patch(facecolor=_color(m), label=m) for m in MODEL_ORDER if m in MODEL_COLORS]
    leg2 = ax.legend(handles=model_patches, fontsize=FONT_LEGEND,
                     loc="upper left", bbox_to_anchor=(0, 1), ncol=3,
                     title="Model", title_fontsize=FONT_LEGEND)
    ax.add_artist(leg2)
    # Re-add subset legend
    ax.legend(handles=legend_patches, fontsize=FONT_LEGEND, loc="upper right")

    _save(fig, out_dir, "mcc_overview_bar")


# ── Figure 2: Family heatmap (full_canonical only) ───────────────────────────
def fig_family_heatmap(by_family_df: pd.DataFrame, hardness_df: pd.DataFrame, out_dir: Path) -> None:
    df = by_family_df[by_family_df["subset"] == "full_canonical"].copy()
    if df.empty:
        df = by_family_df.copy()

    piv = df.pivot_table(index="family", columns="model", values="MCC", aggfunc="mean")

    # Sort families by hardness (from hardness_df)
    hardness_order = list(hardness_df["family"]) if not hardness_df.empty else list(piv.index)
    families = [f for f in hardness_order if f in piv.index] + [f for f in piv.index if f not in hardness_order]
    piv = piv.reindex(families)

    # Sort models by mean MCC desc
    model_ord = sorted([m for m in piv.columns if m in MODEL_ORDER],
                        key=lambda m: MODEL_ORDER.index(m))
    piv = piv.reindex(columns=[m for m in model_ord if m in piv.columns])

    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns)*1.8), max(6, len(families)*0.55)),
                           constrained_layout=True)
    vals = piv.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.8)

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=FONT_TICK, rotation=15, ha="right")
    ax.set_yticks(range(len(families)))
    fam_labels = []
    for f in families:
        lbl = FAMILY_DISPLAY.get(f, f).replace("\n", " ")
        # Add ⚠️ for small-N in full canonical
        n_row = by_family_df[(by_family_df["family"]==f) & (by_family_df["subset"]=="full_canonical")]["N_family"]
        small = any(n_row < 100)
        fam_labels.append(f"{'⚠️ ' if small else ''}{lbl}")
    ax.set_yticklabels(fam_labels, fontsize=FONT_TICK)
    ax.set_xlabel("Model", fontsize=FONT_AXIS)
    ax.set_ylabel("Task Family (hardest → easiest)", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Per-Family MCC (full canonical, N=40,886)", fontsize=FONT_TITLE)

    for i in range(len(families)):
        for j in range(len(piv.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                tc = "black" if 0.0 < v < 0.65 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FONT_ANNOT+1, color=tc, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("MCC", fontsize=FONT_AXIS)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    _save(fig, out_dir, "family_heatmap")


# ── Figure 3: Bias plot ───────────────────────────────────────────────────────
def fig_bias_plot(df: pd.DataFrame, out_dir: Path) -> None:
    # Use full_canonical; fall back to 5k_armored if not available
    prio = ["full_canonical", "5k_armored", "1k_subset"]
    rows = []
    for model in df["model"].unique():
        for subset in prio:
            sub = df[(df["model"]==model) & (df["subset"]==subset)]
            if not sub.empty and pd.notna(sub["gt_TRUE_pct"].iloc[0]):
                rows.append(sub.iloc[0].to_dict())
                break
    if not rows: return
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    ax.plot([0.3,0.75],[0.3,0.75], "k--", alpha=0.25, lw=1, label="Ideal (pred=gt)")
    ax.axhline(0.5, color="gray", lw=0.5, alpha=0.4, ls=":")
    ax.axvline(0.5, color="gray", lw=0.5, alpha=0.4, ls=":")

    used_ys = []
    for _, row in plot_df.iterrows():
        color = _color(row["model"])
        x = float(row["gt_TRUE_pct"])
        y = float(row["pred_TRUE_pct"])
        subset = row["subset"]
        marker = {"full_canonical":"o","5k_armored":"^","1k_subset":"D"}.get(subset,"o")
        ax.scatter(x, y, color=color, s=110, zorder=5,
                   edgecolors="white", linewidths=1, marker=marker)

        tpr = row.get("TPR"); tnr = row.get("TNR")
        lbl = f"{row['model']}"
        if pd.notna(tpr) and pd.notna(tnr):
            lbl += f"\nTPR={tpr:.2f} TNR={tnr:.2f}"
        # Avoid overlap
        offset_y = 0.04
        while any(abs(y + offset_y - uy) < 0.03 for uy in used_ys):
            offset_y += 0.03
        used_ys.append(y + offset_y)
        ax.annotate(lbl, (x, y),
                    xytext=(x + 0.02, y + offset_y),
                    fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

    ax.fill_between([0,1],[0.15,1.15],[0,1], alpha=0.04, color="red")
    ax.fill_between([0,1],[-0.15,0.85],[0,1], alpha=0.04, color="red")
    ax.set_xlim(0.35, 0.70); ax.set_ylim(0.10, 0.85)
    ax.set_xlabel("Ground-truth TRUE rate", fontsize=FONT_AXIS)
    ax.set_ylabel("Predicted TRUE rate", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Label Bias (pred TRUE% vs gt TRUE%)", fontsize=FONT_TITLE)

    # Legend for subsets
    handles = [mpatches.Patch(facecolor="#999", alpha=0.7, label="Full canonical"),
               plt.scatter([], [], marker="^", color="#999", s=80, label="5k armored"),
               plt.scatter([], [], marker="D", color="#999", s=80, label="1k subset")]
    ax.legend(handles=handles, fontsize=FONT_LEGEND, loc="lower right")
    ax.tick_params(labelsize=FONT_TICK)
    _save(fig, out_dir, "bias_plot")


# ── Figure 4: Latency ─────────────────────────────────────────────────────────
def fig_latency(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["latency_mean_s"].notna() & df["model"].isin(MODEL_COLORS)].copy()
    if sub.empty: return

    # Prefer full_canonical, else 5k_armored
    best = {}
    for _, row in sub.iterrows():
        m = row["model"]; s = row["subset"]
        if m not in best or (s == "full_canonical" and best[m]["subset"] != "full_canonical"):
            best[m] = row.to_dict()
    plot_df = pd.DataFrame(list(best.values()))
    plot_df = plot_df.sort_values("latency_mean_s")

    models = plot_df["model"].tolist()
    means  = plot_df["latency_mean_s"].tolist()
    p95s   = plot_df["latency_p95_s"].tolist()
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = [_color(m) for m in models]
    ax.bar(x, means, 0.55, color=colors, alpha=0.85, edgecolor="white")
    ax.scatter(x, p95s, marker="D", s=65, color=colors, zorder=5,
               edgecolors="black", linewidths=0.5)

    for i, (m, p) in enumerate(zip(means, p95s)):
        ax.text(i, m + 0.005, f"{m:.2f}s", ha="center", va="bottom", fontsize=FONT_ANNOT+1)
        if p: ax.text(i, p + 0.005, f"p95={p:.2f}s", ha="center", va="bottom",
                      fontsize=FONT_ANNOT, style="italic", color="#555")

    ax.set_xticks(x)
    xlbls = []
    for m, row in zip(models, [best[m] for m in models]):
        xlbls.append(f"{m}\n({row['subset']})")
    ax.set_xticklabels(xlbls, fontsize=FONT_TICK)
    ax.set_ylabel("Latency per question (seconds)", fontsize=FONT_AXIS)
    ax.set_title("ChaosBench-Logic v2 — Inference Latency (Ollama, local)", fontsize=FONT_TITLE)

    bar_patch = mpatches.Patch(facecolor="#AAAAAA", label="Mean latency")
    dot_patch = plt.scatter([], [], marker="D", color="#AAAAAA", s=50, label="p95 latency")
    ax.legend(handles=[bar_patch, dot_patch], fontsize=FONT_LEGEND)
    ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
    _save(fig, out_dir, "latency_plot")


# ── Figure 5: Per-family grouped bar ─────────────────────────────────────────
def fig_family_bar(by_family_df: pd.DataFrame, out_dir: Path) -> None:
    """One panel per subset tier (5k_armored, full_canonical)."""
    for subset, title_sfx in [("full_canonical", "Full canonical (N=40,886)"),
                               ("5k_armored", "5k armored (N=5,000)")]:
        df = by_family_df[by_family_df["subset"] == subset].copy()
        if df.empty: continue

        models_in = [m for m in MODEL_ORDER if m in df["model"].unique()]
        families = [f for f in [
            "regime_transition","cross_indicator","consistency_paraphrase","perturbation",
            "atomic","adversarial_nearmiss","adversarial_misleading","multi_hop",
            "fol_inference","indicator_diagnostic","extended_systems"
        ] if f in df["family"].unique()]

        n_m = len(models_in); n_f = len(families)
        width = 0.75 / n_m
        x = np.arange(n_f)

        fig, ax = plt.subplots(figsize=(max(12, n_f*1.1), 5), constrained_layout=True)
        for i, model in enumerate(models_in):
            mccs = []
            for fam in families:
                row = df[(df["model"]==model) & (df["family"]==fam)]
                mccs.append(float(row["MCC"].iloc[0]) if not row.empty else np.nan)
            offset = (i - n_m/2 + 0.5) * width
            ax.bar(x + offset, mccs, width*0.9, label=model,
                   color=_color(model), alpha=0.88, edgecolor="white")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_DISPLAY.get(f,f).replace("\n"," ") for f in families],
                           fontsize=FONT_TICK-1, rotation=15, ha="right")
        ax.set_ylabel("MCC", fontsize=FONT_AXIS)
        ax.set_title(f"ChaosBench-Logic v2 — Per-Family MCC ({title_sfx})", fontsize=FONT_TITLE)
        ax.legend(fontsize=FONT_LEGEND, loc="upper left",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0, title="Model")
        ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
        ax.set_ylim(-0.35, 0.85)

        stem = f"family_bar_{'full' if 'full' in subset else '5k'}"
        _save(fig, out_dir, stem)


# ── Figure 6: 5k vs full cross-check ─────────────────────────────────────────
def fig_crosscheck(crosscheck_df: pd.DataFrame, out_dir: Path) -> None:
    if crosscheck_df.empty: return
    cols = [c for c in ("5k_armored", "full_canonical") if c in crosscheck_df.columns]
    if len(cols) < 2: return

    models = crosscheck_df["model"].unique()
    n_m = len(models)

    fig, axes = plt.subplots(1, n_m, figsize=(4*n_m+1, 5), constrained_layout=True,
                             sharey=True)
    if n_m == 1: axes = [axes]

    all_families = crosscheck_df["family"].unique()
    family_labels = [FAMILY_DISPLAY.get(f,f).replace("\n"," ") for f in all_families]

    for ax, model in zip(axes, models):
        sub = crosscheck_df[crosscheck_df["model"]==model]
        families = sub["family"].tolist()
        v5k   = sub["5k_armored"].fillna(0).tolist() if "5k_armored" in sub else [0]*len(families)
        vfull = sub["full_canonical"].fillna(0).tolist() if "full_canonical" in sub else [0]*len(families)

        x = np.arange(len(families))
        ax.barh(x - 0.2, v5k,   0.35, label="5k armored",       color=_color(model), alpha=0.55)
        ax.barh(x + 0.2, vfull, 0.35, label="Full canonical",    color=_color(model), alpha=0.90)
        ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.3)
        ax.set_yticks(x)
        ax.set_yticklabels([FAMILY_DISPLAY.get(f,f).replace("\n"," ") for f in families],
                           fontsize=FONT_TICK-1)
        ax.set_title(model, fontsize=FONT_AXIS, color=_color(model))
        ax.set_xlabel("MCC", fontsize=FONT_AXIS-1)
        ax.xaxis.grid(True, alpha=0.25)
        if ax == axes[0]:
            ax.legend(fontsize=FONT_LEGEND)

    fig.suptitle("ChaosBench-Logic v2 — 5k Armored vs Full Canonical MCC per Family",
                 fontsize=FONT_TITLE)
    _save(fig, out_dir, "subset_crosscheck")


# ── Figure index ─────────────────────────────────────────────────────────────
def write_figure_index(out_dir: Path) -> None:
    lines = [
        "# Figure Index — ChaosBench-Logic v2 Results Pack (v2)",
        "",
        "## Model Alias → Provider Mapping",
        "",
        "| Alias | Full provider | Color |",
        "|-------|--------------|-------|",
        "| Qwen2.5-32B | ollama/qwen2.5:32b | dark blue |",
        "| Qwen2.5-14B | ollama/qwen2.5:14b | light blue |",
        "| Qwen2.5-7B  | ollama/qwen2.5:7b  | very light blue |",
        "| Llama3.1-8B | ollama/llama3.1:8b | orange-red |",
        "| Gemma2-9B   | ollama/gemma2:9b   | green |",
        "| Mistral-7B  | ollama/mistral:7b  | purple |",
        "",
        "## Figures",
        "",
        "| File | Caption |",
        "|------|---------|",
        "| `mcc_overview_bar.{pdf,png}` | Primary headline figure. MCC_micro grouped by model and evaluation subset (1k / 5k / full). Bar shade = subset tier; bar color = model. |",
        "| `family_heatmap.{pdf,png}` | Heatmap: model × task family MCC, full-canonical runs only (N=40,886). Sorted hardest→easiest. ⚠️ marks families with N<100. |",
        "| `bias_plot.{pdf,png}` | Scatter of pred_TRUE% vs gt_TRUE%. Each point = one run (best available subset per model). Annotated with TPR and TNR. Shaded zone = |bias| > 0.15. |",
        "| `latency_plot.{pdf,png}` | Bar chart of mean inference latency (s/q) with p95 markers, Ollama local. |",
        "| `family_bar_full.{pdf,png}` | Grouped bar: per-family MCC for all models on full_canonical (N=40,886). |",
        "| `family_bar_5k.{pdf,png}` | Grouped bar: per-family MCC for all models on 5k_armored (N=5,000). |",
        "| `subset_crosscheck.{pdf,png}` | Side-by-side comparison: 5k_armored vs full_canonical MCC per family for models with both runs (Qwen2.5-14B, Qwen2.5-32B, Mistral-7B). |",
        "",
        "## Subset Notes",
        "",
        "- **5k armored** (N=5,000): proportionally stratified from 40,886 canonical (~62% atomic).",
        "  Models: Gemma2-9B, Mistral-7B, Qwen2.5-14B, Qwen2.5-32B.",
        "- **Full canonical** (N=40,886): complete frozen dataset.",
        "  Models: Llama3.1-8B, Mistral-7B, Qwen2.5-14B, Qwen2.5-32B.",
        "- **1k subset** (N=1,000): balanced 1k sample (56 items per major family).",
        "  Models: Qwen2.5-14B, Qwen2.5-7B, Llama3.1-8B (early runs, kept for historical comparison).",
        "- Cross-validate: Mistral, Qwen14B, Qwen32B appear in both 5k and full. MCC delta < 0.01.",
    ]
    (out_dir / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n")
    print("  Saved: FIGURE_INDEX.md")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", required=True)
    parser.add_argument("--out_dir",    required=True)
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    out_dir    = Path(args.out_dir)

    baselines_path   = tables_dir / "baselines_table.csv"
    by_family_path   = tables_dir / "by_family.csv"
    hardness_path    = tables_dir / "hardness.csv"
    crosscheck_path  = tables_dir / "full_vs_5k_crosscheck.csv"

    if not baselines_path.exists():
        print("ERROR: baselines_table.csv not found. Run build_results_pack.py first.")
        sys.exit(1)

    baselines_df  = pd.read_csv(baselines_path)
    by_family_df  = pd.read_csv(by_family_path)  if by_family_path.exists()  else pd.DataFrame()
    hardness_df   = pd.read_csv(hardness_path)   if hardness_path.exists()   else pd.DataFrame()
    crosscheck_df = pd.read_csv(crosscheck_path) if crosscheck_path.exists() else pd.DataFrame()

    print("[generate_results_figures] Generating figures...")
    print("  Figure 1: MCC overview bar")
    fig_mcc_overview(baselines_df, out_dir)

    if not by_family_df.empty:
        print("  Figure 2: Family heatmap")
        fig_family_heatmap(by_family_df, hardness_df, out_dir)

        print("  Figures 3a/3b: Per-family grouped bars")
        fig_family_bar(by_family_df, out_dir)

    print("  Figure 4: Bias plot")
    fig_bias_plot(baselines_df, out_dir)

    print("  Figure 5: Latency plot")
    fig_latency(baselines_df, out_dir)

    if not crosscheck_df.empty:
        print("  Figure 6: Subset crosscheck")
        fig_crosscheck(crosscheck_df, out_dir)

    print("  Writing FIGURE_INDEX.md")
    write_figure_index(out_dir)

    print(f"[generate_results_figures] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
