#!/usr/bin/env python3
"""Build an EM briefing pack for M6 safe-transfer results."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sort_strict_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = list(rows)
    out.sort(
        key=lambda row: (
            as_float(row.get("strict_pass"), 0.0),
            as_float(row.get("safety_weighted_score"), 0.0),
            as_float(row.get("mean_online_minus_static"), 0.0),
            as_float(row.get("provider_bootstrap_ci_low"), 0.0),
        ),
        reverse=True,
    )
    return out


def aggregate_replay_rows(
    grid_root: Path,
    strict_by_config: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for config_dir in sorted(grid_root.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("replay_"):
            continue
        cycle_dir = config_dir / "cycle"
        lopo_path = cycle_dir / "m6_lopo_replay.csv"
        temporal_path = cycle_dir / "m6_temporal_backtest_replay.csv"
        manifest_path = cycle_dir / "m5_cycle_manifest.json"
        if not lopo_path.exists() or not temporal_path.exists():
            continue

        lopo_rows = read_csv_rows(lopo_path)
        temporal_rows = read_csv_rows(temporal_path)
        if not lopo_rows or not temporal_rows:
            continue

        strict_row = strict_by_config.get(config_dir.name, {})
        provider_step_mults = ""
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            provider_step_mults = str(payload.get("online_provider_step_mults") or "")

        out.append(
            {
                "config": config_dir.name,
                "strict_mean_online_minus_static": as_float(
                    strict_row.get("mean_online_minus_static"), 0.0
                ),
                "strict_pass": as_float(strict_row.get("strict_pass"), 0.0),
                "strict_provider_ci_low": as_float(
                    strict_row.get("provider_bootstrap_ci_low"), 0.0
                ),
                "strict_worst_provider": str(strict_row.get("worst_provider") or ""),
                "strict_worst_provider_mean_diff": as_float(
                    strict_row.get("worst_provider_mean_diff"), 0.0
                ),
                "lopo_rows": float(len(lopo_rows)),
                "lopo_worst_online_minus_static": min(
                    as_float(row.get("mean_online_minus_static"), 0.0)
                    for row in lopo_rows
                ),
                "lopo_best_online_minus_static": max(
                    as_float(row.get("mean_online_minus_static"), 0.0)
                    for row in lopo_rows
                ),
                "temporal_rows": float(len(temporal_rows)),
                "temporal_worst_online_minus_static": min(
                    as_float(row.get("mean_online_minus_static"), 0.0)
                    for row in temporal_rows
                ),
                "temporal_best_online_minus_static": max(
                    as_float(row.get("mean_online_minus_static"), 0.0)
                    for row in temporal_rows
                ),
                "online_provider_step_mults": provider_step_mults,
                "lopo_csv": str(lopo_path),
                "temporal_csv": str(temporal_path),
            }
        )

    out.sort(
        key=lambda row: (
            as_float(row.get("strict_pass"), 0.0),
            as_float(row.get("temporal_worst_online_minus_static"), 0.0),
            as_float(row.get("lopo_worst_online_minus_static"), 0.0),
            as_float(row.get("strict_mean_online_minus_static"), 0.0),
        ),
        reverse=True,
    )
    return out


def choose_candidate(
    replay_rows: Sequence[Dict[str, Any]],
    selection_json_path: Path,
) -> Dict[str, Any]:
    if selection_json_path.exists():
        payload = json.loads(selection_json_path.read_text(encoding="utf-8"))
        selected_name = str(payload.get("selected_candidate") or "")
        for row in replay_rows:
            if str(row.get("config") or "") == selected_name:
                row_out = dict(row)
                row_out["selection_source"] = "selection_json"
                row_out["selection_rationale"] = str(
                    payload.get("selection_rationale") or ""
                )
                return row_out

    if replay_rows:
        row = dict(replay_rows[0])
        row["selection_source"] = "ranked_replay_metrics"
        row["selection_rationale"] = (
            "highest strict/replay worst-case profile under deterministic ranking"
        )
        return row

    return {
        "config": "",
        "selection_source": "none",
        "selection_rationale": "no replay rows discovered",
    }


def plot_strict_top(
    strict_rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    top_k: int,
) -> None:
    if not strict_rows:
        return
    rows = list(strict_rows[: max(1, int(top_k))])
    labels = [str(row.get("config") or "") for row in rows][::-1]
    values = [as_float(row.get("mean_online_minus_static"), 0.0) for row in rows][::-1]
    colors = [
        "#2E7D32" if as_float(row.get("strict_pass"), 0.0) > 0.0 else "#C62828"
        for row in rows
    ][::-1]

    height = max(4.0, 0.5 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height), constrained_layout=True)
    ax.barh(labels, values, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Mean online - static delta MCC")
    ax.set_title("M6 Strict Eval: Top Configs by Safety-Weighted Rank")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    fig.savefig(out_dir / "em_fig_strict_top_configs.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "em_fig_strict_top_configs.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stress_overview(stress_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not stress_rows:
        return
    rows = sorted(
        stress_rows,
        key=lambda row: as_float(row.get("mean_online_minus_static"), 0.0),
        reverse=True,
    )
    labels = [str(row.get("config") or "") for row in rows]
    values = [as_float(row.get("mean_online_minus_static"), 0.0) for row in rows]
    colors = [
        "#2E7D32" if as_float(row.get("strict_pass"), 0.0) > 0.0 else "#C62828"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_ylabel("Mean online - static delta MCC")
    ax.set_title("M6 Stress Suite: Online vs Static")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (0.00015 if value >= 0.0 else -0.00015),
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=8,
        )

    fig.savefig(out_dir / "em_fig_stress_suite.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "em_fig_stress_suite.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_replay_guardrails(
    replay_rows: Sequence[Dict[str, Any]], out_dir: Path
) -> None:
    if not replay_rows:
        return
    labels = [str(row.get("config") or "") for row in replay_rows]
    strict_vals = [
        as_float(row.get("strict_mean_online_minus_static"), 0.0) for row in replay_rows
    ]
    lopo_vals = [
        as_float(row.get("lopo_worst_online_minus_static"), 0.0) for row in replay_rows
    ]
    temporal_vals = [
        as_float(row.get("temporal_worst_online_minus_static"), 0.0)
        for row in replay_rows
    ]

    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(
        [i - width for i in x], strict_vals, width, label="Strict mean", color="#1565C0"
    )
    ax.bar(x, lopo_vals, width, label="LOPO worst", color="#2E7D32")
    ax.bar(
        [i + width for i in x],
        temporal_vals,
        width,
        label="Temporal worst",
        color="#EF6C00",
    )

    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_ylabel("Online - static delta MCC")
    ax.set_title("M6 Replay Guardrails: Strict / LOPO / Temporal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.savefig(out_dir / "em_fig_replay_guardrails.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "em_fig_replay_guardrails.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    out_path: Path,
    strict_rows: Sequence[Dict[str, Any]],
    stress_rows: Sequence[Dict[str, Any]],
    replay_rows: Sequence[Dict[str, Any]],
    selected: Dict[str, Any],
    strict_csv_path: Path,
    stress_csv_path: Path,
) -> None:
    strict_pass_count = sum(
        1 for row in strict_rows if as_float(row.get("strict_pass"), 0.0) > 0.0
    )
    top = strict_rows[0] if strict_rows else {}
    worst = strict_rows[-1] if strict_rows else {}

    lines = [
        "# M6 EM Briefing Pack",
        "",
        "## Snapshot",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Strict configs analyzed: {len(strict_rows)}",
        f"- Strict-pass configs: {strict_pass_count}",
        f"- Top strict config: {str(top.get('config') or '(none)')}",
        f"- Bottom strict config: {str(worst.get('config') or '(none)')}",
        "",
        "## Candidate",
        "",
        f"- Selected config: {str(selected.get('config') or '(none)')}",
        f"- Selection source: {str(selected.get('selection_source') or 'n/a')}",
        f"- Rationale: {str(selected.get('selection_rationale') or 'n/a')}",
        f"- Strict mean online-static: {as_float(selected.get('strict_mean_online_minus_static'), 0.0):+.6f}",
        f"- Replay LOPO worst online-static: {as_float(selected.get('lopo_worst_online_minus_static'), 0.0):+.6f}",
        f"- Replay temporal worst online-static: {as_float(selected.get('temporal_worst_online_minus_static'), 0.0):+.6f}",
        "",
        "## Stress",
        "",
        f"- Stress configs analyzed: {len(stress_rows)}",
    ]

    if stress_rows:
        best_stress = max(
            stress_rows,
            key=lambda row: as_float(row.get("mean_online_minus_static"), 0.0),
        )
        lines.append(
            "- Best stress config: "
            f"{str(best_stress.get('config') or '')} "
            f"({as_float(best_stress.get('mean_online_minus_static'), 0.0):+.6f})"
        )

    lines.extend(
        [
            "",
            "## Data Sources",
            "",
            f"- Strict CSV: {strict_csv_path}",
            f"- Stress CSV: {stress_csv_path}",
            "- Replay aggregate CSV: m6_em_replay_summary.csv",
            "",
            "## Figure Files",
            "",
            "- em_fig_strict_top_configs.{png,pdf}",
            "- em_fig_stress_suite.{png,pdf}",
            "- em_fig_replay_guardrails.{png,pdf}",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build M6 EM briefing pack")
    parser.add_argument(
        "--grid-root",
        default="workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep",
    )
    parser.add_argument(
        "--stress-root",
        default="workspace/deep_survey_2026-03-01/repair_v3/m6_stress_suite",
    )
    parser.add_argument(
        "--out-dir",
        default="workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/m6_em_pack",
    )
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    grid_root = Path(args.grid_root)
    stress_root = Path(args.stress_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strict_csv_path = grid_root / "m6_online_strict_eval.csv"
    stress_csv_path = stress_root / "m6_online_strict_eval.csv"
    selection_json_path = grid_root / "m6_replay_candidate_selection.json"

    strict_rows = sort_strict_rows(read_csv_rows(strict_csv_path))
    strict_by_config = {str(row.get("config") or ""): row for row in strict_rows}
    stress_rows = sort_strict_rows(read_csv_rows(stress_csv_path))

    replay_rows = aggregate_replay_rows(
        grid_root=grid_root, strict_by_config=strict_by_config
    )
    selected = choose_candidate(
        replay_rows=replay_rows, selection_json_path=selection_json_path
    )

    strict_export = [
        {
            "config": str(row.get("config") or ""),
            "strict_pass": as_float(row.get("strict_pass"), 0.0),
            "mean_online_minus_static": as_float(
                row.get("mean_online_minus_static"), 0.0
            ),
            "provider_bootstrap_ci_low": as_float(
                row.get("provider_bootstrap_ci_low"), 0.0
            ),
            "provider_bootstrap_ci_high": as_float(
                row.get("provider_bootstrap_ci_high"), 0.0
            ),
            "worst_provider": str(row.get("worst_provider") or ""),
            "worst_provider_mean_diff": as_float(
                row.get("worst_provider_mean_diff"), 0.0
            ),
            "mean_harm_delta": as_float(row.get("mean_harm_delta"), 0.0),
            "mean_alarm_delta": as_float(row.get("mean_alarm_delta"), 0.0),
            "safety_weighted_score": as_float(row.get("safety_weighted_score"), 0.0),
        }
        for row in strict_rows
    ]
    write_csv(
        out_dir / "m6_em_master_summary.csv",
        strict_export,
        fieldnames=[
            "config",
            "strict_pass",
            "mean_online_minus_static",
            "provider_bootstrap_ci_low",
            "provider_bootstrap_ci_high",
            "worst_provider",
            "worst_provider_mean_diff",
            "mean_harm_delta",
            "mean_alarm_delta",
            "safety_weighted_score",
        ],
    )

    write_csv(
        out_dir / "m6_em_replay_summary.csv",
        replay_rows,
        fieldnames=[
            "config",
            "strict_mean_online_minus_static",
            "strict_pass",
            "strict_provider_ci_low",
            "strict_worst_provider",
            "strict_worst_provider_mean_diff",
            "lopo_rows",
            "lopo_worst_online_minus_static",
            "lopo_best_online_minus_static",
            "temporal_rows",
            "temporal_worst_online_minus_static",
            "temporal_best_online_minus_static",
            "online_provider_step_mults",
            "lopo_csv",
            "temporal_csv",
        ],
    )

    stress_export = [
        {
            "config": str(row.get("config") or ""),
            "strict_pass": as_float(row.get("strict_pass"), 0.0),
            "mean_online_minus_static": as_float(
                row.get("mean_online_minus_static"), 0.0
            ),
            "provider_bootstrap_ci_low": as_float(
                row.get("provider_bootstrap_ci_low"), 0.0
            ),
            "worst_provider": str(row.get("worst_provider") or ""),
            "worst_provider_mean_diff": as_float(
                row.get("worst_provider_mean_diff"), 0.0
            ),
            "mean_harm_delta": as_float(row.get("mean_harm_delta"), 0.0),
            "mean_alarm_delta": as_float(row.get("mean_alarm_delta"), 0.0),
        }
        for row in stress_rows
    ]
    write_csv(
        out_dir / "m6_em_stress_summary.csv",
        stress_export,
        fieldnames=[
            "config",
            "strict_pass",
            "mean_online_minus_static",
            "provider_bootstrap_ci_low",
            "worst_provider",
            "worst_provider_mean_diff",
            "mean_harm_delta",
            "mean_alarm_delta",
        ],
    )

    candidate_rows: List[Dict[str, Any]] = []
    for row in replay_rows:
        candidate_rows.append(
            {
                "config": str(row.get("config") or ""),
                "selected": 1.0
                if str(row.get("config") or "") == str(selected.get("config") or "")
                else 0.0,
                "strict_pass": as_float(row.get("strict_pass"), 0.0),
                "strict_mean_online_minus_static": as_float(
                    row.get("strict_mean_online_minus_static"), 0.0
                ),
                "lopo_worst_online_minus_static": as_float(
                    row.get("lopo_worst_online_minus_static"), 0.0
                ),
                "temporal_worst_online_minus_static": as_float(
                    row.get("temporal_worst_online_minus_static"), 0.0
                ),
                "online_provider_step_mults": str(
                    row.get("online_provider_step_mults") or ""
                ),
            }
        )
    write_csv(
        out_dir / "m6_em_candidate_comparison.csv",
        candidate_rows,
        fieldnames=[
            "config",
            "selected",
            "strict_pass",
            "strict_mean_online_minus_static",
            "lopo_worst_online_minus_static",
            "temporal_worst_online_minus_static",
            "online_provider_step_mults",
        ],
    )

    plot_strict_top(strict_rows=strict_rows, out_dir=out_dir, top_k=int(args.top_k))
    plot_stress_overview(stress_rows=stress_rows, out_dir=out_dir)
    plot_replay_guardrails(replay_rows=replay_rows, out_dir=out_dir)

    write_markdown_summary(
        out_path=out_dir / "M6_EM_BRIEF.md",
        strict_rows=strict_rows,
        stress_rows=stress_rows,
        replay_rows=replay_rows,
        selected=selected,
        strict_csv_path=strict_csv_path,
        stress_csv_path=stress_csv_path,
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "grid_root": str(grid_root),
        "stress_root": str(stress_root),
        "strict_csv": str(strict_csv_path),
        "stress_csv": str(stress_csv_path),
        "selection_json": str(selection_json_path),
        "n_strict_rows": len(strict_rows),
        "n_stress_rows": len(stress_rows),
        "n_replay_rows": len(replay_rows),
        "selected_candidate": str(selected.get("config") or ""),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "m6_em_pack_manifest.json", manifest)

    print("M6 EM pack build complete")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
