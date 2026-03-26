#!/usr/bin/env python3
"""Run M5 instance-level selective guardrail experiments on CARE-v3 runs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TRANSFER_PROVIDER_THRESHOLDS = (
    PROJECT_ROOT / "chaosbench" / "repair" / "m5_provider_thresholds_crossfit_v1.json"
)
DEFAULT_TRANSFER_PROVIDER_REFERENCES = (
    PROJECT_ROOT
    / "chaosbench"
    / "repair"
    / "m5_provider_reference_dists_crossfit_v1.json"
)

from chaosbench.repair.engine import compute_axiom_violation_rate, load_selector_index
from chaosbench.repair.instance_policy import (
    apply_instance_policy,
    fit_instance_policy,
    policy_cell_rows,
    policy_family_rows,
    score_instance_candidate,
)
from chaosbench.repair.online_controller import (
    OnlineControllerConfig,
    OnlineTransferController,
)
from scripts.run_m4_selective_guardrail import (
    as_float,
    load_repair_config,
    load_runs,
    load_split_map,
    metrics_from_records,
    sign_test_two_sided,
    write_csv,
)


def load_provider_thresholds(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("provider threshold map must be a JSON object")
    out: Dict[str, float] = {}
    for key, value in payload.items():
        provider = str(key or "").strip().lower()
        if not provider:
            continue
        if isinstance(value, dict):
            out[provider] = float(value.get("threshold", 0.0))
        else:
            out[provider] = float(value)
    return out


def load_provider_reference_dists(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("provider reference map must be a JSON object")
    out: Dict[str, Dict[str, float]] = {}
    for key, value in payload.items():
        provider = str(key or "").strip().lower()
        if not provider or not isinstance(value, dict):
            continue
        dist = {
            str(cell_key): float(cell_value)
            for cell_key, cell_value in value.items()
            if as_float(cell_value, 0.0) > 0.0
        }
        if dist:
            out[provider] = dist
    return out


def resolve_threshold(
    provider: str,
    default_threshold: float,
    provider_thresholds: Dict[str, float],
) -> float:
    if not provider_thresholds:
        return float(default_threshold)

    provider_lc = str(provider or "").strip().lower()
    if provider_lc in provider_thresholds:
        return float(provider_thresholds[provider_lc])

    base = provider_lc.split("/", 1)[0]
    if base in provider_thresholds:
        return float(provider_thresholds[base])

    return float(default_threshold)


def resolve_provider_thresholds_path(
    provider_thresholds_json: str | None,
    auto_provider_thresholds_json: bool,
) -> tuple[str | None, str]:
    if provider_thresholds_json:
        return provider_thresholds_json, "explicit"

    if auto_provider_thresholds_json and DEFAULT_TRANSFER_PROVIDER_THRESHOLDS.exists():
        return str(DEFAULT_TRANSFER_PROVIDER_THRESHOLDS), "auto_default_for_transfer"

    return None, "none"


def resolve_provider_reference_path(
    provider_reference_json: str | None,
    auto_provider_reference_json: bool,
    provider_thresholds_path: str | None,
) -> tuple[str | None, str]:
    if provider_reference_json:
        return provider_reference_json, "explicit"

    if auto_provider_reference_json:
        if provider_thresholds_path:
            sibling = (
                Path(provider_thresholds_path)
                .resolve()
                .with_name("provider_reference_dists_crossfit_v1.json")
            )
            if sibling.exists():
                return str(sibling), "auto_sibling_of_threshold_map"
        if DEFAULT_TRANSFER_PROVIDER_REFERENCES.exists():
            return str(
                DEFAULT_TRANSFER_PROVIDER_REFERENCES
            ), "auto_default_for_transfer"

    return None, "none"


def parse_provider_step_multipliers(raw: str) -> Dict[str, float]:
    payload = str(raw or "").strip()
    if not payload:
        return {}
    out: Dict[str, float] = {}
    for token in payload.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--online-provider-step-mults entries must be provider=multiplier"
            )
        provider_raw, multiplier_raw = item.split("=", 1)
        provider = provider_raw.strip().lower()
        if not provider:
            raise ValueError(
                "--online-provider-step-mults has empty provider key entry"
            )
        multiplier = as_float(multiplier_raw, -1.0)
        if multiplier < 0.0:
            raise ValueError("--online-provider-step-mults multiplier must be >= 0")
        out[provider] = float(multiplier)
    return out


def online_controller_config_from_args(
    args: argparse.Namespace,
) -> OnlineControllerConfig:
    step_default = float(args.online_provider_step_default)
    if step_default < 0.0:
        raise ValueError("--online-provider-step-default must be >= 0")

    return OnlineControllerConfig(
        threshold_min=float(args.online_threshold_min),
        threshold_max=float(args.online_threshold_max),
        eta0=float(args.online_eta0),
        threshold_step=float(args.online_threshold_step),
        eps_mcc=float(args.online_eps_mcc),
        eps_axiom=float(args.online_eps_axiom),
        axiom_penalty=float(args.online_axiom_penalty),
        harm_axiom_penalty=float(args.online_harm_axiom_penalty),
        risk_budget_b0=float(args.online_risk_budget_b0),
        shift_kappa=float(args.online_shift_kappa),
        shift_target=float(args.online_shift_target),
        lambda_max=float(args.online_lambda_max),
        rho_max=float(args.online_rho_max),
        alarm_drift=float(args.online_alarm_drift),
        alarm_threshold=float(args.online_alarm_threshold),
        emergency_step=float(args.online_emergency_step),
        ema_alpha=float(args.online_ema_alpha),
        sweep_radius=int(args.online_sweep_radius),
        sweep_mix=float(args.online_sweep_mix),
        sweep_min_improvement=float(args.online_sweep_min_improvement),
        provider_step_default=step_default,
        provider_step_multipliers=parse_provider_step_multipliers(
            str(args.online_provider_step_mults)
        ),
        enforce_non_degrade_guard=not bool(args.online_disable_non_degrade_guard),
        non_degrade_margin=float(args.online_non_degrade_margin),
        non_degrade_rollback_step=float(args.online_non_degrade_rollback_step),
    )


def build_report(
    out_path: Path,
    policy: Dict[str, Any],
    family_rows: List[Dict[str, Any]],
    run_rows: List[Dict[str, Any]],
    heldout_sign: Dict[str, float],
) -> None:
    heldout = [row for row in run_rows if row["split"] == "heldout"]
    dev = [row for row in run_rows if row["split"] == "dev"]

    def mean_col(rows: List[Dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return statistics.mean(as_float(row.get(key), 0.0) for row in rows)

    controller_mode = (
        str(run_rows[0].get("controller_mode", "static")) if run_rows else "static"
    )

    lines = [
        "# M5 Instance Guardrail Results",
        "",
        "## Policy summary",
        "",
        f"- Threshold: {as_float(policy.get('threshold'), 0.0):+.6f}",
        f"- Controller mode: {controller_mode}",
        f"- Margin step: {as_float(policy.get('margin_step'), 0.0):.4f}",
        f"- Support cap: {int(as_float(policy.get('support_cap'), 0.0))}",
        f"- Shrinkage: {as_float(policy.get('shrinkage'), 0.0):.4f}",
        f"- Degrade penalty: {as_float(policy.get('degrade_penalty'), 1.0):.4f}",
        "",
        "## Family policy",
        "",
        "| Family | Enabled | Mean utility | Candidates | Improved | Degraded |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in family_rows:
        lines.append(
            f"| {row['family']} | {int(as_float(row['enabled']))} | "
            f"{as_float(row['mean_utility']):+.4f} | "
            f"{int(as_float(row['n_candidates']))} | "
            f"{int(as_float(row['improved']))} | {int(as_float(row['degraded']))} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Dev mean delta MCC (baseline): {mean_col(dev, 'delta_mcc_baseline'):+.4f}",
            f"- Dev mean delta MCC (policy): {mean_col(dev, 'delta_mcc_policy'):+.4f}",
            f"- Held-out mean delta MCC (baseline): {mean_col(heldout, 'delta_mcc_baseline'):+.4f}",
            f"- Held-out mean delta MCC (policy): {mean_col(heldout, 'delta_mcc_policy'):+.4f}",
            f"- Held-out mean (policy - baseline) delta MCC: {mean_col(heldout, 'policy_minus_baseline_mcc'):+.4f}",
            f"- Held-out sign test (policy - baseline): positives={int(heldout_sign['positives'])}/{int(heldout_sign['n'])}, p={heldout_sign['p_value']:.6f}",
            f"- Mean shift score: {mean_col(run_rows, 'shift_score'):.4f}",
            f"- Mean harm loss: {mean_col(run_rows, 'harm_loss'):.4f}",
            f"- Alarm rate: {mean_col(run_rows, 'alarm_triggered'):.4f}",
            "",
            "## Per-run",
            "",
            "| Run ID | Provider | Split | Threshold before | Threshold after | Delta MCC baseline | Delta MCC policy | Policy - baseline | Shift | Harm | Alarm |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in run_rows:
        lines.append(
            f"| {row['run_id']} | {row['provider']} | {row['split']} | "
            f"{as_float(row.get('threshold_before'), as_float(row.get('threshold_used'), 0.0)):.4f} | "
            f"{as_float(row.get('threshold_after'), as_float(row.get('threshold_used'), 0.0)):.4f} | "
            f"{row['delta_mcc_baseline']:+.4f} | {row['delta_mcc_policy']:+.4f} | "
            f"{row['policy_minus_baseline_mcc']:+.4f} | "
            f"{as_float(row.get('shift_score'), 0.0):.4f} | "
            f"{as_float(row.get('harm_loss'), 0.0):.4f} | "
            f"{as_float(row.get('alarm_triggered'), 0.0):.0f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M5 instance guardrail suite")
    parser.add_argument(
        "--repair-dir",
        default=str(
            PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "repair_v3"
        ),
    )
    parser.add_argument(
        "--split-map-csv",
        default=None,
        help="Optional CSV mapping run_id->split (defaults to <repair-dir>/tables/repair_deltas.csv)",
    )
    parser.add_argument(
        "--default-split",
        default="unknown",
        help="Split label used when run_id missing from split map",
    )
    parser.add_argument(
        "--policy-json",
        default=None,
        help="Optional pre-fit policy JSON; if provided, skip dev policy fitting",
    )
    parser.add_argument(
        "--provider-thresholds-json",
        default=None,
        help="Optional JSON map of provider or provider-prefix to threshold",
    )
    parser.add_argument(
        "--auto-provider-thresholds-json",
        action="store_true",
        help="Auto-load default provider threshold map for transfer runs",
    )
    parser.add_argument(
        "--provider-reference-json",
        default=None,
        help="Optional JSON map of provider or provider-prefix to reference distributions",
    )
    parser.add_argument(
        "--auto-provider-reference-json",
        action="store_true",
        help="Auto-load provider reference map from calibration outputs for online mode",
    )
    parser.add_argument(
        "--threshold-override",
        type=float,
        default=None,
        help="Optional score threshold override applied after loading/fitting policy",
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "workspace"
            / "deep_survey_2026-03-01"
            / "repair_v3"
            / "m5_instance"
        ),
    )
    parser.add_argument("--selector", default="data/canonical_v2_files.json")
    parser.add_argument("--margin-step", type=float, default=0.05)
    parser.add_argument("--support-cap", type=int, default=8)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    parser.add_argument("--min-family-samples", type=int, default=20)
    parser.add_argument("--degrade-penalty", type=float, default=1.0)
    parser.add_argument(
        "--controller-mode",
        choices=["static", "online"],
        default="static",
        help="Use static thresholds or online adaptive controller",
    )
    parser.add_argument(
        "--controller-state-json",
        default=None,
        help="Optional JSON path for warm-starting online controller state",
    )
    parser.add_argument("--online-threshold-min", type=float, default=0.0)
    parser.add_argument("--online-threshold-max", type=float, default=1.5)
    parser.add_argument("--online-eta0", type=float, default=0.5)
    parser.add_argument("--online-threshold-step", type=float, default=0.05)
    parser.add_argument("--online-eps-mcc", type=float, default=0.0)
    parser.add_argument("--online-eps-axiom", type=float, default=0.0)
    parser.add_argument("--online-axiom-penalty", type=float, default=0.0)
    parser.add_argument("--online-harm-axiom-penalty", type=float, default=1.0)
    parser.add_argument("--online-risk-budget-b0", type=float, default=0.001)
    parser.add_argument("--online-shift-kappa", type=float, default=2.0)
    parser.add_argument("--online-shift-target", type=float, default=0.05)
    parser.add_argument("--online-lambda-max", type=float, default=50.0)
    parser.add_argument("--online-rho-max", type=float, default=50.0)
    parser.add_argument("--online-alarm-drift", type=float, default=0.0)
    parser.add_argument("--online-alarm-threshold", type=float, default=0.02)
    parser.add_argument("--online-emergency-step", type=float, default=0.1)
    parser.add_argument("--online-ema-alpha", type=float, default=0.1)
    parser.add_argument("--online-sweep-radius", type=int, default=2)
    parser.add_argument("--online-sweep-mix", type=float, default=0.5)
    parser.add_argument("--online-sweep-min-improvement", type=float, default=0.0)
    parser.add_argument(
        "--online-provider-step-mults",
        default="",
        help="Comma-separated provider=multiplier dampening map for online updates",
    )
    parser.add_argument("--online-provider-step-default", type=float, default=1.0)
    parser.add_argument("--online-non-degrade-margin", type=float, default=0.0)
    parser.add_argument("--online-non-degrade-rollback-step", type=float, default=0.0)
    parser.add_argument(
        "--online-disable-non-degrade-guard",
        action="store_true",
        help="Disable provider non-degradation threshold rollback guard",
    )
    parser.add_argument(
        "--online-update-splits",
        default="heldout",
        help="Comma-separated splits that can update online controller state",
    )
    parser.add_argument(
        "--online-label-lag-runs",
        type=int,
        default=0,
        help="Delay controller updates by this many runs to simulate delayed labels",
    )
    args = parser.parse_args()

    if int(args.online_label_lag_runs) < 0:
        raise ValueError("--online-label-lag-runs must be >= 0")
    if float(args.online_non_degrade_margin) < 0.0:
        raise ValueError("--online-non-degrade-margin must be >= 0")
    if float(args.online_non_degrade_rollback_step) < 0.0:
        raise ValueError("--online-non-degrade-rollback-step must be >= 0")
    if float(args.online_sweep_min_improvement) < 0.0:
        raise ValueError("--online-sweep-min-improvement must be >= 0")

    repair_dir = Path(args.repair_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_repair_config(repair_dir)
    id_to_meta = load_selector_index(
        selector_path=Path(args.selector),
        project_root=PROJECT_ROOT,
    )
    split_map_path = (
        Path(args.split_map_csv)
        if args.split_map_csv
        else repair_dir / "tables" / "repair_deltas.csv"
    )
    split_map = load_split_map(split_map_path)

    runs = load_runs(
        repair_dir=repair_dir,
        split_map=split_map,
        id_to_meta=id_to_meta,
        config=config,
        default_split=str(args.default_split),
    )
    if not runs:
        raise RuntimeError("No runs discovered under repair_dir/{repair_runs,runs}")

    threshold_rows: List[Dict[str, Any]] = []
    if args.policy_json:
        policy = json.loads(Path(args.policy_json).read_text(encoding="utf-8"))
        family_rows = policy_family_rows(policy)
        cell_rows = policy_cell_rows(policy)
    else:
        dev_candidates = [
            candidate
            for run in runs
            if run.split == "dev"
            for candidate in run.candidates
        ]
        if not dev_candidates:
            raise RuntimeError(
                "No dev candidates available to fit policy. "
                "Provide --policy-json or a split map with dev runs."
            )

        policy, family_rows, cell_rows, threshold_rows = fit_instance_policy(
            candidates=dev_candidates,
            margin_step=args.margin_step,
            support_cap=args.support_cap,
            shrinkage=args.shrinkage,
            min_family_samples=args.min_family_samples,
            degrade_penalty=args.degrade_penalty,
        )

    if args.threshold_override is not None:
        policy["threshold"] = float(args.threshold_override)

    provider_thresholds_path, provider_thresholds_source = (
        resolve_provider_thresholds_path(
            provider_thresholds_json=args.provider_thresholds_json,
            auto_provider_thresholds_json=bool(args.auto_provider_thresholds_json),
        )
    )
    provider_thresholds: Dict[str, float] = {}
    if provider_thresholds_path:
        provider_thresholds = load_provider_thresholds(Path(provider_thresholds_path))

    provider_reference_path, provider_reference_source = (
        resolve_provider_reference_path(
            provider_reference_json=args.provider_reference_json,
            auto_provider_reference_json=bool(args.auto_provider_reference_json),
            provider_thresholds_path=provider_thresholds_path,
        )
    )
    provider_reference_dists: Dict[str, Dict[str, float]] = {}
    if provider_reference_path:
        provider_reference_dists = load_provider_reference_dists(
            Path(provider_reference_path)
        )

    controller: OnlineTransferController | None = None
    controller_config: OnlineControllerConfig | None = None
    if args.controller_mode == "online":
        controller_config = online_controller_config_from_args(args)
        controller = OnlineTransferController.from_policy(
            policy=policy,
            config=controller_config,
            provider_reference_dists=provider_reference_dists,
        )
        if args.controller_state_json:
            payload = json.loads(
                Path(args.controller_state_json).read_text(encoding="utf-8")
            )
            controller.load_dict(payload)

    online_update_splits = {
        token.strip()
        for token in str(args.online_update_splits).split(",")
        if token.strip()
    }

    run_rows: List[Dict[str, Any]] = []
    online_trace_rows: List[Dict[str, Any]] = []
    online_update_events: List[Dict[str, Any]] = []
    pending_updates: List[Dict[str, Any]] = []
    label_lag_runs = int(args.online_label_lag_runs)
    provider_override_hits = 0
    for run_index, run in enumerate(runs):
        updates_applied_before_decision = 0.0
        if controller is not None and label_lag_runs > 0 and pending_updates:
            remaining_updates: List[Dict[str, Any]] = []
            for payload in pending_updates:
                if int(as_float(payload.get("due_run_index"), -1)) <= int(run_index):
                    update = controller.update(
                        provider=str(payload["provider"]),
                        fallback_threshold=as_float(payload["fallback_threshold"], 0.0),
                        shift_score=as_float(payload["shift_score"], 0.0),
                        delta_mcc=as_float(payload["delta_mcc"], 0.0),
                        baseline_axiom_rate=as_float(
                            payload["baseline_axiom_rate"], 0.0
                        ),
                        policy_axiom_rate=as_float(payload["policy_axiom_rate"], 0.0),
                        candidate_rows=payload.get("candidate_rows"),
                        degrade_penalty=as_float(payload.get("degrade_penalty"), 1.0),
                        online_minus_static=as_float(
                            payload.get("online_minus_static"),
                            0.0,
                        ),
                    )
                    online_update_events.append(
                        {
                            "source_run_id": str(payload["source_run_id"]),
                            "source_provider": str(payload["provider"]),
                            "source_split": str(payload["split"]),
                            "source_run_index": float(payload["source_run_index"]),
                            "due_run_index": float(payload["due_run_index"]),
                            "applied_at_run_id": run.run_id,
                            "applied_at_run_index": float(run_index),
                            "post_stream": 0.0,
                            "controller_mode": args.controller_mode,
                            "update_applied": 1.0,
                            **update,
                        }
                    )
                    updates_applied_before_decision += 1.0
                else:
                    remaining_updates.append(payload)
            pending_updates = remaining_updates

        pre_metrics = metrics_from_records(run.records, "parsed_label")
        baseline_metrics = metrics_from_records(run.baseline_records, "repaired_label")

        default_threshold = as_float(policy.get("threshold"), 0.0)
        static_threshold = resolve_threshold(
            provider=run.provider,
            default_threshold=default_threshold,
            provider_thresholds=provider_thresholds,
        )
        if static_threshold != default_threshold:
            provider_override_hits += 1

        margin_step = as_float(policy.get("margin_step"), 0.05)
        support_cap = int(as_float(policy.get("support_cap"), 8.0))
        shift_score = 0.0
        if controller is not None:
            shift_score = controller.shift_score(
                candidates=run.candidates,
                margin_step=margin_step,
                support_cap=support_cap,
                provider=run.provider,
            )
            threshold_before = controller.threshold_for_provider(
                provider=run.provider,
                fallback_threshold=static_threshold,
            )
        else:
            threshold_before = float(static_threshold)

        threshold_used = float(threshold_before)
        policy_for_run = dict(policy)
        policy_for_run["threshold"] = threshold_used
        families = policy.get("families", {})
        online_candidate_rows = [
            {
                "score": float(score_instance_candidate(candidate, policy)),
                "improved": float(candidate.improved),
                "degraded": float(candidate.degraded),
                "enabled": 1.0
                if as_float(
                    families.get(str(candidate.family or "unknown"), {}).get(
                        "enabled", 0.0
                    ),
                    0.0,
                )
                > 0.0
                else 0.0,
            }
            for candidate in run.candidates
        ]

        policy_records, policy_stats = apply_instance_policy(
            repaired_records=run.baseline_records,
            candidates=run.candidates,
            policy=policy_for_run,
        )
        policy_metrics = metrics_from_records(policy_records, "repaired_label")

        pre_v_count, pre_v_rate = compute_axiom_violation_rate(
            records=run.records,
            label_key="parsed_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )
        baseline_v_count, baseline_v_rate = compute_axiom_violation_rate(
            records=run.baseline_records,
            label_key="repaired_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )
        policy_v_count, policy_v_rate = compute_axiom_violation_rate(
            records=policy_records,
            label_key="repaired_label",
            id_to_meta=id_to_meta,
            config=config,
            gate_families=None,
        )

        policy_row_flips = sum(
            1 for row in policy_records if row.get("was_flipped") is True
        )
        policy_row_flip_rate = (
            policy_row_flips / policy_metrics["valid"]
            if policy_metrics["valid"]
            else 0.0
        )

        delta_mcc_baseline = baseline_metrics["mcc"] - pre_metrics["mcc"]
        delta_mcc_policy = policy_metrics["mcc"] - pre_metrics["mcc"]
        policy_minus_baseline = delta_mcc_policy - delta_mcc_baseline
        static_policy_minus_baseline = float(policy_minus_baseline)
        online_minus_static = 0.0
        if controller is not None:
            if abs(float(threshold_used) - float(static_threshold)) <= 1e-12:
                static_delta_mcc_policy = float(delta_mcc_policy)
            else:
                static_policy_for_run = dict(policy)
                static_policy_for_run["threshold"] = float(static_threshold)
                static_policy_records, _ = apply_instance_policy(
                    repaired_records=run.baseline_records,
                    candidates=run.candidates,
                    policy=static_policy_for_run,
                )
                static_policy_metrics = metrics_from_records(
                    static_policy_records,
                    "repaired_label",
                )
                static_delta_mcc_policy = (
                    static_policy_metrics["mcc"] - pre_metrics["mcc"]
                )
            static_policy_minus_baseline = float(static_delta_mcc_policy) - float(
                delta_mcc_baseline
            )
            online_minus_static = float(policy_minus_baseline) - float(
                static_policy_minus_baseline
            )

        controller_update = {
            "threshold_before": float(threshold_used),
            "threshold_after": float(threshold_used),
            "delta_threshold": 0.0,
            "threshold_after_primal": float(threshold_used),
            "shift_score": float(shift_score),
            "utility": float(policy_minus_baseline),
            "harm_loss": max(0.0, -float(policy_minus_baseline)),
            "risk_budget": 0.0,
            "lambda_harm": 0.0,
            "rho_shift": 0.0,
            "sweep_applied": 0.0,
            "sweep_threshold": float(threshold_used),
            "sweep_objective": 0.0,
            "sweep_center_objective": 0.0,
            "sweep_objective_gain": 0.0,
            "sweep_mix_requested": 0.0,
            "sweep_mix_effective": 0.0,
            "step_multiplier": 0.0,
            "eta_effective": 0.0,
            "non_degrade_signal": float(online_minus_static),
            "non_degrade_guard_triggered": 0.0,
            "non_degrade_rollback_applied": 0.0,
            "alarm_triggered": 0.0,
            "cusum_harm": 0.0,
            "ema_harm": 0.0,
            "ema_shift": 0.0,
            "seen_batches": 0.0,
            "alarms": 0.0,
            "update_applied": 0.0,
            "update_scheduled": 0.0,
            "lag_runs": float(label_lag_runs),
            "due_run_index": float(run_index),
            "updates_applied_before_decision": float(updates_applied_before_decision),
            "pending_queue_size": float(len(pending_updates)),
        }

        if controller is not None and run.split in online_update_splits:
            if label_lag_runs <= 0:
                immediate_update = controller.update(
                    provider=run.provider,
                    fallback_threshold=static_threshold,
                    shift_score=shift_score,
                    delta_mcc=policy_minus_baseline,
                    baseline_axiom_rate=baseline_v_rate,
                    policy_axiom_rate=policy_v_rate,
                    candidate_rows=online_candidate_rows,
                    degrade_penalty=as_float(policy.get("degrade_penalty"), 1.0),
                    online_minus_static=float(online_minus_static),
                )
                controller_update.update(immediate_update)
                controller_update["update_applied"] = 1.0
                controller_update["update_scheduled"] = 0.0
                controller_update["lag_runs"] = 0.0
                controller_update["due_run_index"] = float(run_index)
                controller_update["pending_queue_size"] = float(len(pending_updates))

                online_update_events.append(
                    {
                        "source_run_id": run.run_id,
                        "source_provider": run.provider,
                        "source_split": run.split,
                        "source_run_index": float(run_index),
                        "due_run_index": float(run_index),
                        "applied_at_run_id": run.run_id,
                        "applied_at_run_index": float(run_index),
                        "post_stream": 0.0,
                        "controller_mode": args.controller_mode,
                        "update_applied": 1.0,
                        **immediate_update,
                    }
                )
            else:
                due_run_index = int(run_index + label_lag_runs)
                pending_updates.append(
                    {
                        "source_run_id": run.run_id,
                        "provider": run.provider,
                        "split": run.split,
                        "source_run_index": float(run_index),
                        "due_run_index": float(due_run_index),
                        "fallback_threshold": float(static_threshold),
                        "shift_score": float(shift_score),
                        "delta_mcc": float(policy_minus_baseline),
                        "baseline_axiom_rate": float(baseline_v_rate),
                        "policy_axiom_rate": float(policy_v_rate),
                        "candidate_rows": online_candidate_rows,
                        "degrade_penalty": as_float(policy.get("degrade_penalty"), 1.0),
                        "online_minus_static": float(online_minus_static),
                    }
                )
                controller_update["update_scheduled"] = 1.0
                controller_update["due_run_index"] = float(due_run_index)
                controller_update["pending_queue_size"] = float(len(pending_updates))

        online_trace_rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "controller_mode": args.controller_mode,
                **controller_update,
                "delta_mcc_baseline": float(delta_mcc_baseline),
                "delta_mcc_policy": float(delta_mcc_policy),
                "policy_minus_baseline_mcc": float(policy_minus_baseline),
                "static_policy_minus_baseline_mcc": float(static_policy_minus_baseline),
                "online_minus_static_mcc": float(online_minus_static),
                "baseline_axiom_violation_rate": float(baseline_v_rate),
                "policy_axiom_violation_rate": float(policy_v_rate),
                "update_applied": float(controller_update["update_applied"]),
            }
        )

        run_rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "controller_mode": args.controller_mode,
                "n_items": pre_metrics["total"],
                "delta_mcc_baseline": delta_mcc_baseline,
                "delta_mcc_policy": delta_mcc_policy,
                "policy_minus_baseline_mcc": policy_minus_baseline,
                "static_policy_minus_baseline_mcc": float(static_policy_minus_baseline),
                "online_minus_static_mcc": float(online_minus_static),
                "threshold_used": float(threshold_used),
                "threshold_before": float(controller_update["threshold_before"]),
                "threshold_after": float(controller_update["threshold_after"]),
                "delta_threshold": float(controller_update["delta_threshold"]),
                "threshold_after_primal": float(
                    controller_update["threshold_after_primal"]
                ),
                "shift_score": float(controller_update["shift_score"]),
                "utility": float(controller_update["utility"]),
                "harm_loss": float(controller_update["harm_loss"]),
                "risk_budget": float(controller_update["risk_budget"]),
                "lambda_harm": float(controller_update["lambda_harm"]),
                "rho_shift": float(controller_update["rho_shift"]),
                "sweep_applied": float(controller_update["sweep_applied"]),
                "sweep_threshold": float(controller_update["sweep_threshold"]),
                "sweep_objective": float(controller_update["sweep_objective"]),
                "sweep_center_objective": float(
                    controller_update["sweep_center_objective"]
                ),
                "sweep_objective_gain": float(
                    controller_update["sweep_objective_gain"]
                ),
                "sweep_mix_requested": float(controller_update["sweep_mix_requested"]),
                "sweep_mix_effective": float(controller_update["sweep_mix_effective"]),
                "step_multiplier": float(controller_update["step_multiplier"]),
                "eta_effective": float(controller_update["eta_effective"]),
                "non_degrade_signal": float(controller_update["non_degrade_signal"]),
                "non_degrade_guard_triggered": float(
                    controller_update["non_degrade_guard_triggered"]
                ),
                "non_degrade_rollback_applied": float(
                    controller_update["non_degrade_rollback_applied"]
                ),
                "alarm_triggered": float(controller_update["alarm_triggered"]),
                "update_applied": float(controller_update["update_applied"]),
                "update_scheduled": float(controller_update["update_scheduled"]),
                "lag_runs": float(controller_update["lag_runs"]),
                "due_run_index": float(controller_update["due_run_index"]),
                "updates_applied_before_decision": float(
                    controller_update["updates_applied_before_decision"]
                ),
                "pending_queue_size": float(controller_update["pending_queue_size"]),
                "cusum_harm": float(controller_update["cusum_harm"]),
                "ema_harm": float(controller_update["ema_harm"]),
                "ema_shift": float(controller_update["ema_shift"]),
                "delta_ba_baseline": baseline_metrics["balanced_accuracy"]
                - pre_metrics["balanced_accuracy"],
                "delta_ba_policy": policy_metrics["balanced_accuracy"]
                - pre_metrics["balanced_accuracy"],
                "row_flip_rate_baseline": run.baseline_row_flip_rate,
                "row_flip_rate_policy": policy_row_flip_rate,
                "veto_rate": policy_stats["veto_rate"],
                "kept_flips": policy_stats["kept_flips"],
                "vetoed_flips": policy_stats["vetoed_flips"],
                "mean_score_kept": policy_stats["mean_score_kept"],
                "mean_score_vetoed": policy_stats["mean_score_vetoed"],
                "pre_axiom_violation_rate": pre_v_rate,
                "baseline_axiom_violation_rate": baseline_v_rate,
                "policy_axiom_violation_rate": policy_v_rate,
                "baseline_axiom_violations": float(pre_v_count - baseline_v_count),
                "policy_axiom_violations": float(pre_v_count - policy_v_count),
            }
        )

    post_stream_updates = 0
    if controller is not None and label_lag_runs > 0 and pending_updates:
        for payload in sorted(
            pending_updates,
            key=lambda row: (
                as_float(row.get("due_run_index"), 0.0),
                as_float(row.get("source_run_index"), 0.0),
            ),
        ):
            update = controller.update(
                provider=str(payload["provider"]),
                fallback_threshold=as_float(payload["fallback_threshold"], 0.0),
                shift_score=as_float(payload["shift_score"], 0.0),
                delta_mcc=as_float(payload["delta_mcc"], 0.0),
                baseline_axiom_rate=as_float(payload["baseline_axiom_rate"], 0.0),
                policy_axiom_rate=as_float(payload["policy_axiom_rate"], 0.0),
                candidate_rows=payload.get("candidate_rows"),
                degrade_penalty=as_float(payload.get("degrade_penalty"), 1.0),
                online_minus_static=as_float(payload.get("online_minus_static"), 0.0),
            )
            online_update_events.append(
                {
                    "source_run_id": str(payload["source_run_id"]),
                    "source_provider": str(payload["provider"]),
                    "source_split": str(payload["split"]),
                    "source_run_index": float(payload["source_run_index"]),
                    "due_run_index": float(payload["due_run_index"]),
                    "applied_at_run_id": "__post_stream__",
                    "applied_at_run_index": float(len(runs)),
                    "post_stream": 1.0,
                    "controller_mode": args.controller_mode,
                    "update_applied": 1.0,
                    **update,
                }
            )
            post_stream_updates += 1
        pending_updates = []

    heldout_improvements = [
        as_float(row["policy_minus_baseline_mcc"])
        for row in run_rows
        if row["split"] == "heldout"
    ]
    heldout_sign = sign_test_two_sided(heldout_improvements)

    write_csv(
        out_dir / "m5_family_policy.csv",
        family_rows,
        fieldnames=[
            "family",
            "enabled",
            "n_candidates",
            "improved",
            "degraded",
            "neutral",
            "mean_utility",
        ],
    )
    write_csv(
        out_dir / "m5_cell_policy.csv",
        cell_rows,
        fieldnames=[
            "family",
            "margin_bucket",
            "support_bucket",
            "n_candidates",
            "improved",
            "degraded",
            "neutral",
            "mean_utility",
            "smoothed_utility",
        ],
    )
    if threshold_rows:
        write_csv(
            out_dir / "m5_threshold_sweep.csv",
            threshold_rows,
            fieldnames=[
                "threshold",
                "n_accepted",
                "accepted_improved",
                "accepted_degraded",
                "accepted_objective",
            ],
        )

    write_csv(
        out_dir / "m5_run_deltas.csv",
        run_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "controller_mode",
            "n_items",
            "threshold_used",
            "threshold_before",
            "threshold_after",
            "delta_threshold",
            "threshold_after_primal",
            "shift_score",
            "utility",
            "harm_loss",
            "risk_budget",
            "lambda_harm",
            "rho_shift",
            "sweep_applied",
            "sweep_threshold",
            "sweep_objective",
            "sweep_center_objective",
            "sweep_objective_gain",
            "sweep_mix_requested",
            "sweep_mix_effective",
            "step_multiplier",
            "eta_effective",
            "non_degrade_signal",
            "non_degrade_guard_triggered",
            "non_degrade_rollback_applied",
            "alarm_triggered",
            "update_applied",
            "update_scheduled",
            "lag_runs",
            "due_run_index",
            "updates_applied_before_decision",
            "pending_queue_size",
            "cusum_harm",
            "ema_harm",
            "ema_shift",
            "delta_mcc_baseline",
            "delta_mcc_policy",
            "policy_minus_baseline_mcc",
            "static_policy_minus_baseline_mcc",
            "online_minus_static_mcc",
            "delta_ba_baseline",
            "delta_ba_policy",
            "row_flip_rate_baseline",
            "row_flip_rate_policy",
            "veto_rate",
            "kept_flips",
            "vetoed_flips",
            "mean_score_kept",
            "mean_score_vetoed",
            "pre_axiom_violation_rate",
            "baseline_axiom_violation_rate",
            "policy_axiom_violation_rate",
            "baseline_axiom_violations",
            "policy_axiom_violations",
        ],
    )

    write_csv(
        out_dir / "m6_online_trace.csv",
        online_trace_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "controller_mode",
            "threshold_before",
            "threshold_after",
            "delta_threshold",
            "threshold_after_primal",
            "shift_score",
            "utility",
            "harm_loss",
            "risk_budget",
            "lambda_harm",
            "rho_shift",
            "sweep_applied",
            "sweep_threshold",
            "sweep_objective",
            "sweep_center_objective",
            "sweep_objective_gain",
            "sweep_mix_requested",
            "sweep_mix_effective",
            "step_multiplier",
            "eta_effective",
            "non_degrade_signal",
            "non_degrade_guard_triggered",
            "non_degrade_rollback_applied",
            "alarm_triggered",
            "update_applied",
            "update_scheduled",
            "lag_runs",
            "due_run_index",
            "updates_applied_before_decision",
            "pending_queue_size",
            "cusum_harm",
            "ema_harm",
            "ema_shift",
            "seen_batches",
            "alarms",
            "delta_mcc_baseline",
            "delta_mcc_policy",
            "policy_minus_baseline_mcc",
            "static_policy_minus_baseline_mcc",
            "online_minus_static_mcc",
            "baseline_axiom_violation_rate",
            "policy_axiom_violation_rate",
        ],
    )

    write_csv(
        out_dir / "m6_online_update_events.csv",
        online_update_events,
        fieldnames=[
            "source_run_id",
            "source_provider",
            "source_split",
            "source_run_index",
            "due_run_index",
            "applied_at_run_id",
            "applied_at_run_index",
            "post_stream",
            "controller_mode",
            "update_applied",
            "threshold_before",
            "threshold_after",
            "delta_threshold",
            "threshold_after_primal",
            "shift_score",
            "utility",
            "harm_loss",
            "risk_budget",
            "lambda_harm",
            "rho_shift",
            "sweep_applied",
            "sweep_threshold",
            "sweep_objective",
            "sweep_center_objective",
            "sweep_objective_gain",
            "sweep_mix_requested",
            "sweep_mix_effective",
            "step_multiplier",
            "eta_effective",
            "non_degrade_signal",
            "non_degrade_guard_triggered",
            "non_degrade_rollback_applied",
            "alarm_triggered",
            "cusum_harm",
            "ema_harm",
            "ema_shift",
            "seen_batches",
            "alarms",
        ],
    )

    policy_path = out_dir / "m5_policy.json"
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    controller_state_path: str | None = None
    if controller is not None:
        controller_state_json = out_dir / "m6_controller_state.json"
        controller_state_json.write_text(
            json.dumps(controller.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        controller_state_path = str(controller_state_json)

    controller_metric_rows = online_update_events if online_update_events else run_rows

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repair_dir": str(repair_dir),
        "split_map_csv": str(split_map_path),
        "default_split": str(args.default_split),
        "controller_mode": str(args.controller_mode),
        "controller_state_json_in": str(args.controller_state_json)
        if args.controller_state_json
        else None,
        "controller_state_json_out": controller_state_path,
        "online_update_splits": sorted(online_update_splits),
        "controller_config": controller_config.to_dict()
        if controller_config is not None
        else None,
        "online_label_lag_runs": int(label_lag_runs),
        "controller_total_updates_applied": float(len(online_update_events)),
        "controller_total_updates_scheduled": float(
            sum(as_float(row.get("update_scheduled"), 0.0) for row in run_rows)
        ),
        "controller_post_stream_updates": float(post_stream_updates),
        "controller_pending_updates_after_flush": float(len(pending_updates)),
        "controller_total_alarms": float(
            sum(
                as_float(row.get("alarm_triggered"), 0.0)
                for row in controller_metric_rows
            )
        ),
        "controller_mean_shift": float(
            statistics.mean(
                as_float(row.get("shift_score"), 0.0) for row in controller_metric_rows
            )
            if controller_metric_rows
            else 0.0
        ),
        "controller_mean_harm": float(
            statistics.mean(
                as_float(row.get("harm_loss"), 0.0) for row in controller_metric_rows
            )
            if controller_metric_rows
            else 0.0
        ),
        "policy_json": str(args.policy_json) if args.policy_json else None,
        "auto_provider_thresholds_json": bool(args.auto_provider_thresholds_json),
        "provider_thresholds_json": provider_thresholds_path,
        "provider_thresholds_source": provider_thresholds_source,
        "provider_thresholds": provider_thresholds,
        "provider_threshold_overrides_used": provider_override_hits,
        "auto_provider_reference_json": bool(args.auto_provider_reference_json),
        "provider_reference_json": provider_reference_path,
        "provider_reference_source": provider_reference_source,
        "provider_reference_keys": sorted(provider_reference_dists.keys()),
        "fitted_policy_json": str(policy_path),
        "policy_type": str(policy.get("policy_type", "instance_v1")),
        "policy_threshold": as_float(policy.get("threshold"), 0.0),
        "threshold_override": args.threshold_override,
        "policy_margin_step": as_float(policy.get("margin_step"), 0.0),
        "policy_support_cap": int(as_float(policy.get("support_cap"), 0.0)),
        "policy_shrinkage": as_float(policy.get("shrinkage"), 0.0),
        "policy_degrade_penalty": as_float(policy.get("degrade_penalty"), 1.0),
        "policy_min_family_samples": int(
            as_float(policy.get("min_family_samples"), args.min_family_samples)
        ),
        "n_policy_families": len(family_rows),
        "n_policy_cells": len(cell_rows),
        "n_runs": len(runs),
        "n_dev_runs": sum(1 for run in runs if run.split == "dev"),
        "n_heldout_runs": sum(1 for run in runs if run.split == "heldout"),
        "heldout_sign_test_policy_minus_baseline": heldout_sign,
        "config": config.to_dict(),
    }
    (out_dir / "m5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    build_report(
        out_path=out_dir / "M5_RESULTS.md",
        policy=policy,
        family_rows=family_rows,
        run_rows=run_rows,
        heldout_sign=heldout_sign,
    )

    print("M5 instance guardrail complete")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
