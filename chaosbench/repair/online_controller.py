"""Online controller for safe transfer under context shift."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from chaosbench.repair.selective import FlipCandidate


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _provider_key(provider: str) -> str:
    text = str(provider or "").strip().lower()
    return text if text else "unknown"


def _cell_key(family: str, margin_bucket: int, support_bucket: int) -> str:
    return f"{family}|m{margin_bucket}|s{support_bucket}"


def _margin_bucket(margin: float, margin_step: float) -> int:
    clipped = min(1.0, max(0.0, float(margin)))
    n_steps = int(round(1.0 / margin_step))
    bucket = int(clipped / margin_step)
    return min(bucket, n_steps)


def _support_bucket(support: int, support_cap: int) -> int:
    return min(max(0, int(support)), support_cap)


def normalize_distribution(raw: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, _as_float(value, 0.0)) for value in raw.values())
    if total <= 0.0:
        return {}
    return {
        str(key): max(0.0, _as_float(value, 0.0)) / total
        for key, value in raw.items()
        if max(0.0, _as_float(value, 0.0)) > 0.0
    }


def normalize_provider_distributions(
    raw: Mapping[str, Mapping[str, float]] | None,
) -> Dict[str, Dict[str, float]]:
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for key, value in raw.items():
        provider = _provider_key(str(key))
        if not isinstance(value, Mapping):
            continue
        dist = normalize_distribution(value)
        if dist:
            out[provider] = dist
    return out


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    p_norm = normalize_distribution(p)
    q_norm = normalize_distribution(q)
    if not p_norm and not q_norm:
        return 0.0

    keys = set(p_norm.keys()) | set(q_norm.keys())
    m: Dict[str, float] = {}
    for key in keys:
        m[key] = 0.5 * (p_norm.get(key, 0.0) + q_norm.get(key, 0.0))

    def kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        out = 0.0
        for key in keys:
            ai = max(0.0, _as_float(a.get(key, 0.0), 0.0))
            bi = max(1e-12, _as_float(b.get(key, 0.0), 0.0))
            if ai > 0.0:
                out += ai * math.log(ai / bi)
        return out

    jsd = 0.5 * kl(p_norm, m) + 0.5 * kl(q_norm, m)
    return float(jsd / math.log(2.0))


def reference_distribution_from_policy(policy: Mapping[str, Any]) -> Dict[str, float]:
    cells = policy.get("cells", {})
    if not isinstance(cells, dict):
        return {}
    raw: Dict[str, float] = {}
    for row in cells.values():
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "unknown")
        margin_bucket = _as_int(row.get("margin_bucket"), 0)
        support_bucket = _as_int(row.get("support_bucket"), 0)
        count = max(0.0, _as_float(row.get("n_candidates"), 0.0))
        if count <= 0.0:
            continue
        key = _cell_key(family, margin_bucket, support_bucket)
        raw[key] = raw.get(key, 0.0) + count
    return normalize_distribution(raw)


def candidate_distribution(
    candidates: Sequence[FlipCandidate],
    margin_step: float,
    support_cap: int,
) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for candidate in candidates:
        family = str(candidate.family or "unknown")
        margin_bucket = _margin_bucket(candidate.margin, margin_step)
        support_bucket = _support_bucket(candidate.support, support_cap)
        key = _cell_key(family, margin_bucket, support_bucket)
        raw[key] = raw.get(key, 0.0) + 1.0
    return normalize_distribution(raw)


@dataclass
class OnlineControllerConfig:
    threshold_min: float = 0.0
    threshold_max: float = 1.5
    eta0: float = 0.5
    threshold_step: float = 0.05
    eps_mcc: float = 0.0
    eps_axiom: float = 0.0
    axiom_penalty: float = 0.0
    harm_axiom_penalty: float = 1.0
    risk_budget_b0: float = 0.001
    shift_kappa: float = 2.0
    shift_target: float = 0.05
    lambda_max: float = 50.0
    rho_max: float = 50.0
    alarm_drift: float = 0.0
    alarm_threshold: float = 0.02
    emergency_step: float = 0.1
    ema_alpha: float = 0.1
    sweep_radius: int = 2
    sweep_mix: float = 0.5
    sweep_min_improvement: float = 0.0
    provider_step_default: float = 1.0
    provider_step_multipliers: Dict[str, float] = field(default_factory=dict)
    enforce_non_degrade_guard: bool = True
    non_degrade_margin: float = 0.0
    non_degrade_rollback_step: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold_min": float(self.threshold_min),
            "threshold_max": float(self.threshold_max),
            "eta0": float(self.eta0),
            "threshold_step": float(self.threshold_step),
            "eps_mcc": float(self.eps_mcc),
            "eps_axiom": float(self.eps_axiom),
            "axiom_penalty": float(self.axiom_penalty),
            "harm_axiom_penalty": float(self.harm_axiom_penalty),
            "risk_budget_b0": float(self.risk_budget_b0),
            "shift_kappa": float(self.shift_kappa),
            "shift_target": float(self.shift_target),
            "lambda_max": float(self.lambda_max),
            "rho_max": float(self.rho_max),
            "alarm_drift": float(self.alarm_drift),
            "alarm_threshold": float(self.alarm_threshold),
            "emergency_step": float(self.emergency_step),
            "ema_alpha": float(self.ema_alpha),
            "sweep_radius": float(self.sweep_radius),
            "sweep_mix": float(self.sweep_mix),
            "sweep_min_improvement": float(self.sweep_min_improvement),
            "provider_step_default": float(self.provider_step_default),
            "provider_step_multipliers": {
                str(key): float(value)
                for key, value in sorted(self.provider_step_multipliers.items())
            },
            "enforce_non_degrade_guard": bool(self.enforce_non_degrade_guard),
            "non_degrade_margin": float(self.non_degrade_margin),
            "non_degrade_rollback_step": float(self.non_degrade_rollback_step),
        }


@dataclass
class ProviderControllerState:
    threshold: float
    lambda_harm: float = 0.0
    rho_shift: float = 0.0
    cusum_harm: float = 0.0
    seen_batches: int = 0
    alarms: int = 0
    ema_harm: float = 0.0
    ema_shift: float = 0.0
    last_harm: float = 0.0
    last_shift: float = 0.0
    last_budget: float = 0.0
    last_utility: float = 0.0
    last_alarm: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "threshold": float(self.threshold),
            "lambda_harm": float(self.lambda_harm),
            "rho_shift": float(self.rho_shift),
            "cusum_harm": float(self.cusum_harm),
            "seen_batches": float(self.seen_batches),
            "alarms": float(self.alarms),
            "ema_harm": float(self.ema_harm),
            "ema_shift": float(self.ema_shift),
            "last_harm": float(self.last_harm),
            "last_shift": float(self.last_shift),
            "last_budget": float(self.last_budget),
            "last_utility": float(self.last_utility),
            "last_alarm": float(self.last_alarm),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderControllerState":
        return cls(
            threshold=_as_float(payload.get("threshold"), 0.0),
            lambda_harm=_as_float(payload.get("lambda_harm"), 0.0),
            rho_shift=_as_float(payload.get("rho_shift"), 0.0),
            cusum_harm=_as_float(payload.get("cusum_harm"), 0.0),
            seen_batches=_as_int(payload.get("seen_batches"), 0),
            alarms=_as_int(payload.get("alarms"), 0),
            ema_harm=_as_float(payload.get("ema_harm"), 0.0),
            ema_shift=_as_float(payload.get("ema_shift"), 0.0),
            last_harm=_as_float(payload.get("last_harm"), 0.0),
            last_shift=_as_float(payload.get("last_shift"), 0.0),
            last_budget=_as_float(payload.get("last_budget"), 0.0),
            last_utility=_as_float(payload.get("last_utility"), 0.0),
            last_alarm=_as_float(payload.get("last_alarm"), 0.0),
        )


class OnlineTransferController:
    """Primal-dual controller for online safe transfer adaptation."""

    def __init__(
        self,
        config: OnlineControllerConfig,
        reference_dist: Mapping[str, float] | None = None,
        provider_reference_dists: Mapping[str, Mapping[str, float]] | None = None,
    ):
        self.config = config
        self.reference_dist = normalize_distribution(reference_dist or {})
        self.provider_reference_dists = normalize_provider_distributions(
            provider_reference_dists
        )
        self.states: Dict[str, ProviderControllerState] = {}

    @classmethod
    def from_policy(
        cls,
        policy: Mapping[str, Any],
        config: OnlineControllerConfig,
        provider_reference_dists: Mapping[str, Mapping[str, float]] | None = None,
    ) -> "OnlineTransferController":
        return cls(
            config=config,
            reference_dist=reference_distribution_from_policy(policy),
            provider_reference_dists=provider_reference_dists,
        )

    def _reference_for_provider(self, provider: str) -> Mapping[str, float]:
        key = _provider_key(provider)
        direct = self.provider_reference_dists.get(key)
        if direct:
            return direct
        base = key.split("/", 1)[0]
        by_base = self.provider_reference_dists.get(base)
        if by_base:
            return by_base
        return self.reference_dist

    def _state_for_provider(
        self,
        provider: str,
        fallback_threshold: float,
    ) -> ProviderControllerState:
        key = _provider_key(provider)
        state = self.states.get(key)
        if state is None:
            threshold = _clip(
                float(fallback_threshold),
                self.config.threshold_min,
                self.config.threshold_max,
            )
            state = ProviderControllerState(threshold=threshold)
            self.states[key] = state
        return state

    def _step_multiplier_for_provider(self, provider: str) -> float:
        key = _provider_key(provider)
        multiplier = self.config.provider_step_multipliers.get(key)
        if multiplier is None:
            base = key.split("/", 1)[0]
            multiplier = self.config.provider_step_multipliers.get(
                base,
                self.config.provider_step_default,
            )
        return max(0.0, float(multiplier))

    def threshold_for_provider(
        self,
        provider: str,
        fallback_threshold: float,
    ) -> float:
        state = self._state_for_provider(provider, fallback_threshold)
        return float(state.threshold)

    def shift_score(
        self,
        candidates: Sequence[FlipCandidate],
        margin_step: float,
        support_cap: int,
        provider: str | None = None,
    ) -> float:
        reference = self._reference_for_provider(str(provider or ""))
        if not reference:
            return 0.0
        batch_dist = candidate_distribution(candidates, margin_step, support_cap)
        if not batch_dist:
            return 0.0
        return float(js_divergence(batch_dist, reference))

    def update(
        self,
        provider: str,
        fallback_threshold: float,
        shift_score: float,
        delta_mcc: float,
        baseline_axiom_rate: float,
        policy_axiom_rate: float,
        candidate_rows: Sequence[Mapping[str, Any]] | None = None,
        degrade_penalty: float = 1.0,
        online_minus_static: float | None = None,
    ) -> Dict[str, float]:
        state = self._state_for_provider(provider, fallback_threshold)
        threshold_before = float(state.threshold)
        shift = max(0.0, float(shift_score))

        axiom_regression = max(
            0.0, float(policy_axiom_rate) - float(baseline_axiom_rate)
        )
        utility = float(delta_mcc) - self.config.axiom_penalty * axiom_regression
        harm = max(0.0, -float(delta_mcc) - self.config.eps_mcc)
        harm += self.config.harm_axiom_penalty * max(
            0.0,
            axiom_regression - self.config.eps_axiom,
        )
        budget = self.config.risk_budget_b0 * math.exp(-self.config.shift_kappa * shift)

        step_multiplier = self._step_multiplier_for_provider(provider)
        eta_base = self.config.eta0 / math.sqrt(max(1.0, 1.0 + state.seen_batches))
        eta = eta_base * step_multiplier
        state.lambda_harm = _clip(
            state.lambda_harm + eta * (harm - budget),
            0.0,
            self.config.lambda_max,
        )
        state.rho_shift = _clip(
            state.rho_shift + eta * (shift - self.config.shift_target),
            0.0,
            self.config.rho_max,
        )

        tighten_force = state.lambda_harm + state.rho_shift * max(
            0.0,
            shift - self.config.shift_target,
        )
        relax_force = max(0.0, utility)
        step = self.config.threshold_step * eta
        delta_threshold = step * (tighten_force - relax_force)
        threshold_after_primal = _clip(
            state.threshold + delta_threshold,
            self.config.threshold_min,
            self.config.threshold_max,
        )
        state.threshold = threshold_after_primal

        sweep_applied = 0.0
        sweep_threshold = float(state.threshold)
        sweep_objective = 0.0
        sweep_center_objective = 0.0
        sweep_objective_gain = 0.0
        sweep_mix_requested = _clip(float(self.config.sweep_mix), 0.0, 1.0)
        sweep_mix_effective = 0.0
        if candidate_rows:
            candidate_list = [row for row in candidate_rows if isinstance(row, Mapping)]
            sweep_eval = self._local_threshold_sweep(
                threshold_center=float(state.threshold),
                lambda_harm=float(state.lambda_harm),
                rho_shift=float(state.rho_shift),
                shift=float(shift),
                budget=float(budget),
                candidate_rows=candidate_list,
                degrade_penalty=float(degrade_penalty),
            )
            sweep_threshold = float(sweep_eval["best_threshold"])
            sweep_objective = float(sweep_eval["best_objective"])
            sweep_center_objective = float(sweep_eval["center_objective"])
            sweep_objective_gain = float(sweep_eval["objective_gain"])
            sweep_mix_effective = _clip(
                float(sweep_mix_requested) * float(step_multiplier),
                0.0,
                1.0,
            )
            if sweep_mix_effective > 0.0 and sweep_objective_gain >= float(
                self.config.sweep_min_improvement
            ):
                blended = (1.0 - sweep_mix_effective) * float(
                    state.threshold
                ) + sweep_mix_effective * sweep_threshold
                state.threshold = _clip(
                    blended,
                    self.config.threshold_min,
                    self.config.threshold_max,
                )
                sweep_applied = 1.0

        non_degrade_signal = (
            float(online_minus_static)
            if online_minus_static is not None
            else float(delta_mcc)
        )
        non_degrade_guard_triggered = 0.0
        non_degrade_rollback_applied = 0.0
        if (
            bool(self.config.enforce_non_degrade_guard)
            and float(non_degrade_signal) < -float(self.config.non_degrade_margin)
            and float(state.threshold) < float(threshold_before)
        ):
            rollback_target = max(
                float(threshold_before),
                float(threshold_before) + float(self.config.non_degrade_rollback_step),
            )
            rollback_target = _clip(
                rollback_target,
                self.config.threshold_min,
                self.config.threshold_max,
            )
            non_degrade_rollback_applied = max(
                0.0,
                float(rollback_target) - float(state.threshold),
            )
            state.threshold = rollback_target
            non_degrade_guard_triggered = 1.0

        state.cusum_harm = max(
            0.0,
            state.cusum_harm + harm - budget - self.config.alarm_drift,
        )
        alarm = state.cusum_harm > self.config.alarm_threshold
        if alarm:
            state.alarms += 1
            state.threshold = _clip(
                state.threshold + self.config.emergency_step,
                self.config.threshold_min,
                self.config.threshold_max,
            )
            state.cusum_harm = 0.0

        alpha = _clip(self.config.ema_alpha, 0.0, 1.0)
        state.ema_harm = alpha * harm + (1.0 - alpha) * state.ema_harm
        state.ema_shift = alpha * shift + (1.0 - alpha) * state.ema_shift
        state.last_harm = float(harm)
        state.last_shift = float(shift)
        state.last_budget = float(budget)
        state.last_utility = float(utility)
        state.last_alarm = 1.0 if alarm else 0.0
        state.seen_batches += 1

        return {
            "threshold_before": float(threshold_before),
            "threshold_after": float(state.threshold),
            "delta_threshold": float(state.threshold - threshold_before),
            "threshold_after_primal": float(threshold_after_primal),
            "shift_score": float(shift),
            "utility": float(utility),
            "harm_loss": float(harm),
            "risk_budget": float(budget),
            "lambda_harm": float(state.lambda_harm),
            "rho_shift": float(state.rho_shift),
            "sweep_applied": float(sweep_applied),
            "sweep_threshold": float(sweep_threshold),
            "sweep_objective": float(sweep_objective),
            "sweep_center_objective": float(sweep_center_objective),
            "sweep_objective_gain": float(sweep_objective_gain),
            "sweep_mix_requested": float(sweep_mix_requested),
            "sweep_mix_effective": float(sweep_mix_effective),
            "step_multiplier": float(step_multiplier),
            "eta_effective": float(eta),
            "non_degrade_signal": float(non_degrade_signal),
            "non_degrade_guard_triggered": float(non_degrade_guard_triggered),
            "non_degrade_rollback_applied": float(non_degrade_rollback_applied),
            "alarm_triggered": 1.0 if alarm else 0.0,
            "cusum_harm": float(state.cusum_harm),
            "ema_harm": float(state.ema_harm),
            "ema_shift": float(state.ema_shift),
            "seen_batches": float(state.seen_batches),
            "alarms": float(state.alarms),
        }

    def _local_threshold_sweep(
        self,
        threshold_center: float,
        lambda_harm: float,
        rho_shift: float,
        shift: float,
        budget: float,
        candidate_rows: Sequence[Mapping[str, Any]],
        degrade_penalty: float,
    ) -> Dict[str, float]:
        step = max(1e-9, float(self.config.threshold_step))
        radius = max(0, int(self.config.sweep_radius))
        thresholds = [
            _clip(
                float(threshold_center) + float(offset) * step,
                self.config.threshold_min,
                self.config.threshold_max,
            )
            for offset in range(-radius, radius + 1)
        ]
        thresholds = sorted(set(thresholds))
        if not thresholds:
            thresholds = [float(threshold_center)]

        rho_penalty = max(0.0, float(self.config.shift_target) - float(shift))
        objective_by_threshold: Dict[float, float] = {}
        best = {
            "best_threshold": float(threshold_center),
            "best_objective": -1e9,
        }
        for threshold in thresholds:
            accepted = [
                row
                for row in candidate_rows
                if _as_float(row.get("enabled"), 0.0) > 0.0
                and _as_float(row.get("score"), -1e9) >= float(threshold)
            ]
            utility = sum(
                _as_float(row.get("improved"), 0.0)
                - float(degrade_penalty) * _as_float(row.get("degraded"), 0.0)
                for row in accepted
            )
            degraded = sum(_as_float(row.get("degraded"), 0.0) for row in accepted)
            denom = max(1.0, float(len(candidate_rows)))
            harm_rate = float(degraded) / denom
            risk_over = max(0.0, harm_rate - float(budget))
            objective = (
                float(utility)
                - float(lambda_harm) * risk_over
                - float(rho_shift) * rho_penalty
            )
            objective_by_threshold[float(threshold)] = float(objective)
            tie_break = float(threshold)
            score = (float(objective), tie_break)
            best_score = (float(best["best_objective"]), float(best["best_threshold"]))
            if score > best_score:
                best = {
                    "best_threshold": float(threshold),
                    "best_objective": float(objective),
                }
        closest_center_threshold = min(
            thresholds,
            key=lambda threshold: abs(float(threshold) - float(threshold_center)),
        )
        center_objective = float(
            objective_by_threshold.get(float(closest_center_threshold), 0.0)
        )
        best["center_threshold"] = float(closest_center_threshold)
        best["center_objective"] = float(center_objective)
        best["objective_gain"] = float(best["best_objective"] - center_objective)
        return best

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "reference_dist": self.reference_dist,
            "provider_reference_dists": self.provider_reference_dists,
            "states": {
                key: state.to_dict() for key, state in sorted(self.states.items())
            },
        }

    def load_dict(self, payload: Mapping[str, Any]) -> None:
        reference_dist = payload.get("reference_dist", {})
        if isinstance(reference_dist, Mapping):
            self.reference_dist = normalize_distribution(reference_dist)

        provider_reference_dists = payload.get("provider_reference_dists", {})
        if isinstance(provider_reference_dists, Mapping):
            self.provider_reference_dists = normalize_provider_distributions(
                provider_reference_dists
            )

        states = payload.get("states", {})
        if not isinstance(states, dict):
            return
        self.states = {}
        for key, row in states.items():
            if not isinstance(row, dict):
                continue
            self.states[str(key)] = ProviderControllerState.from_dict(row)
