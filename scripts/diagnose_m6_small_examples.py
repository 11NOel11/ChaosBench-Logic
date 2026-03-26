#!/usr/bin/env python3
"""Small deterministic examples to explain M6 behavior.

This script is intentionally tiny and does not depend on full repair artifacts.
It demonstrates why threshold cliffs matter and how conservative sweep behavior helps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from chaosbench.repair.online_controller import (
    OnlineControllerConfig,
    OnlineTransferController,
)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def utility_at_threshold(
    candidate_rows: List[Mapping[str, Any]],
    threshold: float,
    degrade_penalty: float,
) -> float:
    accepted = [
        row
        for row in candidate_rows
        if as_float(row.get("enabled"), 0.0) > 0.0
        and as_float(row.get("score"), -1e9) >= float(threshold)
    ]
    return float(
        sum(
            as_float(row.get("improved"), 0.0)
            - float(degrade_penalty) * as_float(row.get("degraded"), 0.0)
            for row in accepted
        )
    )


def main() -> int:
    out_dir = (
        Path("workspace") / "deep_survey_2026-03-01" / "repair_v3" / "m6_small_examples"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cliff_rows = [
        {"score": 0.120000, "improved": 1.0, "degraded": 0.0, "enabled": 1.0},
        {"score": 0.102509, "improved": 0.0, "degraded": 1.0, "enabled": 1.0},
    ]
    utility_t011 = utility_at_threshold(cliff_rows, threshold=0.11, degrade_penalty=1.0)
    utility_t010 = utility_at_threshold(cliff_rows, threshold=0.10, degrade_penalty=1.0)

    tie_controller = OnlineTransferController(
        config=OnlineControllerConfig(
            threshold_step=0.1,
            sweep_radius=1,
            sweep_mix=1.0,
            eta0=0.0,
        )
    )
    tie_controller.threshold_for_provider("openai/gpt-4o", fallback_threshold=0.3)
    tie_update = tie_controller.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.3,
        shift_score=0.0,
        delta_mcc=0.0,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
        candidate_rows=[
            {"score": 0.0, "improved": 0.0, "degraded": 0.0, "enabled": 0.0}
        ],
        online_minus_static=0.0,
    )

    gated_controller = OnlineTransferController(
        config=OnlineControllerConfig(
            threshold_step=0.1,
            sweep_radius=1,
            sweep_mix=1.0,
            sweep_min_improvement=0.01,
            eta0=0.0,
        )
    )
    gated_controller.threshold_for_provider("openai/gpt-4o", fallback_threshold=0.3)
    gated_update = gated_controller.update(
        provider="openai/gpt-4o",
        fallback_threshold=0.3,
        shift_score=0.0,
        delta_mcc=0.0,
        baseline_axiom_rate=0.0,
        policy_axiom_rate=0.0,
        candidate_rows=[
            {"score": 0.0, "improved": 0.0, "degraded": 0.0, "enabled": 0.0}
        ],
        online_minus_static=0.0,
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cliff_case": {
            "description": "Single degraded candidate just below 0.11 threshold",
            "utility_at_t011": utility_t011,
            "utility_at_t010": utility_t010,
            "utility_drop_when_crossing_cliff": utility_t010 - utility_t011,
            "candidate_rows": cliff_rows,
        },
        "tie_break_case": {
            "description": "Flat sweep objective selects higher threshold on tie",
            "threshold_before": tie_update["threshold_before"],
            "sweep_threshold": tie_update["sweep_threshold"],
            "threshold_after": tie_update["threshold_after"],
            "sweep_objective_gain": tie_update["sweep_objective_gain"],
        },
        "min_gain_gate_case": {
            "description": "Sweep move blocked without objective gain",
            "threshold_before": gated_update["threshold_before"],
            "sweep_threshold": gated_update["sweep_threshold"],
            "threshold_after": gated_update["threshold_after"],
            "sweep_applied": gated_update["sweep_applied"],
            "sweep_objective_gain": gated_update["sweep_objective_gain"],
        },
    }

    out_path = out_dir / "m6_small_example_diagnostics.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("M6 small-example diagnostics complete")
    print(json.dumps({"out_path": str(out_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
