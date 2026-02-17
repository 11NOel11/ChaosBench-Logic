"""Indicator computation pipeline for all benchmark systems.

Computes 0-1 test, permutation entropy, and MEGNO for each of the
30 benchmark systems and writes results as separate JSON files.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from chaosbench.data.indicators.time_series import (
    SYSTEM_REGISTRY,
    _DEFAULT_PARAMS,
    generate_map_trajectory,
    generate_ode_trajectory,
    get_default_ic,
    get_system_type,
)
from chaosbench.data.indicators.zero_one_test import zero_one_test
from chaosbench.data.indicators.permutation_entropy import permutation_entropy
from chaosbench.data.indicators.megno import compute_megno


ALL_SYSTEMS = [
    "lorenz63",
    "rossler",
    "chen_system",
    "chua_circuit",
    "double_pendulum",
    "duffing_chaotic",
    "fitzhugh_nagumo",
    "hindmarsh_rose",
    "lorenz84",
    "lorenz96",
    "lotka_volterra",
    "mackey_glass",
    "brusselator",
    "damped_oscillator",
    "damped_driven_pendulum_nonchaotic",
    "oregonator",
    "rikitake_dynamo",
    "shm",
    "vdp",
    "logistic_r4",
    "logistic_r2_8",
    "henon",
    "ikeda_map",
    "arnold_cat_map",
    "bakers_map",
    "circle_map_quasiperiodic",
    "standard_map",
    "kuramoto_sivashinsky",
    "stochastic_ou",
    "sine_gordon",
]


def get_dysts_systems(dysts_dir: str = "systems/dysts") -> list[str]:
    """Get list of dysts system IDs from disk.

    Args:
        dysts_dir: Directory containing dysts system JSON files.

    Returns:
        Sorted list of dysts system IDs.
    """
    if not os.path.isdir(dysts_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(dysts_dir)
        if f.endswith(".json") and f.startswith("dysts_")
    )


def _generate_scalar_series(
    system_id: str,
    params: Dict[str, float],
    seed: int,
) -> np.ndarray:
    """Generate a 1D scalar time series from the first component.

    Args:
        system_id: System identifier.
        params: System parameters.
        seed: Random seed for reproducibility.

    Returns:
        1D numpy array of the first state variable over time.
    """
    sys_type = get_system_type(system_id)
    if sys_type == "ode":
        traj = generate_ode_trajectory(
            system_id, params, t_span=(0, 100), n_points=10000, seed=seed
        )
    else:
        traj = generate_map_trajectory(
            system_id, params, n_iter=10000, seed=seed
        )
    return traj[:, 0]


def compute_all_indicators(
    system_id: str,
    params: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute all chaos indicators for a single system.

    Runs 0-1 test, permutation entropy, and MEGNO (where applicable).
    Uses the first component of the trajectory for scalar indicators.

    Args:
        system_id: System identifier.
        params: System parameters. Uses defaults from time_series module if None.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: system_id, zero_one_K, permutation_entropy, megno,
        system_type, seed, timestamp.
        Values may be None if computation is not applicable or fails.
    """
    if params is None:
        params = dict(_DEFAULT_PARAMS.get(system_id, {}))

    sys_type = get_system_type(system_id)
    series = _generate_scalar_series(system_id, params, seed)

    zero_one_K = None
    try:
        zero_one_K = zero_one_test(series, seed=seed)
    except Exception:
        pass

    perm_entropy = None
    try:
        perm_entropy = permutation_entropy(series, order=3, delay=1)
    except Exception:
        pass

    megno_val = None
    megno_reason = None
    try:
        ic = get_default_ic(system_id)
        megno_val = compute_megno(
            system_id, params, ic, seed=seed, validate=True, max_abs_megno=50.0
        )
        if megno_val is None:
            megno_reason = "computation_failed_or_invalid"
    except Exception as e:
        megno_reason = f"exception_{type(e).__name__}"

    result = {
        "system_id": system_id,
        "zero_one_K": zero_one_K,
        "permutation_entropy": perm_entropy,
        "megno": megno_val,
        "system_type": sys_type,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add metadata about why MEGNO failed if applicable
    if megno_val is None and megno_reason is not None:
        result["megno_failure_reason"] = megno_reason

    return result


def populate_all_systems(
    systems_dir: str = "systems",
    output_dir: str = "systems/indicators",
    seed: int = 42,
) -> None:
    """Compute indicators for all systems and save as JSON files.

    Does NOT modify original system JSONs. Writes separate indicator files
    as {system_id}_indicators.json in the output directory.

    Args:
        systems_dir: Directory containing system JSON files.
        output_dir: Directory to write indicator JSON files.
        seed: Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    for system_id in ALL_SYSTEMS:
        json_path = os.path.join(systems_dir, f"{system_id}.json")
        params = None
        if os.path.isfile(json_path):
            with open(json_path, "r") as fh:
                sys_data = json.load(fh)
            params = sys_data.get("parameters", sys_data.get("params", None))

        result = compute_all_indicators(system_id, params=params, seed=seed)

        out_path = os.path.join(output_dir, f"{system_id}_indicators.json")
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
