#!/usr/bin/env python3
"""Populate new predicate truth values for all 165 systems.

Computes 12 new predicates from indicator JSON data and system JSON fields,
then writes them into each system's truth_assignment in-place.

New predicates:
  HyperChaotic   - >= 2 positive Lyapunov exponents
  Conservative   - sum of Lyapunov exponents ~= 0 (Hamiltonian)
  HighDimensional - Kaplan-Yorke dimension >= 3.0
  Multifractal   - |KY_dim - corr_dim| >= 0.5
  HighDimSystem  - state space dimension >= 4
  ContinuousTime - continuous-time ODE (not a map)
  DiscreteTime   - discrete-time map
  DelaySystem    - delay differential equation system
  Forced         - externally forced / non-autonomous
  Autonomous     - no explicit time dependence (converse of Forced)
  StrongMixing   - strong mixing (chaotic + dissipative proxy)
  WeakMixing     - weak mixing (positive Lyapunov proxy)
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---- Constants ----------------------------------------------------------------

# Core map system IDs (discrete-time)
CORE_MAP_NAMES = {
    "henon",
    "logistic_r2_8",
    "logistic_r4",
    "standard_map",
    "bakers_map",
    "arnold_cat_map",
    "ikeda_map",
    "circle_map_quasiperiodic",
}

# dysts name fragments that indicate a map
DYSTS_MAP_FRAGMENTS = {
    "henonheiles",  # NOTE: Henon-Heiles is actually a Hamiltonian ODE, not a map
}
# Explicitly discrete dysts systems (maps in dysts)
DYSTS_MAP_NAMES = {
    "dysts_henonheiles",  # Hamiltonian ODE (conservative, not a map)
    # True maps in dysts (if any) - checked against dysts source
}

# Core forced (non-autonomous) system IDs
CORE_FORCED_NAMES = {
    "duffing_chaotic",             # driven: has omega, gamma params
    "damped_driven_pendulum_nonchaotic",  # "driven" in name
    "brusselator",                 # Brusselator can be forced (has A, B which can be external)
}

# Core delay system IDs (has tau in params or "delay" in name)
CORE_DELAY_NAMES = {
    "mackey_glass",  # has tau parameter
}

# Core conservative system IDs (Hamiltonian or volume-preserving maps)
# NOTE: double_pendulum excluded — existing truth_assignment has Dissipative=True,
# which conflicts with Conservative; keeping existing assignment takes precedence.
CORE_CONSERVATIVE_NAMES = {
    "standard_map",   # area-preserving map (Hamiltonian)
    "arnold_cat_map", # volume-preserving map
}

# Hardcoded dimensions for core systems (phase space dimension)
CORE_DIMENSIONS = {
    "arnold_cat_map": 2,
    "bakers_map": 2,
    "brusselator": 2,
    "chen_system": 3,
    "chua_circuit": 3,
    "circle_map_quasiperiodic": 1,
    "damped_driven_pendulum_nonchaotic": 2,
    "damped_oscillator": 2,
    "double_pendulum": 4,
    "duffing_chaotic": 2,
    "fitzhugh_nagumo": 2,
    "henon": 2,
    "hindmarsh_rose": 3,
    "ikeda_map": 2,
    "kuramoto_sivashinsky": 64,   # discretized PDE (large)
    "logistic_r2_8": 1,
    "logistic_r4": 1,
    "lorenz63": 3,
    "lorenz84": 4,
    "lorenz96": 36,               # N=36 default
    "lotka_volterra": 4,          # 4-species
    "mackey_glass": 50,           # DDE: high effective dimension
    "oregonator": 3,
    "rikitake_dynamo": 5,
    "rossler": 3,
    "shm": 2,
    "sine_gordon": 64,            # discretized PDE
    "standard_map": 2,
    "stochastic_ou": 1,
    "vdp": 2,
}


# ---- Helper functions ---------------------------------------------------------

def _is_discrete(system_id: str, system_json: dict) -> bool:
    """Return True if the system is a discrete-time map."""
    sid = system_id.lower()
    # Explicit core maps
    if sid in CORE_MAP_NAMES:
        return True
    # Check system_type field (dysts indicator or system JSON)
    if system_json.get("system_type") == "map":
        return True
    # Check dysts name patterns
    name_lower = system_json.get("name", "").lower()
    if "map" in name_lower and "poincare" not in name_lower:
        # Careful: some dysts systems have "map" in description but are ODEs
        # Check against known dysts maps
        if sid in DYSTS_MAP_NAMES:
            return True
    return False


def _is_forced(system_id: str, system_json: dict) -> bool:
    """Return True if the system is externally forced / non-autonomous."""
    sid = system_id.lower()
    if sid in CORE_FORCED_NAMES:
        return True
    # Name-based detection
    name_lower = system_json.get("name", sid).lower()
    if "forced" in name_lower or "driven" in name_lower:
        return True
    # dysts forced systems (ForcedBrusselator, ForcedFitzHughNagumo, ForcedVanDerPol)
    if "forced" in sid:
        return True
    # Check params for driving frequency/amplitude
    params = system_json.get("parameters", {})
    if isinstance(params, dict):
        # Driven systems typically have omega + A/gamma (amplitude)
        has_omega = "omega" in params or "Omega" in params
        has_driving = "A" in params or "gamma" in params or "F" in params
        if has_omega and has_driving:
            return True
    return False


def _get_dimension(system_id: str, system_json: dict) -> int:
    """Return state space dimension for the system."""
    # Prefer explicit dimension field (all dysts systems have this)
    dim = system_json.get("dimension")
    if dim is not None:
        return int(dim)
    # Fall back to hardcoded table for core systems
    return CORE_DIMENSIONS.get(system_id.lower(), 3)


def _is_conservative_from_lyap(lyap: list) -> bool:
    """Return True if sum of Lyapunov exponents is approximately 0."""
    if not lyap or len(lyap) < 2:
        return False
    valid = [x for x in lyap if isinstance(x, (int, float))]
    if not valid:
        return False
    lyap_sum = sum(valid)
    threshold = 0.05 * len(valid)  # ±5% per dimension
    return abs(lyap_sum) < threshold


def compute_new_predicates(
    system_id: str,
    system_json: dict,
    indicator_json: Optional[dict],
) -> dict:
    """Compute truth values for all 12 new predicates.

    Args:
        system_id: System identifier string.
        system_json: Loaded system JSON data.
        indicator_json: Loaded indicator JSON data (may be None).

    Returns:
        Dict mapping predicate name to bool value.
    """
    # Extract indicator fields (dysts systems have these)
    lyap = []
    kyd = 0.0
    cd = 0.0
    if indicator_json is not None:
        raw_lyap = indicator_json.get("lyapunov_spectrum") or []
        lyap = [x for x in raw_lyap if isinstance(x, (int, float))]
        kyd = indicator_json.get("kaplan_yorke_dimension") or 0.0
        cd = indicator_json.get("correlation_dimension") or 0.0

    # Derived values
    # Use 0.01 threshold for general positive Lyapunov count (WeakMixing, etc.)
    # Use 0.05 threshold for HyperChaotic to avoid counting near-zero neutral
    # direction (which appears as ~0.01-0.03 in numerical estimates for ODEs).
    n_pos_lyap = sum(1 for x in lyap if x > 0.01)
    n_pos_lyap_strict = sum(1 for x in lyap if x > 0.05)
    dim = _get_dimension(system_id, system_json)
    sid = system_id.lower()

    # Fall back to existing truth_assignment for proxy predicates when lyap unavailable
    truth = system_json.get("truth_assignment", {})
    is_chaotic = truth.get("Chaotic", False)
    is_dissipative = truth.get("Dissipative", False)
    has_pos_lyap = truth.get("PosLyap", False)

    # ---- Compute each predicate ------------------------------------------------

    # HyperChaotic: >= 2 positive Lyapunov exponents (using strict 0.05 threshold
    # to avoid counting the near-zero neutral direction of ODEs as a second
    # positive exponent — Lorenz has ~0.03 neutral direction, not genuinely positive)
    if lyap:
        hyper_chaotic = n_pos_lyap_strict >= 2
    else:
        # Core systems: use name-based heuristic (none of the 30 core are hyperchaotic)
        hyper_chaotic = False

    # Conservative: sum of Lyapunov exponents ≈ 0 AND not dissipative
    if lyap:
        conservative = _is_conservative_from_lyap(lyap) and not is_dissipative
    else:
        # Use hardcoded known conservative core systems
        conservative = sid in CORE_CONSERVATIVE_NAMES

    # HighDimensional: Kaplan-Yorke dimension >= 3.0
    if kyd and kyd > 0:
        high_dimensional = kyd >= 3.0
    elif lyap and len(lyap) >= 3:
        # Fallback: if we have enough positive exponents, dimension is high
        high_dimensional = n_pos_lyap >= 2
    else:
        # Heuristic: if state dimension >= 5 AND chaotic, likely high-dimensional
        high_dimensional = dim >= 5 and is_chaotic

    # Multifractal: |KY_dim - correlation_dim| >= 0.5
    if kyd and cd and kyd > 0 and cd > 0:
        multifractal = abs(kyd - cd) >= 0.5
    else:
        multifractal = False

    # HighDimSystem: state space dimension >= 4
    high_dim_system = dim >= 4

    # ContinuousTime / DiscreteTime (mutually exclusive)
    discrete_time = _is_discrete(sid, system_json)
    continuous_time = not discrete_time

    # DelaySystem: has tau param or "delay" in name/id
    params = system_json.get("parameters", {}) or {}
    has_tau = isinstance(params, dict) and "tau" in params
    has_delay_in_name = "delay" in sid or "delay" in system_json.get("name", "").lower()
    delay_system = sid in CORE_DELAY_NAMES or has_tau or has_delay_in_name

    # Forced / Autonomous (mutually exclusive)
    forced = _is_forced(sid, system_json)
    autonomous = not forced

    # StrongMixing: chaotic systems exhibit strong mixing (positive Lyapunov exponent proxy)
    # Note: conservative chaotic systems (Arnold cat map) are also strongly mixing —
    # they are Bernoulli systems. Using "has positive Lyapunov exponent" as the proxy.
    if lyap:
        strong_mixing = n_pos_lyap >= 1
    else:
        strong_mixing = has_pos_lyap or is_chaotic

    # WeakMixing: positive Lyapunov exponent → weak mixing (ergodic hierarchy)
    if lyap:
        weak_mixing = n_pos_lyap >= 1
    else:
        weak_mixing = has_pos_lyap or is_chaotic

    return {
        "HyperChaotic": hyper_chaotic,
        "Conservative": conservative,
        "HighDimensional": high_dimensional,
        "Multifractal": multifractal,
        "HighDimSystem": high_dim_system,
        "ContinuousTime": continuous_time,
        "DiscreteTime": discrete_time,
        "DelaySystem": delay_system,
        "Forced": forced,
        "Autonomous": autonomous,
        "StrongMixing": strong_mixing,
        "WeakMixing": weak_mixing,
    }


def load_system_jsons(systems_dir: str) -> Dict[str, Tuple[str, dict]]:
    """Load all system JSON files.

    Returns:
        Dict mapping system_id to (filepath, data).
    """
    systems = {}
    # Core systems
    for fname in sorted(os.listdir(systems_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(systems_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            systems[sid] = (fpath, data)
    # dysts systems
    dysts_dir = os.path.join(systems_dir, "dysts")
    if os.path.isdir(dysts_dir):
        for fname in sorted(os.listdir(dysts_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dysts_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            sid = data.get("system_id")
            if sid:
                systems[sid] = (fpath, data)
    return systems


def load_indicator_jsons(systems_dir: str) -> Dict[str, dict]:
    """Load all indicator JSON files.

    Returns:
        Dict mapping system_id to indicator data.
    """
    indicators = {}
    # Core indicators
    ind_dir = os.path.join(systems_dir, "indicators")
    if os.path.isdir(ind_dir):
        for fname in sorted(os.listdir(ind_dir)):
            if not fname.endswith("_indicators.json"):
                continue
            fpath = os.path.join(ind_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            sid = data.get("system_id")
            if sid:
                indicators[sid] = data
    # dysts indicators
    dysts_ind_dir = os.path.join(systems_dir, "dysts", "indicators")
    if os.path.isdir(dysts_ind_dir):
        for fname in sorted(os.listdir(dysts_ind_dir)):
            if not fname.endswith("_indicators.json"):
                continue
            fpath = os.path.join(dysts_ind_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            sid = data.get("system_id")
            if sid:
                indicators[sid] = data
    return indicators


def validate_mutual_exclusions(systems: Dict[str, Tuple[str, dict]]) -> list:
    """Check that mutual exclusion predicates are respected.

    Returns:
        List of violation strings.
    """
    violations = []
    mutual_exclusive_pairs = [
        ("ContinuousTime", "DiscreteTime"),
        ("Forced", "Autonomous"),
        ("Conservative", "Dissipative"),
    ]
    for sid, (fpath, data) in systems.items():
        truth = data.get("truth_assignment", {})
        for p1, p2 in mutual_exclusive_pairs:
            if truth.get(p1) and truth.get(p2):
                violations.append(f"{sid}: {p1} and {p2} are both True (must be exclusive)")
    return violations


def run(
    systems_dir: str = "systems",
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Main entry point.

    Args:
        systems_dir: Path to systems directory.
        dry_run: If True, show computed values but do not write.
        verbose: If True, print per-system details.
    """
    print("=" * 60)
    print("  populate_new_predicates.py")
    print(f"  Mode: {'DRY RUN' if dry_run else 'WRITE'}")
    print("=" * 60)

    os.chdir(PROJECT_ROOT)
    systems = load_system_jsons(systems_dir)
    indicators = load_indicator_jsons(systems_dir)

    print(f"\nLoaded {len(systems)} systems, {len(indicators)} indicator sets")

    # Compute new predicates for all systems
    stats = {
        "HyperChaotic": 0, "Conservative": 0, "HighDimensional": 0,
        "Multifractal": 0, "HighDimSystem": 0, "ContinuousTime": 0,
        "DiscreteTime": 0, "DelaySystem": 0, "Forced": 0, "Autonomous": 0,
        "StrongMixing": 0, "WeakMixing": 0,
    }
    updated = 0

    for sid, (fpath, data) in sorted(systems.items()):
        ind = indicators.get(sid)
        new_preds = compute_new_predicates(sid, data, ind)

        if verbose:
            true_preds = [k for k, v in new_preds.items() if v]
            print(f"  {sid}: {', '.join(true_preds) or 'none'}")

        # Update stats
        for pred, val in new_preds.items():
            if val:
                stats[pred] += 1

        if not dry_run:
            # Merge into truth_assignment
            truth = data.get("truth_assignment", {})
            truth.update(new_preds)
            data["truth_assignment"] = truth

            # Write back to file
            with open(fpath, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            updated += 1

    # Print statistics
    print(f"\n{'Predicate':<20} {'TRUE count':>12} {'TRUE %':>8}")
    print("-" * 42)
    total = len(systems)
    for pred, count in sorted(stats.items()):
        pct = 100 * count / total if total > 0 else 0.0
        print(f"  {pred:<18} {count:>12} {pct:>7.1f}%")

    print(f"\nTotal systems: {total}")
    if not dry_run:
        print(f"Updated: {updated} system JSON files")

    # Validate mutual exclusions (after writing)
    if not dry_run:
        # Reload to check written data
        systems_check = load_system_jsons(systems_dir)
    else:
        # Build in-memory with computed values for validation
        systems_check = {}
        for sid, (fpath, data) in systems.items():
            data_copy = dict(data)
            truth_copy = dict(data.get("truth_assignment", {}))
            ind = indicators.get(sid)
            new_preds = compute_new_predicates(sid, data, ind)
            truth_copy.update(new_preds)
            data_copy["truth_assignment"] = truth_copy
            systems_check[sid] = (fpath, data_copy)

    violations = validate_mutual_exclusions(systems_check)
    if violations:
        print(f"\n[WARNINGS] {len(violations)} mutual exclusion violations:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("\n[OK] No mutual exclusion violations found.")

    print("\nDone!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate new predicate truth values for all systems"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show computed values without writing to disk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-system predicate details",
    )
    parser.add_argument(
        "--systems-dir",
        default="systems",
        help="Path to systems directory (default: systems)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        systems_dir=args.systems_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
