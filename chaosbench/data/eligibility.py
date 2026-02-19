"""System eligibility checking for v2.2 scaled generation.

Determines which systems are eligible for each question family based on
their available metadata (truth_assignment, indicators, regime metadata).

Eligibility Requirements per Family
------------------------------------
- atomic: truth_assignment with all 11 predicates
- consistency_paraphrase: truth_assignment
- perturbation_robustness: truth_assignment
- multi_hop: truth_assignment
- fol_inference: truth_assignment
- adversarial: truth_assignment
- indicator_diagnostics: truth_assignment + at least 1 numeric indicator
- cross_indicator: truth_assignment + at least 2 numeric indicators
- regime_transition: truth_assignment + bifurcation metadata
- extended_systems: truth_assignment
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from chaosbench.data.bifurcations import BIFURCATION_DATA
from chaosbench.logic.ontology import PREDICATES


# Original 11 predicates used for eligibility checks.
# Systems must have these 11 in their truth_assignment to be eligible.
# The full PREDICATES list (15) includes v2.2 extensions (Dissipative, Bounded,
# Mixing, Ergodic) which are NOT required for eligibility.
ELIGIBILITY_PREDICATES = [
    "Chaotic", "Deterministic", "PosLyap", "Sensitive", "StrangeAttr",
    "PointUnpredictable", "StatPredictable", "QuasiPeriodic", "Random",
    "FixedPointAttr", "Periodic",
]

# Core indicators used by indicator-dependent families
CORE_INDICATOR_KEYS = {
    "zero_one_K", "permutation_entropy", "megno",
    "lyapunov_spectrum", "kaplan_yorke_dimension",
    "correlation_dimension", "multiscale_entropy",
    "maximum_lyapunov_estimated",
}

# Dysts indicator keys (different from core)
DYSTS_INDICATOR_KEYS = {
    "lyapunov_spectrum",
    "kaplan_yorke_dimension",
    "correlation_dimension",
    "multiscale_entropy",
    "maximum_lyapunov_estimated",
}

# All recognized numeric indicator keys
ALL_INDICATOR_KEYS = CORE_INDICATOR_KEYS | DYSTS_INDICATOR_KEYS

# Systems with bifurcation metadata for regime_transition
REGIME_ELIGIBLE_SYSTEMS: Set[str] = set(BIFURCATION_DATA.keys())

# Families and their requirements
FAMILY_REQUIREMENTS = {
    "atomic": {"truth_assignment": True, "indicators": False, "regime_metadata": False},
    "consistency_paraphrase": {"truth_assignment": True, "indicators": False, "regime_metadata": False},
    "perturbation_robustness": {"truth_assignment": True, "indicators": False, "regime_metadata": False},
    "multi_hop": {"truth_assignment": True, "indicators": False, "regime_metadata": False},
    "fol_inference": {"truth_assignment": True, "indicators": False, "regime_metadata": False},
    "adversarial": {"truth_assignment": True, "indicators": False, "regime_metadata": False},
    "indicator_diagnostics": {"truth_assignment": True, "indicators": True, "min_indicators": 1, "regime_metadata": False},
    "cross_indicator": {"truth_assignment": True, "indicators": True, "min_indicators": 2, "regime_metadata": False},
    "regime_transition": {"truth_assignment": True, "indicators": False, "regime_metadata": True},
    "extended_systems": {"truth_assignment": True, "indicators": False, "regime_metadata": False},
}


def _count_numeric_indicators(
    indicators: Dict[str, Any],
) -> int:
    """Count how many numeric indicator values a system has.

    Args:
        indicators: Indicator dict for a system.

    Returns:
        Number of non-None numeric indicator values.
    """
    count = 0
    for key in ALL_INDICATOR_KEYS:
        val = indicators.get(key)
        if val is not None and isinstance(val, (int, float)):
            count += 1
        elif key == "lyapunov_spectrum" and isinstance(val, list) and len(val) > 0:
            count += 1
    return count


def _has_complete_truth_assignment(system: Dict[str, Any]) -> bool:
    """Check if a system has a complete truth_assignment with all 11 core predicates.

    Uses ELIGIBILITY_PREDICATES (original 11) rather than full PREDICATES (15)
    to avoid breaking eligibility when v2.2 extension predicates are absent.

    Args:
        system: System data dict.

    Returns:
        True if all 11 core predicates are present.
    """
    truth = system.get("truth_assignment", {})
    if not truth:
        return False
    return all(pred in truth for pred in ELIGIBILITY_PREDICATES)


def check_system_eligibility(
    system: Dict[str, Any],
    family: str,
    indicators: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Check whether a system is eligible for a given question family.

    Args:
        system: System data dict with at least system_id and truth_assignment.
        family: Question family name (e.g. "atomic", "indicator_diagnostics").
        indicators: Optional indicator data dict for this system.

    Returns:
        Tuple of (eligible: bool, reason: str).
        If eligible, reason is "eligible".
        If not eligible, reason describes why.
    """
    if family not in FAMILY_REQUIREMENTS:
        return False, f"unknown family: {family}"

    reqs = FAMILY_REQUIREMENTS[family]
    system_id = system.get("system_id", "unknown")

    # Check truth_assignment
    if reqs["truth_assignment"]:
        if not _has_complete_truth_assignment(system):
            missing = set(ELIGIBILITY_PREDICATES) - set(system.get("truth_assignment", {}).keys())
            return False, f"missing predicates in truth_assignment: {sorted(missing)}"

    # Check indicators
    if reqs.get("indicators", False):
        if indicators is None:
            return False, "no indicator data available"
        min_ind = reqs.get("min_indicators", 1)
        n_ind = _count_numeric_indicators(indicators)
        if n_ind < min_ind:
            return False, f"needs {min_ind} numeric indicators, has {n_ind}"

    # Check regime metadata
    if reqs.get("regime_metadata", False):
        if system_id not in REGIME_ELIGIBLE_SYSTEMS:
            return False, f"no bifurcation metadata for system {system_id}"

    return True, "eligible"


def get_eligible_systems(
    systems: Dict[str, Dict[str, Any]],
    family: str,
    indicators: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Get list of system IDs eligible for a given family.

    Args:
        systems: Dict mapping system_id to system data.
        family: Question family name.
        indicators: Optional dict mapping system_id to indicator data.

    Returns:
        Sorted list of eligible system IDs.
    """
    eligible = []
    for sid in sorted(systems.keys()):
        ind = (indicators or {}).get(sid)
        ok, _ = check_system_eligibility(systems[sid], family, ind)
        if ok:
            eligible.append(sid)
    return eligible


def generate_eligibility_report(
    systems: Dict[str, Dict[str, Any]],
    indicators: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive eligibility report for all systems and families.

    Args:
        systems: Dict mapping system_id to system data.
        indicators: Optional dict mapping system_id to indicator data.

    Returns:
        Dict with per-family eligible counts, per-system eligibility matrix,
        and summary statistics.
    """
    families = sorted(FAMILY_REQUIREMENTS.keys())
    system_ids = sorted(systems.keys())

    per_family: Dict[str, Dict[str, Any]] = {}
    for family in families:
        eligible_ids = get_eligible_systems(systems, family, indicators)
        per_family[family] = {
            "eligible_count": len(eligible_ids),
            "total_systems": len(system_ids),
            "eligible_systems": eligible_ids,
        }

    per_system: Dict[str, Dict[str, Any]] = {}
    for sid in system_ids:
        sys_data = systems[sid]
        ind = (indicators or {}).get(sid)
        eligibility: Dict[str, bool] = {}
        reasons: Dict[str, str] = {}
        for family in families:
            ok, reason = check_system_eligibility(sys_data, family, ind)
            eligibility[family] = ok
            reasons[family] = reason
        per_system[sid] = {
            "eligible_families": sum(1 for v in eligibility.values() if v),
            "total_families": len(families),
            "eligibility": eligibility,
            "reasons": reasons,
        }

    return {
        "total_systems": len(system_ids),
        "total_families": len(families),
        "per_family": per_family,
        "per_system": per_system,
    }
