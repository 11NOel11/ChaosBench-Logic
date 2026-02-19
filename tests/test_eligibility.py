"""Tests for the system eligibility module."""

import pytest

from chaosbench.data.eligibility import (
    check_system_eligibility,
    get_eligible_systems,
    generate_eligibility_report,
    FAMILY_REQUIREMENTS,
    REGIME_ELIGIBLE_SYSTEMS,
)


def _make_system(system_id="lorenz63", chaotic=True, all_predicates=True):
    """Create a minimal system dict for testing."""
    truth = {
        "Chaotic": chaotic,
        "Deterministic": True,
        "PosLyap": chaotic,
        "Sensitive": chaotic,
        "StrangeAttr": chaotic,
        "PointUnpredictable": chaotic,
        "StatPredictable": True,
        "QuasiPeriodic": False,
        "Random": False,
        "FixedPointAttr": not chaotic,
        "Periodic": not chaotic,
    }
    if not all_predicates:
        del truth["Periodic"]
    return {
        "system_id": system_id,
        "name": f"{system_id} system",
        "truth_assignment": truth,
    }


def _make_indicators(system_id="lorenz63", has_k=True, has_pe=True, has_megno=False):
    """Create a minimal indicator dict for testing."""
    ind = {"system_id": system_id}
    if has_k:
        ind["zero_one_K"] = 0.95
    if has_pe:
        ind["permutation_entropy"] = 0.85
    if has_megno:
        ind["megno"] = 5.0
    return ind


class TestCheckSystemEligibility:
    def test_atomic_eligible_with_complete_truth(self):
        system = _make_system()
        ok, reason = check_system_eligibility(system, "atomic")
        assert ok
        assert reason == "eligible"

    def test_atomic_ineligible_missing_predicates(self):
        system = _make_system(all_predicates=False)
        ok, reason = check_system_eligibility(system, "atomic")
        assert not ok
        assert "missing predicates" in reason

    def test_indicator_diagnostics_eligible(self):
        system = _make_system()
        indicators = _make_indicators(has_k=True)
        ok, reason = check_system_eligibility(system, "indicator_diagnostics", indicators)
        assert ok

    def test_indicator_diagnostics_no_indicators(self):
        system = _make_system()
        ok, reason = check_system_eligibility(system, "indicator_diagnostics", None)
        assert not ok
        assert "no indicator data" in reason

    def test_cross_indicator_needs_2_indicators(self):
        system = _make_system()
        ind_1 = _make_indicators(has_k=True, has_pe=False)
        ok, reason = check_system_eligibility(system, "cross_indicator", ind_1)
        assert not ok
        assert "needs 2" in reason

        ind_2 = _make_indicators(has_k=True, has_pe=True)
        ok, reason = check_system_eligibility(system, "cross_indicator", ind_2)
        assert ok

    def test_regime_transition_eligible_system(self):
        system = _make_system(system_id="logistic")
        ok, reason = check_system_eligibility(system, "regime_transition")
        assert ok

    def test_regime_transition_ineligible_system(self):
        system = _make_system(system_id="unknown_system")
        ok, reason = check_system_eligibility(system, "regime_transition")
        assert not ok
        assert "no bifurcation metadata" in reason

    def test_unknown_family(self):
        system = _make_system()
        ok, reason = check_system_eligibility(system, "nonexistent_family")
        assert not ok
        assert "unknown family" in reason

    def test_all_truth_only_families_eligible(self):
        """All families that only need truth_assignment should pass."""
        system = _make_system()
        truth_only_families = [
            "atomic", "consistency_paraphrase", "perturbation_robustness",
            "multi_hop", "fol_inference", "adversarial", "extended_systems",
        ]
        for family in truth_only_families:
            ok, reason = check_system_eligibility(system, family)
            assert ok, f"{family} should be eligible but got: {reason}"


class TestGetEligibleSystems:
    def test_all_eligible_for_atomic(self):
        systems = {
            "sys1": _make_system("sys1"),
            "sys2": _make_system("sys2"),
        }
        eligible = get_eligible_systems(systems, "atomic")
        assert len(eligible) == 2

    def test_filters_ineligible(self):
        systems = {
            "sys1": _make_system("sys1"),
            "sys2": _make_system("sys2", all_predicates=False),
        }
        eligible = get_eligible_systems(systems, "atomic")
        assert len(eligible) == 1
        assert eligible[0] == "sys1"


class TestEligibilityReport:
    def test_report_structure(self):
        systems = {
            "lorenz63": _make_system("lorenz63"),
            "logistic": _make_system("logistic"),
        }
        indicators = {
            "lorenz63": _make_indicators("lorenz63"),
        }
        report = generate_eligibility_report(systems, indicators)

        assert "total_systems" in report
        assert report["total_systems"] == 2
        assert "per_family" in report
        assert "per_system" in report
        assert "atomic" in report["per_family"]
        assert report["per_family"]["atomic"]["eligible_count"] == 2

    def test_regime_eligibility_counts(self):
        systems = {
            "logistic": _make_system("logistic"),
            "unknown": _make_system("unknown"),
        }
        report = generate_eligibility_report(systems)
        # logistic is in BIFURCATION_DATA, unknown is not
        assert report["per_family"]["regime_transition"]["eligible_count"] == 1
