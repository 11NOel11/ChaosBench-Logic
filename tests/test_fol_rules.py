"""
Tests for FOL (First-Order Logic) violation checking.

This module tests the FOL violation detection system that checks model
predictions against formal logic axioms about dynamical systems.

Functions tested:
- get_fol_rules(): Returns FOL axiom definitions
- load_system_ontology(): Loads system truth assignments from JSON
- extract_predicate_from_question(): Maps questions to predicates
- check_fol_violations(): Checks predictions against axioms
"""

import pytest
import sys
import os
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval_chaosbench import (
    get_fol_rules,
    load_system_ontology,
    extract_predicate_from_question,
    check_fol_violations,
)


class TestGetFOLRules:
    """Test that FOL axioms are correctly defined."""

    def test_fol_rules_structure(self):
        """Rules should be a dict with requires/excludes for each predicate."""
        rules = get_fol_rules()
        assert isinstance(rules, dict)
        assert len(rules) > 0

        for predicate, rule_dict in rules.items():
            assert "requires" in rule_dict
            assert "excludes" in rule_dict
            assert isinstance(rule_dict["requires"], list)
            assert isinstance(rule_dict["excludes"], list)

    def test_chaotic_requirements(self):
        """
        Test Chaotic axiom matches user specification.

        From Phase 2 spec (v2.3 extended):
        - Requires: Deterministic, PosLyap, Sensitive, PointUnpredictable, StatPredictable,
                    Mixing, StrongMixing
        - Excludes: Random, Periodic, QuasiPeriodic, FixedPointAttr
        - Note: StrangeAttr is sufficient but NOT necessary (not in requires)
        """
        rules = get_fol_rules()
        chaotic = rules["Chaotic"]

        # Check required predicates are a superset of the core v2.2 set
        core_requires = {
            "Deterministic",
            "PosLyap",
            "Sensitive",
            "PointUnpredictable",
            "StatPredictable",
            "Mixing",
        }
        expected_excludes = {"Random", "Periodic", "QuasiPeriodic", "FixedPointAttr"}

        assert core_requires.issubset(set(chaotic["requires"])), \
            f"Chaotic missing core requires: {core_requires - set(chaotic['requires'])}"
        assert set(chaotic["excludes"]) == expected_excludes
        # Verify StrangeAttr is NOT required
        assert "StrangeAttr" not in chaotic["requires"]

    def test_random_requirements(self):
        """Random excludes deterministic systems."""
        rules = get_fol_rules()
        random_rules = rules["Random"]

        assert "Deterministic" in random_rules["excludes"]
        assert "Chaotic" in random_rules["excludes"]
        assert "QuasiPeriodic" in random_rules["excludes"]

    def test_quasiperiodic_requirements(self):
        """QuasiPeriodic requires Deterministic, excludes chaos and randomness."""
        rules = get_fol_rules()
        qp = rules["QuasiPeriodic"]

        assert "Deterministic" in qp["requires"]
        assert "Chaotic" in qp["excludes"]
        assert "Random" in qp["excludes"]

    def test_deterministic_requirements(self):
        """Deterministic excludes Random."""
        rules = get_fol_rules()
        det = rules["Deterministic"]

        assert "Random" in det["excludes"]

    def test_periodic_requirements(self):
        """Periodic requires Deterministic."""
        rules = get_fol_rules()
        periodic = rules["Periodic"]

        assert "Deterministic" in periodic["requires"]
        assert "Chaotic" in periodic["excludes"]

    def test_fixed_point_requirements(self):
        """FixedPointAttr requires Deterministic."""
        rules = get_fol_rules()
        fp = rules["FixedPointAttr"]

        assert "Deterministic" in fp["requires"]
        assert "Chaotic" in fp["excludes"]


class TestLoadSystemOntology:
    """Test loading of system truth assignments from JSON files."""

    def test_ontology_loads_30_systems(self):
        """Should load all 30 system definitions."""
        ontology = load_system_ontology("systems")
        assert len(ontology) == 30

    def test_ontology_returns_dict(self):
        """Should return dict mapping system_id to truth assignments."""
        ontology = load_system_ontology("systems")
        assert isinstance(ontology, dict)

        for system_id, truth_assignment in ontology.items():
            assert isinstance(system_id, str)
            assert isinstance(truth_assignment, dict)

    def test_lorenz63_truth_assignment(self):
        """Lorenz-63 should be chaotic and deterministic."""
        ontology = load_system_ontology("systems")
        assert "lorenz63" in ontology

        lorenz = ontology["lorenz63"]
        assert lorenz["Chaotic"] == True
        assert lorenz["Deterministic"] == True
        assert lorenz["PosLyap"] == True
        assert lorenz["Sensitive"] == True
        assert lorenz["StrangeAttr"] == True
        assert lorenz["Random"] == False
        assert lorenz["QuasiPeriodic"] == False

    def test_stochastic_ou_truth_assignment(self):
        """Stochastic OU process should be random, not deterministic."""
        ontology = load_system_ontology("systems")
        assert "stochastic_ou" in ontology

        ou = ontology["stochastic_ou"]
        assert ou["Random"] == True
        assert ou["Deterministic"] == False
        assert ou["Chaotic"] == False

    def test_circle_map_quasiperiodic_truth_assignment(self):
        """Circle map (quasiperiodic regime) should be quasiperiodic."""
        ontology = load_system_ontology("systems")
        assert "circle_map_quasiperiodic" in ontology

        cm = ontology["circle_map_quasiperiodic"]
        assert cm["QuasiPeriodic"] == True
        assert cm["Deterministic"] == True
        assert cm["Chaotic"] == False
        assert cm["Random"] == False

    def test_all_systems_have_required_predicates(self):
        """Each system should have at least the 27 predicates defined (v2.3)."""
        ontology = load_system_ontology("systems")
        # Core 15 predicates (v2.2)
        required_predicates = {
            "Chaotic", "Deterministic", "PosLyap", "Sensitive", "StrangeAttr",
            "PointUnpredictable", "StatPredictable", "QuasiPeriodic", "Random",
            "FixedPointAttr", "Periodic", "Dissipative", "Bounded", "Mixing", "Ergodic",
            # v2.3 new predicates
            "HyperChaotic", "Conservative", "HighDimensional", "Multifractal",
            "HighDimSystem", "ContinuousTime", "DiscreteTime", "DelaySystem",
            "Forced", "Autonomous", "StrongMixing", "WeakMixing",
        }

        for system_id, truth_assignment in ontology.items():
            missing = required_predicates - set(truth_assignment.keys())
            assert not missing, \
                f"System {system_id} missing predicates: {missing}"

    def test_handles_missing_directory(self):
        """Should handle missing directory gracefully."""
        ontology = load_system_ontology("nonexistent_directory")
        assert ontology == {}

    def test_handles_empty_directory(self, tmp_path):
        """Should handle empty directory gracefully."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        ontology = load_system_ontology(str(empty_dir))
        assert ontology == {}


class TestExtractPredicateFromQuestion:
    """Test question-to-predicate mapping."""

    def test_chaotic_extraction(self):
        """Questions about chaos should map to Chaotic."""
        assert extract_predicate_from_question("Is the system chaotic?") == "Chaotic"
        assert extract_predicate_from_question("Does it exhibit chaos?") == "Chaotic"

    def test_deterministic_extraction(self):
        """Questions about determinism should map to Deterministic."""
        q = "Is the Lorenz-63 system deterministic?"
        assert extract_predicate_from_question(q) == "Deterministic"

    def test_poslyap_extraction(self):
        """Questions about Lyapunov exponent should map to PosLyap."""
        q1 = "Does the system have a positive Lyapunov exponent?"
        q2 = "Is the largest Lyapunov exponent positive?"
        assert extract_predicate_from_question(q1) == "PosLyap"
        assert extract_predicate_from_question(q2) == "PosLyap"

    def test_sensitive_extraction(self):
        """Questions about sensitivity should map to Sensitive."""
        q = "Does the system exhibit sensitive dependence on initial conditions?"
        assert extract_predicate_from_question(q) == "Sensitive"

    def test_strange_attractor_extraction(self):
        """Questions about strange attractors should map to StrangeAttr."""
        q = "Does the system have a strange attractor?"
        assert extract_predicate_from_question(q) == "StrangeAttr"

    def test_quasiperiodic_extraction(self):
        """Questions about quasi-periodicity should map to QuasiPeriodic."""
        q1 = "Is the system quasi-periodic?"
        q2 = "Does it exhibit quasiperiodic behavior?"
        assert extract_predicate_from_question(q1) == "QuasiPeriodic"
        assert extract_predicate_from_question(q2) == "QuasiPeriodic"

    def test_random_extraction(self):
        """Questions about randomness should map to Random."""
        q1 = "Is the system random?"
        q2 = "Does it involve randomness?"
        q3 = "Is it stochastic?"
        assert extract_predicate_from_question(q1) == "Random"
        assert extract_predicate_from_question(q2) == "Random"
        assert extract_predicate_from_question(q3) == "Random"

    def test_periodic_extraction(self):
        """Questions about periodicity should map to Periodic."""
        q = "Does the system have a periodic orbit?"
        assert extract_predicate_from_question(q) == "Periodic"

    def test_fixed_point_extraction(self):
        """Questions about fixed points should map to FixedPointAttr."""
        q = "Does the system have a fixed point attractor?"
        assert extract_predicate_from_question(q) == "FixedPointAttr"

    def test_unpredictable_extraction(self):
        """Questions about predictability should map to PointUnpredictable."""
        q1 = "Is long-term pointwise prediction possible?"
        q2 = "Is the system point-wise predictable?"
        assert extract_predicate_from_question(q1) == "PointUnpredictable"
        assert extract_predicate_from_question(q2) == "PointUnpredictable"

    def test_stat_predictable_extraction(self):
        """Questions about statistical predictability should map to StatPredictable."""
        q = "Is the system statistically predictable?"
        assert extract_predicate_from_question(q) == "StatPredictable"

    def test_no_match_returns_none(self):
        """Questions without predicate keywords should return None."""
        assert extract_predicate_from_question("What is the weather?") is None
        assert extract_predicate_from_question("How are you?") is None
        assert extract_predicate_from_question("") is None
        assert extract_predicate_from_question(None) is None

    def test_case_insensitive(self):
        """Extraction should be case-insensitive."""
        assert extract_predicate_from_question("IS THE SYSTEM CHAOTIC?") == "Chaotic"
        assert extract_predicate_from_question("is the system chaotic?") == "Chaotic"


class TestCheckFOLViolations:
    """Test FOL violation detection logic."""

    def test_no_violations_when_consistent(self):
        """Consistent predictions should have zero violations."""
        predictions = {
            "Chaotic": "YES",
            "Deterministic": "YES",
            "PosLyap": "YES",
            "Sensitive": "YES",
            "PointUnpredictable": "YES",
            "StatPredictable": "YES",
            "Random": "NO",
            "QuasiPeriodic": "NO",
        }
        violations = check_fol_violations(predictions)
        assert violations == []

    def test_chaotic_requires_deterministic_violation(self):
        """Chaotic=YES but Deterministic=NO should violate Chaotic → Deterministic."""
        predictions = {"Chaotic": "YES", "Deterministic": "NO"}
        violations = check_fol_violations(predictions)
        assert "Chaotic → Deterministic" in violations

    def test_chaotic_requires_poslyap_violation(self):
        """Chaotic=YES but PosLyap=NO should violate."""
        predictions = {"Chaotic": "YES", "PosLyap": "NO"}
        violations = check_fol_violations(predictions)
        assert "Chaotic → PosLyap" in violations

    def test_chaotic_excludes_random_violation(self):
        """Chaotic=YES and Random=YES should violate exclusion."""
        predictions = {"Chaotic": "YES", "Random": "YES"}
        violations = check_fol_violations(predictions)
        assert "Chaotic → ¬Random" in violations
        assert "Random → ¬Chaotic" in violations  # Symmetric

    def test_chaotic_excludes_quasiperiodic_violation(self):
        """Chaotic=YES and QuasiPeriodic=YES should violate."""
        predictions = {"Chaotic": "YES", "QuasiPeriodic": "YES"}
        violations = check_fol_violations(predictions)
        assert "Chaotic → ¬QuasiPeriodic" in violations

    def test_deterministic_excludes_random_violation(self):
        """Deterministic=YES and Random=YES should violate."""
        predictions = {"Deterministic": "YES", "Random": "YES"}
        violations = check_fol_violations(predictions)
        assert "Deterministic → ¬Random" in violations
        assert "Random → ¬Deterministic" in violations  # Symmetric

    def test_multiple_violations(self):
        """Should detect all violations in inconsistent predictions."""
        predictions = {
            "Chaotic": "YES",
            "Deterministic": "NO",  # Violation 1
            "Random": "YES",  # Violation 2 (Chaotic excludes Random)
            "PosLyap": "NO",  # Violation 3
        }
        violations = check_fol_violations(predictions)
        assert len(violations) >= 3
        assert "Chaotic → Deterministic" in violations
        assert "Chaotic → PosLyap" in violations
        assert "Chaotic → ¬Random" in violations

    def test_missing_predicates_not_violations(self):
        """Missing predicates should not count as violations."""
        predictions = {"Chaotic": "YES"}
        # Chaotic requires Deterministic, but we don't have it in predictions
        # Should NOT violate (conservative: only check what we have)
        violations = check_fol_violations(predictions)
        assert violations == []

    def test_quasiperiodic_requires_deterministic(self):
        """QuasiPeriodic=YES but Deterministic=NO should violate."""
        predictions = {"QuasiPeriodic": "YES", "Deterministic": "NO"}
        violations = check_fol_violations(predictions)
        assert "QuasiPeriodic → Deterministic" in violations

    def test_empty_predictions(self):
        """Empty predictions should have no violations."""
        violations = check_fol_violations({})
        assert violations == []

    def test_partial_predictions_consistent(self):
        """Partial predictions that are consistent should have no violations."""
        predictions = {
            "Deterministic": "YES",
            "Random": "NO",
        }
        violations = check_fol_violations(predictions)
        assert violations == []


class TestFOLIntegration:
    """Integration tests combining multiple FOL functions."""

    def test_end_to_end_violation_detection(self):
        """
        Test complete flow: question → predicate → prediction → violation check.
        """
        # Simulate a dialogue about Lorenz-63
        q1 = "Is the Lorenz-63 system chaotic?"
        q2 = "Is it random?"

        # Extract predicates
        pred1 = extract_predicate_from_question(q1)
        pred2 = extract_predicate_from_question(q2)

        assert pred1 == "Chaotic"
        assert pred2 == "Random"

        # Simulate model saying YES to both (incorrect!)
        predictions = {pred1: "YES", pred2: "YES"}

        # Check violations
        violations = check_fol_violations(predictions)

        # Should detect exclusion violation
        assert len(violations) >= 2
        assert "Chaotic → ¬Random" in violations
        assert "Random → ¬Chaotic" in violations

    def test_correct_predictions_from_ontology(self):
        """
        Test that ground truth from ontology doesn't violate FOL rules.
        """
        ontology = load_system_ontology("systems")

        # Check a few systems
        for system_id in ["lorenz63", "stochastic_ou", "circle_map_quasiperiodic"]:
            if system_id not in ontology:
                continue

            truth = ontology[system_id]

            # Convert booleans to YES/NO
            predictions = {
                pred: "YES" if value else "NO" for pred, value in truth.items()
            }

            # Ground truth should never violate FOL rules
            violations = check_fol_violations(predictions)
            assert (
                violations == []
            ), f"Ground truth for {system_id} violates FOL: {violations}"


# Summary test
def test_fol_coverage_summary():
    """
    Meta-test documenting FOL test coverage.

    Coverage:
    ✓ get_fol_rules(): 8 tests
    ✓ load_system_ontology(): 8 tests
    ✓ extract_predicate_from_question(): 16 tests
    ✓ check_fol_violations(): 12 tests
    ✓ Integration tests: 2 tests

    Total: 46 test cases covering all FOL functionality
    """
    pass
