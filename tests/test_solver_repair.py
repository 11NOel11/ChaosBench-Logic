"""Tests for MaxSAT solver repair module.

Tests cover CNF encoding, minimal-flip repair, ground truth validation,
edge cases, performance bounds, and utility functions.
"""

import json
import os
import time

import pytest

from chaosbench.logic.axioms import check_fol_violations, get_fol_rules
from chaosbench.logic.ontology import PREDICATES
from chaosbench.logic.solver_repair import (
    count_flips,
    encode_fol_to_cnf,
    repair_assignment,
    validate_repair,
)

SYSTEMS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "systems")

ALL_PREDICATES = [
    "Chaotic",
    "Deterministic",
    "PosLyap",
    "Sensitive",
    "StrangeAttr",
    "PointUnpredictable",
    "StatPredictable",
    "QuasiPeriodic",
    "Random",
    "FixedPointAttr",
    "Periodic",
]


def _bool_to_yesno(truth: dict) -> dict:
    """Convert bool truth assignment to YES/NO string dict."""
    return {k: "YES" if v else "NO" for k, v in truth.items()}


def _load_system(filename: str) -> dict:
    """Load a single system JSON and return its truth assignment as YES/NO."""
    path = os.path.join(SYSTEMS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _bool_to_yesno(data["truth_assignment"])


def _load_all_systems() -> dict:
    """Load all system JSONs, return dict of system_id -> YES/NO assignment."""
    systems = {}
    for fname in os.listdir(SYSTEMS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SYSTEMS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        systems[data["system_id"]] = _bool_to_yesno(data["truth_assignment"])
    return systems


class TestEncodeFolToCnf:
    """Tests for FOL-to-CNF encoding."""

    def test_clauses_generated(self):
        """Encoding should produce a non-empty list of clauses."""
        rules = get_fol_rules()
        clauses, var_map = encode_fol_to_cnf(rules, list(PREDICATES))
        assert len(clauses) > 0

    def test_var_map_has_all_predicates(self):
        """Variable map should contain an entry for every predicate."""
        rules = get_fol_rules()
        clauses, var_map = encode_fol_to_cnf(rules, list(PREDICATES))
        for pred in PREDICATES:
            assert pred in var_map
            assert var_map[pred] >= 1

    def test_var_map_ids_unique(self):
        """Each predicate should map to a unique positive integer."""
        rules = get_fol_rules()
        _, var_map = encode_fol_to_cnf(rules, list(PREDICATES))
        ids = list(var_map.values())
        assert len(ids) == len(set(ids))
        assert all(v > 0 for v in ids)

    def test_requires_clause_structure(self):
        """Chaotic requires Deterministic should produce clause [-chaotic, deterministic]."""
        rules = get_fol_rules()
        clauses, var_map = encode_fol_to_cnf(rules, list(PREDICATES))
        chaotic_id = var_map["Chaotic"]
        det_id = var_map["Deterministic"]
        assert [-chaotic_id, det_id] in clauses

    def test_excludes_clause_structure(self):
        """Chaotic excludes Random should produce clause [-chaotic, -random]."""
        rules = get_fol_rules()
        clauses, var_map = encode_fol_to_cnf(rules, list(PREDICATES))
        chaotic_id = var_map["Chaotic"]
        random_id = var_map["Random"]
        assert [-chaotic_id, -random_id] in clauses

    def test_all_chaotic_requires_encoded(self):
        """All five Chaotic requires clauses should appear."""
        rules = get_fol_rules()
        clauses, var_map = encode_fol_to_cnf(rules, list(PREDICATES))
        chaotic_id = var_map["Chaotic"]
        for req in ["Deterministic", "PosLyap", "Sensitive", "PointUnpredictable", "StatPredictable"]:
            assert [-chaotic_id, var_map[req]] in clauses

    def test_empty_rules_no_clauses(self):
        """Empty rules dict should produce zero clauses."""
        clauses, var_map = encode_fol_to_cnf({}, list(PREDICATES))
        assert clauses == []
        assert len(var_map) == len(PREDICATES)


class TestConsistentAssignmentZeroFlips:
    """Consistent ground truths should require zero repair flips."""

    def test_lorenz63_zero_flips(self):
        """Lorenz-63 (chaotic) ground truth is consistent."""
        assignment = _load_system("lorenz63.json")
        repaired, n_flips = repair_assignment(assignment)
        assert n_flips == 0
        assert repaired == assignment

    def test_damped_oscillator_zero_flips(self):
        """Damped oscillator (FixedPointAttr) ground truth is consistent."""
        assignment = _load_system("damped_oscillator.json")
        repaired, n_flips = repair_assignment(assignment)
        assert n_flips == 0
        assert repaired == assignment

    def test_stochastic_ou_zero_flips(self):
        """Stochastic OU (Random) ground truth is consistent."""
        assignment = _load_system("stochastic_ou.json")
        repaired, n_flips = repair_assignment(assignment)
        assert n_flips == 0
        assert repaired == assignment

    def test_circle_map_quasiperiodic_zero_flips(self):
        """Circle map (QuasiPeriodic) ground truth is consistent."""
        assignment = _load_system("circle_map_quasiperiodic.json")
        repaired, n_flips = repair_assignment(assignment)
        assert n_flips == 0
        assert repaired == assignment


class TestAllGroundTruthSystems:
    """Ground truth system JSONs should be FOL-consistent and require zero flips.

    The standard_map system intentionally has Chaotic=YES and QuasiPeriodic=YES
    (mixed phase space), which violates the Chaotic-excludes-QuasiPeriodic axiom.
    It is excluded from strict consistency checks.
    """

    KNOWN_INCONSISTENT = {"standard_map"}

    def test_all_30_systems_loaded(self):
        """All 30 system JSONs should load successfully."""
        systems = _load_all_systems()
        assert len(systems) == 30

    def test_consistent_systems_zero_flips(self):
        """All FOL-consistent systems require zero repair flips."""
        systems = _load_all_systems()
        for system_id, assignment in systems.items():
            if system_id in self.KNOWN_INCONSISTENT:
                continue
            repaired, n_flips = repair_assignment(assignment)
            assert n_flips == 0, (
                f"System {system_id} required {n_flips} flips"
            )
            assert repaired == assignment

    def test_consistent_systems_validate(self):
        """All FOL-consistent ground truths pass validate_repair."""
        systems = _load_all_systems()
        for system_id, assignment in systems.items():
            if system_id in self.KNOWN_INCONSISTENT:
                continue
            assert validate_repair(assignment), (
                f"System {system_id} failed validation"
            )

    def test_standard_map_now_consistent(self):
        """standard_map was updated in v2.2 to resolve the mixed phase space inconsistency."""
        systems = _load_all_systems()
        sm = systems["standard_map"]
        assert sm["Chaotic"] == "YES"
        assert sm["QuasiPeriodic"] == "NO"
        assert validate_repair(sm)
        repaired, n_flips = repair_assignment(sm)
        assert validate_repair(repaired)
        assert n_flips == 0


class TestSingleViolationMinimalRepair:
    """A single rule violation should require minimal flips to repair."""

    def test_chaotic_without_deterministic(self):
        """Chaotic=YES, Deterministic=NO should flip at least one predicate."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assignment["Chaotic"] = "YES"
        repaired, n_flips = repair_assignment(assignment)
        assert validate_repair(repaired)
        assert n_flips >= 1

    def test_chaotic_without_deterministic_minimal(self):
        """Flipping Chaotic to NO is cheaper than satisfying all its requires."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assignment["Chaotic"] = "YES"
        repaired, n_flips = repair_assignment(assignment)
        assert repaired["Chaotic"] == "NO"
        assert n_flips == 1

    def test_random_with_deterministic(self):
        """Random=YES + Deterministic=YES should flip exactly one."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assignment["Random"] = "YES"
        assignment["Deterministic"] = "YES"
        repaired, n_flips = repair_assignment(assignment)
        assert validate_repair(repaired)
        assert n_flips == 1

    def test_periodic_without_deterministic(self):
        """Periodic=YES, Deterministic=NO should flip one predicate."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assignment["Periodic"] = "YES"
        repaired, n_flips = repair_assignment(assignment)
        assert validate_repair(repaired)
        assert n_flips <= 2


class TestMultipleViolationsMinimalRepair:
    """Multiple violations should be repaired with minimal global flips."""

    def test_chaotic_and_random_both_yes(self):
        """Chaotic=YES and Random=YES requires fixing at least one."""
        assignment = _load_system("lorenz63.json")
        assignment["Random"] = "YES"
        repaired, n_flips = repair_assignment(assignment)
        assert validate_repair(repaired)
        assert n_flips >= 1

    def test_chaotic_and_random_repairs_random(self):
        """With full chaotic support, solver should flip Random rather than cascade."""
        assignment = _load_system("lorenz63.json")
        assignment["Random"] = "YES"
        repaired, n_flips = repair_assignment(assignment)
        assert repaired["Random"] == "NO"
        assert n_flips == 1

    def test_many_contradictions(self):
        """Several wrong predicates should still converge to valid assignment."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assignment["Chaotic"] = "YES"
        assignment["Random"] = "YES"
        assignment["Periodic"] = "YES"
        repaired, n_flips = repair_assignment(assignment)
        assert validate_repair(repaired)
        assert n_flips >= 1


class TestEdgeCases:
    """Edge cases: all YES, all NO, empty, single predicate."""

    def test_all_yes_repaired(self):
        """All predicates YES violates rules (Chaotic excludes Random, etc.)."""
        assignment = {p: "YES" for p in ALL_PREDICATES}
        assert not validate_repair(assignment)
        repaired, n_flips = repair_assignment(assignment)
        assert validate_repair(repaired)
        assert n_flips >= 1

    def test_all_no_zero_flips(self):
        """All predicates NO is trivially consistent."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assert validate_repair(assignment)
        repaired, n_flips = repair_assignment(assignment)
        assert n_flips == 0
        assert repaired == assignment

    def test_empty_dict_zero_flips(self):
        """Empty prediction dict should return empty with zero flips."""
        repaired, n_flips = repair_assignment({})
        assert n_flips == 0
        assert repaired == {}

    def test_single_predicate_true(self):
        """Single predicate YES (no rule triggers on missing preds) -> 0 flips."""
        assignment = {"Sensitive": "YES"}
        repaired, n_flips = repair_assignment(assignment)
        assert n_flips == 0

    def test_single_predicate_false(self):
        """Single predicate NO -> 0 flips."""
        assignment = {"Chaotic": "NO"}
        repaired, n_flips = repair_assignment(assignment)
        assert n_flips == 0


class TestPerformance:
    """Solver should complete quickly for any input."""

    def test_repair_under_one_second_all_yes(self):
        """Repairing all-YES assignment completes in under 1 second."""
        assignment = {p: "YES" for p in ALL_PREDICATES}
        start = time.monotonic()
        repair_assignment(assignment)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    def test_repair_under_one_second_all_no(self):
        """Repairing all-NO assignment completes in under 1 second."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        start = time.monotonic()
        repair_assignment(assignment)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    def test_repair_under_one_second_mixed(self):
        """Repairing a contradictory mixed assignment completes in under 1 second."""
        assignment = {p: ("YES" if i % 2 == 0 else "NO") for i, p in enumerate(ALL_PREDICATES)}
        start = time.monotonic()
        repair_assignment(assignment)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


class TestValidateRepair:
    """Tests for the validate_repair function."""

    def test_valid_chaotic_assignment(self):
        """Full chaotic assignment with all requires met is valid."""
        assignment = {
            "Chaotic": "YES",
            "Deterministic": "YES",
            "PosLyap": "YES",
            "Sensitive": "YES",
            "StrangeAttr": "YES",
            "PointUnpredictable": "YES",
            "StatPredictable": "YES",
            "QuasiPeriodic": "NO",
            "Random": "NO",
            "FixedPointAttr": "NO",
            "Periodic": "NO",
        }
        assert validate_repair(assignment) is True

    def test_invalid_chaotic_missing_deterministic(self):
        """Chaotic=YES without Deterministic=YES is invalid."""
        assignment = {
            "Chaotic": "YES",
            "Deterministic": "NO",
            "PosLyap": "YES",
            "Sensitive": "YES",
            "PointUnpredictable": "YES",
            "StatPredictable": "YES",
            "Random": "NO",
        }
        assert validate_repair(assignment) is False

    def test_valid_all_no(self):
        """All NO is valid (no antecedent fires)."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assert validate_repair(assignment) is True

    def test_invalid_random_and_deterministic(self):
        """Random=YES + Deterministic=YES violates exclusion."""
        assignment = {p: "NO" for p in ALL_PREDICATES}
        assignment["Random"] = "YES"
        assignment["Deterministic"] = "YES"
        assert validate_repair(assignment) is False

    def test_valid_empty(self):
        """Empty assignment is trivially valid."""
        assert validate_repair({}) is True


class TestCountFlips:
    """Tests for the count_flips utility function."""

    def test_identical_zero_flips(self):
        """Identical dicts have zero flips."""
        a = {"Chaotic": "YES", "Random": "NO"}
        assert count_flips(a, dict(a)) == 0

    def test_one_flip(self):
        """Changing one value produces one flip."""
        a = {"Chaotic": "YES", "Random": "NO"}
        b = {"Chaotic": "NO", "Random": "NO"}
        assert count_flips(a, b) == 1

    def test_all_flipped(self):
        """Flipping every predicate counts all."""
        a = {p: "YES" for p in ALL_PREDICATES}
        b = {p: "NO" for p in ALL_PREDICATES}
        assert count_flips(a, b) == len(ALL_PREDICATES)

    def test_missing_key_in_repaired_not_counted(self):
        """Keys missing from repaired dict are not counted as flips."""
        a = {"Chaotic": "YES", "Random": "NO"}
        b = {"Chaotic": "YES"}
        assert count_flips(a, b) == 0

    def test_empty_dicts(self):
        """Empty dicts have zero flips."""
        assert count_flips({}, {}) == 0
