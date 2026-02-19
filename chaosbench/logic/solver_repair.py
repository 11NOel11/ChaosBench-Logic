"""MaxSAT solver repair for predicate assignments using python-sat RC2."""

from typing import Dict, List, Optional, Tuple

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from chaosbench.logic.axioms import check_fol_violations, get_fol_rules
from chaosbench.logic.ontology import PREDICATES


def encode_fol_to_cnf(
    rules: Dict[str, Dict[str, List[str]]],
    predicates: List[str],
) -> Tuple[List[List[int]], Dict[str, int]]:
    """Convert FOL axioms to CNF clauses for SAT solver.

    Each "requires" rule: A=True -> B=True becomes clause [-a, b]
    Each "excludes" rule: A=True -> B=False becomes clause [-a, -b]

    Args:
        rules: FOL rules dict from get_fol_rules().
        predicates: List of predicate names.

    Returns:
        Tuple of (list of CNF clauses, variable mapping {pred_name: var_id}).
    """
    var_map = {pred: idx + 1 for idx, pred in enumerate(predicates)}
    clauses: List[List[int]] = []

    for pred, constraints in rules.items():
        if pred not in var_map:
            continue
        a = var_map[pred]

        for req in constraints.get("requires", []):
            if req not in var_map:
                continue
            b = var_map[req]
            clauses.append([-a, b])

        for exc in constraints.get("excludes", []):
            if exc not in var_map:
                continue
            b = var_map[exc]
            clauses.append([-a, -b])

    return clauses, var_map


def repair_assignment(
    predictions: Dict[str, str],
    rules: Optional[Dict] = None,
) -> Tuple[Dict[str, str], int]:
    """Find minimal-flip repair using MaxSAT.

    Hard constraints = FOL axioms (must be satisfied).
    Soft constraints = keep original predictions (each with weight 1).

    Args:
        predictions: Model predictions as {predicate: "YES"|"NO"}.
        rules: Optional FOL rules (defaults to get_fol_rules()).

    Returns:
        Tuple of (repaired predictions dict, number of flips).
    """
    if rules is None:
        rules = get_fol_rules()

    predicates = list(PREDICATES)
    hard_clauses, var_map = encode_fol_to_cnf(rules, predicates)

    wcnf = WCNF()

    for clause in hard_clauses:
        wcnf.append(clause)

    for pred in predicates:
        if pred not in var_map or pred not in predictions:
            continue
        v = var_map[pred]
        if predictions[pred] == "YES":
            wcnf.append([v], weight=1)
        else:
            wcnf.append([-v], weight=1)

    solver = RC2(wcnf)
    model = solver.compute()
    solver.delete()

    if model is None:
        return dict(predictions), 0

    true_vars = set(model)
    repaired: Dict[str, str] = {}
    for pred in predicates:
        if pred not in predictions:
            continue
        v = var_map[pred]
        repaired[pred] = "YES" if v in true_vars else "NO"

    flips = count_flips(predictions, repaired)
    return repaired, flips


def validate_repair(assignment: Dict[str, str]) -> bool:
    """Check that a repaired assignment has zero FOL violations.

    Args:
        assignment: Predicate assignment {predicate: "YES"|"NO"}.

    Returns:
        True if zero violations.
    """
    violations = check_fol_violations(assignment)
    return len(violations) == 0


def count_flips(original: Dict[str, str], repaired: Dict[str, str]) -> int:
    """Count number of predicates that changed between original and repaired.

    Args:
        original: Original predictions.
        repaired: Repaired predictions.

    Returns:
        Number of changed predicates.
    """
    flips = 0
    for pred in original:
        if pred in repaired and original[pred] != repaired[pred]:
            flips += 1
    return flips
