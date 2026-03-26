"""CARE-v3 repair utilities for post-hoc constraint-aware label repair."""

from chaosbench.repair.engine import (
    load_selector_index,
    read_jsonl,
    repair_records,
    write_jsonl,
)
from chaosbench.repair.hashing import constraint_hash
from chaosbench.repair.controls import (
    budget_match_candidate_labels,
    inject_parser_noise,
    random_flip_labels,
    shuffled_gate_families,
)
from chaosbench.repair.selective import (
    apply_margin_policy,
    collect_flip_candidates,
    fit_margin_policy,
    policy_map,
)
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
    candidate_distribution,
    js_divergence,
    reference_distribution_from_policy,
)
from chaosbench.repair.types import RepairConfig, RepairResult, RepairStats

__all__ = [
    "RepairConfig",
    "RepairResult",
    "RepairStats",
    "budget_match_candidate_labels",
    "constraint_hash",
    "collect_flip_candidates",
    "fit_margin_policy",
    "policy_map",
    "apply_margin_policy",
    "fit_instance_policy",
    "score_instance_candidate",
    "apply_instance_policy",
    "policy_family_rows",
    "policy_cell_rows",
    "OnlineControllerConfig",
    "OnlineTransferController",
    "reference_distribution_from_policy",
    "candidate_distribution",
    "js_divergence",
    "inject_parser_noise",
    "load_selector_index",
    "random_flip_labels",
    "read_jsonl",
    "repair_records",
    "shuffled_gate_families",
    "write_jsonl",
]
