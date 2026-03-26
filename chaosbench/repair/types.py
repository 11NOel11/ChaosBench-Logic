"""Typed structures for CARE-v3 repair configuration and outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VALID_LABELS = {"TRUE", "FALSE"}


@dataclass(frozen=True)
class RepairConfig:
    """Configuration for one deterministic repair pass."""

    name: str = "care_v3"
    gate_families: Optional[Tuple[str, ...]] = None
    extractor_strategy: str = "last_mention"
    polarity_mode: str = "rule_based"
    leave_invalid_unchanged: bool = True
    enable_group_consistency: bool = False
    seed: int = 42

    def allows_family(self, family: Optional[str]) -> bool:
        """Return True when a task family is eligible for repair."""
        if self.gate_families is None:
            return True
        if family is None:
            return False
        return family in set(self.gate_families)

    def to_dict(self) -> Dict[str, Any]:
        """Return stable JSON-serializable view for hashing/manifests."""
        payload = asdict(self)
        if payload["gate_families"] is not None:
            payload["gate_families"] = sorted(payload["gate_families"])
        return payload


@dataclass
class RepairStats:
    """Operational statistics emitted for one repaired run."""

    total_records: int = 0
    valid_records: int = 0
    eligible_records: int = 0
    eligible_with_predicate: int = 0
    systems_repaired: int = 0
    predicate_assignments: int = 0
    predicate_flips: int = 0
    row_flips: int = 0
    axiom_violations_pre: int = 0
    axiom_violations_post: int = 0
    group_inconsistency_pre: float = 0.0
    group_inconsistency_post: float = 0.0
    flip_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly statistics payload."""
        payload = asdict(self)
        payload["flip_reasons"] = dict(sorted(self.flip_reasons.items()))
        return payload


@dataclass
class RepairResult:
    """Return object for one repair execution."""

    records: List[Dict[str, Any]]
    stats: RepairStats
    constraint_hash: str
    config: RepairConfig
