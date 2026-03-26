"""Hashing helpers for CARE-v3 manifests and reproducibility checks."""

from __future__ import annotations

import hashlib
import json
from typing import Dict

from chaosbench.logic.axioms import get_fol_rules
from chaosbench.repair.extraction import NEGATION_PATTERNS
from chaosbench.repair.types import RepairConfig


def _stable_json(payload: Dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def constraint_hash(config: RepairConfig) -> str:
    """Hash repair configuration and hard constraints."""
    payload = {
        "protocol": "care_v3",
        "config": config.to_dict(),
        "negation_patterns": list(NEGATION_PATTERNS),
        "fol_rules": get_fol_rules(),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return digest
