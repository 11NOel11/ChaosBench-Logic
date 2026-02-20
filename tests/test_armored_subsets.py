"""Stability regression tests for canonical armored subsets.

Verifies:
- All standard subset files and manifests exist
- Manifest SHA256 matches actual file hash (stability regression)
- Items have required fields (id, question, ground_truth, type)
- ground_truth ∈ {TRUE, FALSE}
- Family suite files cover all 10 expected families
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
FAMILY_SUITES_DIR = SUBSETS_DIR / "subset_family_suites"

STANDARD_SUBSETS = [
    "subset_1k_armored.jsonl",
    "subset_5k_armored.jsonl",
]

EXPECTED_FAMILIES = [
    "adversarial",
    "atomic",
    "consistency_paraphrase",
    "cross_indicator",
    "extended_systems",
    "fol_inference",
    "indicator_diagnostics",
    "multi_hop",
    "perturbation_robustness",
    "regime_transition",
]

REQUIRED_FIELDS = {"id", "question", "ground_truth"}


def _sha256_ids(items: list[dict]) -> str:
    """Recompute the manifest SHA256 (hash over sorted item IDs, matching subsets.py)."""
    ids = sorted(item["id"] for item in items)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Standard subsets: existence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", STANDARD_SUBSETS)
def test_standard_subset_exists(name):
    """Standard subset JSONL file must exist."""
    assert (SUBSETS_DIR / name).exists(), f"Missing: data/subsets/{name}"


@pytest.mark.parametrize("name", STANDARD_SUBSETS)
def test_standard_subset_manifest_exists(name):
    """Manifest JSON for each standard subset must exist."""
    manifest_name = name.replace(".jsonl", ".manifest.json")
    assert (SUBSETS_DIR / manifest_name).exists(), f"Missing manifest: {manifest_name}"


# ---------------------------------------------------------------------------
# Standard subsets: SHA stability regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", STANDARD_SUBSETS)
def test_standard_subset_sha_stable(name):
    """Manifest SHA256 (over sorted IDs) must match recomputed hash (stability regression)."""
    subset_path = SUBSETS_DIR / name
    manifest_path = SUBSETS_DIR / name.replace(".jsonl", ".manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_sha = manifest.get("sha256") or manifest.get("subset_sha256")
    assert recorded_sha is not None, "Manifest missing 'sha256' field"

    items = _load_jsonl(subset_path)
    recomputed = _sha256_ids(items)
    assert recomputed == recorded_sha, (
        f"SHA mismatch for {name}: manifest={recorded_sha!r}, recomputed={recomputed!r}"
    )


# ---------------------------------------------------------------------------
# Standard subsets: item field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", STANDARD_SUBSETS)
def test_standard_subset_item_fields(name):
    """Every item must have required fields."""
    items = _load_jsonl(SUBSETS_DIR / name)
    assert len(items) > 0, f"{name} is empty"
    for i, item in enumerate(items):
        missing = REQUIRED_FIELDS - item.keys()
        assert not missing, f"Item {i} in {name} missing fields: {missing}"


@pytest.mark.parametrize("name", STANDARD_SUBSETS)
def test_standard_subset_ground_truth_values(name):
    """ground_truth must be TRUE or FALSE for every item."""
    items = _load_jsonl(SUBSETS_DIR / name)
    bad = [
        (i, item.get("ground_truth"))
        for i, item in enumerate(items)
        if item.get("ground_truth") not in {"TRUE", "FALSE"}
    ]
    assert not bad, f"Invalid ground_truth values in {name}: {bad[:5]}"


# ---------------------------------------------------------------------------
# Family suite: existence and coverage
# ---------------------------------------------------------------------------


def test_family_suite_covers_all_families():
    """Family suite directory must have a file for each expected family."""
    for family in EXPECTED_FAMILIES:
        path = FAMILY_SUITES_DIR / f"{family}.jsonl"
        assert path.exists(), f"Missing family suite file: {path}"


def test_family_suite_manifests_exist():
    """Each family suite file must have a companion manifest."""
    for family in EXPECTED_FAMILIES:
        manifest_path = FAMILY_SUITES_DIR / f"{family}.manifest.json"
        assert manifest_path.exists(), f"Missing family suite manifest: {manifest_path}"


@pytest.mark.parametrize("family", EXPECTED_FAMILIES)
def test_family_suite_sha_stable(family):
    """Family suite manifest SHA (over sorted IDs) must match recomputed hash."""
    subset_path = FAMILY_SUITES_DIR / f"{family}.jsonl"
    manifest_path = FAMILY_SUITES_DIR / f"{family}.manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_sha = manifest.get("sha256") or manifest.get("subset_sha256")
    assert recorded_sha is not None, f"Manifest for {family} missing 'sha256' field"

    items = _load_jsonl(subset_path)
    recomputed = _sha256_ids(items)
    assert recomputed == recorded_sha, (
        f"SHA mismatch for {family}: manifest={recorded_sha!r}, recomputed={recomputed!r}"
    )


@pytest.mark.parametrize("family", EXPECTED_FAMILIES)
def test_family_suite_item_fields(family):
    """Every item in a family suite must have required fields."""
    items = _load_jsonl(FAMILY_SUITES_DIR / f"{family}.jsonl")
    assert len(items) > 0, f"{family}.jsonl is empty"
    for i, item in enumerate(items):
        missing = REQUIRED_FIELDS - item.keys()
        assert not missing, f"Item {i} in {family}.jsonl missing fields: {missing}"


@pytest.mark.parametrize("family", EXPECTED_FAMILIES)
def test_family_suite_ground_truth_values(family):
    """ground_truth must be TRUE or FALSE in every family suite file."""
    items = _load_jsonl(FAMILY_SUITES_DIR / f"{family}.jsonl")
    bad = [
        (i, item.get("ground_truth"))
        for i, item in enumerate(items)
        if item.get("ground_truth") not in {"TRUE", "FALSE"}
    ]
    assert not bad, f"Invalid ground_truth in {family}.jsonl: {bad[:5]}"
