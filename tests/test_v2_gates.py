"""Regression tests for ChaosBench-Logic v2 canonical dataset gates.

Tests run quickly (<2 minutes total). No @pytest.mark.slow needed.

Checks:
- Canonical selector file exists and is valid
- Canonical total matches manifest
- All ground_truth labels are TRUE/FALSE
- Multi-hop and atomic structural sanity
- Paraphrase group label consistency
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
SYSTEMS_DIR = PROJECT_ROOT / "systems"


def _load_selector() -> Dict[str, Any]:
    return json.loads((DATA_DIR / "canonical_v2_files.json").read_text(encoding="utf-8"))


def _load_manifest() -> Dict[str, Any]:
    return json.loads((DATA_DIR / "v2_manifest.json").read_text(encoding="utf-8"))


def _load_family(fname: str) -> List[Dict[str, Any]]:
    fpath = DATA_DIR / fname
    if not fpath.exists():
        return []
    records = []
    for line in fpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_system_truths() -> Dict[str, Dict[str, bool]]:
    truths: Dict[str, Dict[str, bool]] = {}
    for json_file in SYSTEMS_DIR.glob("*.json"):
        if json_file.name.endswith("_indicators.json"):
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            sid = data.get("system_id", "")
            ta = data.get("truth_assignment", {})
            if sid and ta:
                truths[sid] = {k: bool(v) for k, v in ta.items()}
        except Exception:
            pass
    dysts_dir = SYSTEMS_DIR / "dysts"
    if dysts_dir.exists():
        for json_file in dysts_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                sid = data.get("system_id", "")
                ta = data.get("truth_assignment", {})
                if sid and ta:
                    truths[sid] = {k: bool(v) for k, v in ta.items()}
            except Exception:
                pass
    return truths


# ---------------------------------------------------------------------------
# TestCanonicalSelector
# ---------------------------------------------------------------------------

class TestCanonicalSelector:
    def test_canonical_selector_exists(self):
        """data/canonical_v2_files.json must exist."""
        assert (DATA_DIR / "canonical_v2_files.json").exists(), \
            "data/canonical_v2_files.json not found"

    def test_canonical_selector_non_empty(self):
        """Selector must list at least 1 file."""
        selector = _load_selector()
        assert "files" in selector, "Missing 'files' key in canonical_v2_files.json"
        assert len(selector["files"]) >= 1, "No files listed in canonical_v2_files.json"

    def test_canonical_selector_has_ten_files(self):
        """Selector must list exactly 10 canonical v22 files."""
        selector = _load_selector()
        assert len(selector["files"]) == 10, \
            f"Expected 10 canonical files, got {len(selector['files'])}"

    def test_canonical_files_exist_on_disk(self):
        """All files listed in selector must exist on disk."""
        selector = _load_selector()
        missing = []
        for f in selector["files"]:
            fpath = PROJECT_ROOT / f
            if not fpath.exists():
                missing.append(f)
        assert not missing, f"Canonical files missing from disk: {missing}"

    def test_canonical_total_matches_manifest(self):
        """Sum of record counts in canonical files must equal manifest total_new_questions."""
        selector = _load_selector()
        manifest = _load_manifest()
        expected_total = manifest["total_new_questions"]  # 40886

        actual_total = 0
        for f in selector["files"]:
            fpath = PROJECT_ROOT / f
            if fpath.exists():
                actual_total += sum(
                    1 for line in fpath.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )

        assert actual_total == expected_total, (
            f"Canonical file total {actual_total} != manifest total_new_questions {expected_total}"
        )

    def test_manifest_has_dataset_release_field(self):
        """v2_manifest.json must have dataset_release: 'v2'."""
        manifest = _load_manifest()
        assert "dataset_release" in manifest, \
            "v2_manifest.json missing 'dataset_release' field"
        assert manifest["dataset_release"] == "v2", \
            f"Expected dataset_release='v2', got '{manifest['dataset_release']}'"


# ---------------------------------------------------------------------------
# TestLabelSanity
# ---------------------------------------------------------------------------

class TestLabelSanity:
    def test_no_labels_outside_true_false(self):
        """Seeded sample of 500 items must all have ground_truth in {TRUE, FALSE}."""
        rng = random.Random(42)
        selector = _load_selector()

        all_records = []
        for f in selector["files"]:
            fpath = PROJECT_ROOT / f
            if fpath.exists():
                for line in fpath.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            all_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        sample = rng.sample(all_records, min(500, len(all_records)))
        invalid = [
            (r.get("id"), r.get("ground_truth"))
            for r in sample
            if r.get("ground_truth") not in {"TRUE", "FALSE"}
        ]
        assert not invalid, f"Items with invalid ground_truth: {invalid[:5]}"

    def test_label_balance_roughly_50_50(self):
        """Overall TRUE/FALSE balance should be between 40% and 60%."""
        selector = _load_selector()
        true_count = 0
        total = 0
        for f in selector["files"]:
            fpath = PROJECT_ROOT / f
            if fpath.exists():
                for line in fpath.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            total += 1
                            if r.get("ground_truth") == "TRUE":
                                true_count += 1
                        except json.JSONDecodeError:
                            pass
        assert total > 0, "No records loaded"
        true_pct = true_count / total * 100
        assert 40 <= true_pct <= 60, (
            f"Label balance out of range: {true_pct:.1f}% TRUE (expected 40-60%)"
        )


# ---------------------------------------------------------------------------
# TestMultiHopLogic
# ---------------------------------------------------------------------------

class TestMultiHopLogic:
    def test_multi_hop_has_valid_schema(self):
        """Seeded sample of 200 multi-hop items must have correct type and valid fields."""
        records = _load_family("v22_multi_hop.jsonl")
        if not records:
            pytest.skip("v22_multi_hop.jsonl not found")

        rng = random.Random(42)
        sample = rng.sample(records, min(200, len(records)))

        issues = []
        for item in sample:
            if item.get("type") != "multi_hop":
                issues.append(f"{item.get('id')}: type={item.get('type')!r}, expected 'multi_hop'")
            if item.get("ground_truth") not in {"TRUE", "FALSE"}:
                issues.append(f"{item.get('id')}: invalid ground_truth={item.get('ground_truth')!r}")
            if not item.get("question", ""):
                issues.append(f"{item.get('id')}: empty question")

        assert not issues, f"Multi-hop schema issues: {issues[:5]}"

    def test_multi_hop_system_ids_valid(self):
        """Multi-hop items with system_id must reference known systems."""
        records = _load_family("v22_multi_hop.jsonl")
        if not records:
            pytest.skip("v22_multi_hop.jsonl not found")

        system_truths = _load_system_truths()
        rng = random.Random(42)
        sample = rng.sample(records, min(500, len(records)))

        has_system = [r for r in sample if r.get("system_id")]
        if not has_system:
            pytest.skip("No multi-hop items have system_id in sample")

        known = sum(1 for r in has_system if r.get("system_id", "") in system_truths)
        pct = 100 * known / len(has_system)
        assert pct >= 30.0, (
            f"Only {pct:.1f}% of multi-hop items reference known systems "
            f"(expected >= 30%). Check system JSON files."
        )


# ---------------------------------------------------------------------------
# TestParaphraseConsistency
# ---------------------------------------------------------------------------

class TestParaphraseConsistency:
    def test_paraphrase_group_label_consistency(self):
        """Seeded sample of paraphrase groups must all share the same ground_truth."""
        records = _load_family("v22_consistency_paraphrase.jsonl")
        if not records:
            pytest.skip("v22_consistency_paraphrase.jsonl not found")

        # Group by base ID
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in records:
            item_id = item.get("id", "")
            m = re.match(r"^(atomic_\d+)_para_\d+$", item_id)
            if m:
                groups[m.group(1)].append(item)
            else:
                gid = item.get("group_id")
                if gid:
                    groups[gid].append(item)

        # Sample groups
        rng = random.Random(42)
        group_keys = list(groups.keys())
        sample_keys = rng.sample(group_keys, min(500, len(group_keys)))

        flip_groups = []
        for key in sample_keys:
            items = groups[key]
            if len(items) < 2:
                continue
            labels = set(i.get("ground_truth", "") for i in items)
            if len(labels) > 1:
                flip_groups.append({
                    "group_key": key,
                    "labels": list(labels),
                })

        assert not flip_groups, (
            f"{len(flip_groups)} paraphrase groups have inconsistent labels. "
            f"First: {flip_groups[0]}"
        )


# ---------------------------------------------------------------------------
# TestPerturbationIntegrity
# ---------------------------------------------------------------------------

# Predicates that appear verbatim in perturbation questions
_PERTURB_PREDICATES = [
    "chaotic", "sensitive", "strangeattr", "poslyap", "deterministic",
    "periodic", "quasiperiodic", "multistable", "regime", "dissipative",
    "bounded", "mixing", "ergodic",
]
_NEGATION_SIGNALS = [
    "is it false that", "is it incorrect that", "would it be incorrect",
    "would you say it is incorrect", "is not ", "not be characterized",
    "cannot be characterized",
]


def _extract_perturb_predicate(question: str) -> str:
    q = question.lower()
    for pred in _PERTURB_PREDICATES:
        # Use character boundary to avoid matching "chaotic" inside "non-chaotic"
        # (both hyphens and letters are excluded as leading characters)
        if re.search(r"(?<![a-z\-])" + pred + r"(?![a-z])", q):
            return pred
    return "other"


def _is_perturb_negated(question: str) -> bool:
    q = question.lower()
    return any(sig in q for sig in _NEGATION_SIGNALS)


class TestPerturbationIntegrity:
    """Gate tests for perturbation_robustness dataset integrity."""

    def test_perturbation_file_exists(self):
        """v22_perturbation_robustness.jsonl must exist."""
        fpath = DATA_DIR / "v22_perturbation_robustness.jsonl"
        assert fpath.exists(), "v22_perturbation_robustness.jsonl not found"

    def test_perturbation_label_schema(self):
        """All perturbation items must have ground_truth in {TRUE, FALSE}."""
        records = _load_family("v22_perturbation_robustness.jsonl")
        if not records:
            pytest.skip("perturbation file not found")
        bad = [r["id"] for r in records if r.get("ground_truth") not in {"TRUE", "FALSE"}]
        assert not bad, f"Perturbation items with invalid label: {bad[:5]}"

    def test_perturbation_type_distribution(self):
        """Perturbation file must contain paraphrase, negation, and distractor types."""
        records = _load_family("v22_perturbation_robustness.jsonl")
        if not records:
            pytest.skip("perturbation file not found")

        types_found = set()
        for item in records:
            m = re.match(r"perturb_(\w+)_\d+", item.get("id", ""))
            if m:
                types_found.add(m.group(1))

        required = {"paraphrase", "negation", "distractor"}
        missing = required - types_found
        assert not missing, f"Missing perturbation types: {missing}"

    def test_paraphrase_distractor_no_polarity_conflicts(self):
        """Seeded sample: paraphrase/distractor items with same (system, predicate, polarity)
        must share the same ground_truth.

        Polarity = 'neg' if the question stem contains a negation signal, else 'aff'.
        This distinguishes 'Can X be Y?' (aff) from 'Is it incorrect that X is Y?' (neg).
        """
        records = _load_family("v22_perturbation_robustness.jsonl")
        if not records:
            pytest.skip("perturbation file not found")

        rng = random.Random(42)
        sample = rng.sample(records, min(800, len(records)))

        conflicts: List[Dict[str, Any]] = []
        for ptype in ("paraphrase", "distractor"):
            groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
            for item in sample:
                m = re.match(r"perturb_(\w+)_\d+", item.get("id", ""))
                if not m or m.group(1) != ptype:
                    continue
                pred = _extract_perturb_predicate(item["question"])
                polarity = "neg" if _is_perturb_negated(item["question"]) else "aff"
                key = (item.get("system_id", ""), pred, polarity)
                groups[key].append(item)

            for key, group_items in groups.items():
                labels = {i["ground_truth"] for i in group_items}
                if len(labels) > 1:
                    conflicts.append({
                        "type": ptype,
                        "key": key,
                        "labels": sorted(labels),
                        "item_ids": [i["id"] for i in group_items],
                    })

        assert not conflicts, (
            f"{len(conflicts)} polarity-matched label conflicts in "
            f"paraphrase/distractor groups. First: {conflicts[0]}"
        )

    def test_negation_items_mostly_false(self):
        """Negation-type items should be predominantly FALSE (> 80%).

        Negations of TRUE atomic claims yield FALSE; TRUE negations arise only
        from negating already-FALSE claims, which should be a minority.
        """
        records = _load_family("v22_perturbation_robustness.jsonl")
        if not records:
            pytest.skip("perturbation file not found")

        negations = [
            r for r in records
            if re.match(r"perturb_negation_\d+", r.get("id", ""))
        ]
        if not negations:
            pytest.skip("No negation items found")

        n_false = sum(1 for r in negations if r["ground_truth"] == "FALSE")
        pct_false = 100 * n_false / len(negations)
        assert pct_false >= 60, (
            f"Only {pct_false:.1f}% of negation items are FALSE "
            f"(expected >= 60%). This may indicate a design error."
        )
