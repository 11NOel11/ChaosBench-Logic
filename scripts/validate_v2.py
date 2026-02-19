"""Validation script for ChaosBench-Logic v2 installation and data integrity."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict


def check_import():
    """Verify chaosbench package imports cleanly."""
    try:
        import chaosbench

        assert chaosbench.__version__ == "2.0.0"
        return True, f"chaosbench v{chaosbench.__version__} imported"
    except Exception as e:
        return False, f"Import failed: {e}"


def check_systems(systems_dir="systems"):
    """Verify all 30 system JSON files exist and parse correctly."""
    if not os.path.isdir(systems_dir):
        return False, f"Systems directory not found: {systems_dir}"

    json_files = sorted(f for f in os.listdir(systems_dir) if f.endswith(".json"))
    if len(json_files) != 30:
        return False, f"Expected 30 system files, found {len(json_files)}"

    errors = []
    for fname in json_files:
        fpath = os.path.join(systems_dir, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            if "system_id" not in data:
                errors.append(f"{fname}: missing system_id")
            if "truth_assignment" not in data:
                errors.append(f"{fname}: missing truth_assignment")
        except json.JSONDecodeError as e:
            errors.append(f"{fname}: parse error: {e}")

    if errors:
        return False, f"{len(errors)} system file errors: {'; '.join(errors[:3])}"
    return True, f"All {len(json_files)} system files valid"


def iter_canonical_files(data_dir="data"):
    """Yield canonical v22_*.jsonl file paths."""
    selector_path = os.path.join(data_dir, "canonical_v2_files.json")
    if os.path.isfile(selector_path):
        with open(selector_path) as f:
            files = json.load(f).get("files", [])
        project_root = os.path.dirname(os.path.abspath(data_dir))
        for rel_path in files:
            fpath = os.path.join(project_root, rel_path)
            if os.path.isfile(fpath):
                yield fpath
    else:
        import glob
        for path in sorted(glob.glob(os.path.join(data_dir, "v22_*.jsonl"))):
            yield path


def iter_batch_files(data_dir="data"):
    """Yield dataset file paths — now delegates to canonical v22_*.jsonl files."""
    yield from iter_canonical_files(data_dir)


def compute_sha256(path):
    """Compute SHA-256 hash for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def check_manifest_integrity(data_dir="data", manifest_path="data/v2_manifest.json"):
    """Validate v2 manifest counts against canonical files on disk."""
    if not os.path.isfile(manifest_path):
        return False, f"Manifest not found: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify dataset_release field
    dataset_release = manifest.get("dataset_release")
    if dataset_release != "v2":
        return False, f"Expected dataset_release='v2', got {dataset_release!r}"

    # Count items across canonical v22_*.jsonl files
    selector_path = os.path.join(data_dir, "canonical_v2_files.json")
    if os.path.isfile(selector_path):
        with open(selector_path) as f:
            canonical_files = json.load(f).get("files", [])
    else:
        import glob
        canonical_files = sorted(glob.glob(os.path.join(data_dir, "v22_*.jsonl")))

    computed_total = 0
    missing = []
    for rel_path in canonical_files:
        # canonical_v2_files.json lists paths relative to project root
        fpath = rel_path if os.path.isabs(rel_path) else os.path.join(os.path.dirname(data_dir.rstrip("/")), rel_path)
        if not os.path.isfile(fpath):
            # Try relative to data_dir directly
            fpath2 = os.path.join(data_dir, os.path.basename(rel_path))
            if os.path.isfile(fpath2):
                fpath = fpath2
            else:
                missing.append(rel_path)
                continue
        with open(fpath, "r", encoding="utf-8") as f:
            computed_total += sum(1 for line in f if line.strip())

    if missing:
        return False, f"Missing canonical files: {'; '.join(missing[:3])}"

    expected_total = manifest.get("total_new_questions")
    if expected_total is not None and expected_total != computed_total:
        return (
            False,
            f"total_new_questions mismatch: manifest={expected_total}, files={computed_total}",
        )

    return True, f"Manifest integrity OK: {computed_total} questions across {len(canonical_files)} files"


def check_unique_item_ids(data_dir="data"):
    """Ensure all item IDs are unique across all batch files."""
    seen = {}
    duplicates = []

    for batch_path in iter_batch_files(data_dir):
        with open(batch_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                item_id = record.get("id")
                if not item_id:
                    duplicates.append(f"{batch_path}:{i} missing id")
                    continue
                if item_id in seen:
                    duplicates.append(
                        f"duplicate id '{item_id}' in {batch_path}:{i} (first seen {seen[item_id]})"
                    )
                else:
                    seen[item_id] = f"{batch_path}:{i}"

    if duplicates:
        return False, f"Found {len(duplicates)} ID issues: {'; '.join(duplicates[:3])}"
    return True, f"All item IDs unique ({len(seen)} total)"


def check_question_contamination(data_dir="data", max_duplicates=0):
    """Check for exact duplicate question texts across batches."""
    question_to_locs = defaultdict(list)

    for batch_path in iter_batch_files(data_dir):
        batch_name = os.path.basename(batch_path)
        with open(batch_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                q = (record.get("question") or "").strip()
                if q:
                    question_to_locs[q].append(f"{batch_name}:{i}")

    duplicates = [locs for locs in question_to_locs.values() if len(locs) > 1]
    duplicate_count = sum(len(locs) - 1 for locs in duplicates)
    total_questions = len(question_to_locs)

    if duplicate_count > max_duplicates:
        sample = duplicates[0][:3]
        return False, (
            f"Found {duplicate_count} duplicate question rows (limit {max_duplicates}) across {total_questions} unique texts; "
            f"sample locations: {sample}"
        )

    return True, (
        f"Duplicate question rows within limit ({duplicate_count}/{max_duplicates}) "
        f"across {total_questions} unique texts"
    )


def check_run_manifest_registry(runs_dir="runs"):
    """Validate run manifest registry files if present."""
    if not os.path.isdir(runs_dir):
        return True, "No runs/ registry yet (skipped)"

    manifest_files = sorted(
        p
        for p in os.listdir(runs_dir)
        if p.startswith("manifest_") and p.endswith(".json")
    )
    if not manifest_files:
        return True, "No run manifests found (skipped)"

    required = {
        "manifest_version",
        "timestamp",
        "model",
        "mode",
        "run_name",
        "run_out_dir",
        "data_dir",
        "batches",
        "total_items_before_sharding",
        "items_in_this_run",
        "sharding",
    }

    errors = []
    for fname in manifest_files:
        path = os.path.join(runs_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            missing = required - set(data.keys())
            if missing:
                errors.append(f"{fname} missing fields: {sorted(missing)}")
        except Exception as e:
            errors.append(f"{fname} parse error: {e}")

    if errors:
        return False, f"Run manifest errors ({len(errors)}): {'; '.join(errors[:3])}"
    return True, f"Run manifests valid ({len(manifest_files)} file(s))"


def check_fol_consistency(systems_dir="systems"):
    """Verify FOL ground truths are consistent for all systems."""
    from chaosbench.logic.axioms import check_fol_violations, load_system_ontology

    ontology = load_system_ontology(systems_dir)
    if not ontology:
        return False, "No ontology loaded"

    violations = {}
    for system_id, truth in ontology.items():
        preds = {k: ("YES" if v else "NO") for k, v in truth.items()}
        v = check_fol_violations(preds)
        if v:
            violations[system_id] = v

    known_inconsistent = {"standard_map"}
    unexpected = {k: v for k, v in violations.items() if k not in known_inconsistent}

    if unexpected:
        details = "; ".join(f"{k}: {v}" for k, v in list(unexpected.items())[:3])
        return False, f"{len(unexpected)} unexpected FOL violations: {details}"

    msg = f"{len(ontology)} systems checked, {len(violations)} known inconsistent"
    return True, msg


def check_indicators():
    """Verify indicator modules load and produce results for lorenz63."""
    try:
        from chaosbench.data.indicators.populate import compute_all_indicators

        result = compute_all_indicators("lorenz63", seed=42)
        expected_keys = {
            "system_id",
            "zero_one_K",
            "permutation_entropy",
            "megno",
            "system_type",
            "seed",
            "timestamp",
        }
        missing = expected_keys - set(result.keys())
        if missing:
            return False, f"Missing indicator keys: {missing}"
        return True, f"lorenz63 indicators computed: K={result['zero_one_K']:.3f}"
    except Exception as e:
        return False, f"Indicator computation failed: {e}"


def check_solver_repair():
    """Verify MaxSAT solver repair works on a simple case."""
    try:
        from chaosbench.logic.solver_repair import repair_assignment, validate_repair

        preds = {"Chaotic": "YES", "Deterministic": "NO"}
        repaired, flips = repair_assignment(preds)
        if not validate_repair(repaired):
            return False, "Repair did not produce consistent assignment"
        return True, f"Solver repair works ({flips} flips for test case)"
    except Exception as e:
        return False, f"Solver repair failed: {e}"


def check_regime_transitions():
    """Verify regime transition task generates questions."""
    try:
        from chaosbench.tasks.regime_transition import RegimeTransitionTask

        task = RegimeTransitionTask()
        items = task.generate_items()
        if len(items) < 10:
            return False, f"Only {len(items)} questions generated (expected 10+)"
        return True, f"Regime transition task: {len(items)} questions"
    except Exception as e:
        return False, f"Regime transition task failed: {e}"


def check_adversarial():
    """Verify adversarial question generation works."""
    try:
        from chaosbench.data.adversarial import generate_adversarial_set

        systems = {
            "lorenz63": {
                "name": "Lorenz system",
                "truth": {
                    "Chaotic": True,
                    "Deterministic": True,
                    "PosLyap": True,
                    "Sensitive": True,
                    "StrangeAttr": True,
                    "PointUnpredictable": True,
                    "StatPredictable": True,
                    "QuasiPeriodic": False,
                    "Random": False,
                    "FixedPointAttr": False,
                    "Periodic": False,
                },
            },
        }
        questions = generate_adversarial_set(systems, n_per_type=2, seed=42)
        if len(questions) != 6:
            return False, f"Expected 6 adversarial questions, got {len(questions)}"
        return True, f"Adversarial generation: {len(questions)} questions"
    except Exception as e:
        return False, f"Adversarial generation failed: {e}"


def check_test_suite():
    """Run pytest programmatically and report results."""
    try:
        import subprocess

        result = subprocess.run(
            ["uv", "run", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        last_line = result.stdout.strip().split("\n")[-1] if result.stdout else ""
        if result.returncode == 0:
            return True, last_line
        return False, f"Tests failed: {last_line}"
    except Exception as e:
        return False, f"Test execution failed: {e}"


def parse_args():
    """Parse CLI options for validator."""
    parser = argparse.ArgumentParser(
        description="Validate ChaosBench-Logic v2 release state"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict checks (question contamination gate)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest suite",
    )
    parser.add_argument(
        "--max-duplicate-questions",
        type=int,
        default=0,
        help="Maximum allowed exact duplicate question rows for contamination check",
    )
    return parser.parse_args()


def main():
    """Run all validation checks and print results."""
    args = parse_args()

    checks = [
        ("Import", check_import),
        ("System files", lambda: check_systems("systems")),
        (
            "Manifest integrity",
            lambda: check_manifest_integrity("data", "data/v2_manifest.json"),
        ),
        ("Unique item IDs", lambda: check_unique_item_ids("data")),
        ("Run manifest registry", lambda: check_run_manifest_registry("runs")),
        ("FOL consistency", lambda: check_fol_consistency("systems")),
        ("Indicators", check_indicators),
        ("Solver repair", check_solver_repair),
        ("Regime transitions", check_regime_transitions),
        ("Adversarial", check_adversarial),
    ]

    if args.strict:
        checks.append(
            (
                "Question contamination",
                lambda: check_question_contamination(
                    "data", max_duplicates=args.max_duplicate_questions
                ),
            )
        )

    if not args.skip_tests:
        checks.append(("Test suite", check_test_suite))

    results = []
    for name, check_fn in checks:
        passed, msg = check_fn()
        status = "PASS" if passed else "FAIL"
        results.append((name, passed, msg))
        print(f"  [{status}] {name}: {msg}")

    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} checks passed")

    if n_pass < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
