#!/usr/bin/env python3
"""Build the v2 dataset: generates batch8-18 JSONL files and manifest.

Supports both core (30 systems) and expanded (dysts) system pools.
Batches 8-14 use core systems. Batches 15-18 use dysts-expanded systems.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import yaml

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from chaosbench.data.schemas import Question
from chaosbench.data.adversarial import generate_adversarial_set
from chaosbench.data.indicators.populate import populate_all_systems
from chaosbench.tasks.regime_transition import RegimeTransitionTask
from chaosbench.tasks.indicator_diagnostics import generate_indicator_questions
from chaosbench.tasks.fol_inference import FOLInferenceTask
from chaosbench.tasks.extended_systems import ExtendedSystemsTask
from chaosbench.tasks.cross_indicator import CrossIndicatorTask
from chaosbench.tasks.consistency import generate_paraphrase_set
from chaosbench.tasks.atomic import generate_atomic_questions
from chaosbench.tasks.multi_hop import generate_multi_hop_questions
from chaosbench.tasks.perturbation_robustness import generate_perturbation_questions
from chaosbench.eval.runner import load_jsonl


DEFAULT_GENERATION_CONFIG = {
    "seed": 42,
    "template": "V2",
    "indicators": {
        "seed": 42,
    },
    "adversarial": {
        "n_per_type": 50,
        "drop_unknown": True,
    },
    "consistency": {
        "batch2_take": 50,
        "paraphrase_variants": 3,
    },
}


def question_to_jsonl(q: Question, template: str = "V2") -> dict:
    """Convert a Question dataclass to JSONL dict format.

    Args:
        q: Question object with YES/NO ground_truth.
        template: Template label for the JSONL record.

    Returns:
        Dict with id, question, ground_truth (TRUE/FALSE), type, system_id, template.
    """
    gt_map = {"YES": "TRUE", "NO": "FALSE"}
    return {
        "id": q.item_id,
        "question": q.question_text,
        "ground_truth": gt_map.get(q.ground_truth, q.ground_truth),
        "type": q.task_family,
        "system_id": q.system_id,
        "template": template,
    }


def write_batch(questions: list, filepath: str, template: str = "V2") -> int:
    """Write questions to a JSONL file.

    Args:
        questions: List of Question objects.
        filepath: Output JSONL path.
        template: Template label.

    Returns:
        Number of lines written.
    """
    with open(filepath, "w") as f:
        for q in questions:
            line = json.dumps(question_to_jsonl(q, template))
            f.write(line + "\n")
    return len(questions)


def load_systems(systems_dir: str = "systems") -> dict:
    """Load core system JSONs (the original 30).

    Args:
        systems_dir: Path to systems directory.

    Returns:
        Dict mapping system_id to system data.
    """
    systems = {}
    for fname in sorted(os.listdir(systems_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(systems_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            systems[sid] = data
    return systems


def load_dysts_systems(dysts_dir: str = "systems/dysts") -> dict:
    """Load dysts-imported system JSONs.

    Args:
        dysts_dir: Path to dysts systems directory.

    Returns:
        Dict mapping system_id to system data. Empty dict if dir missing.
    """
    systems = {}
    if not os.path.isdir(dysts_dir):
        return systems
    for fname in sorted(os.listdir(dysts_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(dysts_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            systems[sid] = data
    return systems


def load_all_systems(systems_dir: str = "systems") -> dict:
    """Load both core and dysts systems.

    Args:
        systems_dir: Path to top-level systems directory.

    Returns:
        Dict mapping system_id to system data (core + dysts combined).
    """
    all_sys = load_systems(systems_dir)
    dysts_sys = load_dysts_systems(os.path.join(systems_dir, "dysts"))
    all_sys.update(dysts_sys)
    return all_sys


def load_indicators(indicators_dir: str = "systems/indicators") -> dict:
    """Load all indicator JSONs.

    Args:
        indicators_dir: Path to indicators directory.

    Returns:
        Dict mapping system_id to indicator data.
    """
    indicators = {}
    if not os.path.isdir(indicators_dir):
        return indicators
    for fname in sorted(os.listdir(indicators_dir)):
        if not fname.endswith("_indicators.json"):
            continue
        fpath = os.path.join(indicators_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            # Filter out None values so downstream code can use "key in dict" checks
            filtered = {k: v for k, v in data.items() if v is not None}
            indicators[sid] = filtered
    return indicators


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file contents.

    Args:
        filepath: Path to file.

    Returns:
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def validate_jsonl(filepath: str) -> bool:
    """Validate that every line in a JSONL file is valid JSON with TRUE/FALSE ground_truth.

    Args:
        filepath: Path to JSONL file.

    Returns:
        True if all lines are valid.
    """
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gt = record.get("ground_truth")
            if gt not in ("TRUE", "FALSE"):
                print(f"  [ERROR] {filepath}:{i} ground_truth={gt!r}")
                return False
    return True


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested dictionaries recursively."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_generation_config(config_path: str | None) -> Dict[str, Any]:
    """Load YAML config and merge onto defaults."""
    if config_path is None:
        return copy.deepcopy(DEFAULT_GENERATION_CONFIG)

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Generation config must be a YAML mapping/object")

    return _deep_update(DEFAULT_GENERATION_CONFIG, loaded)


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build ChaosBench-Logic v2 dataset batches and manifest"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML generation config (default: internal defaults)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override top-level generation seed from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan without generating data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_generation_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("indicators", {})
    cfg["indicators"].setdefault("seed", cfg["seed"])

    os.chdir(PROJECT_ROOT)
    data_dir = "data"
    systems_dir = "systems"
    indicators_dir = "systems/indicators"

    os.makedirs(data_dir, exist_ok=True)

    print("=" * 70)
    print("  ChaosBench-Logic v2 Dataset Builder")
    print("=" * 70)
    print(f"  Config: {args.config if args.config else '<defaults>'}")
    print(f"  Seed:   {cfg['seed']}")

    if args.dry_run:
        print("\n  [DRY RUN] Config validated. Would generate batches 8-18.")
        dysts_dir = os.path.join(systems_dir, "dysts")
        n_dysts = len([f for f in os.listdir(dysts_dir) if f.endswith(".json")]) if os.path.isdir(dysts_dir) else 0
        print(f"  Core systems: 30")
        print(f"  Dysts systems: {n_dysts}")
        print(f"  Batches 15-18: {'YES' if n_dysts > 0 else 'SKIP (no dysts systems)'}")
        sys.exit(0)

    # Step 1: Populate indicators
    print("\n[1/7] Computing indicators for all 30 systems...")
    populate_all_systems(
        systems_dir=systems_dir,
        output_dir=indicators_dir,
        seed=cfg["indicators"]["seed"],
    )
    print(f"  Wrote indicator files to {indicators_dir}/")

    systems = load_systems(systems_dir)
    indicators = load_indicators(indicators_dir)
    print(f"  Loaded {len(systems)} systems, {len(indicators)} indicator sets")

    batch_counts = {}

    # Step 2: Batch 8 - Indicator diagnostics
    print("\n[2/7] Generating batch8_indicator_diagnostics...")
    ind_questions = generate_indicator_questions(systems, indicators, seed=cfg["seed"])
    count = write_batch(
        ind_questions,
        os.path.join(data_dir, "batch8_indicator_diagnostics.jsonl"),
        template=cfg["template"],
    )
    batch_counts["batch8_indicator_diagnostics"] = count
    print(f"  Wrote {count} questions")

    # Step 3: Batch 9 - Regime transitions
    print("\n[3/7] Generating batch9_regime_transitions...")
    task = RegimeTransitionTask(seed=cfg["seed"])
    regime_questions = task.generate_items()
    count = write_batch(
        regime_questions,
        os.path.join(data_dir, "batch9_regime_transitions.jsonl"),
        template=cfg["template"],
    )
    batch_counts["batch9_regime_transitions"] = count
    print(f"  Wrote {count} questions")

    # Step 4: Batch 10 - Adversarial
    print("\n[4/7] Generating batch10_adversarial...")
    adv_systems = {}
    for sid, sdata in systems.items():
        adv_systems[sid] = {
            "name": sdata.get("name", sid),
            "truth": sdata.get("truth_assignment", {}),
        }
    adv_questions = generate_adversarial_set(
        adv_systems,
        n_per_type=cfg["adversarial"]["n_per_type"],
        seed=cfg["seed"],
    )
    if cfg["adversarial"]["drop_unknown"]:
        adv_questions = [q for q in adv_questions if q.ground_truth != "UNKNOWN"]
    count = write_batch(
        adv_questions,
        os.path.join(data_dir, "batch10_adversarial.jsonl"),
        template=cfg["template"],
    )
    batch_counts["batch10_adversarial"] = count
    print(f"  Wrote {count} questions")

    # Step 5: Batch 11 - Consistency paraphrase
    print("\n[5/7] Generating batch11_consistency_paraphrase...")
    batch1_path = os.path.join(data_dir, "batch1_atomic_implication.jsonl")
    batch2_path = os.path.join(data_dir, "batch2_multiHop_crossSystem.jsonl")
    batch1_items = load_jsonl(batch1_path)
    batch2_items = load_jsonl(batch2_path)

    base_questions = []
    for item in batch1_items:
        base_questions.append(
            Question(
                item_id=item["id"],
                question_text=item["question"],
                system_id=item.get("system_id", "unknown"),
                task_family=item.get("type", "atomic"),
                ground_truth="YES" if item["ground_truth"] == "TRUE" else "NO",
                predicates=[],
            )
        )
    for item in batch2_items[: cfg["consistency"]["batch2_take"]]:
        base_questions.append(
            Question(
                item_id=item["id"],
                question_text=item["question"],
                system_id=item.get("system_id", "unknown"),
                task_family=item.get("type", "multi_hop"),
                ground_truth="YES" if item["ground_truth"] == "TRUE" else "NO",
                predicates=[],
            )
        )

    para_questions = []
    for bq in base_questions:
        cset = generate_paraphrase_set(
            bq,
            n_variants=cfg["consistency"]["paraphrase_variants"],
            seed=cfg["seed"],
        )
        for q in cset.questions[1:]:
            para_questions.append(q)

    count = write_batch(
        para_questions,
        os.path.join(data_dir, "batch11_consistency_paraphrase.jsonl"),
        template=cfg["template"],
    )
    batch_counts["batch11_consistency_paraphrase"] = count
    print(f"  Wrote {count} questions")

    # Step 6: Batch 12 - FOL inference
    print("\n[6/7] Generating batch12_fol_inference...")
    fol_task = FOLInferenceTask(systems=systems, seed=cfg["seed"])
    fol_questions = fol_task.generate_items()
    count = write_batch(
        fol_questions,
        os.path.join(data_dir, "batch12_fol_inference.jsonl"),
        template=cfg["template"],
    )
    batch_counts["batch12_fol_inference"] = count
    print(f"  Wrote {count} questions")

    # Step 7: Batch 13 - Extended systems
    print("\n[7a/7] Generating batch13_extended_systems...")
    ext_task = ExtendedSystemsTask(systems=systems, seed=cfg["seed"])
    ext_questions = ext_task.generate_items()
    count = write_batch(
        ext_questions,
        os.path.join(data_dir, "batch13_extended_systems.jsonl"),
        template=cfg["template"],
    )
    batch_counts["batch13_extended_systems"] = count
    print(f"  Wrote {count} questions")

    # Batch 14 - Cross indicator
    print("\n[7b/7] Generating batch14_cross_indicator...")
    cross_task = CrossIndicatorTask(
        systems=systems,
        indicators=indicators,
        seed=cfg["seed"],
    )
    cross_questions = cross_task.generate_items()
    count = write_batch(
        cross_questions,
        os.path.join(data_dir, "batch14_cross_indicator.jsonl"),
        template=cfg["template"],
    )
    batch_counts["batch14_cross_indicator"] = count
    print(f"  Wrote {count} questions")

    # ---- Batches 15-18: Dysts-expanded system pool ----
    dysts_systems = load_dysts_systems("systems/dysts")
    all_systems = {**systems, **dysts_systems}
    n_dysts = len(dysts_systems)

    if n_dysts > 0:
        print(f"\n  Loaded {n_dysts} dysts systems ({len(all_systems)} total)")

        # Step 8: Batch 15 - Atomic questions on all systems
        print("\n[8/11] Generating batch15_atomic_dysts...")
        atomic_target = cfg.get("batches", {}).get("batch15_atomic_dysts", {}).get("target_count", 3000)
        atomic_qs = generate_atomic_questions(
            all_systems,
            seed=cfg["seed"],
            target_count=atomic_target,
        )
        count = write_batch(
            atomic_qs,
            os.path.join(data_dir, "batch15_atomic_dysts.jsonl"),
            template=cfg["template"],
        )
        batch_counts["batch15_atomic_dysts"] = count
        print(f"  Wrote {count} questions")

        # Step 9: Batch 16 - Multi-hop on all systems
        print("\n[9/11] Generating batch16_multi_hop_dysts...")
        mh_target = cfg.get("batches", {}).get("batch16_multi_hop_dysts", {}).get("target_count", 2000)
        mh_qs = generate_multi_hop_questions(
            all_systems,
            seed=cfg["seed"],
            target_count=mh_target,
        )
        count = write_batch(
            mh_qs,
            os.path.join(data_dir, "batch16_multi_hop_dysts.jsonl"),
            template=cfg["template"],
        )
        batch_counts["batch16_multi_hop_dysts"] = count
        print(f"  Wrote {count} questions")

        # Step 10: Batch 17 - Perturbation robustness on mixed pool
        print("\n[10/11] Generating batch17_perturbation_robustness...")
        perturb_target = cfg.get("batches", {}).get("batch17_perturbation_robustness", {}).get("target_count", 3000)
        perturb_qs = generate_perturbation_questions(
            all_systems,
            seed=cfg["seed"],
            target_count=perturb_target,
        )
        count = write_batch(
            perturb_qs,
            os.path.join(data_dir, "batch17_perturbation_robustness.jsonl"),
            template=cfg["template"],
        )
        batch_counts["batch17_perturbation_robustness"] = count
        print(f"  Wrote {count} questions")

        # Step 11: Batch 18 - FOL inference on all systems
        print("\n[11/11] Generating batch18_fol_dysts...")
        fol_dysts_target = cfg.get("batches", {}).get("batch18_fol_dysts", {}).get("target_count", 2500)
        fol_dysts_task = FOLInferenceTask(systems=all_systems, seed=cfg["seed"])
        fol_dysts_qs = fol_dysts_task.generate_items()
        if fol_dysts_target and len(fol_dysts_qs) > fol_dysts_target:
            fol_dysts_qs = fol_dysts_qs[:fol_dysts_target]
        count = write_batch(
            fol_dysts_qs,
            os.path.join(data_dir, "batch18_fol_dysts.jsonl"),
            template=cfg["template"],
        )
        batch_counts["batch18_fol_dysts"] = count
        print(f"  Wrote {count} questions")
    else:
        print("\n  [SKIP] No dysts systems found - skipping batches 15-18")
        print("  Run: python scripts/import_dysts_systems.py to import dysts systems")

    # Validation
    print("\n" + "=" * 70)
    print("  VALIDATION")
    print("=" * 70)

    all_valid = True
    for batch_name in sorted(batch_counts.keys()):
        fpath = os.path.join(data_dir, f"{batch_name}.jsonl")
        valid = validate_jsonl(fpath)
        status = "OK" if valid else "FAIL"
        print(f"  [{status}] {batch_name}.jsonl ({batch_counts[batch_name]} lines)")
        if not valid:
            all_valid = False

    # Manifest
    manifest = {
        "version": "2.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_config": cfg,
        "batches": {},
        "total_new_questions": sum(batch_counts.values()),
    }

    # Count existing batches
    existing_total = 0
    for i in range(1, 8):
        batch_files = {
            1: "batch1_atomic_implication.jsonl",
            2: "batch2_multiHop_crossSystem.jsonl",
            3: "batch3_pde_chem_bio.jsonl",
            4: "batch4_maps_advanced.jsonl",
            5: "batch5_counterfactual_high_difficulty.jsonl",
            6: "batch6_deep_bias_probes.jsonl",
            7: "batch7_multiturn_advanced.jsonl",
        }
        fpath = os.path.join(data_dir, batch_files[i])
        if os.path.isfile(fpath):
            with open(fpath, "r") as f:
                n = sum(1 for _ in f)
            existing_total += n

    manifest["total_existing_questions"] = existing_total
    manifest["grand_total"] = existing_total + sum(batch_counts.values())

    for batch_name, count in sorted(batch_counts.items()):
        fpath = os.path.join(data_dir, f"{batch_name}.jsonl")
        manifest["batches"][batch_name] = {
            "count": count,
            "sha256": compute_file_hash(fpath),
        }

    manifest_path = os.path.join(data_dir, "v2_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    max_batch = max(int(k.split("_")[0].replace("batch", "")) for k in batch_counts.keys())
    print(f"  Existing questions (batch1-7): {existing_total}")
    print(f"  New questions (batch8-{max_batch}):    {sum(batch_counts.values())}")
    print(f"  Grand total:                   {manifest['grand_total']}")
    print(f"  Manifest:                      {manifest_path}")
    print(f"  All valid:                     {all_valid}")

    if not all_valid:
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
