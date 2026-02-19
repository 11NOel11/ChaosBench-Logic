#!/usr/bin/env python3
"""Build the v2 dataset: generates batch8+ JSONL files and manifest.

Supports both core (30 systems) and expanded (dysts) system pools.

v2.1 mode (default): Batches 8-14 use core systems. Batches 15-18 use dysts.
v2.2 mode (--config v2_2_scale.yaml): Per-system quota generation across all
    165 systems for scalable families. Replaces batches 8-18 with unified
    family-based output files.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from chaosbench.data.schemas import Question
from chaosbench.data.adversarial import generate_adversarial_set
from chaosbench.data.eligibility import get_eligible_systems, generate_eligibility_report
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
from chaosbench.data.grouping import _normalize_text


def dedupe_exact(questions: list, dedupe_log_path: str = None) -> tuple:
    """Remove exact duplicates from question list.

    Keeps first occurrence of each unique (question, system_id, ground_truth) tuple.

    Args:
        questions: List of Question objects or dicts.
        dedupe_log_path: Optional path to write removed duplicates log.

    Returns:
        Tuple of (deduplicated_questions, removed_count, removed_items).
    """
    seen_keys = {}
    kept = []
    removed = []

    for i, q in enumerate(questions):
        # Handle both Question dataclass and dict
        if hasattr(q, 'question_text'):
            q_text = q.question_text
            sys_id = q.system_id
            gt = q.ground_truth
            q_id = q.item_id
            family = q.task_family
        else:
            q_text = q.get('question', '')
            sys_id = q.get('system_id', '')
            gt = q.get('ground_truth', '')
            q_id = q.get('id', '')
            family = q.get('type', '')

        # Compute uniqueness key
        norm_text = _normalize_text(q_text)
        key = f"{family}:{sys_id}:{norm_text}:{gt}"

        if key in seen_keys:
            # Duplicate found - log it
            removed.append({
                'removed_id': q_id,
                'kept_id': seen_keys[key]['id'],
                'key': key,
                'family': family,
                'system_id': sys_id,
                'question': q_text,
            })
        else:
            # First occurrence - keep it
            seen_keys[key] = {'id': q_id, 'index': i}
            kept.append(q)

    # Write dedupe log if requested
    if dedupe_log_path and removed:
        import json
        import os
        os.makedirs(os.path.dirname(dedupe_log_path), exist_ok=True)
        with open(dedupe_log_path, 'w') as f:
            for item in removed:
                f.write(json.dumps(item) + '\n')

    return kept, len(removed), removed


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
    """Convert a Question dataclass to JSONL dict format."""
    gt_map = {"YES": "TRUE", "NO": "FALSE"}
    return {
        "id": q.item_id,
        "question": q.question_text,
        "ground_truth": gt_map.get(q.ground_truth, q.ground_truth),
        "type": q.task_family,
        "system_id": q.system_id,
        "template": template,
    }


def write_batch(questions: list, filepath: str, template: str = "V2", dedupe: bool = False) -> tuple:
    """Write questions to a JSONL file.

    Args:
        questions: List of Question objects.
        filepath: Output JSONL file path.
        template: Template version string.
        dedupe: If True, remove exact duplicates before writing.

    Returns:
        Tuple of (count_written, count_removed) if dedupe=True, else (count_written, 0).
    """
    removed_count = 0

    if dedupe:
        # Dedupe before writing
        batch_name = os.path.basename(filepath).replace('.jsonl', '')
        dedupe_log = f"reports/dedupe/{batch_name}_removed.jsonl"
        questions, removed_count, _ = dedupe_exact(questions, dedupe_log_path=dedupe_log)
        if removed_count > 0:
            print(f"    [DEDUPE] Removed {removed_count} duplicates (logged to {dedupe_log})")

    with open(filepath, "w") as f:
        for q in questions:
            line = json.dumps(question_to_jsonl(q, template))
            f.write(line + "\n")

    return len(questions), removed_count


def load_systems(systems_dir: str = "systems") -> dict:
    """Load core system JSONs (the original 30)."""
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
    """Load dysts-imported system JSONs."""
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
    """Load both core and dysts systems."""
    all_sys = load_systems(systems_dir)
    dysts_sys = load_dysts_systems(os.path.join(systems_dir, "dysts"))
    all_sys.update(dysts_sys)
    return all_sys


def load_indicators(indicators_dir: str = "systems/indicators") -> dict:
    """Load all indicator JSONs from a directory."""
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
            filtered = {k: v for k, v in data.items() if v is not None}
            indicators[sid] = filtered
    return indicators


def load_all_indicators(systems_dir: str = "systems") -> dict:
    """Load indicators from both core and dysts indicator directories."""
    indicators = load_indicators(os.path.join(systems_dir, "indicators"))
    dysts_ind = load_indicators(os.path.join(systems_dir, "dysts", "indicators"))
    indicators.update(dysts_ind)
    return indicators


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def validate_jsonl(filepath: str) -> bool:
    """Validate that every line in a JSONL file is valid JSON with TRUE/FALSE ground_truth."""
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


def _is_v22_config(cfg: Dict[str, Any]) -> bool:
    """Check if config is v2.2 format (has family_targets or per_system_items_by_family)."""
    return "family_targets" in cfg or "per_system_items_by_family" in cfg


def _balance_and_shuffle(
    questions: List[Question],
    seed: int,
    target_count: Optional[int] = None,
) -> List[Question]:
    """Balance YES/NO answers and shuffle, optionally truncating to target_count."""
    rng = random.Random(seed)
    yes_qs = [q for q in questions if q.ground_truth == "YES"]
    no_qs = [q for q in questions if q.ground_truth == "NO"]
    rng.shuffle(yes_qs)
    rng.shuffle(no_qs)

    # Interleave for balance
    balanced = []
    max_len = max(len(yes_qs), len(no_qs))
    for i in range(max_len):
        if i < len(yes_qs):
            balanced.append(yes_qs[i])
        if i < len(no_qs):
            balanced.append(no_qs[i])

    rng.shuffle(balanced)

    if target_count is not None:
        balanced = balanced[:target_count]
    return balanced


# ============================================================================
# v2.2 Scaled Generation Functions
# ============================================================================

def _generate_v22_atomic(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate atomic questions across all eligible systems with per-system quota."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("atomic", 5445)
    templates_per_predicate = cfg.get("templates_per_predicate", 1)
    enable_multiframe = cfg.get("enable_multiframe", False)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    questions = generate_atomic_questions(
        eligible_systems,
        seed=seed,
        target_count=target,
        templates_per_predicate=templates_per_predicate,
        enable_multiframe=enable_multiframe,
    )
    return questions


def _generate_v22_consistency(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
    data_dir: str,
) -> List[Question]:
    """Generate consistency paraphrase questions scaled to all eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("consistency_paraphrase", 3300)
    n_variants = cfg.get("paraphrase_variants", cfg.get("consistency", {}).get("paraphrase_variants", 3))

    # Use atomic questions as base for paraphrasing across all systems
    # Use only Frame A for consistency base (simpler, more natural paraphrase targets)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    base_atomic = generate_atomic_questions(eligible_systems, seed=seed + 100, enable_multiframe=False)

    para_questions = []
    for bq in base_atomic:
        cset = generate_paraphrase_set(bq, n_variants=n_variants, seed=seed)
        for q in cset.questions[1:]:
            para_questions.append(q)
        if len(para_questions) >= target:
            break

    return para_questions[:target]


def _generate_v22_perturbation(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate perturbation robustness questions across all eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("perturbation_robustness", 5280)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    questions = generate_perturbation_questions(
        eligible_systems, seed=seed, target_count=target,
    )
    return questions


def _generate_v22_multi_hop(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate multi-hop questions across all eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("multi_hop", 1650)
    # v2.3: default to 6-hop when config supports it (new axiom edges)
    max_hop_count = cfg.get("max_hop_count", 4)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    questions = generate_multi_hop_questions(
        eligible_systems,
        seed=seed,
        target_count=target,
        max_hop_count=max_hop_count,
    )
    return questions


def _generate_v22_fol(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate FOL inference questions across all eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("fol_inference", 1000)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    fol_task = FOLInferenceTask(systems=eligible_systems, seed=seed, target_count=target)
    questions = fol_task.generate_items()
    if len(questions) > target:
        questions = questions[:target]
    return questions


def _generate_v22_adversarial(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate adversarial questions across all eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("adversarial", 800)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}

    adv_systems = {}
    for sid in eligible_ids:
        sdata = eligible_systems[sid]
        adv_systems[sid] = {
            "name": sdata.get("name", sid),
            "truth": sdata.get("truth_assignment", {}),
        }

    # Scale n_per_type based on target
    n_per_type = max(50, target // 3)
    questions = generate_adversarial_set(adv_systems, n_per_type=n_per_type, seed=seed)

    if cfg.get("adversarial", {}).get("drop_unknown", True):
        questions = [q for q in questions if q.ground_truth != "UNKNOWN"]

    # Deduplicate
    seen = set()
    deduped = []
    for q in questions:
        norm = q.question_text.strip().lower()
        if norm not in seen:
            seen.add(norm)
            deduped.append(q)
    questions = deduped

    if len(questions) > target:
        questions = questions[:target]
    return questions


def _generate_v22_indicator_diagnostics(
    all_systems: Dict[str, Dict],
    all_indicators: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate indicator diagnostic questions for eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("indicator_diagnostics", 1500)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    eligible_indicators = {sid: all_indicators[sid] for sid in eligible_ids if sid in all_indicators}
    questions = generate_indicator_questions(eligible_systems, eligible_indicators, seed=seed)
    if len(questions) > target:
        questions = questions[:target]
    return questions


def _generate_v22_cross_indicator(
    all_systems: Dict[str, Dict],
    all_indicators: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate cross-indicator questions for eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("cross_indicator", 400)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    eligible_indicators = {sid: all_indicators[sid] for sid in eligible_ids if sid in all_indicators}
    cross_task = CrossIndicatorTask(
        systems=eligible_systems, indicators=eligible_indicators, seed=seed,
    )
    questions = cross_task.generate_items()
    if len(questions) > target:
        questions = questions[:target]
    return questions


def _generate_v22_regime_transition(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate regime transition questions for eligible systems."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("regime_transition", 400)
    task = RegimeTransitionTask(seed=seed)
    questions = task.generate_items()
    if len(questions) > target:
        questions = questions[:target]
    return questions


def _generate_v22_extended_systems(
    all_systems: Dict[str, Dict],
    eligible_ids: List[str],
    cfg: Dict[str, Any],
) -> List[Question]:
    """Generate extended systems questions."""
    seed = cfg["seed"]
    target = cfg.get("family_targets", {}).get("extended_systems", 300)
    eligible_systems = {sid: all_systems[sid] for sid in eligible_ids}
    ext_task = ExtendedSystemsTask(systems=eligible_systems, seed=seed)
    questions = ext_task.generate_items()
    if len(questions) > target:
        questions = questions[:target]
    return questions


def build_v22(cfg: Dict[str, Any], data_dir: str, systems_dir: str, verbose: bool = False, args=None):
    """Build v2.2 scaled dataset with per-system quota generation.

    Generates one output file per family instead of numbered batches.
    All 165 systems are used where eligible.
    """
    seed = cfg["seed"]
    template = cfg.get("template", "V2")

    print("\n" + "=" * 70)
    print("  ChaosBench-Logic v2.2 Scaled Dataset Builder")
    print("=" * 70)
    print(f"  Seed: {seed}")

    # Load all systems
    all_systems = load_all_systems(systems_dir)
    all_indicators = load_all_indicators(systems_dir)
    n_core = len(load_systems(systems_dir))
    n_dysts = len(all_systems) - n_core
    print(f"  Systems: {len(all_systems)} total ({n_core} core + {n_dysts} dysts)")
    print(f"  Indicators: {len(all_indicators)} systems with indicator data")

    # Apply max_systems cap
    max_sys = cfg.get("max_systems")
    if max_sys and len(all_systems) > max_sys:
        rng = random.Random(seed)
        keep = sorted(rng.sample(sorted(all_systems.keys()), max_sys))
        all_systems = {sid: all_systems[sid] for sid in keep}
        print(f"  Capped to {max_sys} systems")

    # Compute eligibility
    if verbose:
        report = generate_eligibility_report(all_systems, all_indicators)
        print("\n  Eligibility Report:")
        for family, info in sorted(report["per_family"].items()):
            print(f"    {family}: {info['eligible_count']}/{info['total_systems']} eligible")

    # Define generation pipeline
    families = [
        ("atomic", _generate_v22_atomic),
        ("perturbation_robustness", _generate_v22_perturbation),
        ("consistency_paraphrase", None),  # Special handling
        ("multi_hop", _generate_v22_multi_hop),
        ("fol_inference", _generate_v22_fol),
        ("adversarial", _generate_v22_adversarial),
        ("indicator_diagnostics", None),  # Needs indicators
        ("cross_indicator", None),  # Needs indicators
        ("regime_transition", _generate_v22_regime_transition),
        ("extended_systems", _generate_v22_extended_systems),
    ]

    batch_counts = {}
    total_removed = 0  # Track dedupe removals
    generation_stats = {}  # Track generation statistics per family
    all_questions: List[Dict[str, Any]] = []

    for family_name, generator in families:
        eligible_ids = get_eligible_systems(all_systems, family_name, all_indicators)
        target = cfg.get("family_targets", {}).get(family_name, 0)

        print(f"\n  Generating {family_name} ({len(eligible_ids)} eligible systems, target={target})...")

        if family_name == "consistency_paraphrase":
            questions = _generate_v22_consistency(all_systems, eligible_ids, cfg, data_dir)
        elif family_name == "indicator_diagnostics":
            questions = _generate_v22_indicator_diagnostics(
                all_systems, all_indicators, eligible_ids, cfg,
            )
        elif family_name == "cross_indicator":
            questions = _generate_v22_cross_indicator(
                all_systems, all_indicators, eligible_ids, cfg,
            )
        elif generator is not None:
            questions = generator(all_systems, eligible_ids, cfg)
        else:
            questions = []

        if not questions:
            print(f"    [SKIP] No questions generated for {family_name}")
            generation_stats[family_name] = {
                "requested_target": target,
                "eligible_systems": len(eligible_ids),
                "generated_count": 0,
                "final_count": 0,
                "dedupe_removed": 0,
                "gap_from_target": target,
                "achievement_pct": 0.0,
            }
            continue

        # Write family output file
        output_name = f"v22_{family_name}"
        filepath = os.path.join(data_dir, f"{output_name}.jsonl")
        generated_count = len(questions)
        count, removed = write_batch(questions, filepath, template=template, dedupe=args.dedupe_exact if args else False)
        batch_counts[output_name] = count
        total_removed += removed

        # Track statistics
        generation_stats[family_name] = {
            "requested_target": target,
            "eligible_systems": len(eligible_ids),
            "generated_count": generated_count,
            "final_count": count,
            "dedupe_removed": removed,
            "gap_from_target": target - count if target > 0 else 0,
            "achievement_pct": (count / target * 100) if target > 0 else 100.0,
        }

        print(f"    Wrote {count} questions to {output_name}.jsonl")

        # Collect for quality gates
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_questions.append(json.loads(line))

    # Count existing v1 batches
    existing_total = 0
    v1_archive_dir = os.path.join(data_dir, "archive", "v1")
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
        fpath = os.path.join(v1_archive_dir, batch_files[i])
        if os.path.isfile(fpath):
            with open(fpath, "r") as f:
                n = sum(1 for _ in f)
            existing_total += n

    # Run quality gates
    print("\n" + "=" * 70)
    print("  QUALITY GATES")
    print("=" * 70)

    from chaosbench.quality.gates import run_all_gates
    gate_config = cfg.get("quality_gates", {})
    gate_results = run_all_gates(all_questions, gate_config)

    all_gates_passed = True
    for result in gate_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.gate_name}: {result.details}")
        if result.violations:
            for v in result.violations[:3]:
                print(f"         - {v}")
        if not result.passed:
            all_gates_passed = False

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
    new_total = sum(batch_counts.values())
    manifest = {
        "schema_version": "v2",
        "version": cfg.get("version", "2.3.0"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_config": cfg,
        "batches": {},
        "total_new_questions": new_total,
        "total_existing_questions": existing_total,
        "total_questions": existing_total + new_total,
        "dedupe_exact_enabled": args.dedupe_exact if args else False,
        "dedupe_exact_removed": total_removed,
        "dedupe_policy": "keep_first_sorted_by_id" if (args and args.dedupe_exact) else "none",
        "quality_gates": {
            r.gate_name: {"passed": r.passed, "details": r.details}
            for r in gate_results
        },
    }

    for batch_name, count in sorted(batch_counts.items()):
        fpath = os.path.join(data_dir, f"{batch_name}.jsonl")
        manifest["batches"][batch_name] = {
            "count": count,
            "sha256": compute_file_hash(fpath),
        }

    manifest_path = os.path.join(data_dir, "v2_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save generation statistics for diagnostic analysis
    stats_dir = os.path.join("reports", "scale_diag")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_file": args.config if args and args.config else "default",
            "seed": seed,
            "total_requested": sum(s["requested_target"] for s in generation_stats.values()),
            "total_generated": sum(s["generated_count"] for s in generation_stats.values()),
            "total_final": sum(s["final_count"] for s in generation_stats.values()),
            "total_dedupe_removed": total_removed,
            "per_family": generation_stats,
        }, f, indent=2)
    print(f"\n  Generation statistics saved to: {stats_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Existing questions (batch1-7): {existing_total}")
    print(f"  New v2.2 questions:            {new_total}")
    print(f"  Total questions:               {manifest['total_questions']}")
    print(f"  Families generated:            {len(batch_counts)}")
    print(f"  Manifest:                      {manifest_path}")
    print(f"  All valid:                     {all_valid}")
    print(f"  Quality gates:                 {'ALL PASSED' if all_gates_passed else 'SOME FAILED'}")

    if not all_valid:
        sys.exit(1)

    print("\nDone!")
    return manifest


def build_v21(cfg: Dict[str, Any], data_dir: str, systems_dir: str, args=None):
    """Build v2.1 dataset (original batch-based generation)."""
    indicators_dir = os.path.join(systems_dir, "indicators")

    print("\n" + "=" * 70)
    print("  ChaosBench-Logic v2.1 Dataset Builder")
    print("=" * 70)

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
    total_removed = 0  # Track dedupe removals

    # Step 2: Batch 8 - Indicator diagnostics
    print("\n[2/7] Generating batch8_indicator_diagnostics...")
    ind_questions = generate_indicator_questions(systems, indicators, seed=cfg["seed"])
    count, removed = write_batch(
        ind_questions,
        os.path.join(data_dir, "batch8_indicator_diagnostics.jsonl"),
        template=cfg["template"],
        dedupe=args.dedupe_exact if args else False,
    )
    batch_counts["batch8_indicator_diagnostics"] = count
    total_removed += removed
    print(f"  Wrote {count} questions")

    # Step 3: Batch 9 - Regime transitions
    print("\n[3/7] Generating batch9_regime_transitions...")
    task = RegimeTransitionTask(seed=cfg["seed"])
    regime_questions = task.generate_items()
    count, removed = write_batch(
        regime_questions,
        os.path.join(data_dir, "batch9_regime_transitions.jsonl"),
        template=cfg["template"],
        dedupe=args.dedupe_exact if args else False,
    )
    batch_counts["batch9_regime_transitions"] = count
    total_removed += removed
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
    seen_texts = set()
    deduped = []
    for q in adv_questions:
        norm = q.question_text.strip().lower()
        if norm not in seen_texts:
            seen_texts.add(norm)
            deduped.append(q)
    if len(deduped) < len(adv_questions):
        print(f"  Deduped: {len(adv_questions)} -> {len(deduped)} (removed {len(adv_questions) - len(deduped)} duplicates)")
    adv_questions = deduped
    count, removed = write_batch(
        adv_questions,
        os.path.join(data_dir, "batch10_adversarial.jsonl"),
        template=cfg["template"],
        dedupe=args.dedupe_exact if args else False,
    )
    batch_counts["batch10_adversarial"] = count
    total_removed += removed
    print(f"  Wrote {count} questions")

    # Step 5: Batch 11 - Consistency paraphrase
    print("\n[5/7] Generating batch11_consistency_paraphrase...")
    v1_archive_dir = os.path.join(data_dir, "archive", "v1")
    batch1_path = os.path.join(v1_archive_dir, "batch1_atomic_implication.jsonl")
    batch2_path = os.path.join(v1_archive_dir, "batch2_multiHop_crossSystem.jsonl")
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

    count, removed = write_batch(
        para_questions,
        os.path.join(data_dir, "batch11_consistency_paraphrase.jsonl"),
        template=cfg["template"],
        dedupe=args.dedupe_exact if args else False,
    )
    batch_counts["batch11_consistency_paraphrase"] = count
    total_removed += removed
    print(f"  Wrote {count} questions")

    # Step 6: Batch 12 - FOL inference
    print("\n[6/7] Generating batch12_fol_inference...")
    fol_task = FOLInferenceTask(systems=systems, seed=cfg["seed"])
    fol_questions = fol_task.generate_items()
    count, removed = write_batch(
        fol_questions,
        os.path.join(data_dir, "batch12_fol_inference.jsonl"),
        template=cfg["template"],
        dedupe=args.dedupe_exact if args else False,
    )
    batch_counts["batch12_fol_inference"] = count
    total_removed += removed
    print(f"  Wrote {count} questions")

    # Step 7: Batch 13 - Extended systems
    print("\n[7a/7] Generating batch13_extended_systems...")
    ext_task = ExtendedSystemsTask(systems=systems, seed=cfg["seed"])
    ext_questions = ext_task.generate_items()
    count, removed = write_batch(
        ext_questions,
        os.path.join(data_dir, "batch13_extended_systems.jsonl"),
        template=cfg["template"],
        dedupe=args.dedupe_exact if args else False,
    )
    batch_counts["batch13_extended_systems"] = count
    total_removed += removed
    print(f"  Wrote {count} questions")

    # Batch 14 - Cross indicator
    print("\n[7b/7] Generating batch14_cross_indicator...")
    cross_task = CrossIndicatorTask(
        systems=systems,
        indicators=indicators,
        seed=cfg["seed"],
    )
    cross_questions = cross_task.generate_items()
    count, removed = write_batch(
        cross_questions,
        os.path.join(data_dir, "batch14_cross_indicator.jsonl"),
        template=cfg["template"],
        dedupe=args.dedupe_exact if args else False,
    )
    batch_counts["batch14_cross_indicator"] = count
    total_removed += removed
    print(f"  Wrote {count} questions")

    # Batches 15-18: Dysts-expanded system pool
    dysts_systems = load_dysts_systems("systems/dysts")
    all_systems = {**systems, **dysts_systems}
    n_dysts = len(dysts_systems)

    if n_dysts > 0:
        print(f"\n  Loaded {n_dysts} dysts systems ({len(all_systems)} total)")

        print("\n[8/11] Generating batch15_atomic_dysts...")
        dysts_only = {sid: s for sid, s in all_systems.items() if sid not in systems}
        atomic_target = cfg.get("batches", {}).get("batch15_atomic_dysts", {}).get("target_count", 3000)
        atomic_qs = generate_atomic_questions(
            dysts_only, seed=cfg["seed"], target_count=atomic_target,
        )
        count, removed = write_batch(
            atomic_qs,
            os.path.join(data_dir, "batch15_atomic_dysts.jsonl"),
            template=cfg["template"],
            dedupe=args.dedupe_exact if args else False,
        )
        batch_counts["batch15_atomic_dysts"] = count
        total_removed += removed
        print(f"  Wrote {count} questions")

        print("\n[9/11] Generating batch16_multi_hop_dysts...")
        mh_target = cfg.get("batches", {}).get("batch16_multi_hop_dysts", {}).get("target_count", 500)
        dysts_only_mh = {sid: s for sid, s in all_systems.items() if sid not in systems}
        mh_qs = generate_multi_hop_questions(
            dysts_only_mh, seed=cfg["seed"], target_count=mh_target,
        )
        for q in mh_qs:
            q.item_id = f"dysts_{q.item_id}"
        count, removed = write_batch(
            mh_qs,
            os.path.join(data_dir, "batch16_multi_hop_dysts.jsonl"),
            template=cfg["template"],
            dedupe=args.dedupe_exact if args else False,
        )
        batch_counts["batch16_multi_hop_dysts"] = count
        total_removed += removed
        print(f"  Wrote {count} questions")

        print("\n[10/11] Generating batch17_perturbation_robustness...")
        perturb_target = cfg.get("batches", {}).get("batch17_perturbation_robustness", {}).get("target_count", 3000)
        perturb_qs = generate_perturbation_questions(
            all_systems, seed=cfg["seed"], target_count=perturb_target,
        )
        count, removed = write_batch(
            perturb_qs,
            os.path.join(data_dir, "batch17_perturbation_robustness.jsonl"),
            template=cfg["template"],
            dedupe=args.dedupe_exact if args else False,
        )
        batch_counts["batch17_perturbation_robustness"] = count
        total_removed += removed
        print(f"  Wrote {count} questions")

        print("\n[11/11] Generating batch18_fol_dysts...")
        fol_dysts_target = cfg.get("batches", {}).get("batch18_fol_dysts", {}).get("target_count", 2500)
        dysts_only_systems = {sid: s for sid, s in all_systems.items() if sid not in systems}
        fol_dysts_task = FOLInferenceTask(systems=dysts_only_systems, seed=cfg["seed"])
        fol_dysts_qs = fol_dysts_task.generate_items()
        if fol_dysts_target and len(fol_dysts_qs) > fol_dysts_target:
            fol_dysts_qs = fol_dysts_qs[:fol_dysts_target]
        fol_dysts_qs = [q for q in fol_dysts_qs if q.system_id != "generic"]
        for q in fol_dysts_qs:
            q.item_id = f"dysts_{q.item_id}"
        count, removed = write_batch(
            fol_dysts_qs,
            os.path.join(data_dir, "batch18_fol_dysts.jsonl"),
            template=cfg["template"],
            dedupe=args.dedupe_exact if args else False,
        )
        batch_counts["batch18_fol_dysts"] = count
        total_removed += removed
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
        "schema_version": "v2",
        "version": "2.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_config": cfg,
        "batches": {},
        "total_new_questions": sum(batch_counts.values()),
        "dedupe_exact_enabled": args.dedupe_exact if args else False,
        "dedupe_exact_removed": total_removed,
        "dedupe_policy": "keep_first_sorted_by_id" if (args and args.dedupe_exact) else "none",
    }

    existing_total = 0
    v1_archive_dir = os.path.join(data_dir, "archive", "v1")
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
        fpath = os.path.join(v1_archive_dir, batch_files[i])
        if os.path.isfile(fpath):
            with open(fpath, "r") as f:
                n = sum(1 for _ in f)
            existing_total += n

    manifest["total_existing_questions"] = existing_total
    manifest["total_questions"] = existing_total + sum(batch_counts.values())

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
    print(f"  Total questions:               {manifest['total_questions']}")
    print(f"  Manifest:                      {manifest_path}")
    print(f"  All valid:                     {all_valid}")

    if not all_valid:
        sys.exit(1)

    print("\nDone!")


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-system eligibility and quota allocation",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Regenerate manifest with hashes from existing files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: data/)",
    )
    parser.add_argument(
        "--dedupe_exact",
        action="store_true",
        help="Remove exact duplicates (keep first occurrence)",
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
    data_dir = args.output_dir or "data"
    systems_dir = "systems"

    os.makedirs(data_dir, exist_ok=True)

    print("=" * 70)
    print("  ChaosBench-Logic Dataset Builder")
    print("=" * 70)
    print(f"  Config: {args.config if args.config else '<defaults>'}")
    print(f"  Seed:   {cfg['seed']}")
    print(f"  Mode:   {'v2.2 (scaled)' if _is_v22_config(cfg) else 'v2.1 (batch-based)'}")

    if args.dry_run:
        print("\n  [DRY RUN] Config validated.")
        all_systems = load_all_systems(systems_dir)
        all_indicators = load_all_indicators(systems_dir)
        n_core = len(load_systems(systems_dir))
        n_dysts = len(all_systems) - n_core
        print(f"  Core systems: {n_core}")
        print(f"  Dysts systems: {n_dysts}")
        print(f"  Total systems: {len(all_systems)}")
        if _is_v22_config(cfg):
            print(f"\n  v2.2 Family Targets:")
            for family, target in sorted(cfg.get("family_targets", {}).items()):
                eligible = get_eligible_systems(all_systems, family, all_indicators)
                print(f"    {family}: target={target}, eligible={len(eligible)} systems")
        sys.exit(0)

    if args.manifest_only:
        print("\n  [MANIFEST ONLY] Regenerating manifest from existing files...")
        # Regenerate manifest from existing JSONL files
        import glob
        batch_counts = {}
        for fpath in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
            fname = os.path.basename(fpath).replace(".jsonl", "")
            with open(fpath, "r") as f:
                n = sum(1 for line in f if line.strip())
            batch_counts[fname] = n

        manifest = {
            "version": cfg.get("version", "2.3.0"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batches": {},
            "total_questions": sum(batch_counts.values()),
        }
        for name, count in sorted(batch_counts.items()):
            fpath = os.path.join(data_dir, f"{name}.jsonl")
            manifest["batches"][name] = {
                "count": count,
                "sha256": compute_file_hash(fpath),
            }
        manifest_path = os.path.join(data_dir, "v2_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest: {manifest_path}")
        print(f"  Total: {sum(batch_counts.values())} questions across {len(batch_counts)} files")
        sys.exit(0)

    # Route to appropriate builder
    if _is_v22_config(cfg):
        build_v22(cfg, data_dir, systems_dir, verbose=args.verbose, args=args)
    else:
        build_v21(cfg, data_dir, systems_dir, args=args)


if __name__ == "__main__":
    main()
