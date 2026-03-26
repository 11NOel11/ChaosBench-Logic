#!/usr/bin/env python3
"""Build CARE-v3 repair pack with boundary analysis and audited artifacts.

Usage:
    python scripts/build_repair_pack.py --mode smoke
    python scripts/build_repair_pack.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.repair.constraints import (
    build_truth_assignments,
    compute_group_inconsistency_rate,
    count_axiom_violations,
)
from chaosbench.repair.controls import (
    budget_match_candidate_labels,
    inject_parser_noise,
    random_flip_labels,
    shuffled_gate_families,
)
from chaosbench.repair.engine import (
    compute_axiom_violation_rate,
    load_selector_index,
    read_jsonl,
    repair_records,
    write_jsonl,
)
from chaosbench.repair.types import RepairConfig

OUT_DEFAULT = PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "repair_v3"

DEV_PROVIDERS = {
    "ollama/llama3.1:8b",
    "ollama/mistral:7b",
    "ollama/qwen2.5:14b",
    "ollama/qwen2.5:32b",
}

HELDOUT_PROVIDERS = {
    "openai/gpt-4o",
    "openai/gpt-5.2",
    "anthropic/claude-sonnet-4-6",
    "deepseek/deepseek-chat",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
}

GATE_VARIANTS: List[Tuple[str, Optional[Tuple[str, ...]]]] = [
    ("all", None),
    ("multi_hop", ("multi_hop",)),
    ("multi_hop+fol", ("multi_hop", "fol_inference")),
    ("multi_hop+consistency", ("multi_hop", "consistency_paraphrase")),
    (
        "multi_hop+fol+consistency",
        ("multi_hop", "fol_inference", "consistency_paraphrase"),
    ),
    ("multi_hop+fol+atomic", ("multi_hop", "fol_inference", "atomic")),
]

EXTRACTOR_VARIANTS = ("first_match", "last_mention", "tail_clause")
POLARITY_VARIANTS = ("none", "rule_based")

FIGURE_STEMS = (
    "repair_boundary_scatter",
    "repair_violation_vs_delta",
    "repair_delta_by_family",
    "repair_flip_breakdown",
)

VALID_BINARY_LABELS = {"TRUE", "FALSE"}


@dataclass
class RunEntry:
    run_id: str
    provider: str
    run_dir: Path
    total_items: int
    canonical_selector: str
    prompt_hash: str
    dataset_sha: str
    split: str


def short_model_name(provider: str) -> str:
    mapping = {
        "ollama/llama3.1:8b": "Llama3.1-8B",
        "ollama/mistral:7b": "Mistral-7B",
        "ollama/qwen2.5:14b": "Qwen2.5-14B",
        "ollama/qwen2.5:32b": "Qwen2.5-32B",
        "openai/gpt-4o": "GPT-4o",
        "openai/gpt-5.2": "GPT-5.2",
        "anthropic/claude-sonnet-4-6": "Claude-Sonnet-4.6",
        "deepseek/deepseek-chat": "DeepSeek-Chat",
        "openrouter/google/gemini-2.5-flash": "Gemini-2.5-Flash",
        "openrouter/meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B",
    }
    return mapping.get(provider, provider.replace("/", "_"))


def to_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def write_pre_registered_protocol(out_dir: Path) -> Path:
    path = out_dir / "PRE_REGISTERED_PROTOCOL.md"
    lines = [
        "# CARE-v3 Pre-Registered Protocol",
        "",
        "## Fixed Hypotheses",
        "",
        "1. Global un-gated axiom repair has a non-trivial help/hurt boundary across models.",
        "2. Eligibility-gated repair can produce robust positive MCC deltas on held-out models.",
        "3. Boundary shape is associated with baseline model strength and violation reduction.",
        "",
        "## Fixed Split",
        "",
        "- Dev providers (for config selection only):",
        "  - ollama/llama3.1:8b",
        "  - ollama/mistral:7b",
        "  - ollama/qwen2.5:14b",
        "  - ollama/qwen2.5:32b",
        "- Held-out providers (for headline claims):",
        "  - openai/gpt-4o",
        "  - openai/gpt-5.2",
        "  - anthropic/claude-sonnet-4-6",
        "  - deepseek/deepseek-chat",
        "  - openrouter/google/gemini-2.5-flash",
        "  - openrouter/meta-llama/llama-3.3-70b-instruct",
        "",
        "## Fixed Selection Objective",
        "",
        "Choose a single frozen config by maximizing:",
        "1. minimum dev-model delta MCC,",
        "2. then mean dev-model delta MCC,",
        "3. then minimizing average row-flip rate.",
        "",
        "## Acceptance Criteria",
        "",
        "- Frozen config must be selected before held-out reporting.",
        "- Held-out report must include all discovered full canonical runs for held-out providers.",
        "- Include paired bootstrap CIs (item-level and system-cluster) and sign test.",
        "- Include M1 falsification controls and stability checks (order seeds, parser noise).",
        "- Train any M2 adaptive eligibility predictor on dev runs only.",
        "- Report M2 family-aware and family-agnostic variants side by side.",
        "",
        "## Failure Criteria",
        "",
        "- If held-out minimum delta MCC < 0, claim is boundary-oriented, not universal improvement.",
        "- If no candidate has positive mean delta on dev, report null result.",
        "",
        "## No-Ground-Truth Rule",
        "",
        "Repair decisions may not read reference labels. Ground-truth is only used in post-hoc evaluation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def discover_full_runs(
    runs_dir: Path,
    selector_rel: str,
    expected_total: int,
) -> Tuple[List[RunEntry], List[Dict[str, str]]]:
    discovered: List[RunEntry] = []
    skipped: List[Dict[str, str]] = []

    for manifest_path in sorted(runs_dir.glob("**/run_manifest.json")):
        if "_archive_excluded" in str(manifest_path):
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = manifest_path.parent
        run_id = manifest.get("run_id", run_dir.name)
        provider = manifest.get("provider", "")

        pred_path = run_dir / "predictions.jsonl"
        if not pred_path.exists():
            skipped.append({"run_id": run_id, "reason": "predictions_missing"})
            continue

        total_items = int(manifest.get("total_items_evaluated", 0) or 0)
        canonical_selector = str(manifest.get("canonical_selector", ""))
        if total_items != expected_total:
            skipped.append(
                {
                    "run_id": run_id,
                    "reason": f"not_full_canonical_total={total_items}",
                }
            )
            continue

        if canonical_selector != selector_rel:
            skipped.append(
                {
                    "run_id": run_id,
                    "reason": f"selector_mismatch={canonical_selector}",
                }
            )
            continue

        if provider in DEV_PROVIDERS:
            split = "dev"
        elif provider in HELDOUT_PROVIDERS:
            split = "heldout"
        else:
            split = "unassigned"

        discovered.append(
            RunEntry(
                run_id=run_id,
                provider=provider,
                run_dir=run_dir,
                total_items=total_items,
                canonical_selector=canonical_selector,
                prompt_hash=str(manifest.get("prompt_hash", "")),
                dataset_sha=str(manifest.get("dataset_global_sha256", "")),
                split=split,
            )
        )

    return discovered, skipped


def discover_canonical_runs_min_items(
    runs_dir: Path,
    selector_rel: str,
    min_items: int,
) -> List[RunEntry]:
    """Discover canonical runs with at least a minimum item count."""
    out: List[RunEntry] = []
    for manifest_path in sorted(runs_dir.glob("**/run_manifest.json")):
        if "_archive_excluded" in str(manifest_path):
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = manifest_path.parent
        pred_path = run_dir / "predictions.jsonl"
        if not pred_path.exists():
            continue

        total_items = int(manifest.get("total_items_evaluated", 0) or 0)
        selector = str(manifest.get("canonical_selector", ""))
        if total_items < min_items or selector != selector_rel:
            continue

        provider = str(manifest.get("provider", ""))
        out.append(
            RunEntry(
                run_id=str(manifest.get("run_id", run_dir.name)),
                provider=provider,
                run_dir=run_dir,
                total_items=total_items,
                canonical_selector=selector,
                prompt_hash=str(manifest.get("prompt_hash", "")),
                dataset_sha=str(manifest.get("dataset_global_sha256", "")),
                split=provider_split(provider),
            )
        )
    return out


def select_expanded_panel_runs(
    candidates: List[RunEntry],
    target_count: int,
) -> List[RunEntry]:
    """Select one best run per provider and keep up to target_count providers.

    Selection keeps all core held-out providers first, then highest-coverage extras.
    """
    best_by_provider: Dict[str, RunEntry] = {}
    for run in candidates:
        current = best_by_provider.get(run.provider)
        if current is None:
            best_by_provider[run.provider] = run
            continue
        if run.total_items > current.total_items:
            best_by_provider[run.provider] = run
            continue
        if run.total_items == current.total_items and run.run_id > current.run_id:
            best_by_provider[run.provider] = run

    providers = sorted(best_by_provider.keys())
    if len(providers) <= target_count:
        return sorted(best_by_provider.values(), key=lambda item: item.provider)

    core = [
        best_by_provider[provider]
        for provider in sorted(HELDOUT_PROVIDERS)
        if provider in best_by_provider
    ]
    core_providers = {run.provider for run in core}

    extras = [
        run for run in best_by_provider.values() if run.provider not in core_providers
    ]
    extras.sort(key=lambda item: (-item.total_items, item.provider, item.run_id))

    chosen = list(core)
    remaining = max(0, target_count - len(chosen))
    chosen.extend(extras[:remaining])
    return sorted(chosen, key=lambda item: item.provider)


def load_predictions_for_mode(
    pred_path: Path,
    mode: str,
    smoke_per_family: int,
    seed: int,
) -> List[Dict[str, Any]]:
    records = read_jsonl(pred_path)
    if mode != "smoke":
        return records

    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_family[row.get("task_family") or "unknown"].append(row)

    sampled: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for family in sorted(by_family.keys()):
        family_rows = by_family[family]
        if len(family_rows) <= smoke_per_family:
            sampled.extend(family_rows)
            continue

        order = np.arange(len(family_rows))
        rng.shuffle(order)
        keep = sorted(order[:smoke_per_family].tolist())
        sampled.extend([family_rows[idx] for idx in keep])

    return sampled


def confusion_from_labels(
    records: List[Dict[str, Any]],
    label_key: str,
) -> Tuple[int, int, int, int, int, int]:
    tp = tn = fp = fn = 0
    total = len(records)
    valid = 0

    for row in records:
        gt = row.get("ground_truth")
        pred = row.get(label_key)
        if gt not in {"TRUE", "FALSE"}:
            continue
        if pred not in {"TRUE", "FALSE"}:
            continue
        valid += 1

        if gt == "TRUE" and pred == "TRUE":
            tp += 1
        elif gt == "FALSE" and pred == "FALSE":
            tn += 1
        elif gt == "FALSE" and pred == "TRUE":
            fp += 1
        elif gt == "TRUE" and pred == "FALSE":
            fn += 1

    return tp, fp, tn, fn, valid, total


def metrics_from_records(
    records: List[Dict[str, Any]], label_key: str
) -> Dict[str, float]:
    tp, fp, tn, fn, valid, total = confusion_from_labels(records, label_key)
    coverage = (valid / total) if total else 0.0
    invalid_rate = 1.0 - coverage if total else 0.0
    accuracy_valid = (tp + tn) / valid if valid else 0.0
    effective_accuracy = coverage * accuracy_valid

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0

    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else 0.0

    pred_true = 0
    for row in records:
        if row.get(label_key) == "TRUE":
            pred_true += 1
    pred_true_pct = pred_true / valid if valid else 0.0

    return {
        "total": float(total),
        "valid": float(valid),
        "coverage": float(coverage),
        "invalid_rate": float(invalid_rate),
        "accuracy_valid": float(accuracy_valid),
        "effective_accuracy": float(effective_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "pred_true_pct": float(pred_true_pct),
    }


def metrics_from_explicit_labels(
    records: List[Dict[str, Any]], labels: Sequence[str]
) -> Dict[str, float]:
    """Compute metrics for record ground truth against explicit predictions."""
    if len(records) != len(labels):
        raise ValueError("records and labels length mismatch")

    tp = tn = fp = fn = 0
    total = len(records)
    valid = 0
    pred_true = 0

    for row, pred in zip(records, labels):
        gt = row.get("ground_truth")
        if gt not in VALID_BINARY_LABELS:
            continue
        if pred not in VALID_BINARY_LABELS:
            continue
        valid += 1
        if pred == "TRUE":
            pred_true += 1

        if gt == "TRUE" and pred == "TRUE":
            tp += 1
        elif gt == "FALSE" and pred == "FALSE":
            tn += 1
        elif gt == "FALSE" and pred == "TRUE":
            fp += 1
        elif gt == "TRUE" and pred == "FALSE":
            fn += 1

    coverage = (valid / total) if total else 0.0
    invalid_rate = 1.0 - coverage if total else 0.0
    accuracy_valid = (tp + tn) / valid if valid else 0.0
    effective_accuracy = coverage * accuracy_valid
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else 0.0
    pred_true_pct = pred_true / valid if valid else 0.0

    return {
        "total": float(total),
        "valid": float(valid),
        "coverage": float(coverage),
        "invalid_rate": float(invalid_rate),
        "accuracy_valid": float(accuracy_valid),
        "effective_accuracy": float(effective_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "pred_true_pct": float(pred_true_pct),
    }


def per_family_mcc(
    records: List[Dict[str, Any]], label_key: str
) -> Dict[str, Dict[str, float]]:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        family = row.get("task_family") or "unknown"
        by_family[family].append(row)

    out: Dict[str, Dict[str, float]] = {}
    for family in sorted(by_family.keys()):
        metrics = metrics_from_records(by_family[family], label_key)
        out[family] = {
            "n_total": metrics["total"],
            "n_valid": metrics["valid"],
            "mcc": metrics["mcc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy_valid": metrics["accuracy_valid"],
        }
    return out


def mcc_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float((tp * tn - fp * fn) / denom) if denom else 0.0


def prepare_binary_arrays(
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true: List[int] = []
    y_pre: List[int] = []
    y_post: List[int] = []
    cluster: List[str] = []

    for row in records:
        gt = row.get("ground_truth")
        pre = row.get("parsed_label")
        post = row.get("repaired_label")
        if gt not in {"TRUE", "FALSE"}:
            continue
        if pre not in {"TRUE", "FALSE"}:
            continue
        if post not in {"TRUE", "FALSE"}:
            continue

        y_true.append(1 if gt == "TRUE" else 0)
        y_pre.append(1 if pre == "TRUE" else 0)
        y_post.append(1 if post == "TRUE" else 0)

        item_id = row.get("id", row.get("item_id", ""))
        system_id = id_to_meta.get(item_id, {}).get("system_id")
        cluster.append(system_id or f"unknown:{item_id}")

    return (
        np.asarray(y_true, dtype=np.int8),
        np.asarray(y_pre, dtype=np.int8),
        np.asarray(y_post, dtype=np.int8),
        np.asarray(cluster),
    )


def bootstrap_delta_mcc(
    y_true: np.ndarray,
    y_pre: np.ndarray,
    y_post: np.ndarray,
    n_bootstrap: int,
    seed: int,
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    deltas: List[float] = []

    if cluster_ids is None:
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            pre_mcc = mcc_from_binary(y_true[idx], y_pre[idx])
            post_mcc = mcc_from_binary(y_true[idx], y_post[idx])
            deltas.append(post_mcc - pre_mcc)
    else:
        cluster_map: Dict[str, np.ndarray] = {}
        for cluster in np.unique(cluster_ids):
            cluster_map[str(cluster)] = np.where(cluster_ids == cluster)[0]

        keys = np.asarray(sorted(cluster_map.keys()))
        n_clusters = len(keys)
        for _ in range(n_bootstrap):
            sampled_keys = rng.choice(keys, size=n_clusters, replace=True)
            sampled_idx = np.concatenate(
                [cluster_map[str(key)] for key in sampled_keys]
            )
            pre_mcc = mcc_from_binary(y_true[sampled_idx], y_pre[sampled_idx])
            post_mcc = mcc_from_binary(y_true[sampled_idx], y_post[sampled_idx])
            deltas.append(post_mcc - pre_mcc)

    if not deltas:
        return 0.0, 0.0

    low = float(np.percentile(deltas, 2.5))
    high = float(np.percentile(deltas, 97.5))
    return low, high


def sign_test_two_sided(deltas: Sequence[float]) -> Dict[str, float]:
    positives = sum(1 for value in deltas if value > 0)
    negatives = sum(1 for value in deltas if value < 0)
    n = positives + negatives
    if n == 0:
        return {"n": 0.0, "positives": 0.0, "negatives": 0.0, "p_value": 1.0}

    tail = min(positives, negatives)
    cumulative = 0.0
    for i in range(tail + 1):
        cumulative += math.comb(n, i) * (0.5**n)
    p_value = min(1.0, 2.0 * cumulative)

    return {
        "n": float(n),
        "positives": float(positives),
        "negatives": float(negatives),
        "p_value": float(p_value),
    }


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    return values


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    return values


def stable_seed_from_text(text: str) -> int:
    """Build deterministic integer seed from text content."""
    total = 0
    for idx, char in enumerate(text):
        total += (idx + 1) * ord(char)
    return total % 1_000_000_007


def provider_split(provider: str) -> str:
    if provider in DEV_PROVIDERS:
        return "dev"
    if provider in HELDOUT_PROVIDERS:
        return "heldout"
    return "unassigned"


def build_candidate_configs() -> List[RepairConfig]:
    configs: List[RepairConfig] = []
    for gate_name, gate_families in GATE_VARIANTS:
        for extractor in EXTRACTOR_VARIANTS:
            for polarity_mode in POLARITY_VARIANTS:
                name = f"{gate_name}__{extractor}__{polarity_mode}"
                configs.append(
                    RepairConfig(
                        name=name,
                        gate_families=gate_families,
                        extractor_strategy=extractor,
                        polarity_mode=polarity_mode,
                        leave_invalid_unchanged=True,
                        enable_group_consistency=False,
                        seed=42,
                    )
                )
    return configs


def evaluate_config_on_run(
    run: RunEntry,
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
    n_bootstrap: int,
    with_bootstrap: bool,
    include_records: bool,
) -> Dict[str, Any]:
    repair = repair_records(records=records, id_to_meta=id_to_meta, config=config)

    pre_metrics = metrics_from_records(records, "parsed_label")
    post_metrics = metrics_from_records(repair.records, "repaired_label")

    pre_family = per_family_mcc(records, "parsed_label")
    post_family = per_family_mcc(repair.records, "repaired_label")

    pre_viol_count, pre_viol_rate = compute_axiom_violation_rate(
        records=records,
        label_key="parsed_label",
        id_to_meta=id_to_meta,
        config=config,
        gate_families=None,
    )
    post_viol_count, post_viol_rate = compute_axiom_violation_rate(
        records=repair.records,
        label_key="repaired_label",
        id_to_meta=id_to_meta,
        config=config,
        gate_families=None,
    )

    id_to_system = {
        item_id: str(meta.get("system_id") or "")
        for item_id, meta in id_to_meta.items()
    }
    group_pre = compute_group_inconsistency_rate(
        records=records,
        label_key="parsed_label",
        id_to_system=id_to_system,
        extractor_strategy=config.extractor_strategy,
        polarity_mode=config.polarity_mode,
    )
    group_post = compute_group_inconsistency_rate(
        records=repair.records,
        label_key="repaired_label",
        id_to_system=id_to_system,
        extractor_strategy=config.extractor_strategy,
        polarity_mode=config.polarity_mode,
    )

    ci_item_low = ci_item_high = 0.0
    ci_cluster_low = ci_cluster_high = 0.0
    if with_bootstrap:
        y_true, y_pre, y_post, clusters = prepare_binary_arrays(
            repair.records, id_to_meta
        )
        ci_item_low, ci_item_high = bootstrap_delta_mcc(
            y_true=y_true,
            y_pre=y_pre,
            y_post=y_post,
            n_bootstrap=n_bootstrap,
            seed=config.seed,
            cluster_ids=None,
        )
        ci_cluster_low, ci_cluster_high = bootstrap_delta_mcc(
            y_true=y_true,
            y_pre=y_pre,
            y_post=y_post,
            n_bootstrap=n_bootstrap,
            seed=config.seed + 17,
            cluster_ids=clusters,
        )

    delta = {
        "mcc": post_metrics["mcc"] - pre_metrics["mcc"],
        "balanced_accuracy": post_metrics["balanced_accuracy"]
        - pre_metrics["balanced_accuracy"],
        "accuracy_valid": post_metrics["accuracy_valid"]
        - pre_metrics["accuracy_valid"],
        "tpr": post_metrics["tpr"] - pre_metrics["tpr"],
        "tnr": post_metrics["tnr"] - pre_metrics["tnr"],
        "pred_true_pct": post_metrics["pred_true_pct"] - pre_metrics["pred_true_pct"],
        "axiom_violation_rate": post_viol_rate - pre_viol_rate,
        "group_inconsistency_rate": group_post - group_pre,
    }

    row_flip_rate = (
        repair.stats.row_flips / repair.stats.valid_records
        if repair.stats.valid_records
        else 0.0
    )

    return {
        "run_id": run.run_id,
        "provider": run.provider,
        "split": run.split,
        "config_name": config.name,
        "constraint_hash": repair.constraint_hash,
        "config": config.to_dict(),
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "delta": delta,
        "pre_family": pre_family,
        "post_family": post_family,
        "pre_axiom_violations": pre_viol_count,
        "post_axiom_violations": post_viol_count,
        "pre_axiom_violation_rate": pre_viol_rate,
        "post_axiom_violation_rate": post_viol_rate,
        "pre_group_inconsistency": group_pre,
        "post_group_inconsistency": group_post,
        "row_flip_rate": row_flip_rate,
        "repair_stats": repair.stats.to_dict(),
        "ci_item_low": ci_item_low,
        "ci_item_high": ci_item_high,
        "ci_cluster_low": ci_cluster_low,
        "ci_cluster_high": ci_cluster_high,
        "repaired_records": repair.records if include_records else None,
    }


def evaluate_falsification_controls(
    run: RunEntry,
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    frozen_config: RepairConfig,
    frozen_result: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    """Evaluate M1 falsification controls for one run."""
    rows: List[Dict[str, Any]] = []

    pre = frozen_result["pre_metrics"]
    target_flips = int(frozen_result["repair_stats"].get("row_flips", 0))
    valid = int(pre.get("valid", 0.0) or 0)
    target_rate = (target_flips / valid) if valid else 0.0

    rows.append(
        {
            "run_id": run.run_id,
            "provider": run.provider,
            "split": run.split,
            "method": "care_frozen",
            "mcc_pre": pre["mcc"],
            "mcc_post": frozen_result["post_metrics"]["mcc"],
            "delta_mcc": frozen_result["delta"]["mcc"],
            "row_flip_rate": frozen_result["row_flip_rate"],
            "budget_target_rate": target_rate,
            "viol_reduction": frozen_result["pre_axiom_violation_rate"]
            - frozen_result["post_axiom_violation_rate"],
            "details": "frozen",
        }
    )

    # Control 1: ungated all-family repair.
    all_cfg = RepairConfig(
        name="all_family_control",
        gate_families=None,
        extractor_strategy=frozen_config.extractor_strategy,
        polarity_mode=frozen_config.polarity_mode,
        leave_invalid_unchanged=frozen_config.leave_invalid_unchanged,
        enable_group_consistency=frozen_config.enable_group_consistency,
        seed=frozen_config.seed,
    )
    all_result = evaluate_config_on_run(
        run=run,
        records=records,
        id_to_meta=id_to_meta,
        config=all_cfg,
        n_bootstrap=0,
        with_bootstrap=False,
        include_records=False,
    )
    rows.append(
        {
            "run_id": run.run_id,
            "provider": run.provider,
            "split": run.split,
            "method": "all_family_repair",
            "mcc_pre": pre["mcc"],
            "mcc_post": all_result["post_metrics"]["mcc"],
            "delta_mcc": all_result["delta"]["mcc"],
            "row_flip_rate": all_result["row_flip_rate"],
            "budget_target_rate": target_rate,
            "viol_reduction": all_result["pre_axiom_violation_rate"]
            - all_result["post_axiom_violation_rate"],
            "details": "ungated",
        }
    )

    # Control 2: matched-rate random flips.
    random_labels, random_flips = random_flip_labels(
        records,
        target_flips=target_flips,
        seed=seed + 101,
        label_key="parsed_label",
    )
    random_metrics = metrics_from_explicit_labels(records, random_labels)
    rows.append(
        {
            "run_id": run.run_id,
            "provider": run.provider,
            "split": run.split,
            "method": "random_flip_matched",
            "mcc_pre": pre["mcc"],
            "mcc_post": random_metrics["mcc"],
            "delta_mcc": random_metrics["mcc"] - pre["mcc"],
            "row_flip_rate": (random_flips / valid) if valid else 0.0,
            "budget_target_rate": target_rate,
            "viol_reduction": 0.0,
            "details": "random_budget_matched",
        }
    )

    # Control 3: shuffled-gate control with budget match.
    observed_families = [str(row.get("task_family") or "unknown") for row in records]
    wrong_gate = shuffled_gate_families(
        observed_families,
        gate_families=tuple(frozen_config.gate_families or ()),
        seed=seed + 203,
    )
    wrong_cfg = RepairConfig(
        name="shuffled_gate_control",
        gate_families=wrong_gate,
        extractor_strategy=frozen_config.extractor_strategy,
        polarity_mode=frozen_config.polarity_mode,
        leave_invalid_unchanged=frozen_config.leave_invalid_unchanged,
        enable_group_consistency=frozen_config.enable_group_consistency,
        seed=frozen_config.seed,
    )
    wrong_result = evaluate_config_on_run(
        run=run,
        records=records,
        id_to_meta=id_to_meta,
        config=wrong_cfg,
        n_bootstrap=0,
        with_bootstrap=False,
        include_records=True,
    )

    candidate_labels = [
        str(row.get("repaired_label", row.get("parsed_label", "")))
        for row in (wrong_result["repaired_records"] or records)
    ]
    matched_labels, applied, used, synthetic = budget_match_candidate_labels(
        records=records,
        candidate_labels=candidate_labels,
        target_flips=target_flips,
        seed=seed + 307,
        label_key="parsed_label",
    )
    matched_metrics = metrics_from_explicit_labels(records, matched_labels)
    rows.append(
        {
            "run_id": run.run_id,
            "provider": run.provider,
            "split": run.split,
            "method": "shuffled_gate_budget_matched",
            "mcc_pre": pre["mcc"],
            "mcc_post": matched_metrics["mcc"],
            "delta_mcc": matched_metrics["mcc"] - pre["mcc"],
            "row_flip_rate": (applied / valid) if valid else 0.0,
            "budget_target_rate": target_rate,
            "viol_reduction": 0.0,
            "details": (
                f"gate={','.join(wrong_gate)};"
                f"used_candidate={used};synthetic={synthetic}"
            ),
        }
    )

    return rows


def evaluate_order_seed_stability(
    run: RunEntry,
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    """Run deterministic order-sensitivity checks by permuting record order."""
    rows: List[Dict[str, Any]] = []
    if not records:
        return rows

    index = np.arange(len(records))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = np.array(index)
        rng.shuffle(perm)
        permuted = [records[idx] for idx in perm.tolist()]
        result = evaluate_config_on_run(
            run=run,
            records=permuted,
            id_to_meta=id_to_meta,
            config=config,
            n_bootstrap=0,
            with_bootstrap=False,
            include_records=False,
        )
        rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "seed": int(seed),
                "delta_mcc": result["delta"]["mcc"],
                "delta_balanced_accuracy": result["delta"]["balanced_accuracy"],
                "row_flip_rate": result["row_flip_rate"],
            }
        )

    return rows


def evaluate_parser_noise_sensitivity(
    run: RunEntry,
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    config: RepairConfig,
    clean_delta_mcc: float,
    noise_rates: Sequence[float],
    seed: int,
) -> List[Dict[str, Any]]:
    """Evaluate robustness under synthetic parser-label noise."""
    rows: List[Dict[str, Any]] = []

    for rate in noise_rates:
        noisy_records, n_flipped = inject_parser_noise(
            records,
            noise_rate=rate,
            seed=seed,
            label_key="parsed_label",
        )
        result = evaluate_config_on_run(
            run=run,
            records=noisy_records,
            id_to_meta=id_to_meta,
            config=config,
            n_bootstrap=0,
            with_bootstrap=False,
            include_records=False,
        )
        rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "noise_rate": float(rate),
                "n_noisy_labels": int(n_flipped),
                "delta_mcc_noisy": result["delta"]["mcc"],
                "delta_mcc_clean": clean_delta_mcc,
                "delta_shift": result["delta"]["mcc"] - clean_delta_mcc,
                "row_flip_rate_noisy": result["row_flip_rate"],
            }
        )
        seed += 1

    return rows


M2_FEATURES = (
    "axiom_violation_rate",
    "group_inconsistency_rate",
    "error_rate",
    "bias_abs",
    "invalid_rate",
    "tpr_tnr_gap",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _binary_confusion_from_flags(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def _classification_summary(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    tp, tn, fp, fn = _binary_confusion_from_flags(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    positive_rate = (
        (sum(1 for value in y_pred if value == 1) / len(y_pred)) if y_pred else 0.0
    )
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "positive_rate": float(positive_rate),
    }


def _fit_feature_norms(
    rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    norms: Dict[str, Tuple[float, float]] = {}
    for feature in feature_names:
        values = [_safe_float(row.get(feature, 0.0), default=0.0) for row in rows]
        if not values:
            norms[feature] = (0.0, 1.0)
            continue
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            norms[feature] = (minimum, minimum + 1.0)
        else:
            norms[feature] = (minimum, maximum)
    return norms


def _norm_value(value: float, bounds: Tuple[float, float]) -> float:
    low, high = bounds
    if high <= low:
        return 0.0
    value = min(max(value, low), high)
    return (value - low) / (high - low)


def _family_prior_from_training(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    by_family: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(
            _safe_float(row.get("delta_mcc_family", 0.0))
        )

    if not by_family:
        return {}

    raw = {family: statistics.mean(values) for family, values in by_family.items()}
    values = list(raw.values())
    low = min(values)
    high = max(values)
    if high <= low:
        return {family: 0.5 for family in raw}
    return {family: (value - low) / (high - low) for family, value in raw.items()}


def compute_family_feature_rows(
    run: RunEntry,
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    base_config: RepairConfig,
) -> List[Dict[str, Any]]:
    """Compute family-level pre-repair features for adaptive gating."""
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        family = str(row.get("task_family") or "unknown")
        by_family[family].append(row)

    total_records = len(records)
    id_to_system = {
        item_id: str(meta.get("system_id") or "")
        for item_id, meta in id_to_meta.items()
    }

    out: List[Dict[str, Any]] = []
    for family in sorted(by_family.keys()):
        family_rows = by_family[family]
        metrics = metrics_from_records(family_rows, "parsed_label")

        gt_true = 0
        valid = 0
        parse_conf_vals: List[float] = []
        for row in family_rows:
            label = row.get("parsed_label")
            if (
                label in VALID_BINARY_LABELS
                and row.get("ground_truth") in VALID_BINARY_LABELS
            ):
                valid += 1
                if row.get("ground_truth") == "TRUE":
                    gt_true += 1
            meta = row.get("meta")
            if isinstance(meta, dict):
                parse_conf = _safe_float(meta.get("parse_confidence"), default=0.0)
                parse_conf_vals.append(parse_conf)

        gt_true_pct = (gt_true / valid) if valid else 0.0
        pred_true_pct = metrics["pred_true_pct"]
        bias_abs = abs(pred_true_pct - gt_true_pct)
        error_rate = max(0.0, 1.0 - metrics["mcc"])

        family_assignments = build_truth_assignments(
            records=family_rows,
            label_key="parsed_label",
            id_to_system=id_to_system,
            extractor_strategy=base_config.extractor_strategy,
            polarity_mode=base_config.polarity_mode,
            gate_families=None,
        )
        _, axiom_violation_rate = count_axiom_violations(family_assignments)

        group_inconsistency_rate = compute_group_inconsistency_rate(
            records=family_rows,
            label_key="parsed_label",
            id_to_system=id_to_system,
            extractor_strategy=base_config.extractor_strategy,
            polarity_mode=base_config.polarity_mode,
        )

        out.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "total_items": float(run.total_items),
                "family": family,
                "coverage_fraction": (len(family_rows) / total_records)
                if total_records
                else 0.0,
                "n_total_family": float(len(family_rows)),
                "n_valid_family": metrics["valid"],
                "mcc_pre_family": metrics["mcc"],
                "balanced_accuracy_pre_family": metrics["balanced_accuracy"],
                "tpr_pre_family": metrics["tpr"],
                "tnr_pre_family": metrics["tnr"],
                "invalid_rate": metrics["invalid_rate"],
                "pred_true_pct": pred_true_pct,
                "gt_true_pct": gt_true_pct,
                "bias_abs": bias_abs,
                "error_rate": error_rate,
                "axiom_violation_rate": axiom_violation_rate,
                "group_inconsistency_rate": group_inconsistency_rate,
                "tpr_tnr_gap": abs(metrics["tpr"] - metrics["tnr"]),
                "parse_conf_mean": statistics.mean(parse_conf_vals)
                if parse_conf_vals
                else 0.0,
            }
        )

    return out


def _gate_config_from_base(
    base_config: RepairConfig, gate_families: Sequence[str], name: str
) -> RepairConfig:
    gate_tuple = tuple(sorted(set(gate_families)))
    return RepairConfig(
        name=name,
        gate_families=gate_tuple,
        extractor_strategy=base_config.extractor_strategy,
        polarity_mode=base_config.polarity_mode,
        leave_invalid_unchanged=base_config.leave_invalid_unchanged,
        enable_group_consistency=base_config.enable_group_consistency,
        seed=base_config.seed,
    )


def evaluate_gate_cached(
    run: RunEntry,
    records: List[Dict[str, Any]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    base_config: RepairConfig,
    gate_families: Sequence[str],
    cache: Dict[Tuple[str, str, str, Tuple[str, ...]], Dict[str, Any]],
) -> Dict[str, Any]:
    gate_tuple = tuple(sorted(set(gate_families)))
    key = (
        run.run_id,
        base_config.extractor_strategy,
        base_config.polarity_mode,
        gate_tuple,
    )
    if key not in cache:
        cfg = _gate_config_from_base(
            base_config=base_config,
            gate_families=gate_tuple,
            name=f"cached_gate:{','.join(gate_tuple)}",
        )
        cache[key] = evaluate_config_on_run(
            run=run,
            records=records,
            id_to_meta=id_to_meta,
            config=cfg,
            n_bootstrap=0,
            with_bootstrap=False,
            include_records=False,
        )
    return cache[key]


def build_m2_dev_training_rows(
    dev_runs: Sequence[RunEntry],
    records_by_run: Dict[str, List[Dict[str, Any]]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    base_config: RepairConfig,
    gate_cache: Dict[Tuple[str, str, str, Tuple[str, ...]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build family-level supervised data on dev runs only."""
    rows: List[Dict[str, Any]] = []
    for run in dev_runs:
        records = records_by_run[run.run_id]
        feature_rows = compute_family_feature_rows(
            run=run,
            records=records,
            id_to_meta=id_to_meta,
            base_config=base_config,
        )
        for feature_row in feature_rows:
            family = str(feature_row["family"])
            family_eval = evaluate_gate_cached(
                run=run,
                records=records,
                id_to_meta=id_to_meta,
                base_config=base_config,
                gate_families=(family,),
                cache=gate_cache,
            )
            delta = _safe_float(family_eval["delta"]["mcc"], default=0.0)
            row = dict(feature_row)
            row["delta_mcc_family"] = delta
            row["helpful_family"] = 1.0 if delta > 0 else 0.0
            rows.append(row)
    return rows


def _score_family_row(
    row: Dict[str, Any],
    predictor: Dict[str, Any],
    norms: Dict[str, Tuple[float, float]],
    family_prior: Dict[str, float],
) -> float:
    n = {
        feature: _norm_value(_safe_float(row.get(feature, 0.0)), norms[feature])
        for feature in M2_FEATURES
    }
    score = (
        predictor["w_violation"] * n["axiom_violation_rate"]
        + predictor["w_group"] * n["group_inconsistency_rate"]
        + predictor["w_error"] * n["error_rate"]
        + predictor["w_bias"] * n["bias_abs"]
        + predictor["w_gap"] * n["tpr_tnr_gap"]
        - predictor["w_invalid"] * n["invalid_rate"]
    )
    if predictor.get("use_family_prior", 0.0) > 0.0:
        family = str(row.get("family", ""))
        prior = family_prior.get(family, 0.5)
        score += predictor["w_prior"] * prior
    return float(score)


def _search_predictor(
    training_rows: Sequence[Dict[str, Any]],
    use_family_prior: bool,
    min_balanced_accuracy: float,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Tuple[float, float]],
    Dict[str, float],
]:
    if not training_rows:
        raise RuntimeError("No training rows for M2 predictor")

    norms = _fit_feature_norms(training_rows, M2_FEATURES)
    family_prior = _family_prior_from_training(training_rows)
    y_true = [int(_safe_float(row.get("helpful_family", 0.0))) for row in training_rows]
    true_positive_rate = sum(y_true) / len(y_true)

    weight_values_main = (0.5, 1.0, 1.5)
    weight_values_aux = (0.0, 0.5, 1.0)
    threshold_values = (0.6, 1.0, 1.4, 1.8, 2.2)
    prior_values = (0.0, 0.25, 0.5, 1.0) if use_family_prior else (0.0,)

    best: Optional[Dict[str, Any]] = None
    leaderboard: List[Dict[str, Any]] = []

    for (
        w_violation,
        w_group,
        w_error,
        w_bias,
        w_gap,
        w_invalid,
        w_prior,
        threshold,
    ) in itertools.product(
        weight_values_main,
        weight_values_main,
        weight_values_main,
        weight_values_aux,
        weight_values_aux,
        weight_values_aux,
        prior_values,
        threshold_values,
    ):
        predictor = {
            "w_violation": float(w_violation),
            "w_group": float(w_group),
            "w_error": float(w_error),
            "w_bias": float(w_bias),
            "w_gap": float(w_gap),
            "w_invalid": float(w_invalid),
            "w_prior": float(w_prior),
            "threshold": float(threshold),
            "use_family_prior": 1.0 if use_family_prior else 0.0,
        }

        y_pred: List[int] = []
        for row in training_rows:
            score = _score_family_row(row, predictor, norms, family_prior)
            y_pred.append(1 if score >= predictor["threshold"] else 0)

        cls = _classification_summary(y_true, y_pred)
        objective = (
            cls["balanced_accuracy"],
            cls["f1"],
            -abs(cls["positive_rate"] - true_positive_rate),
        )

        leaderboard.append(
            {
                "w_violation": predictor["w_violation"],
                "w_group": predictor["w_group"],
                "w_error": predictor["w_error"],
                "w_bias": predictor["w_bias"],
                "w_gap": predictor["w_gap"],
                "w_invalid": predictor["w_invalid"],
                "w_prior": predictor["w_prior"],
                "threshold": predictor["threshold"],
                "use_family_prior": predictor["use_family_prior"],
                "balanced_accuracy": cls["balanced_accuracy"],
                "f1": cls["f1"],
                "positive_rate": cls["positive_rate"],
                "objective_1": objective[0],
                "objective_2": objective[1],
                "objective_3": objective[2],
            }
        )

        if cls["balanced_accuracy"] < min_balanced_accuracy:
            continue

        if best is None or objective > best["objective"]:
            best = {
                "predictor": predictor,
                "objective": objective,
                "classification": cls,
            }

    if best is None:
        leaderboard.sort(
            key=lambda row: (
                row["balanced_accuracy"],
                row["f1"],
                -abs(row["positive_rate"] - true_positive_rate),
            ),
            reverse=True,
        )
        top = leaderboard[0]
        best = {
            "predictor": {
                "w_violation": top["w_violation"],
                "w_group": top["w_group"],
                "w_error": top["w_error"],
                "w_bias": top["w_bias"],
                "w_gap": top["w_gap"],
                "w_invalid": top["w_invalid"],
                "w_prior": top["w_prior"],
                "threshold": top["threshold"],
                "use_family_prior": top["use_family_prior"],
            },
            "objective": (
                top["balanced_accuracy"],
                top["f1"],
                -abs(top["positive_rate"] - true_positive_rate),
            ),
            "classification": {
                "balanced_accuracy": top["balanced_accuracy"],
                "f1": top["f1"],
                "positive_rate": top["positive_rate"],
            },
        }

    leaderboard.sort(
        key=lambda row: (
            row["balanced_accuracy"],
            row["f1"],
            -abs(row["positive_rate"] - true_positive_rate),
        ),
        reverse=True,
    )
    return best, leaderboard, norms, family_prior


def predict_gate_for_run(
    feature_rows: Sequence[Dict[str, Any]],
    predictor: Dict[str, Any],
    norms: Dict[str, Tuple[float, float]],
    family_prior: Dict[str, float],
    force_nonempty: bool,
) -> Tuple[Tuple[str, ...], List[Dict[str, Any]]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in feature_rows:
        score = _score_family_row(row, predictor, norms, family_prior)
        scored.append((score, row))

    selected = [
        str(row["family"]) for score, row in scored if score >= predictor["threshold"]
    ]
    if force_nonempty and not selected and scored:
        best_score, best_row = max(scored, key=lambda item: item[0])
        if best_score > 0.0:
            selected = [str(best_row["family"])]

    selected_set = set(selected)
    details: List[Dict[str, Any]] = []
    for score, row in sorted(scored, key=lambda item: item[0], reverse=True):
        details.append(
            {
                "run_id": str(row["run_id"]),
                "provider": str(row["provider"]),
                "split": str(row["split"]),
                "family": str(row["family"]),
                "score": float(score),
                "threshold": float(predictor["threshold"]),
                "selected": 1.0 if str(row["family"]) in selected_set else 0.0,
            }
        )

    return tuple(sorted(selected_set)), details


def evaluate_m2_variant(
    variant_name: str,
    runs: Sequence[RunEntry],
    records_by_run: Dict[str, List[Dict[str, Any]]],
    id_to_meta: Dict[str, Dict[str, Optional[str]]],
    base_config: RepairConfig,
    predictor: Dict[str, Any],
    norms: Dict[str, Tuple[float, float]],
    family_prior: Dict[str, float],
    force_nonempty: bool,
    gate_cache: Dict[Tuple[str, str, str, Tuple[str, ...]], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    feature_rows_all: List[Dict[str, Any]] = []

    for run in runs:
        records = records_by_run[run.run_id]
        feature_rows = compute_family_feature_rows(
            run=run,
            records=records,
            id_to_meta=id_to_meta,
            base_config=base_config,
        )
        feature_rows_all.extend(feature_rows)

        gate_families, details = predict_gate_for_run(
            feature_rows=feature_rows,
            predictor=predictor,
            norms=norms,
            family_prior=family_prior,
            force_nonempty=force_nonempty,
        )
        for detail in details:
            detail["variant"] = variant_name
            detail["gate_size"] = float(len(gate_families))
        gate_rows.extend(details)

        eval_row = evaluate_gate_cached(
            run=run,
            records=records,
            id_to_meta=id_to_meta,
            base_config=base_config,
            gate_families=gate_families,
            cache=gate_cache,
        )

        run_rows.append(
            {
                "variant": variant_name,
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "total_items": float(run.total_items),
                "gate_families": ",".join(gate_families),
                "gate_size": float(len(gate_families)),
                "mcc_pre": eval_row["pre_metrics"]["mcc"],
                "mcc_post": eval_row["post_metrics"]["mcc"],
                "delta_mcc": eval_row["delta"]["mcc"],
                "delta_balanced_accuracy": eval_row["delta"]["balanced_accuracy"],
                "delta_tpr": eval_row["delta"]["tpr"],
                "delta_tnr": eval_row["delta"]["tnr"],
                "row_flip_rate": eval_row["row_flip_rate"],
                "axiom_violation_reduction": eval_row["pre_axiom_violation_rate"]
                - eval_row["post_axiom_violation_rate"],
            }
        )

    return run_rows, gate_rows, feature_rows_all


def write_m2_report(
    out_dir: Path,
    frozen_name: str,
    aware_predictor: Dict[str, Any],
    agnostic_predictor: Dict[str, Any],
    aware_rows: List[Dict[str, Any]],
    agnostic_rows: List[Dict[str, Any]],
    static_rows: List[Dict[str, Any]],
    training_rows: List[Dict[str, Any]],
) -> Path:
    report_path = out_dir / "M2_RESULTS.md"

    static_by_run = {row["run_id"]: row for row in static_rows}
    aware_by_run = {row["run_id"]: row for row in aware_rows}
    agn_by_run = {row["run_id"]: row for row in agnostic_rows}

    heldout_ids = [row["run_id"] for row in static_rows if row["split"] == "heldout"]
    aware_heldout = [aware_by_run[rid] for rid in heldout_ids if rid in aware_by_run]
    agn_heldout = [agn_by_run[rid] for rid in heldout_ids if rid in agn_by_run]
    static_heldout = [static_by_run[rid] for rid in heldout_ids if rid in static_by_run]

    def mean_delta(rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return statistics.mean(_safe_float(row.get("delta_mcc"), 0.0) for row in rows)

    aware_sign = sign_test_two_sided(
        [_safe_float(row["delta_mcc"]) for row in aware_heldout]
    )
    agn_sign = sign_test_two_sided(
        [_safe_float(row["delta_mcc"]) for row in agn_heldout]
    )

    aware_vs_static = [
        _safe_float(aware_by_run[rid]["delta_mcc"])
        - _safe_float(static_by_run[rid]["delta_mcc"])
        for rid in heldout_ids
        if rid in aware_by_run and rid in static_by_run
    ]
    agn_vs_static = [
        _safe_float(agn_by_run[rid]["delta_mcc"])
        - _safe_float(static_by_run[rid]["delta_mcc"])
        for rid in heldout_ids
        if rid in agn_by_run and rid in static_by_run
    ]

    lines = [
        "# M2 Results",
        "",
        "## Setup",
        "",
        f"- Static baseline config: `{frozen_name}`",
        f"- Dev family training rows: {len(training_rows)}",
        "- Variants: adaptive_family_aware and adaptive_family_agnostic",
        "",
        "## Predictor snapshots",
        "",
        f"- Family-aware threshold: {aware_predictor['threshold']:.3f}",
        f"- Family-agnostic threshold: {agnostic_predictor['threshold']:.3f}",
        "",
        "## Held-out run deltas",
        "",
        "| Provider | Static delta MCC | Aware delta MCC | Agnostic delta MCC | Aware - Static | Agnostic - Static |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for rid in heldout_ids:
        if rid not in static_by_run:
            continue
        provider = static_by_run[rid]["provider"]
        static_delta = _safe_float(static_by_run[rid]["delta_mcc"])
        aware_delta = _safe_float(aware_by_run.get(rid, {}).get("delta_mcc", 0.0))
        agn_delta = _safe_float(agn_by_run.get(rid, {}).get("delta_mcc", 0.0))
        lines.append(
            f"| {provider} | {static_delta:+.4f} | {aware_delta:+.4f} | {agn_delta:+.4f} | "
            f"{(aware_delta - static_delta):+.4f} | {(agn_delta - static_delta):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate held-out summary",
            "",
            f"- Mean delta MCC (static): {mean_delta(static_heldout):+.4f}",
            f"- Mean delta MCC (aware): {mean_delta(aware_heldout):+.4f}",
            f"- Mean delta MCC (agnostic): {mean_delta(agn_heldout):+.4f}",
            f"- Sign test aware: positives={int(aware_sign['positives'])}/{int(aware_sign['n'])}, p={aware_sign['p_value']:.6f}",
            f"- Sign test agnostic: positives={int(agn_sign['positives'])}/{int(agn_sign['n'])}, p={agn_sign['p_value']:.6f}",
        ]
    )

    if aware_vs_static:
        lines.append(
            f"- Mean (aware - static) delta MCC: {statistics.mean(aware_vs_static):+.4f}"
        )
    if agn_vs_static:
        lines.append(
            f"- Mean (agnostic - static) delta MCC: {statistics.mean(agn_vs_static):+.4f}"
        )

    aware_tpr = [_safe_float(row.get("delta_tpr", 0.0)) for row in aware_heldout]
    aware_tnr = [_safe_float(row.get("delta_tnr", 0.0)) for row in aware_heldout]
    aware_delta = [_safe_float(row.get("delta_mcc", 0.0)) for row in aware_heldout]
    if len(aware_delta) >= 2:
        corr_tpr = float(
            np.corrcoef(np.asarray(aware_delta), np.asarray(aware_tpr))[0, 1]
        )
        corr_tnr = float(
            np.corrcoef(np.asarray(aware_delta), np.asarray(aware_tnr))[0, 1]
        )
        lines.extend(
            [
                "",
                "## Analytical decomposition",
                "",
                f"- Corr(delta MCC, delta TPR) for aware held-out: {corr_tpr:+.4f}",
                f"- Corr(delta MCC, delta TNR) for aware held-out: {corr_tnr:+.4f}",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def select_frozen_config(
    dev_rows: List[Dict[str, Any]],
    max_row_flip_rate: float,
) -> Tuple[str, Dict[str, float], Dict[str, Any], List[Dict[str, float]]]:
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in dev_rows:
        by_config[row["config_name"]].append(row)

    ranked_all: List[Tuple[Tuple[float, float, float], str]] = []
    score_map: Dict[str, Dict[str, float]] = {}
    repr_map: Dict[str, Any] = {}

    for config_name, rows in sorted(by_config.items()):
        deltas = [row["delta"]["mcc"] for row in rows]
        flip_rates = [row["row_flip_rate"] for row in rows]
        score = {
            "min_delta_mcc": float(min(deltas)),
            "mean_delta_mcc": float(statistics.mean(deltas)),
            "mean_row_flip_rate": float(statistics.mean(flip_rates)),
            "n_models": float(len(rows)),
            "feasible": float(statistics.mean(flip_rates) <= max_row_flip_rate),
        }
        score_map[config_name] = score
        repr_map[config_name] = rows[0]["config"]
        ranked_all.append(
            (
                (
                    score["min_delta_mcc"],
                    score["mean_delta_mcc"],
                    -score["mean_row_flip_rate"],
                ),
                config_name,
            )
        )

    if not ranked_all:
        raise RuntimeError("No dev rows available for frozen config selection")

    feasible_names = {
        name
        for name, score in score_map.items()
        if score["mean_row_flip_rate"] <= max_row_flip_rate
    }
    ranked = [entry for entry in ranked_all if entry[1] in feasible_names]
    if not ranked:
        ranked = ranked_all

    ranked.sort(reverse=True)

    best_name = ranked[0][1]
    score_rows = [
        {
            "config_name": config_name,
            **score_map[config_name],
        }
        for _, config_name in sorted(ranked_all, reverse=True)
    ]
    return best_name, score_map[best_name], repr_map[best_name], score_rows


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_figure(fig: Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_figures(
    run_rows: List[Dict[str, Any]],
    family_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    heldout = [row for row in run_rows if row["split"] == "heldout"]
    metadata: Dict[str, Dict[str, Any]] = {}

    if heldout:
        x = [row["mcc_pre"] for row in heldout]
        y = [row["delta_mcc"] for row in heldout]
        labels = [short_model_name(row["provider"]) for row in heldout]
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.scatter(x, y, s=90, c="#1f77b4", edgecolors="white", linewidths=1.0)
        for idx, label in enumerate(labels):
            ax.annotate(
                label,
                (x[idx], y[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("Baseline MCC")
        ax.set_ylabel("Delta MCC (post - pre)")
        ax.set_title("CARE-v3 boundary: baseline strength vs delta MCC")
        _save_figure(fig, figures_dir, "repair_boundary_scatter")
        metadata["repair_boundary_scatter"] = {
            "x_tick_rotation": 0,
            "label_max_len": max(len(lbl) for lbl in labels),
            "n_points": len(labels),
        }

        x2 = [row["axiom_violation_reduction"] for row in heldout]
        y2 = [row["delta_mcc"] for row in heldout]
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.scatter(x2, y2, s=90, c="#2ca02c", edgecolors="white", linewidths=1.0)
        for idx, label in enumerate(labels):
            ax.annotate(
                label,
                (x2[idx], y2[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("Axiom violation rate reduction")
        ax.set_ylabel("Delta MCC (post - pre)")
        ax.set_title("Boundary: violation reduction vs delta MCC")
        _save_figure(fig, figures_dir, "repair_violation_vs_delta")
        metadata["repair_violation_vs_delta"] = {
            "x_tick_rotation": 0,
            "label_max_len": max(len(lbl) for lbl in labels),
            "n_points": len(labels),
        }

    heldout_family = [row for row in family_rows if row["split"] == "heldout"]
    if heldout_family:
        by_family: Dict[str, List[float]] = defaultdict(list)
        for row in heldout_family:
            by_family[row["family"]].append(row["delta_mcc"])
        families = sorted(
            by_family.keys(), key=lambda fam: statistics.mean(by_family[fam])
        )
        values = [statistics.mean(by_family[fam]) for fam in families]
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]

        wrapped = [fam if len(fam) <= 20 else fam[:17] + "..." for fam in families]
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        ax.bar(np.arange(len(values)), values, color=colors, alpha=0.9)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(wrapped, rotation=35, ha="right")
        ax.set_ylabel("Mean delta MCC (held-out)")
        ax.set_title("CARE-v3 delta MCC by family (held-out mean)")
        _save_figure(fig, figures_dir, "repair_delta_by_family")
        metadata["repair_delta_by_family"] = {
            "x_tick_rotation": 35,
            "label_max_len": max(len(label) for label in wrapped),
            "n_bars": len(values),
        }

    flip_counter: Counter = Counter()
    for row in run_rows:
        if row["split"] != "heldout":
            continue
        for reason, count in row.get("flip_reasons", {}).items():
            flip_counter[reason] += int(count)

    if flip_counter:
        reasons = sorted(flip_counter.keys())
        counts = [flip_counter[reason] for reason in reasons]
        wrapped = [
            reason if len(reason) <= 24 else reason[:21] + "..." for reason in reasons
        ]
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.bar(np.arange(len(counts)), counts, color="#9467bd", alpha=0.9)
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(wrapped, rotation=25, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("CARE-v3 flip reason breakdown (held-out)")
        _save_figure(fig, figures_dir, "repair_flip_breakdown")
        metadata["repair_flip_breakdown"] = {
            "x_tick_rotation": 25,
            "label_max_len": max(len(label) for label in wrapped),
            "n_bars": len(counts),
        }

    manifest_path = figures_dir / "figures_manifest.json"
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def write_no_gt_audit(out_dir: Path) -> Path:
    repair_dir = PROJECT_ROOT / "chaosbench" / "repair"
    hits: List[Tuple[str, int, str]] = []
    for py_path in sorted(repair_dir.glob("*.py")):
        for line_no, line in enumerate(
            py_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "ground_truth" in line:
                hits.append(
                    (str(py_path.relative_to(PROJECT_ROOT)), line_no, line.strip())
                )

    out_path = out_dir / "NO_GT_AUDIT.md"
    lines = [
        "# No-Ground-Truth Audit",
        "",
        "## Scope",
        "",
        "- Directory scanned: `chaosbench/repair/*.py`",
        "- Token scanned: `ground_truth`",
        "",
        "## Result",
        "",
        f"- Matches found: {len(hits)}",
    ]

    if hits:
        lines.extend(["", "## Matches", ""])
        for rel_path, line_no, content in hits:
            lines.append(f"- `{rel_path}:{line_no}` -> `{content}`")
    else:
        lines.extend(
            [
                "",
                "No `ground_truth` references are present in repair implementation files.",
                "Repair decisions are therefore structurally separated from reference labels.",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def summarize_prompt_variant_availability(
    candidates: List[RunEntry],
) -> List[Dict[str, Any]]:
    """Summarize prompt-hash diversity per provider for robustness checks."""
    by_provider: Dict[str, List[RunEntry]] = defaultdict(list)
    for run in candidates:
        by_provider[run.provider].append(run)

    rows: List[Dict[str, Any]] = []
    for provider in sorted(by_provider.keys()):
        runs = by_provider[provider]
        prompt_hashes = sorted({run.prompt_hash for run in runs})
        totals = sorted({run.total_items for run in runs}, reverse=True)
        rows.append(
            {
                "provider": provider,
                "n_runs": float(len(runs)),
                "n_prompt_hashes": float(len(prompt_hashes)),
                "prompt_hashes": ",".join(prompt_hashes),
                "item_scales": ",".join(str(value) for value in totals),
                "status": "available"
                if len(prompt_hashes) >= 2
                else "single_prompt_hash",
            }
        )
    return rows


def write_m1_report(
    out_dir: Path,
    frozen_name: str,
    heldout_rows: List[Dict[str, Any]],
    expanded_rows: List[Dict[str, Any]],
    expanded_target: int,
    controls_rows: List[Dict[str, Any]],
    order_rows: List[Dict[str, Any]],
    noise_rows: List[Dict[str, Any]],
    prompt_rows: List[Dict[str, Any]],
) -> Path:
    """Write dedicated M1 report with controls, stability, and panel expansion."""
    report_path = out_dir / "M1_RESULTS.md"

    heldout_deltas = [row["delta_mcc"] for row in heldout_rows]
    heldout_sign = sign_test_two_sided(heldout_deltas)

    expanded_deltas = [row["delta_mcc"] for row in expanded_rows]
    expanded_sign = sign_test_two_sided(expanded_deltas)

    lines = [
        "# M1 Results",
        "",
        "## Core setup",
        "",
        f"- Frozen config: `{frozen_name}`",
        f"- Core held-out models: {len(heldout_rows)}",
        f"- Expanded panel models: {len(expanded_rows)} (target {expanded_target})",
        "",
        "## Core held-out outcomes",
        "",
        "| Provider | MCC pre | MCC post | Delta MCC | Violation reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(heldout_rows, key=lambda item: item["provider"]):
        lines.append(
            f"| {row['provider']} | {row['mcc_pre']:.4f} | {row['mcc_post']:.4f} | "
            f"{row['delta_mcc']:+.4f} | {row['axiom_violation_reduction']:+.4f} |"
        )

    lines.extend(
        [
            "",
            f"- Sign test (core held-out): positives={int(heldout_sign['positives'])}/{int(heldout_sign['n'])}, p={heldout_sign['p_value']:.6f}",
            "",
            "## Expanded panel",
            "",
            "| Provider | Total items | Delta MCC |",
            "|---|---:|---:|",
        ]
    )
    for row in sorted(expanded_rows, key=lambda item: item["provider"]):
        lines.append(
            f"| {row['provider']} | {int(row['total_items'])} | {row['delta_mcc']:+.4f} |"
        )

    lines.extend(
        [
            "",
            f"- Sign test (expanded panel): positives={int(expanded_sign['positives'])}/{int(expanded_sign['n'])}, p={expanded_sign['p_value']:.6f}",
            "",
            "## Falsification controls (core held-out mean delta MCC)",
            "",
            "| Method | Mean delta MCC | Mean flip rate |",
            "|---|---:|---:|",
        ]
    )

    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in controls_rows:
        if row["split"] == "heldout":
            by_method[row["method"]].append(row)
    for method in sorted(by_method.keys()):
        method_rows = by_method[method]
        mean_delta = statistics.mean(r["delta_mcc"] for r in method_rows)
        mean_flip = statistics.mean(r["row_flip_rate"] for r in method_rows)
        lines.append(f"| {method} | {mean_delta:+.4f} | {mean_flip:.4f} |")

    lines.extend(["", "## Stability checks", ""])

    if order_rows:
        deltas = [row["delta_mcc"] for row in order_rows if row["split"] == "heldout"]
        if deltas:
            lines.append(
                f"- Order-seed delta MCC (held-out): min={min(deltas):+.4f}, max={max(deltas):+.4f}, std={statistics.pstdev(deltas):.4f}"
            )
    if noise_rows:
        shifts = [row["delta_shift"] for row in noise_rows if row["split"] == "heldout"]
        if shifts:
            lines.append(
                f"- Parser-noise delta shift (held-out): min={min(shifts):+.4f}, max={max(shifts):+.4f}, mean={statistics.mean(shifts):+.4f}"
            )

    if prompt_rows:
        available = sum(1 for row in prompt_rows if row["status"] == "available")
        lines.append(
            f"- Prompt-variant availability: {available}/{len(prompt_rows)} providers with >=2 prompt hashes"
        )

    lines.extend(
        [
            "",
            "## Boundary artifact",
            "",
            "- Table: `tables/boundary_table.csv` (baseline MCC, delta MCC, violation reduction)",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_repair_results_report(
    out_dir: Path,
    run_rows: List[Dict[str, Any]],
    family_rows: List[Dict[str, Any]],
    frozen_name: str,
    frozen_score: Dict[str, float],
    max_row_flip_rate: float,
    sign_test: Dict[str, float],
    pearson: float,
    processed_runs: List[RunEntry],
    skipped_runs: List[Dict[str, str]],
) -> Path:
    report_path = out_dir / "REPAIR_RESULTS.md"

    heldout_rows = [row for row in run_rows if row["split"] == "heldout"]
    dev_rows = [row for row in run_rows if row["split"] == "dev"]

    family_help: Dict[str, List[float]] = defaultdict(list)
    for row in family_rows:
        if row["split"] == "heldout":
            family_help[row["family"]].append(row["delta_mcc"])

    family_mean = {
        family: statistics.mean(values) for family, values in family_help.items()
    }
    best_families = sorted(family_mean.items(), key=lambda kv: kv[1], reverse=True)[:5]
    worst_families = sorted(family_mean.items(), key=lambda kv: kv[1])[:5]

    lines: List[str] = [
        "# CARE-v3 Repair Results",
        "",
        "## Protocol",
        "",
        f"- Frozen config: `{frozen_name}`",
        f"- Feasibility cap (mean row flip rate): <= {max_row_flip_rate:.3f}",
        f"- Dev min delta MCC: {frozen_score['min_delta_mcc']:+.4f}",
        f"- Dev mean delta MCC: {frozen_score['mean_delta_mcc']:+.4f}",
        f"- Dev mean row flip rate: {frozen_score['mean_row_flip_rate']:.4f}",
        "",
        "## Held-out model deltas",
        "",
        "| Provider | MCC pre | MCC post | Delta MCC | Delta BA | Delta TPR | Delta TNR | Axiom reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(heldout_rows, key=lambda item: item["provider"]):
        lines.append(
            "| "
            + f"{row['provider']} | {row['mcc_pre']:.4f} | {row['mcc_post']:.4f} | "
            + f"{row['delta_mcc']:+.4f} | {row['delta_balanced_accuracy']:+.4f} | "
            + f"{row['delta_tpr']:+.4f} | {row['delta_tnr']:+.4f} | "
            + f"{row['axiom_violation_reduction']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Statistical checks (held-out)",
            "",
            f"- Sign test: n={int(sign_test['n'])}, positives={int(sign_test['positives'])}, negatives={int(sign_test['negatives'])}, p={sign_test['p_value']:.6f}",
            f"- Pearson correlation (baseline MCC vs delta MCC): {pearson:+.4f}",
            "",
            "## Boundary summary",
            "",
            "Top positive held-out mean family deltas:",
        ]
    )
    for family, value in best_families:
        lines.append(f"- `{family}`: {value:+.4f}")

    lines.extend(["", "Top negative held-out mean family deltas:"])
    for family, value in worst_families:
        lines.append(f"- `{family}`: {value:+.4f}")

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Processed runs: {len(processed_runs)}",
            f"- Dev runs: {len(dev_rows)}",
            f"- Held-out runs: {len(heldout_rows)}",
            f"- Skipped manifests: {len(skipped_runs)}",
            "",
            "## Processed run IDs",
            "",
        ]
    )
    for run in sorted(processed_runs, key=lambda item: item.run_id):
        lines.append(f"- `{run.run_id}` ({run.provider}, {run.split})")

    if skipped_runs:
        lines.extend(["", "## Skipped run manifests", ""])
        for row in skipped_runs[:50]:
            lines.append(f"- `{row['run_id']}`: {row['reason']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pre_registered_protocol(out_dir)

    selector_path = Path(args.selector)
    id_to_meta = load_selector_index(
        selector_path=selector_path, project_root=PROJECT_ROOT
    )

    runs_dir = Path(args.runs_dir)
    discovered, skipped = discover_full_runs(
        runs_dir=runs_dir,
        selector_rel=str(args.selector),
        expected_total=args.expected_total,
    )

    inventory_path = out_dir / "run_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "selector": str(args.selector),
                "expected_total": args.expected_total,
                "discovered": [
                    asdict(run) | {"run_dir": str(run.run_dir)} for run in discovered
                ],
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if not discovered:
        raise RuntimeError("No full canonical runs discovered")

    dev_runs = [run for run in discovered if run.split == "dev"]
    heldout_runs = [run for run in discovered if run.split == "heldout"]
    if not dev_runs:
        raise RuntimeError("No dev runs discovered")
    if not heldout_runs:
        raise RuntimeError("No held-out runs discovered")

    order_seeds = parse_int_list(args.order_seeds)
    noise_rates = parse_float_list(args.parser_noise_rates)

    panel_candidates = discover_canonical_runs_min_items(
        runs_dir=runs_dir,
        selector_rel=str(args.selector),
        min_items=args.expanded_panel_min_items,
    )
    expanded_panel_runs = select_expanded_panel_runs(
        panel_candidates,
        target_count=args.expanded_panel_target,
    )

    prompt_rows = summarize_prompt_variant_availability(panel_candidates)
    write_csv(
        out_dir / "tables" / "prompt_variant_robustness.csv",
        prompt_rows,
        fieldnames=[
            "provider",
            "n_runs",
            "n_prompt_hashes",
            "prompt_hashes",
            "item_scales",
            "status",
        ],
    )

    dev_records_cache: Dict[str, List[Dict[str, Any]]] = {}
    for run in dev_runs:
        dev_records_cache[run.run_id] = load_predictions_for_mode(
            pred_path=run.run_dir / "predictions.jsonl",
            mode=args.mode,
            smoke_per_family=args.smoke_per_family,
            seed=args.seed,
        )

    records_cache_all: Dict[str, List[Dict[str, Any]]] = dict(dev_records_cache)

    candidate_rows: List[Dict[str, Any]] = []
    for config in build_candidate_configs():
        for run in dev_runs:
            eval_row = evaluate_config_on_run(
                run=run,
                records=dev_records_cache[run.run_id],
                id_to_meta=id_to_meta,
                config=config,
                n_bootstrap=0,
                with_bootstrap=False,
                include_records=False,
            )
            candidate_rows.append(eval_row)

    dev_candidate_per_run_rows = [
        {
            "config_name": row["config_name"],
            "run_id": row["run_id"],
            "provider": row["provider"],
            "delta_mcc": row["delta"]["mcc"],
            "delta_balanced_accuracy": row["delta"]["balanced_accuracy"],
            "row_flip_rate": row["row_flip_rate"],
        }
        for row in candidate_rows
    ]
    write_csv(
        out_dir / "tables" / "dev_candidate_per_run.csv",
        dev_candidate_per_run_rows,
        fieldnames=[
            "config_name",
            "run_id",
            "provider",
            "delta_mcc",
            "delta_balanced_accuracy",
            "row_flip_rate",
        ],
    )

    frozen_name, frozen_score, frozen_cfg_payload, score_rows = select_frozen_config(
        candidate_rows,
        max_row_flip_rate=args.max_row_flip_rate,
    )

    write_csv(
        out_dir / "tables" / "dev_candidate_scores.csv",
        score_rows,
        fieldnames=[
            "config_name",
            "min_delta_mcc",
            "mean_delta_mcc",
            "mean_row_flip_rate",
            "n_models",
            "feasible",
        ],
    )
    frozen_config = RepairConfig(
        name=frozen_cfg_payload["name"],
        gate_families=tuple(frozen_cfg_payload["gate_families"])
        if frozen_cfg_payload["gate_families"]
        else None,
        extractor_strategy=frozen_cfg_payload["extractor_strategy"],
        polarity_mode=frozen_cfg_payload["polarity_mode"],
        leave_invalid_unchanged=bool(frozen_cfg_payload["leave_invalid_unchanged"]),
        enable_group_consistency=bool(frozen_cfg_payload["enable_group_consistency"]),
        seed=int(frozen_cfg_payload["seed"]),
    )

    (out_dir / "frozen_config.json").write_text(
        json.dumps(
            {
                "frozen_config_name": frozen_name,
                "score": frozen_score,
                "max_row_flip_rate": args.max_row_flip_rate,
                "config": frozen_config.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    all_runs = sorted(discovered, key=lambda run: run.run_id)
    run_rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    controls_rows: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    noise_rows: List[Dict[str, Any]] = []

    for run in all_runs:
        records = (
            dev_records_cache[run.run_id]
            if run.run_id in dev_records_cache
            else load_predictions_for_mode(
                pred_path=run.run_dir / "predictions.jsonl",
                mode=args.mode,
                smoke_per_family=args.smoke_per_family,
                seed=args.seed,
            )
        )
        records_cache_all[run.run_id] = records

        result = evaluate_config_on_run(
            run=run,
            records=records,
            id_to_meta=id_to_meta,
            config=frozen_config,
            n_bootstrap=args.bootstrap,
            with_bootstrap=True,
            include_records=True,
        )

        run_out = out_dir / "runs" / run.run_id
        run_out.mkdir(parents=True, exist_ok=True)
        write_jsonl(run_out / "repaired_predictions.jsonl", result["repaired_records"])

        repaired_metrics = {
            "run_id": run.run_id,
            "provider": run.provider,
            "split": run.split,
            "pre": result["pre_metrics"],
            "post": result["post_metrics"],
            "delta": result["delta"],
            "pre_axiom_violation_rate": result["pre_axiom_violation_rate"],
            "post_axiom_violation_rate": result["post_axiom_violation_rate"],
            "pre_group_inconsistency": result["pre_group_inconsistency"],
            "post_group_inconsistency": result["post_group_inconsistency"],
            "bootstrap": {
                "item_level": [result["ci_item_low"], result["ci_item_high"]],
                "system_cluster": [result["ci_cluster_low"], result["ci_cluster_high"]],
            },
            "pre_family": result["pre_family"],
            "post_family": result["post_family"],
        }
        (run_out / "repaired_metrics.json").write_text(
            json.dumps(repaired_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        run_manifest = {
            "run_id": run.run_id,
            "provider": run.provider,
            "split": run.split,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "constraint_hash": result["constraint_hash"],
            "config": result["config"],
            "stats": result["repair_stats"],
            "selector": str(args.selector),
            "source_run_dir": str(run.run_dir),
        }
        (run_out / "repair_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        run_rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "split": run.split,
                "mcc_pre": result["pre_metrics"]["mcc"],
                "mcc_post": result["post_metrics"]["mcc"],
                "delta_mcc": result["delta"]["mcc"],
                "delta_balanced_accuracy": result["delta"]["balanced_accuracy"],
                "delta_tpr": result["delta"]["tpr"],
                "delta_tnr": result["delta"]["tnr"],
                "delta_pred_true_pct": result["delta"]["pred_true_pct"],
                "pre_axiom_violation_rate": result["pre_axiom_violation_rate"],
                "post_axiom_violation_rate": result["post_axiom_violation_rate"],
                "axiom_violation_reduction": result["pre_axiom_violation_rate"]
                - result["post_axiom_violation_rate"],
                "pre_group_inconsistency": result["pre_group_inconsistency"],
                "post_group_inconsistency": result["post_group_inconsistency"],
                "group_inconsistency_reduction": result["pre_group_inconsistency"]
                - result["post_group_inconsistency"],
                "row_flip_rate": result["row_flip_rate"],
                "ci_item_low": result["ci_item_low"],
                "ci_item_high": result["ci_item_high"],
                "ci_cluster_low": result["ci_cluster_low"],
                "ci_cluster_high": result["ci_cluster_high"],
                "flip_reasons": result["repair_stats"].get("flip_reasons", {}),
                "total_items": run.total_items,
            }
        )

        controls_rows.extend(
            evaluate_falsification_controls(
                run=run,
                records=records,
                id_to_meta=id_to_meta,
                frozen_config=frozen_config,
                frozen_result=result,
                seed=args.seed + stable_seed_from_text(run.run_id),
            )
        )

        if run.split == "heldout":
            order_rows.extend(
                evaluate_order_seed_stability(
                    run=run,
                    records=records,
                    id_to_meta=id_to_meta,
                    config=frozen_config,
                    seeds=order_seeds,
                )
            )
            noise_rows.extend(
                evaluate_parser_noise_sensitivity(
                    run=run,
                    records=records,
                    id_to_meta=id_to_meta,
                    config=frozen_config,
                    clean_delta_mcc=result["delta"]["mcc"],
                    noise_rates=noise_rates,
                    seed=args.seed + stable_seed_from_text(run.run_id) + 9_973,
                )
            )

        families = sorted(
            set(result["pre_family"].keys()) | set(result["post_family"].keys())
        )
        for family in families:
            pre_family = result["pre_family"].get(family, {})
            post_family = result["post_family"].get(family, {})
            family_rows.append(
                {
                    "run_id": run.run_id,
                    "provider": run.provider,
                    "split": run.split,
                    "family": family,
                    "n_valid_pre": float(pre_family.get("n_valid", 0.0)),
                    "n_valid_post": float(post_family.get("n_valid", 0.0)),
                    "mcc_pre": float(pre_family.get("mcc", 0.0)),
                    "mcc_post": float(post_family.get("mcc", 0.0)),
                    "delta_mcc": float(
                        post_family.get("mcc", 0.0) - pre_family.get("mcc", 0.0)
                    ),
                }
            )

    write_csv(
        out_dir / "tables" / "repair_deltas.csv",
        run_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "total_items",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "delta_balanced_accuracy",
            "delta_tpr",
            "delta_tnr",
            "delta_pred_true_pct",
            "pre_axiom_violation_rate",
            "post_axiom_violation_rate",
            "axiom_violation_reduction",
            "pre_group_inconsistency",
            "post_group_inconsistency",
            "group_inconsistency_reduction",
            "row_flip_rate",
            "ci_item_low",
            "ci_item_high",
            "ci_cluster_low",
            "ci_cluster_high",
            "flip_reasons",
        ],
    )
    write_csv(
        out_dir / "tables" / "repair_by_family.csv",
        family_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "family",
            "n_valid_pre",
            "n_valid_post",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
        ],
    )

    write_csv(
        out_dir / "tables" / "falsification_controls.csv",
        controls_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "method",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "row_flip_rate",
            "budget_target_rate",
            "viol_reduction",
            "details",
        ],
    )

    write_csv(
        out_dir / "tables" / "stability_order_seeds.csv",
        order_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "seed",
            "delta_mcc",
            "delta_balanced_accuracy",
            "row_flip_rate",
        ],
    )

    write_csv(
        out_dir / "tables" / "stability_parser_noise.csv",
        noise_rows,
        fieldnames=[
            "run_id",
            "provider",
            "split",
            "noise_rate",
            "n_noisy_labels",
            "delta_mcc_noisy",
            "delta_mcc_clean",
            "delta_shift",
            "row_flip_rate_noisy",
        ],
    )

    # Expanded model panel (M1): evaluate frozen config on larger run inventory.
    run_row_by_id = {row["run_id"]: row for row in run_rows}
    expanded_rows: List[Dict[str, Any]] = []
    for run in expanded_panel_runs:
        if run.run_id in run_row_by_id:
            ref = run_row_by_id[run.run_id]
            expanded_rows.append(
                {
                    "run_id": run.run_id,
                    "provider": run.provider,
                    "total_items": run.total_items,
                    "split": run.split,
                    "mcc_pre": ref["mcc_pre"],
                    "mcc_post": ref["mcc_post"],
                    "delta_mcc": ref["delta_mcc"],
                    "axiom_violation_reduction": ref["axiom_violation_reduction"],
                    "row_flip_rate": ref["row_flip_rate"],
                }
            )
            continue

        records = load_predictions_for_mode(
            pred_path=run.run_dir / "predictions.jsonl",
            mode=args.mode,
            smoke_per_family=args.smoke_per_family,
            seed=args.seed,
        )
        panel_eval = evaluate_config_on_run(
            run=run,
            records=records,
            id_to_meta=id_to_meta,
            config=frozen_config,
            n_bootstrap=0,
            with_bootstrap=False,
            include_records=False,
        )
        expanded_rows.append(
            {
                "run_id": run.run_id,
                "provider": run.provider,
                "total_items": run.total_items,
                "split": run.split,
                "mcc_pre": panel_eval["pre_metrics"]["mcc"],
                "mcc_post": panel_eval["post_metrics"]["mcc"],
                "delta_mcc": panel_eval["delta"]["mcc"],
                "axiom_violation_reduction": panel_eval["pre_axiom_violation_rate"]
                - panel_eval["post_axiom_violation_rate"],
                "row_flip_rate": panel_eval["row_flip_rate"],
            }
        )

    write_csv(
        out_dir / "tables" / "expanded_panel_deltas.csv",
        expanded_rows,
        fieldnames=[
            "run_id",
            "provider",
            "total_items",
            "split",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "axiom_violation_reduction",
            "row_flip_rate",
        ],
    )

    boundary_rows: List[Dict[str, Any]] = []
    for row in run_rows:
        panel = "heldout_core" if row["split"] == "heldout" else "dev_core"
        boundary_rows.append(
            {
                "run_id": row["run_id"],
                "provider": row["provider"],
                "panel": panel,
                "total_items": row["total_items"],
                "baseline_mcc": row["mcc_pre"],
                "delta_mcc": row["delta_mcc"],
                "axiom_violation_reduction": row["axiom_violation_reduction"],
                "row_flip_rate": row["row_flip_rate"],
            }
        )
    for row in expanded_rows:
        boundary_rows.append(
            {
                "run_id": row["run_id"],
                "provider": row["provider"],
                "panel": "expanded",
                "total_items": row["total_items"],
                "baseline_mcc": row["mcc_pre"],
                "delta_mcc": row["delta_mcc"],
                "axiom_violation_reduction": row["axiom_violation_reduction"],
                "row_flip_rate": row["row_flip_rate"],
            }
        )
    write_csv(
        out_dir / "tables" / "boundary_table.csv",
        boundary_rows,
        fieldnames=[
            "run_id",
            "provider",
            "panel",
            "total_items",
            "baseline_mcc",
            "delta_mcc",
            "axiom_violation_reduction",
            "row_flip_rate",
        ],
    )

    heldout_rows = [row for row in run_rows if row["split"] == "heldout"]
    heldout_deltas = [row["delta_mcc"] for row in heldout_rows]
    sign_result = sign_test_two_sided(heldout_deltas)
    expanded_deltas = [row["delta_mcc"] for row in expanded_rows]
    expanded_sign_result = sign_test_two_sided(expanded_deltas)
    pearson = 0.0
    if len(heldout_rows) >= 2:
        baseline = np.asarray([row["mcc_pre"] for row in heldout_rows], dtype=float)
        deltas = np.asarray([row["delta_mcc"] for row in heldout_rows], dtype=float)
        pearson = float(np.corrcoef(baseline, deltas)[0, 1])

    pearson_expanded = 0.0
    if len(expanded_rows) >= 2:
        baseline = np.asarray([row["mcc_pre"] for row in expanded_rows], dtype=float)
        deltas = np.asarray([row["delta_mcc"] for row in expanded_rows], dtype=float)
        pearson_expanded = float(np.corrcoef(baseline, deltas)[0, 1])

    build_figures(run_rows=run_rows, family_rows=family_rows, out_dir=out_dir)
    write_no_gt_audit(out_dir)
    write_repair_results_report(
        out_dir=out_dir,
        run_rows=run_rows,
        family_rows=family_rows,
        frozen_name=frozen_name,
        frozen_score=frozen_score,
        max_row_flip_rate=args.max_row_flip_rate,
        sign_test=sign_result,
        pearson=pearson,
        processed_runs=all_runs,
        skipped_runs=skipped,
    )

    write_m1_report(
        out_dir=out_dir,
        frozen_name=frozen_name,
        heldout_rows=heldout_rows,
        expanded_rows=expanded_rows,
        expanded_target=args.expanded_panel_target,
        controls_rows=controls_rows,
        order_rows=order_rows,
        noise_rows=noise_rows,
        prompt_rows=prompt_rows,
    )

    m2_summary: Dict[str, Any] = {
        "enabled": float(args.enable_m2),
    }
    if args.enable_m2:
        m2_gate_cache: Dict[Tuple[str, str, str, Tuple[str, ...]], Dict[str, Any]] = {}

        for run in expanded_panel_runs:
            if run.run_id in records_cache_all:
                continue
            records_cache_all[run.run_id] = load_predictions_for_mode(
                pred_path=run.run_dir / "predictions.jsonl",
                mode=args.mode,
                smoke_per_family=args.smoke_per_family,
                seed=args.seed,
            )

        m2_dev_training_rows = build_m2_dev_training_rows(
            dev_runs=dev_runs,
            records_by_run=records_cache_all,
            id_to_meta=id_to_meta,
            base_config=frozen_config,
            gate_cache=m2_gate_cache,
        )
        write_csv(
            out_dir / "tables" / "m2_dev_family_training.csv",
            m2_dev_training_rows,
            fieldnames=[
                "run_id",
                "provider",
                "split",
                "total_items",
                "family",
                "coverage_fraction",
                "n_total_family",
                "n_valid_family",
                "mcc_pre_family",
                "balanced_accuracy_pre_family",
                "tpr_pre_family",
                "tnr_pre_family",
                "invalid_rate",
                "pred_true_pct",
                "gt_true_pct",
                "bias_abs",
                "error_rate",
                "axiom_violation_rate",
                "group_inconsistency_rate",
                "tpr_tnr_gap",
                "parse_conf_mean",
                "delta_mcc_family",
                "helpful_family",
            ],
        )

        aware_best, aware_leaderboard, aware_norms, family_prior = _search_predictor(
            m2_dev_training_rows,
            use_family_prior=True,
            min_balanced_accuracy=args.m2_min_balanced_acc,
        )
        agn_best, agn_leaderboard, agn_norms, _ = _search_predictor(
            m2_dev_training_rows,
            use_family_prior=False,
            min_balanced_accuracy=args.m2_min_balanced_acc,
        )

        write_csv(
            out_dir / "tables" / "m2_predictor_search_family_aware.csv",
            aware_leaderboard,
            fieldnames=[
                "w_violation",
                "w_group",
                "w_error",
                "w_bias",
                "w_gap",
                "w_invalid",
                "w_prior",
                "threshold",
                "use_family_prior",
                "balanced_accuracy",
                "f1",
                "positive_rate",
                "objective_1",
                "objective_2",
                "objective_3",
            ],
        )
        write_csv(
            out_dir / "tables" / "m2_predictor_search_family_agnostic.csv",
            agn_leaderboard,
            fieldnames=[
                "w_violation",
                "w_group",
                "w_error",
                "w_bias",
                "w_gap",
                "w_invalid",
                "w_prior",
                "threshold",
                "use_family_prior",
                "balanced_accuracy",
                "f1",
                "positive_rate",
                "objective_1",
                "objective_2",
                "objective_3",
            ],
        )

        (out_dir / "tables" / "m2_predictor_frozen.json").write_text(
            json.dumps(
                {
                    "aware": {
                        "predictor": aware_best["predictor"],
                        "objective": aware_best["objective"],
                        "classification": aware_best["classification"],
                        "norms": {k: [v[0], v[1]] for k, v in aware_norms.items()},
                        "family_prior": family_prior,
                    },
                    "agnostic": {
                        "predictor": agn_best["predictor"],
                        "objective": agn_best["objective"],
                        "classification": agn_best["classification"],
                        "norms": {k: [v[0], v[1]] for k, v in agn_norms.items()},
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        m2_target_runs = sorted(expanded_panel_runs, key=lambda item: item.provider)
        aware_rows, aware_gate_rows, aware_features = evaluate_m2_variant(
            variant_name="adaptive_family_aware",
            runs=m2_target_runs,
            records_by_run=records_cache_all,
            id_to_meta=id_to_meta,
            base_config=frozen_config,
            predictor=aware_best["predictor"],
            norms=aware_norms,
            family_prior=family_prior,
            force_nonempty=args.m2_force_nonempty,
            gate_cache=m2_gate_cache,
        )
        agn_rows, agn_gate_rows, agn_features = evaluate_m2_variant(
            variant_name="adaptive_family_agnostic",
            runs=m2_target_runs,
            records_by_run=records_cache_all,
            id_to_meta=id_to_meta,
            base_config=frozen_config,
            predictor=agn_best["predictor"],
            norms=agn_norms,
            family_prior={},
            force_nonempty=args.m2_force_nonempty,
            gate_cache=m2_gate_cache,
        )

        m2_run_rows = aware_rows + agn_rows
        write_csv(
            out_dir / "tables" / "m2_run_deltas.csv",
            m2_run_rows,
            fieldnames=[
                "variant",
                "run_id",
                "provider",
                "split",
                "total_items",
                "gate_families",
                "gate_size",
                "mcc_pre",
                "mcc_post",
                "delta_mcc",
                "delta_balanced_accuracy",
                "delta_tpr",
                "delta_tnr",
                "row_flip_rate",
                "axiom_violation_reduction",
            ],
        )
        write_csv(
            out_dir / "tables" / "m2_gates_per_run.csv",
            aware_gate_rows + agn_gate_rows,
            fieldnames=[
                "variant",
                "run_id",
                "provider",
                "split",
                "family",
                "score",
                "threshold",
                "selected",
                "gate_size",
            ],
        )
        write_csv(
            out_dir / "tables" / "m2_family_features_eval.csv",
            aware_features + agn_features,
            fieldnames=[
                "run_id",
                "provider",
                "split",
                "total_items",
                "family",
                "coverage_fraction",
                "n_total_family",
                "n_valid_family",
                "mcc_pre_family",
                "balanced_accuracy_pre_family",
                "tpr_pre_family",
                "tnr_pre_family",
                "invalid_rate",
                "pred_true_pct",
                "gt_true_pct",
                "bias_abs",
                "error_rate",
                "axiom_violation_rate",
                "group_inconsistency_rate",
                "tpr_tnr_gap",
                "parse_conf_mean",
            ],
        )

        static_by_run = {row["run_id"]: row for row in expanded_rows}
        aware_by_run = {row["run_id"]: row for row in aware_rows}
        agn_by_run = {row["run_id"]: row for row in agn_rows}
        m2_compare_rows: List[Dict[str, Any]] = []
        for run in m2_target_runs:
            if run.run_id not in static_by_run:
                continue
            static_row = static_by_run[run.run_id]
            aware_row = aware_by_run.get(run.run_id)
            agn_row = agn_by_run.get(run.run_id)
            if not aware_row or not agn_row:
                continue
            m2_compare_rows.append(
                {
                    "run_id": run.run_id,
                    "provider": run.provider,
                    "split": run.split,
                    "total_items": run.total_items,
                    "static_delta_mcc": static_row["delta_mcc"],
                    "aware_delta_mcc": aware_row["delta_mcc"],
                    "agnostic_delta_mcc": agn_row["delta_mcc"],
                    "aware_minus_static": aware_row["delta_mcc"]
                    - static_row["delta_mcc"],
                    "agnostic_minus_static": agn_row["delta_mcc"]
                    - static_row["delta_mcc"],
                    "aware_gate_size": aware_row["gate_size"],
                    "agnostic_gate_size": agn_row["gate_size"],
                }
            )

        write_csv(
            out_dir / "tables" / "m2_variant_vs_static.csv",
            m2_compare_rows,
            fieldnames=[
                "run_id",
                "provider",
                "split",
                "total_items",
                "static_delta_mcc",
                "aware_delta_mcc",
                "agnostic_delta_mcc",
                "aware_minus_static",
                "agnostic_minus_static",
                "aware_gate_size",
                "agnostic_gate_size",
            ],
        )

        write_m2_report(
            out_dir=out_dir,
            frozen_name=frozen_name,
            aware_predictor=aware_best["predictor"],
            agnostic_predictor=agn_best["predictor"],
            aware_rows=aware_rows,
            agnostic_rows=agn_rows,
            static_rows=expanded_rows,
            training_rows=m2_dev_training_rows,
        )

        aware_heldout = [row for row in aware_rows if row["split"] == "heldout"]
        agn_heldout = [row for row in agn_rows if row["split"] == "heldout"]
        aware_sign = sign_test_two_sided([row["delta_mcc"] for row in aware_heldout])
        agn_sign = sign_test_two_sided([row["delta_mcc"] for row in agn_heldout])
        m2_summary = {
            "enabled": 1.0,
            "aware_sign_test": aware_sign,
            "agnostic_sign_test": agn_sign,
            "aware_mean_delta_mcc_heldout": (
                statistics.mean(row["delta_mcc"] for row in aware_heldout)
                if aware_heldout
                else 0.0
            ),
            "agnostic_mean_delta_mcc_heldout": (
                statistics.mean(row["delta_mcc"] for row in agn_heldout)
                if agn_heldout
                else 0.0
            ),
            "aware_training_balanced_accuracy": aware_best["classification"][
                "balanced_accuracy"
            ],
            "agnostic_training_balanced_accuracy": agn_best["classification"][
                "balanced_accuracy"
            ],
            "n_target_runs": len(m2_target_runs),
            "force_nonempty": float(args.m2_force_nonempty),
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "selector": str(args.selector),
        "n_discovered_runs": len(all_runs),
        "n_dev_runs": len(dev_runs),
        "n_heldout_runs": len(heldout_runs),
        "n_skipped_manifests": len(skipped),
        "frozen_config_name": frozen_name,
        "frozen_config": frozen_config.to_dict(),
        "frozen_score": frozen_score,
        "selection_max_row_flip_rate": args.max_row_flip_rate,
        "heldout_sign_test": sign_result,
        "heldout_baseline_delta_pearson": pearson,
        "expanded_panel_n": len(expanded_rows),
        "expanded_panel_target": args.expanded_panel_target,
        "expanded_panel_target_met": float(
            len(expanded_rows) >= args.expanded_panel_target
        ),
        "expanded_panel_sign_test": expanded_sign_result,
        "expanded_panel_baseline_delta_pearson": pearson_expanded,
        "order_seed_count": len(order_seeds),
        "parser_noise_rates": noise_rates,
        "m2": m2_summary,
    }
    (out_dir / "repair_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CARE-v3 repair pack")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--selector", default="data/canonical_v2_files.json")
    parser.add_argument("--out-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--expected-total", type=int, default=40886)
    parser.add_argument("--smoke-per-family", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-row-flip-rate", type=float, default=0.10)
    parser.add_argument("--order-seeds", default="11,29,47")
    parser.add_argument("--parser-noise-rates", default="0.01,0.03,0.05")
    parser.add_argument("--expanded-panel-min-items", type=int, default=5000)
    parser.add_argument("--expanded-panel-target", type=int, default=12)
    parser.add_argument(
        "--enable-m2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable M2 adaptive eligibility training/evaluation",
    )
    parser.add_argument(
        "--m2-min-balanced-acc",
        type=float,
        default=0.50,
        help="Minimum training balanced accuracy for accepted M2 predictors",
    )
    parser.add_argument(
        "--m2-force-nonempty",
        action="store_true",
        help="Force at least one selected family per run for M2 variants",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_pipeline(args)
    print("CARE-v3 repair pack built")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
