"""Consolidated subset generator for ChaosBench-Logic.

Single entrypoint for all dataset subset creation. Supports:
- Family-aware proportional sampling
- Label balancing within families
- Family weighting overrides
- Min-per-family enforcement with cap logging
- Deterministic output via seeded RNG
- SHA256-tracked manifests with full provenance

Usage (Python API)::

    from chaosbench.data.subsets import SubsetConfig, create_subset
    cfg = SubsetConfig(size=1000, seed=42, balance_labels=True, balance_families=True)
    items, manifest = create_subset(cfg)

Usage (CLI)::

    python -m chaosbench.data.subsets --size 1000 --seed 42 --balance-labels --balance-families \\
        --out data/subsets/my_subset.jsonl

"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SubsetConfig:
    """Configuration for a single subset generation run.

    Attributes:
        size: Target number of items.
        seed: RNG seed for reproducibility.
        balance_labels: If True, enforce ~50/50 TRUE/FALSE within each family.
        balance_families: If True, allocate quota proportionally by family size.
        family_weights: Optional dict mapping family name → sampling weight
            multiplier (relative to proportional baseline). E.g. ``{"multi_hop": 2.0}``
            doubles the multi_hop quota.  Missing families use weight 1.0.
        min_per_family: Minimum items to include per family (if enough items
            exist in the source). Default 1.
        source: ``"canonical"`` (uses ``data/canonical_v2_files.json``) or a
            file path to a JSONL file.
        output_dir: Directory to write subset.jsonl and manifest. If None,
            artifacts are returned in-memory only.
        subset_name: Base name used for output file (without extension).
    """

    size: int
    seed: int = 42
    balance_labels: bool = True
    balance_families: bool = True
    family_weights: Dict[str, float] = field(default_factory=dict)
    min_per_family: int = 1
    source: str = "canonical"
    output_dir: Optional[str] = None
    subset_name: str = "subset"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_canonical_items() -> List[Dict[str, Any]]:
    """Load all items from the canonical v2 file list."""
    selector_path = PROJECT_ROOT / "data" / "canonical_v2_files.json"
    if not selector_path.exists():
        raise FileNotFoundError(f"Canonical selector not found: {selector_path}")
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []
    for rel_path in selector["files"]:
        fpath = PROJECT_ROOT / rel_path
        if not fpath.exists():
            logger.warning("Canonical file missing: %s", fpath)
            continue
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item["_source_file"] = fpath.name
                    items.append(item)
    return items


def _load_file_items(path: str) -> List[Dict[str, Any]]:
    """Load items from a single JSONL file."""
    items: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _get_family(item: Dict[str, Any]) -> str:
    """Return the task family for an item."""
    return item.get("type") or item.get("task_family") or item.get("family") or "unknown"


def _get_label(item: Dict[str, Any]) -> str:
    """Return the ground-truth label for an item."""
    return item.get("ground_truth") or item.get("answer") or item.get("gold") or ""


# ---------------------------------------------------------------------------
# Core sampling logic
# ---------------------------------------------------------------------------


def _compute_quotas(
    by_family: Dict[str, List[Dict[str, Any]]],
    target_size: int,
    family_weights: Dict[str, float],
    min_per_family: int,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Compute per-family item quotas.

    Returns:
        quotas: mapping family → target item count.
        log_notes: mapping family → human-readable note for provenance.
    """
    families = list(by_family.keys())
    total_available = sum(len(v) for v in by_family.values())

    # Compute raw weights (proportional × override)
    weights: Dict[str, float] = {}
    for fam in families:
        base = len(by_family[fam]) / total_available if total_available > 0 else 0.0
        weights[fam] = base * family_weights.get(fam, 1.0)

    # Normalise weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        normalised = {fam: 1.0 / len(families) for fam in families}
    else:
        normalised = {fam: w / total_weight for fam, w in weights.items()}

    # Assign initial quotas
    quotas: Dict[str, int] = {}
    notes: Dict[str, str] = {}
    for fam in families:
        raw_quota = target_size * normalised[fam]
        quota = max(min_per_family, int(math.floor(raw_quota)))
        available = len(by_family[fam])

        if quota > available:
            notes[fam] = (
                f"capped at available={available} (requested {quota})"
            )
            quota = available
        else:
            notes[fam] = f"quota={quota} of available={available}"

        quotas[fam] = quota

    # Pad to target_size if under (distribute residuals to largest families)
    total_assigned = sum(quotas.values())
    if total_assigned < target_size:
        shortfall = target_size - total_assigned
        by_size = sorted(
            [(fam, len(by_family[fam]) - quotas[fam]) for fam in families],
            key=lambda x: x[1],
            reverse=True,
        )
        for fam, headroom in by_size:
            if shortfall <= 0:
                break
            extra = min(shortfall, headroom)
            quotas[fam] += extra
            shortfall -= extra
            notes[fam] += f" (+{extra} padding)"

    return quotas, notes


def _sample_family(
    items: List[Dict[str, Any]],
    quota: int,
    balance_labels: bool,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample *quota* items from a single family bucket.

    Returns:
        sampled items, per-family log dict.
    """
    true_items = [i for i in items if _get_label(i) in ("TRUE", "YES")]
    false_items = [i for i in items if _get_label(i) in ("FALSE", "NO")]
    other_items = [i for i in items if _get_label(i) not in ("TRUE", "YES", "FALSE", "NO")]

    rng.shuffle(true_items)
    rng.shuffle(false_items)
    rng.shuffle(other_items)

    sampled: List[Dict[str, Any]] = []
    achieved_balance = "n/a"

    if balance_labels and (true_items or false_items):
        n_true = min(quota // 2, len(true_items))
        n_false = min(quota - n_true, len(false_items))
        # If one class exhausted, take more from the other
        if n_true + n_false < quota:
            remaining = quota - n_true - n_false
            extra_true = min(remaining, len(true_items) - n_true)
            n_true += extra_true
            extra_false = min(remaining - extra_true, len(false_items) - n_false)
            n_false += extra_false
        sampled = true_items[:n_true] + false_items[:n_false]
        if n_true + n_false > 0:
            achieved_balance = f"{n_true/(n_true+n_false)*100:.0f}% TRUE ({n_true}T/{n_false}F)"
    else:
        combined = true_items + false_items + other_items
        rng.shuffle(combined)
        sampled = combined[:quota]
        n_true = sum(1 for i in sampled if _get_label(i) in ("TRUE", "YES"))
        n_false = sum(1 for i in sampled if _get_label(i) in ("FALSE", "NO"))
        if n_true + n_false > 0:
            achieved_balance = f"{n_true/(n_true+n_false)*100:.0f}% TRUE (unbalanced)"

    # Pad with other_items if still short
    if len(sampled) < quota and other_items:
        sampled += other_items[: quota - len(sampled)]

    if not balance_labels and len(true_items) + len(false_items) > 0:
        # Already handled above
        pass

    log = {
        "available": len(items),
        "requested": quota,
        "achieved": len(sampled),
        "label_balance": achieved_balance,
        "true_available": len(true_items),
        "false_available": len(false_items),
    }
    return sampled, log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_subset(
    config: SubsetConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create a dataset subset according to *config*.

    Returns:
        items: list of selected items (no ``_source_file`` key).
        manifest: provenance dict suitable for writing to ``<name>.manifest.json``.
    """
    rng = random.Random(config.seed)

    # Load source
    if config.source == "canonical":
        all_items = _load_canonical_items()
    else:
        all_items = _load_file_items(config.source)

    if not all_items:
        raise ValueError("No items loaded from source.")

    # Group by family
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        by_family[_get_family(item)].append(item)

    family_names = sorted(by_family.keys())

    # Compute quotas
    if config.balance_families:
        quotas, quota_notes = _compute_quotas(
            by_family,
            config.size,
            config.family_weights,
            config.min_per_family,
        )
    else:
        # Simple proportional (no weight overrides)
        total = len(all_items)
        quotas = {}
        quota_notes = {}
        for fam in family_names:
            q = max(config.min_per_family, int(config.size * len(by_family[fam]) / total))
            quotas[fam] = min(q, len(by_family[fam]))
            quota_notes[fam] = f"quota={quotas[fam]} of available={len(by_family[fam])}"

    # Sample per family
    all_sampled: List[Dict[str, Any]] = []
    per_family_log: Dict[str, Any] = {}

    for fam in family_names:
        quota = quotas.get(fam, 0)
        items_sampled, fam_log = _sample_family(
            by_family[fam],
            quota,
            config.balance_labels,
            rng,
        )
        all_sampled.extend(items_sampled)
        per_family_log[fam] = {**fam_log, "quota_note": quota_notes.get(fam, "")}

        # Log any constraint violations
        if fam_log["achieved"] < quota:
            logger.warning(
                "Family '%s': achieved %d < requested %d (source only has %d)",
                fam,
                fam_log["achieved"],
                quota,
                fam_log["available"],
            )
        logger.info(
            "  %-35s: %3d / %-3d  %s",
            fam,
            fam_log["achieved"],
            quota,
            fam_log["label_balance"],
        )

    # Final shuffle and truncate to exact target size
    rng.shuffle(all_sampled)
    all_sampled = all_sampled[: config.size]

    # Strip internal metadata keys
    clean_items = [{k: v for k, v in item.items() if not k.startswith("_")} for item in all_sampled]

    # Compute subset hash (deterministic over sorted IDs)
    ids = sorted(item["id"] for item in clean_items)
    subset_hash = hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]

    # Per-family counts in final subset
    final_family_counts: Dict[str, Dict[str, int]] = {}
    for item in clean_items:
        fam = _get_family(item)
        lbl = _get_label(item)
        if fam not in final_family_counts:
            final_family_counts[fam] = {"total": 0, "TRUE": 0, "FALSE": 0}
        final_family_counts[fam]["total"] += 1
        if lbl in ("TRUE", "YES"):
            final_family_counts[fam]["TRUE"] += 1
        elif lbl in ("FALSE", "NO"):
            final_family_counts[fam]["FALSE"] += 1

    # Overall label stats
    n_true = sum(v["TRUE"] for v in final_family_counts.values())
    n_false = sum(v["FALSE"] for v in final_family_counts.values())

    # Dataset global SHA256 from freeze manifest if available
    dataset_global_sha: Optional[str] = None
    freeze_path = PROJECT_ROOT / "artifacts" / "freeze" / "v2_freeze_manifest.json"
    if freeze_path.exists():
        try:
            freeze_data = json.loads(freeze_path.read_text(encoding="utf-8"))
            dataset_global_sha = freeze_data.get("global_sha256")
        except Exception:
            pass

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subset_name": config.subset_name,
        "size": len(clean_items),
        "target_size": config.size,
        "seed": config.seed,
        "source": config.source,
        "selection_method": (
            "balanced_by_family" if config.balance_families else "proportional"
        ),
        "balance_labels": config.balance_labels,
        "balance_families": config.balance_families,
        "family_weights": config.family_weights,
        "min_per_family": config.min_per_family,
        "sha256": subset_hash,
        "dataset_global_sha256": dataset_global_sha,
        "label_balance": {
            "true_count": n_true,
            "false_count": n_false,
            "true_ratio": round(n_true / len(clean_items), 4) if clean_items else 0.0,
        },
        "per_family_counts": final_family_counts,
        "per_family_sampling_log": per_family_log,
    }

    return clean_items, manifest


def write_subset(
    items: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    output_dir: str,
    subset_name: str,
) -> Tuple[Path, Path]:
    """Write subset JSONL and manifest JSON to *output_dir*.

    Returns:
        (jsonl_path, manifest_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / f"{subset_name}.jsonl"
    manifest_path = out / f"{subset_name}.manifest.json"

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return jsonl_path, manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a ChaosBench-Logic dataset subset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1k balanced subset (default policy, paper-friendly)
  python -m chaosbench.data.subsets --size 1000 --seed 42 \\
      --balance-labels --balance-families \\
      --out data/subsets/api_balanced_1k.jsonl

  # 5k balanced with custom family weights
  python -m chaosbench.data.subsets --size 5000 --seed 42 \\
      --balance-labels --balance-families \\
      --family-weights '{"multi_hop": 1.5, "adversarial_misleading": 2.0}' \\
      --out data/subsets/api_weighted_5k.jsonl

  # Stratified (no label balancing)
  python -m chaosbench.data.subsets --size 2000 --seed 0 \\
      --out data/subsets/stratified_2k.jsonl
""",
    )
    parser.add_argument("--size", type=int, required=True, help="Target subset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--balance-labels",
        action="store_true",
        default=False,
        help="Balance TRUE/FALSE within each family",
    )
    parser.add_argument(
        "--balance-families",
        action="store_true",
        default=False,
        help="Allocate quota proportionally by family size",
    )
    parser.add_argument(
        "--family-weights",
        type=str,
        default=None,
        help="JSON object: family name → weight multiplier (e.g. '{\"multi_hop\": 2.0}')",
    )
    parser.add_argument(
        "--min-per-family",
        type=int,
        default=1,
        help="Minimum items per family (default: 1)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="canonical",
        help="'canonical' or path to a JSONL file (default: canonical)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL path (manifest written alongside)",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    family_weights: Dict[str, float] = {}
    if args.family_weights:
        try:
            family_weights = json.loads(args.family_weights)
        except json.JSONDecodeError as exc:
            parser.error(f"--family-weights is not valid JSON: {exc}")

    out_path = Path(args.out)
    output_dir = str(out_path.parent)
    subset_name = out_path.stem

    config = SubsetConfig(
        size=args.size,
        seed=args.seed,
        balance_labels=args.balance_labels,
        balance_families=args.balance_families,
        family_weights=family_weights,
        min_per_family=args.min_per_family,
        source=args.source,
        output_dir=output_dir,
        subset_name=subset_name,
    )

    logger.info("Loading items (source=%s)...", config.source)
    items, manifest = create_subset(config)
    logger.info("Sampled %d / %d items", len(items), config.size)

    n_true = manifest["label_balance"]["true_count"]
    n_false = manifest["label_balance"]["false_count"]
    logger.info("Label balance: %.1f%% TRUE | %.1f%% FALSE",
                n_true / len(items) * 100 if items else 0,
                n_false / len(items) * 100 if items else 0)

    logger.info("\nPer-family breakdown:")
    for fam, counts in sorted(manifest["per_family_counts"].items()):
        t, f = counts["TRUE"], counts["FALSE"]
        total = counts["total"]
        pct = t / total * 100 if total > 0 else 0
        logger.info("  %-35s: %4d  (%.0f%% TRUE)", fam, total, pct)

    jsonl_path, manifest_path = write_subset(items, manifest, output_dir, subset_name)
    logger.info("\nWrote subset   → %s", jsonl_path)
    logger.info("Wrote manifest → %s", manifest_path)
    logger.info("SHA256         : %s", manifest["sha256"])


if __name__ == "__main__":
    main(sys.argv[1:])
