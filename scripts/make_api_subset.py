#!/usr/bin/env python3
"""Create balanced API eval subsets from ChaosBench-Logic dataset.

Samples items to create subsets with controlled label balance within families.

Usage:
    python scripts/make_api_subset.py --data_dir data/ --out_path data/subsets/api_core_1k.jsonl --size 1000 --seed 42
    python scripts/make_api_subset.py --data_dir data/ --out_path data/subsets/api_balanced_500.jsonl --size 500 --balance --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_canonical_files(data_dir: Path) -> List[Path]:
    """Load canonical file paths from data/canonical_v2_files.json."""
    selector_path = data_dir.parent / "data" / "canonical_v2_files.json"
    if not selector_path.exists():
        selector_path = data_dir / "canonical_v2_files.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    return [data_dir.parent / f for f in selector["files"]]


def load_all_items(data_dir: str) -> List[Dict[str, Any]]:
    """Load all JSONL items from canonical v2 dataset files."""
    items = []
    data_path = Path(data_dir)

    canonical_files = _load_canonical_files(data_path)
    for fpath in canonical_files:
        if not fpath.exists():
            continue
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item["_batch_file"] = fpath.name
                    items.append(item)

    return items


def sample_balanced_by_family(
    items: List[Dict[str, Any]],
    target_size: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample items with balanced labels within each family.

    Args:
        items: List of all items.
        target_size: Target subset size.
        seed: Random seed for reproducibility.

    Returns:
        Sampled subset with balanced labels.
    """
    rng = random.Random(seed)

    # Group by family
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        family = item.get("type", "unknown")
        by_family[family].append(item)

    # Allocate quota per family (proportional to family size)
    total_items = len(items)
    family_quotas = {}
    for family, family_items in by_family.items():
        quota = int(target_size * len(family_items) / total_items)
        family_quotas[family] = max(1, quota)  # Ensure at least 1 item per family

    # Sample from each family with label balance
    sampled = []
    for family, family_items in by_family.items():
        quota = family_quotas[family]

        # Separate by label
        true_items = [item for item in family_items if item.get("ground_truth") in ("TRUE", "YES")]
        false_items = [item for item in family_items if item.get("ground_truth") in ("FALSE", "NO")]

        # Try to balance labels (50/50 split within family)
        n_true = min(quota // 2, len(true_items))
        n_false = min(quota - n_true, len(false_items))

        # If one class is exhausted, take more from the other
        if n_true < quota // 2 and len(false_items) >= quota:
            n_false = quota - n_true
        if n_false < quota // 2 and len(true_items) >= quota:
            n_true = quota - n_false

        # Sample
        rng.shuffle(true_items)
        rng.shuffle(false_items)
        sampled.extend(true_items[:n_true])
        sampled.extend(false_items[:n_false])

    # If we're under target, add more items randomly
    if len(sampled) < target_size:
        remaining = [item for item in items if item not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[:target_size - len(sampled)])

    # Shuffle final subset
    rng.shuffle(sampled)

    return sampled[:target_size]


def sample_stratified(
    items: List[Dict[str, Any]],
    target_size: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample items stratified by family (proportional sampling).

    Args:
        items: List of all items.
        target_size: Target subset size.
        seed: Random seed for reproducibility.

    Returns:
        Stratified sample.
    """
    rng = random.Random(seed)

    # Group by family
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        family = item.get("type", "unknown")
        by_family[family].append(item)

    # Allocate quota per family (proportional to family size)
    total_items = len(items)
    sampled = []
    for family, family_items in by_family.items():
        quota = int(target_size * len(family_items) / total_items)
        quota = max(1, quota)  # Ensure at least 1 item per family

        rng.shuffle(family_items)
        sampled.extend(family_items[:quota])

    # Shuffle and truncate to exact target size
    rng.shuffle(sampled)
    return sampled[:target_size]


def compute_manifest_hash(items: List[Dict[str, Any]]) -> str:
    """Compute deterministic hash of subset for verification."""
    ids = sorted([item["id"] for item in items])
    signature = "|".join(ids)
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Create balanced API eval subset")
    parser.add_argument("--data_dir", default="data/", help="Data directory")
    parser.add_argument("--out_path", required=True, help="Output JSONL file path")
    parser.add_argument("--size", type=int, required=True, help="Target subset size")
    parser.add_argument("--balance", action="store_true", help="Balance labels within families")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    print(f"Loading items from {args.data_dir}...")
    items = load_all_items(args.data_dir)
    print(f"  Loaded {len(items)} items")

    print(f"\nSampling {args.size} items (balance={args.balance}, seed={args.seed})...")
    if args.balance:
        subset = sample_balanced_by_family(items, args.size, seed=args.seed)
    else:
        subset = sample_stratified(items, args.size, seed=args.seed)

    print(f"  Sampled {len(subset)} items")

    # Compute label balance in subset
    n_true = sum(1 for item in subset if item.get("ground_truth") in ("TRUE", "YES"))
    n_false = sum(1 for item in subset if item.get("ground_truth") in ("FALSE", "NO"))
    print(f"  Label balance: {n_true/len(subset)*100:.1f}% TRUE | {n_false/len(subset)*100:.1f}% FALSE")

    # Per-family balance
    by_family: Dict[str, List[str]] = defaultdict(list)
    for item in subset:
        family = item.get("type", "unknown")
        label = item.get("ground_truth", "")
        by_family[family].append(label)

    print("\n  Per-family balance:")
    for family, labels in sorted(by_family.items()):
        n_true_fam = sum(1 for l in labels if l in ("TRUE", "YES"))
        print(f"    {family:30s}: {n_true_fam/len(labels)*100:5.1f}% TRUE ({n_true_fam:3d}/{len(labels):3d})")

    # Write subset
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for item in subset:
            # Remove _batch_file metadata
            item_clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(item_clean) + "\n")

    print(f"\nWrote subset to {out_path}")

    # Write manifest
    manifest_path = out_path.with_suffix(".manifest.json")

    # Per-family counts
    per_family_counts: Dict[str, Dict[str, int]] = {}
    for item in subset:
        fam = item.get("type", "unknown")
        lbl = item.get("ground_truth", "")
        if fam not in per_family_counts:
            per_family_counts[fam] = {"total": 0, "TRUE": 0, "FALSE": 0}
        per_family_counts[fam]["total"] += 1
        if lbl in ("TRUE", "YES"):
            per_family_counts[fam]["TRUE"] += 1
        elif lbl in ("FALSE", "NO"):
            per_family_counts[fam]["FALSE"] += 1

    # Dataset global SHA256 from freeze manifest if available
    dataset_global_sha = None
    freeze_path = PROJECT_ROOT / "artifacts" / "freeze" / "v2_freeze_manifest.json"
    if freeze_path.exists():
        try:
            freeze_data = json.loads(freeze_path.read_text())
            dataset_global_sha = freeze_data.get("global_sha256")
        except Exception:
            pass

    from datetime import datetime, timezone
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "size": len(subset),
        "seed": args.seed,
        "selection_method": "balanced_by_family" if args.balance else "stratified",
        "sha256": compute_manifest_hash(subset),
        "dataset_global_sha256": dataset_global_sha,
        "label_balance": {
            "true_count": n_true,
            "false_count": n_false,
            "true_ratio": n_true / len(subset) if subset else 0.0,
        },
        "per_family_counts": per_family_counts,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest to {manifest_path}")
    print(f"Manifest SHA-256: {manifest['sha256']}")


if __name__ == "__main__":
    main()
