#!/usr/bin/env python3
"""Merge sharded evaluation outputs into one canonical run directory.

Usage:
    python scripts/merge_sharded_runs.py --model gpt4 --mode zeroshot --results-dir results
"""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from chaosbench.eval.metrics import EvalResult, compute_summary
from chaosbench.eval.runner import (
    save_csvs,
    save_figures,
    save_per_item_results,
    save_run_metadata,
    save_summary_json,
)


def discover_shard_dirs(
    results_dir: Path, model: str, mode: str
) -> List[Tuple[int, int, Path]]:
    """Find shard directories for model/mode.

    Returns a list of (shard_index_1based, num_shards, path) sorted by shard index.
    """
    pattern = re.compile(rf"^{re.escape(model)}_{re.escape(mode)}_shard(\d+)of(\d+)$")
    matches: List[Tuple[int, int, Path]] = []

    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            shard_index = int(m.group(1))
            num_shards = int(m.group(2))
            matches.append((shard_index, num_shards, child))

    matches.sort(key=lambda x: x[0])
    return matches


def load_results_jsonl(path: Path) -> List[EvalResult]:
    """Load EvalResult rows from a JSONL file."""
    results: List[EvalResult] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(EvalResult(**json.loads(line)))
    return results


def merge_results_by_item(
    results_per_shard: List[List[EvalResult]],
) -> Tuple[List[EvalResult], Dict[str, int]]:
    """Merge shard results and verify duplicates are identical."""
    merged: Dict[str, EvalResult] = {}
    duplicates = 0

    for shard_results in results_per_shard:
        for result in shard_results:
            if result.item_id in merged:
                duplicates += 1
                existing = merged[result.item_id]
                if asdict(existing) != asdict(result):
                    raise ValueError(
                        f"Conflicting duplicate for item_id={result.item_id}; "
                        "two shards produced different outputs"
                    )
            else:
                merged[result.item_id] = result

    sorted_results = sorted(merged.values(), key=lambda r: r.item_id)
    report = {
        "unique_items": len(sorted_results),
        "duplicate_rows": duplicates,
    }
    return sorted_results, report


def validate_shard_set(shards: List[Tuple[int, int, Path]]) -> int:
    """Validate shard indices and return declared number of shards."""
    if not shards:
        raise ValueError("No shard directories found for requested model/mode")

    declared = {num for _, num, _ in shards}
    if len(declared) != 1:
        raise ValueError(f"Inconsistent shard declarations: {sorted(declared)}")

    num_shards = next(iter(declared))
    expected = set(range(1, num_shards + 1))
    found = {idx for idx, _, _ in shards}
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Missing shard directories: {missing}")

    return num_shards


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded ChaosBench run outputs")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, required=True, choices=["zeroshot", "cot", "tool"]
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-figures", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else results_dir / f"{args.model}_{args.mode}"
    )

    shards = discover_shard_dirs(results_dir, args.model, args.mode)
    num_shards = validate_shard_set(shards)

    print(f"[INFO] Found {len(shards)} shard directories for {args.model}_{args.mode}")

    results_per_shard: List[List[EvalResult]] = []
    shard_counts: Dict[str, int] = {}

    for shard_index, _, shard_path in shards:
        per_item_path = shard_path / "per_item_results.jsonl"
        if not per_item_path.exists():
            raise FileNotFoundError(f"Missing expected file: {per_item_path}")
        shard_results = load_results_jsonl(per_item_path)
        results_per_shard.append(shard_results)
        shard_counts[f"shard{shard_index}"] = len(shard_results)

    merged_results, merge_report = merge_results_by_item(results_per_shard)
    summary = compute_summary(merged_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_per_item_results(merged_results, str(output_dir))
    save_summary_json(summary, str(output_dir))
    save_csvs(summary, str(output_dir), args.model, args.mode)
    save_run_metadata(
        out_dir=str(output_dir),
        model_name=args.model,
        mode=args.mode,
        max_workers=0,
        results=merged_results,
    )
    if args.save_figures:
        save_figures(summary, str(output_dir))

    merge_meta = {
        "model": args.model,
        "mode": args.mode,
        "source_results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "num_shards": num_shards,
        "shard_counts": shard_counts,
        **merge_report,
    }
    merge_meta_path = output_dir / "merge_meta.json"
    with merge_meta_path.open("w", encoding="utf-8") as f:
        json.dump(merge_meta, f, indent=2)

    print(f"[OK] Merged {merge_report['unique_items']} unique items into: {output_dir}")
    print(f"[OK] Merge metadata: {merge_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
