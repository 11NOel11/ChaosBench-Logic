#!/usr/bin/env python3
"""Backward-compatibility shim for run_benchmark.

All functionality has moved to chaosbench.eval.runner.
This file re-exports everything so existing scripts continue to work.
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

from chaosbench.models.prompt import ModelConfig, make_model_client
from chaosbench.eval.runner import (
    load_batches,
    evaluate_items_with_parallelism,
    save_per_item_results,
    save_summary_json,
    save_csvs,
    save_figures,
    save_run_metadata,
)
from chaosbench.eval.metrics import compute_summary
from chaosbench.eval.cache import ResponseCache


SUPPORTED_MODELS = [
    "gpt4",
    "claude3",
    "gemini",
    "llama3",
    "mixtral",
    "openhermes",
    "dummy",
]


def discover_batch_files(data_dir, requested_batches=None):
    """Discover batch files in numeric order.

    Args:
        data_dir: Directory containing batch JSONL files.
        requested_batches: Optional explicit list of batch filenames.

    Returns:
        Ordered list of batch filenames.
    """
    if requested_batches:
        return requested_batches

    pattern = re.compile(r"^batch(\d+)_.*\.jsonl$")
    batches = []
    for name in os.listdir(data_dir):
        match = pattern.match(name)
        if match:
            batches.append((int(match.group(1)), name))

    batches.sort(key=lambda item: item[0])
    return [name for _, name in batches]


def slice_items_for_shard(items, shard_index=0, num_shards=1):
    """Return deterministic shard slice for distributed evaluation."""
    if num_shards <= 1:
        return items
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def compute_file_sha256(path):
    """Compute SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_run_manifest(
    model_name,
    mode,
    run_out_dir,
    data_dir,
    batch_files,
    total_items,
    shard_items,
    workers,
    checkpoint_interval,
    shard_index,
    num_shards,
    max_items,
):
    """Write a reproducibility manifest for a benchmark run."""
    os.makedirs("runs", exist_ok=True)

    batch_entries = []
    for batch_file in batch_files:
        batch_path = os.path.join(data_dir, batch_file)
        entry = {
            "file": batch_file,
            "path": batch_path,
            "sha256": compute_file_sha256(batch_path)
            if os.path.exists(batch_path)
            else None,
        }
        batch_entries.append(entry)

    timestamp = datetime.now(timezone.utc).isoformat()
    stamp_for_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{model_name}_{mode}"
    if num_shards > 1:
        run_name = f"{run_name}_shard{shard_index + 1}of{num_shards}"

    manifest = {
        "manifest_version": "1.0",
        "timestamp": timestamp,
        "model": model_name,
        "mode": mode,
        "run_name": run_name,
        "run_out_dir": run_out_dir,
        "data_dir": data_dir,
        "batches": batch_entries,
        "total_items_before_sharding": total_items,
        "items_in_this_run": shard_items,
        "workers": workers,
        "checkpoint_interval": checkpoint_interval,
        "sharding": {
            "num_shards": num_shards,
            "shard_index": shard_index,
        },
        "max_items": max_items,
    }

    manifest_path = os.path.join("runs", f"manifest_{stamp_for_name}_{run_name}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    latest_path = os.path.join(run_out_dir, "run_manifest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Run manifest saved: {manifest_path}")
    return manifest_path


def run_evaluation(
    model_name,
    mode,
    workers=None,
    clear_checkpoint=False,
    debug=True,
    data_dir="data",
    batch_files=None,
    out_dir="results",
    checkpoint_interval=50,
    shard_index=0,
    num_shards=1,
    max_items=None,
    cache_dir=None,
    split=None,
):
    """Run evaluation for a single model and mode."""
    batch_files = discover_batch_files(data_dir, batch_files)
    if not batch_files:
        raise ValueError(f"No batch files found in: {data_dir}")

    batch_paths = [os.path.join(data_dir, bf) for bf in batch_files]
    items = load_batches(batch_paths)

    total_items = len(items)
    items = slice_items_for_shard(items, shard_index=shard_index, num_shards=num_shards)
    if max_items is not None:
        items = items[:max_items]

    run_name = f"{model_name}_{mode}"
    if num_shards > 1:
        run_name = f"{run_name}_shard{shard_index + 1}of{num_shards}"
    run_out_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_out_dir, exist_ok=True)
    checkpoint_file = os.path.join(run_out_dir, "checkpoint.json")

    if clear_checkpoint and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"[INFO] Cleared checkpoint: {checkpoint_file}")

    print(f"\n{'=' * 70}")
    print(f"   EVALUATING: {model_name.upper()} ({mode})")
    print(f"{'=' * 70}")
    print(f"[INFO] Items: {len(items)}")
    if num_shards > 1:
        print(
            f"[INFO] Shard: {shard_index + 1}/{num_shards} (from {total_items} total items)"
        )
    print(f"[INFO] Output: {run_out_dir}")
    print(f"[INFO] Workers: {workers if workers else 'auto'}")

    run_start = time.time()

    try:
        config = ModelConfig(name=model_name, mode=mode)
        client = make_model_client(config)

        cache = None
        if cache_dir is not None:
            cache = ResponseCache(cache_dir)
            print(f"[INFO] Cache enabled: {cache_dir}")
            cache_stats = cache.stats()
            print(f"[INFO] Cache stats: {cache_stats}")

        results = evaluate_items_with_parallelism(
            items=items,
            client=client,
            numeric_fact_map={},
            model_name=model_name,
            mode=mode,
            max_workers=workers,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=checkpoint_interval,
            debug=debug,
            debug_samples=10,
            output_dir=run_out_dir,
            cache=cache,
        )

        if cache is not None:
            cache.close()

        summary = compute_summary(results)

        print(f"\n{'=' * 70}")
        print("SAVING RESULTS")
        print(f"{'=' * 70}")

        save_per_item_results(results, run_out_dir)
        save_summary_json(summary, run_out_dir)
        save_csvs(summary, run_out_dir, model_name, mode)
        save_figures(summary, run_out_dir)
        save_run_metadata(
            out_dir=run_out_dir,
            model_name=model_name,
            mode=mode,
            max_workers=workers if workers else 0,
            results=results,
        )

        write_run_manifest(
            model_name=model_name,
            mode=mode,
            run_out_dir=run_out_dir,
            data_dir=data_dir,
            batch_files=batch_files,
            total_items=total_items,
            shard_items=len(items),
            workers=workers,
            checkpoint_interval=checkpoint_interval,
            shard_index=shard_index,
            num_shards=num_shards,
            max_items=max_items,
        )

        run_time = time.time() - run_start
        valid_count = len([r for r in results if r.correct is not None])
        failed_count = len([r for r in results if r.correct is None])

        print(
            f"\n[OK] {model_name.upper()} {mode} completed in {run_time:.1f}s ({run_time / 60:.1f}min)"
        )
        if summary.get("overall_accuracy") is not None:
            print(f"    Overall accuracy: {summary['overall_accuracy']:.1%}")
        print(f"    Valid responses: {valid_count}/{len(results)}")
        if failed_count > 0:
            print(f"    Failed items: {failed_count}")

        return True

    except Exception as e:
        print(f"\n[ERROR] {model_name.upper()} {mode} failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run ChaosBench-Logic evaluation on LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --model gpt4 --mode zeroshot
  python run_benchmark.py --model llama3 --mode both --workers 2
  python run_benchmark.py --model all --mode zeroshot
  python run_benchmark.py --model claude3 --mode cot --clear-checkpoints

Supported models: gpt4, claude3, gemini, llama3, mixtral, openhermes, dummy, all
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS + ["all"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zeroshot", "cot", "both"],
        default="zeroshot",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--clear-checkpoints", action="store_true")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batches", type=str, nargs="+", default=None)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for response cache")
    parser.add_argument("--split", type=str, default=None, help="Split name for data filtering")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    print("\n" + "=" * 70)
    print("   CHAOSBENCH-LOGIC: EVALUATION RUNNER")
    print("=" * 70)

    if args.model == "all":
        models_to_run = SUPPORTED_MODELS
        print(f"[INFO] Running all models: {', '.join(models_to_run)}")
    else:
        models_to_run = [args.model]

    if args.mode == "both":
        modes_to_run = ["zeroshot", "cot"]
    else:
        modes_to_run = [args.mode]

    total = len(models_to_run) * len(modes_to_run)
    completed = 0
    failed = 0

    for model in models_to_run:
        for mode in modes_to_run:
            success = run_evaluation(
                model_name=model,
                mode=mode,
                workers=args.workers,
                clear_checkpoint=args.clear_checkpoints,
                debug=args.debug,
                data_dir=args.data_dir,
                batch_files=args.batches,
                out_dir=args.out_dir,
                checkpoint_interval=args.checkpoint_interval,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                max_items=args.max_items,
                cache_dir=args.cache_dir,
                split=args.split,
            )
            if success:
                completed += 1
            else:
                failed += 1

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total runs: {total}")
    print(f"Completed: {completed}")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"\nResults saved in: {args.out_dir}/")
    for model in models_to_run:
        for mode in modes_to_run:
            run_name = f"{model}_{mode}"
            if args.num_shards > 1:
                run_name = f"{run_name}_shard{args.shard_index + 1}of{args.num_shards}"
            print(f"  cat {args.out_dir}/{run_name}/summary.json")


if __name__ == "__main__":
    main()
