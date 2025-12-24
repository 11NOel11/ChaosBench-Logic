#!/usr/bin/env python3
"""
ChaosBench-Logic Evaluation Runner

Unified script to run evaluations on any model (GPT-4, Claude, Gemini, LLaMA-3, Mixtral, OpenHermes)
in either zeroshot or chain-of-thought (CoT) mode.

Usage:
    python run_benchmark.py --model gpt4 --mode zeroshot
    python run_benchmark.py --model llama3 --mode both --workers 2
    python run_benchmark.py --model all --mode zeroshot
"""

import argparse
import os
import time
from eval_chaosbench import (
    ModelConfig,
    make_model_client,
    load_batches,
    evaluate_items_with_parallelism,
    compute_summary,
    save_per_item_results,
    save_summary_json,
    save_csvs,
    save_figures,
    save_run_metadata,
)


SUPPORTED_MODELS = ["gpt4", "claude3", "gemini", "llama3", "mixtral", "openhermes"]


def run_evaluation(model_name, mode, workers=None, clear_checkpoint=False, debug=True):
    """Run evaluation for a single model and mode."""
    
    # Configuration
    data_dir = "data"
    batch_files = [
        "batch1_atomic_implication.jsonl",
        "batch2_multiHop_crossSystem.jsonl",
        "batch3_pde_chem_bio.jsonl",
        "batch4_maps_advanced.jsonl",
        "batch5_counterfactual_high_difficulty.jsonl",
        "batch6_deep_bias_probes.jsonl",
        "batch7_multiturn_advanced.jsonl",
    ]
    base_out_dir = "results"
    
    # Load data
    batch_paths = [os.path.join(data_dir, bf) for bf in batch_files]
    items = load_batches(batch_paths)
    
    # Setup output directory
    run_out_dir = os.path.join(base_out_dir, f"{model_name}_{mode}")
    os.makedirs(run_out_dir, exist_ok=True)
    checkpoint_file = os.path.join(run_out_dir, "checkpoint.json")
    
    # Clear checkpoint if requested
    if clear_checkpoint and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"[INFO] Cleared checkpoint: {checkpoint_file}")
    
    print(f"\n{'='*70}")
    print(f"   EVALUATING: {model_name.upper()} ({mode})")
    print(f"{'='*70}")
    print(f"[INFO] Items: {len(items)}")
    print(f"[INFO] Output: {run_out_dir}")
    print(f"[INFO] Workers: {workers if workers else 'auto'}")
    
    run_start = time.time()
    
    try:
        # Create model config and client
        config = ModelConfig(name=model_name, mode=mode)
        client = make_model_client(config)
        
        # Evaluate
        results = evaluate_items_with_parallelism(
            items=items,
            client=client,
            numeric_fact_map={},
            model_name=model_name,
            mode=mode,
            max_workers=workers,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=50,
            debug=debug,
            debug_samples=10,
            output_dir=run_out_dir,
        )
        
        # Compute and save metrics
        summary = compute_summary(results)
        
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")
        
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
        
        run_time = time.time() - run_start
        valid_count = len([r for r in results if r.correct is not None])
        failed_count = len([r for r in results if r.correct is None])
        
        print(f"\n[✓] {model_name.upper()} {mode} completed in {run_time:.1f}s ({run_time/60:.1f}min)")
        if summary.get('overall_accuracy') is not None:
            print(f"    Overall accuracy: {summary['overall_accuracy']:.1%}")
        print(f"    Valid responses: {valid_count}/{len(results)}")
        if failed_count > 0:
            print(f"    ⚠️  Failed items: {failed_count}")
        
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
  # Run single model
  python run_benchmark.py --model gpt4 --mode zeroshot
  
  # Run LLaMA-3 with 2 workers (slower but stable)
  python run_benchmark.py --model llama3 --mode both --workers 2
  
  # Run all models in zeroshot mode
  python run_benchmark.py --model all --mode zeroshot
  
  # Clear checkpoints and restart
  python run_benchmark.py --model claude3 --mode cot --clear-checkpoints

Supported models: gpt4, claude3, gemini, llama3, mixtral, openhermes, all
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS + ["all"],
        help="Model to evaluate (or 'all' for all models)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zeroshot", "cot", "both"],
        default="zeroshot",
        help="Evaluation mode (default: zeroshot)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto per model)"
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear existing checkpoints and start fresh"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode (saves sample outputs)"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("   CHAOSBENCH-LOGIC: EVALUATION RUNNER")
    print("="*70)
    
    # Determine which models to run
    if args.model == "all":
        models_to_run = SUPPORTED_MODELS
        print(f"[INFO] Running all models: {', '.join(models_to_run)}")
    else:
        models_to_run = [args.model]
    
    # Determine which modes to run
    if args.mode == "both":
        modes_to_run = ["zeroshot", "cot"]
    else:
        modes_to_run = [args.mode]
    
    # Run evaluations
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
                debug=args.debug
            )
            if success:
                completed += 1
            else:
                failed += 1
    
    # Final summary
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print(f"Total runs: {total}")
    print(f"Completed: {completed}")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"\nResults saved in: results/")
    print("\nView results:")
    for model in models_to_run:
        for mode in modes_to_run:
            print(f"  cat results/{model}_{mode}/summary.json")


if __name__ == "__main__":
    main()
