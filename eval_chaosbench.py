"""Backward-compatibility shim for eval_chaosbench.

All functionality has moved to the chaosbench package.
This file re-exports everything so existing imports continue to work.
"""

from chaosbench.models.prompt import ModelConfig, ModelClient, DummyEchoModel, make_model_client
from chaosbench.eval.runner import (
    load_jsonl,
    load_batches,
    evaluate_single_item_robust,
    evaluate_items_with_parallelism,
    evaluate_items,
    get_provider_policy,
    retry_with_backoff,
    save_run_metadata,
    save_per_item_results,
    save_summary_json,
    save_csvs,
    save_figures,
    PROVIDER_POLICIES,
)
from chaosbench.eval.metrics import (
    normalize_label,
    EvalResult,
    compute_summary,
    YES_SET,
    NO_SET,
)
from chaosbench.logic.axioms import (
    get_fol_rules,
    check_fol_violations,
    load_system_ontology,
)
from chaosbench.logic.extract import extract_predicate_from_question


def parse_args():
    """Legacy CLI argument parser."""
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Evaluate ChaosBench-Logic on a given model.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--mode", type=str, default="zeroshot", choices=["zeroshot", "cot", "tool"])
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument(
        "--batches",
        type=str,
        nargs="+",
        default=[
            "batch1_atomic_implication.jsonl",
            "batch2_multiHop_crossSystem.jsonl",
            "batch3_pde_chem_bio.jsonl",
            "batch4_maps_advanced.jsonl",
            "batch5_counterfactual_high_difficulty.jsonl",
            "batch6_deep_bias_probes.jsonl",
            "batch7_multiturn_advanced.jsonl",
            "batch8_indicator_diagnostics.jsonl",
            "batch9_regime_transitions.jsonl",
            "batch10_adversarial.jsonl",
            "batch11_consistency_paraphrase.jsonl",
            "batch12_fol_inference.jsonl",
            "batch13_extended_systems.jsonl",
            "batch14_cross_indicator.jsonl",
        ],
    )
    ap.add_argument("--out_dir", type=str, default="results")
    return ap.parse_args()


def main():
    """Legacy CLI entry point."""
    import os
    import json

    args = parse_args()
    config = ModelConfig(name=args.model, mode=args.mode)
    client = make_model_client(config)

    batch_paths = [os.path.join(args.data_dir, b) for b in args.batches]
    items = load_batches(batch_paths)
    print(f"Loaded {len(items)} items from {len(batch_paths)} batches")

    numeric_fact_map = {}
    results = evaluate_items(items, client, numeric_fact_map=numeric_fact_map)
    summary = compute_summary(results)

    run_out_dir = os.path.join(args.out_dir, f"{args.model}_{args.mode}")
    os.makedirs(run_out_dir, exist_ok=True)

    save_per_item_results(results, run_out_dir)
    save_summary_json(summary, run_out_dir)
    save_csvs(summary, run_out_dir, args.model, args.mode)
    save_figures(summary, run_out_dir)

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
