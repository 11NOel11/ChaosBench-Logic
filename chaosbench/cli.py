"""Unified CLI entry point for ChaosBench-Logic.

Usage:
    chaosbench build --config configs/generation/v2_2_scale.yaml
    chaosbench validate --data data/ --strict
    chaosbench analyze
    chaosbench freeze [--output-dir artifacts/freeze]
    chaosbench eval --provider mock --dataset canonical --max-items 50
    chaosbench eval --provider ollama --model qwen2.5:7b --dataset canonical
    chaosbench eval --provider ollama --model qwen2.5:7b --subset data/subsets/api_balanced_1k.jsonl
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_build(args):
    """Build dataset."""
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    from scripts.build_v2_dataset import load_generation_config, _is_v22_config, build_v22, build_v21

    cfg = load_generation_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("indicators", {})
    cfg["indicators"].setdefault("seed", cfg["seed"])

    data_dir = args.output_dir or "data"
    os.makedirs(data_dir, exist_ok=True)

    if _is_v22_config(cfg):
        build_v22(cfg, data_dir, "systems", verbose=args.verbose)
    else:
        build_v21(cfg, data_dir, "systems")


def cmd_validate(args):
    """Run validation and quality gates."""
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    from scripts.quality_gates import load_all_items
    from chaosbench.quality.gates import run_all_gates

    items = load_all_items(args.data)
    print(f"Loaded {len(items)} items from {args.data}")

    config = {}
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            full_cfg = yaml.safe_load(f)
        config = full_cfg.get("quality_gates", {})

    results = run_all_gates(items, config)
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.gate_name}: {result.details}")
        if not result.passed:
            all_passed = False

    if args.strict and not all_passed:
        sys.exit(1)


def cmd_analyze(args):
    """Run dataset analysis."""
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    os.system("python scripts/analyze_dataset_axes.py")


def cmd_freeze(args):
    """Freeze the canonical v2 dataset and produce a citable artifact."""
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    from scripts.freeze_v2_dataset import freeze
    freeze(output_dir=args.output_dir, selector=args.selector)


def cmd_eval(args):
    """Run evaluation with a provider."""
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)

    from chaosbench.eval.run import EvalRunner, RunConfig

    # Build provider
    provider_name = args.provider.lower()
    if provider_name == "mock":
        from chaosbench.eval.providers import MockProvider
        provider = MockProvider(default="TRUE")
    elif provider_name == "ollama":
        if not args.model:
            print("ERROR: --model is required for --provider ollama", file=sys.stderr)
            sys.exit(1)
        from chaosbench.eval.providers import OllamaProvider
        provider = OllamaProvider(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        print(f"ERROR: unknown provider '{args.provider}'. Supported: mock, ollama", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or "runs"
    cfg = RunConfig(
        provider=provider,
        output_dir=output_dir,
        max_items=args.max_items,
        seed=args.seed,
        workers=args.workers,
        retries=args.retries,
        strict_parsing=not args.lenient,
    )

    runner = EvalRunner(cfg)

    # Determine dataset
    if args.subset:
        dataset = args.subset
    elif args.dataset == "canonical":
        dataset = "canonical"
    else:
        dataset = args.dataset

    result = runner.run(dataset=dataset)
    print(f"\nEvaluation complete.")
    print(f"  Run ID       : {result['run_id']}")
    print(f"  Output dir   : {result['output_dir']}")
    m = result["metrics"]
    print(f"  Total        : {m.get('total', 0)}")
    print(f"  Coverage     : {m.get('coverage', 0):.4f}")
    print(f"  Accuracy     : {m.get('accuracy_valid', 0):.4f}")
    print(f"  Eff. accuracy: {m.get('effective_accuracy', 0):.4f}")
    print(f"  Predictions  : {result['predictions_path']}")
    print(f"  Manifest     : {result['manifest_path']}")


def main():
    parser = argparse.ArgumentParser(
        prog="chaosbench",
        description="ChaosBench-Logic: Benchmark toolkit for LLM reasoning about dynamical systems",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build dataset")
    build_parser.add_argument("--config", type=str, help="Generation config YAML")
    build_parser.add_argument("--seed", type=int, help="Override seed")
    build_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    build_parser.add_argument("--output-dir", type=str, help="Output directory")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--data", required=True, help="Data directory")
    validate_parser.add_argument("--strict", action="store_true", help="Fail on gate violations")
    validate_parser.add_argument("--config", type=str, help="Quality gate config")

    # Analyze command
    subparsers.add_parser("analyze", help="Analyze dataset")

    # Freeze command
    freeze_parser = subparsers.add_parser("freeze", help="Freeze canonical v2 dataset")
    freeze_parser.add_argument(
        "--output-dir", default="artifacts/freeze", help="Freeze artifact output dir"
    )
    freeze_parser.add_argument(
        "--selector",
        default="data/canonical_v2_files.json",
        help="Canonical selector JSON path",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--provider",
        required=True,
        choices=["mock", "ollama"],
        help="Provider: mock (no network) or ollama (local)",
    )
    eval_parser.add_argument("--model", type=str, help="Model name (required for ollama)")
    eval_parser.add_argument(
        "--dataset",
        default="canonical",
        help="Dataset: 'canonical' (all v2 files) or path to a JSONL file",
    )
    eval_parser.add_argument("--subset", type=str, help="Path to a subset JSONL file")
    eval_parser.add_argument("--max-items", type=int, help="Max items to evaluate")
    eval_parser.add_argument("--output-dir", type=str, help="Output directory (default: runs/)")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    eval_parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    eval_parser.add_argument("--retries", type=int, default=1, help="Retry count on INVALID (0 or 1)")
    eval_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    eval_parser.add_argument("--max-tokens", type=int, default=16, dest="max_tokens", help="Max tokens")
    eval_parser.add_argument(
        "--lenient", action="store_true", help="Use lenient parsing (pattern-first)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "build": cmd_build,
        "validate": cmd_validate,
        "analyze": cmd_analyze,
        "freeze": cmd_freeze,
        "eval": cmd_eval,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
