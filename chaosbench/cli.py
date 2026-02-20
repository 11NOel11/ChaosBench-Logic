"""Unified CLI entry point for ChaosBench-Logic.

Usage:
    chaosbench build --config configs/generation/v2_2_scale.yaml
    chaosbench validate --data data/ --strict
    chaosbench analyze
    chaosbench freeze [--output-dir artifacts/freeze]
    chaosbench eval --provider mock --dataset canonical --max-items 50
    chaosbench eval --provider ollama --model qwen2.5:7b --dataset canonical
    chaosbench eval --provider ollama --model qwen2.5:7b --subset data/subsets/api_balanced_1k.jsonl
    chaosbench eval --provider ollama --model qwen2.5:7b --resume runs/20260220T104105Z_ollama_llama3.1:8b
    chaosbench publish-run --run runs/20260220T104105Z_ollama_llama3.1:8b
    chaosbench analyze-runs --runs-dir runs --out-dir artifacts/runs_audit
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
    # Auto-default workers by model size if not explicitly set
    workers = args.workers
    if workers == 1 and provider_name == "ollama" and args.model:
        model_lower = args.model.lower()
        if any(x in model_lower for x in ["14b", "30b", "34b", "70b", "72b"]):
            workers = 2
        elif any(x in model_lower for x in ["7b", "8b", "13b"]):
            workers = 4

    cfg = RunConfig(
        provider=provider,
        output_dir=output_dir,
        max_items=args.max_items,
        seed=args.seed,
        workers=workers,
        retries=args.retries,
        strict_parsing=not args.lenient,
        resume_run_id=getattr(args, "resume", None),
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


def cmd_publish_run(args):
    """Publish a run's lightweight artifacts to published_results/."""
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    from chaosbench.eval.publish import publish_run, update_published_readme
    from pathlib import Path

    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else None

    published = publish_run(
        run_dir=run_dir,
        out_dir=out_dir,
        compress_predictions=args.compress_predictions,
        force=args.force,
    )
    print(f"Published to: {published}")

    # Regenerate the runs index README
    update_published_readme(published.parent)
    print(f"Updated README: {published.parent / 'README.md'}")


def cmd_analyze_runs(args):
    """Run the audit analysis over runs/ and published_results/runs/."""
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    import subprocess
    cmd = [
        sys.executable,
        "scripts/analyze_runs.py",
        "--runs_dir", args.runs_dir,
        "--out_dir", args.out_dir,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)


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
    eval_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Resume an interrupted run by its run_id (must match an existing runs/<run_id>/ dir)",
    )

    # Publish-run command
    pub_parser = subparsers.add_parser("publish-run", help="Publish run artifacts to published_results/")
    pub_parser.add_argument(
        "--run",
        required=True,
        help="Path to the run directory (e.g. runs/20260220T104105Z_ollama_llama3.1:8b)",
    )
    pub_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Destination directory (default: published_results/runs/<run_id>)",
    )
    pub_parser.add_argument(
        "--compress-predictions",
        action="store_true",
        help="Gzip-compress predictions.jsonl for subset runs (max_items <= 5000)",
    )
    pub_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination if it already exists",
    )

    # Analyze-runs command
    ar_parser = subparsers.add_parser("analyze-runs", help="Run audit analysis over runs/")
    ar_parser.add_argument("--runs-dir", default="runs", help="Runs directory")
    ar_parser.add_argument("--out-dir", default="artifacts/runs_audit", help="Audit output directory")

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
        "publish-run": cmd_publish_run,
        "analyze-runs": cmd_analyze_runs,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
