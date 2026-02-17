#!/usr/bin/env python3
"""Generate and submit SLURM batch jobs for ChaosBench-Logic evaluation.

This script generates SLURM job scripts from the template and optionally
submits them to the cluster scheduler. It supports batch submission across
multiple models and evaluation modes.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_MODELS = [
    "gpt4",
    "claude3",
    "gemini",
    "llama3",
    "mixtral",
    "openhermes",
]

DEFAULT_MODES = ["zeroshot", "cot"]


def get_template_path():
    """Get the path to the SLURM template script.

    Returns:
        Path to slurm_template.sh in the scripts directory.
    """
    script_dir = Path(__file__).parent.resolve()
    return script_dir / "slurm_template.sh"


def generate_job_script(
    model, mode, output_dir, seed=42, workers=4, shard_index=0, num_shards=1
):
    """Generate a SLURM job script from the template.

    Args:
        model: Model name (e.g., gpt4, claude3).
        mode: Evaluation mode (zeroshot, cot).
        output_dir: Base output directory for results.
        seed: Random seed for reproducibility.
        workers: Number of parallel workers.

    Returns:
        String containing the filled SLURM job script.
    """
    template_path = get_template_path()

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r") as f:
        template = f.read()

    job_name = f"chaosbench_{model}_{mode}"
    job_output_dir = os.path.join(output_dir, f"{model}_{mode}")

    script = template.format(
        JOB_NAME=job_name,
        OUTPUT_DIR=job_output_dir,
    )

    script = script.replace("MODEL=${MODEL:-gpt4}", f"MODEL=${{MODEL:-{model}}}")
    script = script.replace("MODE=${MODE:-zeroshot}", f"MODE=${{MODE:-{mode}}}")
    script = script.replace("WORKERS=${WORKERS:-4}", f"WORKERS=${{WORKERS:-{workers}}}")
    script = script.replace("SEED=${SEED:-42}", f"SEED=${{SEED:-{seed}}}")
    script = script.replace(
        "OUTPUT_DIR=${OUTPUT_DIR:-results}", f"OUTPUT_DIR=${{OUTPUT_DIR:-{output_dir}}}"
    )
    script = script.replace(
        "SHARD_INDEX=${SHARD_INDEX:-0}", f"SHARD_INDEX=${{SHARD_INDEX:-{shard_index}}}"
    )
    script = script.replace(
        "NUM_SHARDS=${NUM_SHARDS:-1}", f"NUM_SHARDS=${{NUM_SHARDS:-{num_shards}}}"
    )

    return script


def submit_batch(
    models, modes, base_dir, dry_run=False, workers=4, seed=42, num_shards=1
):
    """Generate SLURM job scripts and optionally submit them.

    Args:
        models: List of model names to evaluate.
        modes: List of evaluation modes.
        base_dir: Base directory for output.
        dry_run: If True, only generate scripts without submitting.
        workers: Number of parallel workers per job.
        seed: Random seed for reproducibility.

    Returns:
        List of job IDs (if submitted) or script paths (if dry_run).
    """
    base_dir = Path(base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    script_dir = base_dir / "slurm_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model in models:
        for mode in modes:
            for shard_index in range(num_shards):
                job_name = f"chaosbench_{model}_{mode}"
                if num_shards > 1:
                    job_name = f"{job_name}_shard{shard_index + 1}of{num_shards}"
                script_path = script_dir / f"{job_name}.sh"

                script_content = generate_job_script(
                    model=model,
                    mode=mode,
                    output_dir=str(base_dir),
                    seed=seed,
                    workers=workers,
                    shard_index=shard_index,
                    num_shards=num_shards,
                )

                with open(script_path, "w") as f:
                    f.write(script_content)

                script_path.chmod(0o755)

                if dry_run:
                    print(f"[DRY-RUN] Generated: {script_path}")
                    results.append(str(script_path))
                else:
                    try:
                        result = subprocess.run(
                            ["sbatch", str(script_path)],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        job_id = result.stdout.strip().split()[-1]
                        print(f"[SUBMITTED] {job_name} -> Job ID: {job_id}")
                        results.append(job_id)
                    except subprocess.CalledProcessError as e:
                        print(
                            f"[ERROR] Failed to submit {job_name}: {e.stderr}",
                            file=sys.stderr,
                        )
                        results.append(None)
                    except FileNotFoundError:
                        print(
                            "[ERROR] sbatch command not found. Is SLURM installed?",
                            file=sys.stderr,
                        )
                        return []

    return results


def main():
    """CLI entry point for cluster evaluation script."""
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for ChaosBench-Logic evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run for all default models and modes
  python scripts/run_cluster_eval.py --dry-run --base-dir /tmp/cluster_test/

  # Submit specific models
  python scripts/run_cluster_eval.py --models gpt4 claude3 --modes zeroshot cot --base-dir results/ --submit

  # Generate scripts only (no submission)
  python scripts/run_cluster_eval.py --models llama3 --modes zeroshot --base-dir results/
        """,
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to evaluate (default: {' '.join(DEFAULT_MODELS)})",
    )

    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=DEFAULT_MODES,
        help=f"Evaluation modes (default: {' '.join(DEFAULT_MODES)})",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory for results and generated scripts",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers per job (default: 4)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split each model/mode run into N shards (default: 1)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without submitting to SLURM",
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs to SLURM (default: only generate scripts)",
    )

    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    if not args.dry_run and not args.submit:
        print(
            "[INFO] No --dry-run or --submit flag provided. Defaulting to dry-run mode."
        )
        print("[INFO] Use --submit to actually submit jobs to SLURM.")
        args.dry_run = True

    print("=" * 70)
    print("ChaosBench-Logic: Cluster Evaluation Script")
    print("=" * 70)
    print(f"Models: {', '.join(args.models)}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Base directory: {args.base_dir}")
    print(f"Workers per job: {args.workers}")
    print(f"Seed: {args.seed}")
    print(f"Shards per run: {args.num_shards}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'SUBMIT'}")
    print("=" * 70)

    results = submit_batch(
        models=args.models,
        modes=args.modes,
        base_dir=args.base_dir,
        dry_run=args.dry_run,
        workers=args.workers,
        seed=args.seed,
        num_shards=args.num_shards,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if args.dry_run:
        print(f"Generated {len(results)} job scripts")
        print(f"Scripts saved in: {Path(args.base_dir).resolve() / 'slurm_scripts'}")
    else:
        successful = len([r for r in results if r is not None])
        print(f"Submitted {successful}/{len(results)} jobs successfully")

        if successful > 0:
            print("\nTo monitor jobs:")
            print("  squeue -u $USER")
            print("\nTo cancel all jobs:")
            print("  scancel -u $USER")

    return 0 if all(r is not None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
