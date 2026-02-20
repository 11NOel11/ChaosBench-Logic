"""chaosbench/eval/publish.py — Publish a run to the tracked published_results/ tree.

Copies only the small, lightweight artifacts from a run directory into
published_results/runs/<run_id>/ so they can be committed to the repository
without bloating it with large predictions files.

Published artifacts (always):
    run_manifest.json   — run metadata and dataset SHA
    metrics.json        — aggregate metrics (no raw text)
    summary.md          — markdown summary table

Optionally published (subset runs only, --compress-predictions):
    predictions_subset.jsonl.gz  — gzip-compressed predictions

NOT published:
    predictions.jsonl   — large raw file (gitignored, stays in runs/)
    .eval_checkpoint.jsonl  — internal resume file

CLI usage:
    chaosbench publish-run --run runs/20260220T104105Z_ollama_llama3.1:8b
    chaosbench publish-run --run runs/20260220T104105Z_ollama_llama3.1:8b \\
        --out published_results/runs/20260220T104105Z_ollama_llama3.1:8b \\
        --compress-predictions
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Files always copied
_CORE_ARTIFACTS = [
    "run_manifest.json",
    "metrics.json",
    "summary.md",
]
# Optional small files copied if present
_OPTIONAL_ARTIFACTS = [
    "summary.json",  # some runs may emit this
]
# Never copied
_EXCLUDED = {
    "predictions.jsonl",
    ".eval_checkpoint.jsonl",
}

# Predictions are compressed only if the run is a subset (max_items set)
_SUBSET_MAX_ITEMS_THRESHOLD = 5_000


def publish_run(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    compress_predictions: bool = False,
    force: bool = False,
) -> Path:
    """Publish a run directory to the tracked output location.

    Args:
        run_dir: Source run directory (e.g. runs/20260220T104105Z_...).
        out_dir: Destination directory. Defaults to
                 published_results/runs/<run_dir.name>.
        compress_predictions: If True, also gzip-compress predictions.jsonl
                              for subset runs (max_items <= threshold).
        force: Overwrite destination if it already exists.

    Returns:
        Path to the published directory.
    """
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run_manifest.json not found in {run_dir}")

    manifest = json.loads(manifest_path.read_text())
    run_id = manifest.get("run_id", run_dir.name)

    if out_dir is None:
        out_dir = PROJECT_ROOT / "published_results" / "runs" / run_id
    out_dir = out_dir.resolve()

    if out_dir.exists() and not force:
        raise FileExistsError(
            f"Destination already exists: {out_dir}. Use --force to overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = []

    # Copy core artifacts
    for fname in _CORE_ARTIFACTS:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            copied.append(fname)
        else:
            skipped.append(f"{fname} (missing)")

    # Copy optional artifacts
    for fname in _OPTIONAL_ARTIFACTS:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            copied.append(fname)

    # Conditionally compress predictions
    pred_src = run_dir / "predictions.jsonl"
    max_items = manifest.get("max_items")
    is_subset = max_items is not None and max_items <= _SUBSET_MAX_ITEMS_THRESHOLD
    if compress_predictions and is_subset and pred_src.exists():
        dest_gz = out_dir / "predictions_subset.jsonl.gz"
        with open(pred_src, "rb") as f_in, gzip.open(dest_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        copied.append("predictions_subset.jsonl.gz")
    elif compress_predictions and not is_subset:
        skipped.append("predictions.jsonl.gz (skipped: not a subset run)")

    # Write a publish receipt
    receipt = {
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "run_id": run_id,
        "artifacts_copied": copied,
        "artifacts_skipped": skipped,
        "compress_predictions": compress_predictions,
    }
    (out_dir / "publish_receipt.json").write_text(json.dumps(receipt, indent=2))

    return out_dir


def update_published_readme(
    published_root: Optional[Path] = None,
) -> None:
    """Regenerate the published_results/runs/README.md index."""
    if published_root is None:
        published_root = PROJECT_ROOT / "published_results" / "runs"
    published_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_dir in sorted(published_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_p = run_dir / "run_manifest.json"
        metrics_p = run_dir / "metrics.json"
        if not manifest_p.exists():
            continue
        m = json.loads(manifest_p.read_text())
        metrics = json.loads(metrics_p.read_text()) if metrics_p.exists() else {}
        ms = m.get("metrics_summary", {})
        runs.append({
            "run_id": m.get("run_id", run_dir.name),
            "provider": m.get("provider", "—"),
            "n": m.get("total_items_evaluated", "—"),
            "acc_valid": ms.get("accuracy_valid", metrics.get("accuracy_valid", "—")),
            "bal_acc": ms.get("balanced_accuracy", metrics.get("balanced_accuracy", "—")),
            "mcc": ms.get("mcc", metrics.get("mcc", "—")),
            "created": m.get("created_utc", "—")[:10],
        })

    lines = [
        "# Published Runs — ChaosBench-Logic v2",
        "",
        "Auto-generated index. Edit via `chaosbench publish-run`.",
        "",
        "| Run ID | Provider | N | Acc_valid | Bal_acc | MCC | Date |",
        "|--------|----------|---|-----------|---------|-----|------|",
    ]
    for r in runs:
        acc = f"{r['acc_valid']:.4f}" if isinstance(r["acc_valid"], float) else str(r["acc_valid"])
        bal = f"{r['bal_acc']:.4f}" if isinstance(r["bal_acc"], float) else str(r["bal_acc"])
        mcc = f"{r['mcc']:.4f}" if isinstance(r["mcc"], float) else str(r["mcc"])
        lines.append(
            f"| `{r['run_id']}` | {r['provider']} | {r['n']:,} "
            f"| {acc} | {bal} | {mcc} | {r['created']} |"
            if isinstance(r["n"], int) else
            f"| `{r['run_id']}` | {r['provider']} | — | {acc} | {bal} | {mcc} | {r['created']} |"
        )

    lines += [
        "",
        "## Artifact Contents",
        "",
        "Each run directory contains:",
        "- `run_manifest.json` — run metadata, dataset SHA256, config",
        "- `metrics.json` — aggregate metrics (coverage, accuracy, MCC, per-family)",
        "- `summary.md` — markdown summary table",
        "- `publish_receipt.json` — publish provenance",
        "- `predictions_subset.jsonl.gz` — compressed predictions (subset runs only)",
        "",
        "See `docs/RUNS_POLICY.md` for the full policy.",
    ]
    (published_root / "README.md").write_text("\n".join(lines) + "\n")
