"""Production evaluation runner with provider interface, 3-way outcomes, and retry.

Usage (programmatic):
    from chaosbench.eval.run import EvalRunner, RunConfig
    from chaosbench.eval.providers import MockProvider

    runner = EvalRunner(RunConfig(provider=MockProvider(), output_dir="runs/my_run"))
    results = runner.run(items)

Usage (CLI):
    python -m chaosbench eval --provider ollama --model qwen2.5:7b --dataset canonical
    python -m chaosbench eval --provider ollama --model qwen2.5:7b --dataset canonical --resume runs/my_run_id
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chaosbench.data.hashing import dataset_global_sha256 as _canonical_sha256
from chaosbench.eval.parsing import ParseOutcome, ParsedLabel, parse_label
from chaosbench.eval.prompts import (
    build_prompt,
    build_reprompt,
    get_prompt_hash,
    get_prompt_version,
)
from chaosbench.eval.providers.base import Provider
from chaosbench.eval.providers.types import ProviderResponse

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Flush a checkpoint every this many completed items
_CHECKPOINT_INTERVAL = 200


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration for a single evaluation run."""

    provider: Provider
    output_dir: str = "runs"
    max_items: Optional[int] = None
    seed: int = 42
    workers: int = 1
    retries: int = 1  # 0 or 1; 1 means one reprompt on INVALID
    strict_parsing: bool = True
    run_id: Optional[str] = None  # auto-generated if None
    resume_run_id: Optional[str] = None  # if set, resume an interrupted run


@dataclass
class PredictionRecord:
    """Per-item prediction record written to predictions.jsonl."""

    id: str
    question: str
    ground_truth: str
    pred_text: str
    parsed_label: Optional[str]
    outcome: str  # VALID_TRUE / VALID_FALSE / INVALID
    correct: Optional[bool]  # None if INVALID
    latency_s: float
    task_family: Optional[str] = None
    split: Optional[str] = None
    retry_pred_text: Optional[str] = None
    retry_outcome: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _make_run_id(provider_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{provider_name.replace('/', '_')}"


def _load_canonical_items(selector_path: str = "data/canonical_v2_files.json") -> List[Dict]:
    root = PROJECT_ROOT
    sel = json.loads((root / selector_path).read_text())
    items: List[Dict] = []
    for rel_path in sel["files"]:
        fpath = root / rel_path
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def _load_subset_items(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _normalize_ground_truth(value: str) -> str:
    v = value.strip().upper()
    if v in {"YES", "Y", "TRUE", "T"}:
        return "TRUE"
    if v in {"NO", "N", "FALSE", "F"}:
        return "FALSE"
    return v


def _dataset_global_sha256(selector_path: str = "data/canonical_v2_files.json") -> str:
    """Compute global SHA256 over canonical files.

    Delegates to chaosbench.data.hashing.dataset_global_sha256 so the formula
    is identical to freeze_v2_dataset.py (includes :count per file).
    """
    return _canonical_sha256(
        selector_path=PROJECT_ROOT / selector_path,
        project_root=PROJECT_ROOT,
    )


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def _evaluate_item(
    item: Dict,
    provider: Provider,
    retries: int,
    strict: bool,
) -> PredictionRecord:
    item_id = item.get("id", item.get("item_id", "unknown"))
    question = item.get("question", item.get("prompt", ""))
    raw_gt = (
        item.get("ground_truth")
        or item.get("answer")
        or item.get("gold")
        or item.get("label")
        or ""
    )
    ground_truth = _normalize_ground_truth(str(raw_gt))
    task_family = item.get("task_family") or item.get("family") or item.get("type") or None
    split = item.get("split", None)

    prompt = build_prompt(question)
    resp: ProviderResponse = provider.generate(prompt)
    parsed: ParsedLabel = parse_label(resp.text, strict=strict)

    retry_pred_text = None
    retry_outcome = None

    if parsed.outcome == ParseOutcome.INVALID and retries >= 1:
        reprompt = build_reprompt(prompt, resp.text)
        retry_resp: ProviderResponse = provider.generate(reprompt)
        retry_parsed: ParsedLabel = parse_label(retry_resp.text, strict=strict)
        retry_pred_text = retry_resp.text
        retry_outcome = retry_parsed.outcome.value
        if retry_parsed.outcome != ParseOutcome.INVALID:
            parsed = retry_parsed
            resp = retry_resp

    correct: Optional[bool] = None
    if parsed.outcome != ParseOutcome.INVALID:
        correct = parsed.label == ground_truth

    return PredictionRecord(
        id=item_id,
        question=question,
        ground_truth=ground_truth,
        pred_text=resp.text,
        parsed_label=parsed.label,
        outcome=parsed.outcome.value,
        correct=correct,
        latency_s=resp.latency_s,
        task_family=task_family,
        split=split,
        retry_pred_text=retry_pred_text,
        retry_outcome=retry_outcome,
        meta={"parse_reason": parsed.reason, "parse_confidence": parsed.confidence},
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(records: List[PredictionRecord]) -> Dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"total": 0}

    valid = [r for r in records if r.outcome != ParseOutcome.INVALID.value]
    invalid = [r for r in records if r.outcome == ParseOutcome.INVALID.value]
    correct = [r for r in valid if r.correct]

    coverage = len(valid) / total
    invalid_rate = len(invalid) / total
    accuracy_valid = len(correct) / len(valid) if valid else 0.0
    effective_accuracy = coverage * accuracy_valid

    # Balanced accuracy and MCC
    tp = sum(1 for r in valid if r.parsed_label == "TRUE" and r.ground_truth == "TRUE")
    tn = sum(1 for r in valid if r.parsed_label == "FALSE" and r.ground_truth == "FALSE")
    fp = sum(1 for r in valid if r.parsed_label == "TRUE" and r.ground_truth == "FALSE")
    fn = sum(1 for r in valid if r.parsed_label == "FALSE" and r.ground_truth == "TRUE")

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2

    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    # Per-family metrics
    by_family: Dict[str, List[PredictionRecord]] = defaultdict(list)
    for r in records:
        key = r.task_family or "unknown"
        by_family[key].append(r)

    per_family: Dict[str, Dict] = {}
    for fam, fam_records in by_family.items():
        fam_valid = [r for r in fam_records if r.outcome != ParseOutcome.INVALID.value]
        fam_correct = [r for r in fam_valid if r.correct]
        per_family[fam] = {
            "total": len(fam_records),
            "valid": len(fam_valid),
            "correct": len(fam_correct),
            "coverage": len(fam_valid) / len(fam_records) if fam_records else 0.0,
            "accuracy_valid": len(fam_correct) / len(fam_valid) if fam_valid else 0.0,
        }

    # Per-split metrics
    by_split: Dict[str, List[PredictionRecord]] = defaultdict(list)
    for r in records:
        by_split[r.split or "unknown"].append(r)

    per_split: Dict[str, Dict] = {}
    for sp, sp_records in by_split.items():
        sp_valid = [r for r in sp_records if r.outcome != ParseOutcome.INVALID.value]
        sp_correct = [r for r in sp_valid if r.correct]
        per_split[sp] = {
            "total": len(sp_records),
            "valid": len(sp_valid),
            "accuracy_valid": len(sp_correct) / len(sp_valid) if sp_valid else 0.0,
        }

    return {
        "total": total,
        "valid": len(valid),
        "invalid": len(invalid),
        "correct": len(correct),
        "coverage": round(coverage, 4),
        "invalid_rate": round(invalid_rate, 4),
        "accuracy_valid": round(accuracy_valid, 4),
        "effective_accuracy": round(effective_accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "mcc": round(mcc, 4),
        "per_family": per_family,
        "per_split": per_split,
    }


# ---------------------------------------------------------------------------
# Runner class
# ---------------------------------------------------------------------------


class EvalRunner:
    """Evaluation runner using a Provider instance.

    Features:
    - ETA progress bar (items/sec + rolling estimate)
    - Resume interrupted runs (--resume) via checkpoint file
    - Safe parallel execution with periodic checkpoint flushing
    - Consistent SHA256 with freeze manifest (via chaosbench.data.hashing)

    Args:
        config: RunConfig specifying provider, output dir, etc.
        canonical_selector: Path to canonical_v2_files.json.
    """

    def __init__(
        self,
        config: RunConfig,
        canonical_selector: str = "data/canonical_v2_files.json",
    ):
        self.config = config
        self.canonical_selector = canonical_selector

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_path(out_dir: Path) -> Path:
        return out_dir / ".eval_checkpoint.jsonl"

    @staticmethod
    def _load_checkpoint(out_dir: Path) -> Dict[str, PredictionRecord]:
        """Load already-completed predictions from checkpoint file."""
        cp_path = EvalRunner._checkpoint_path(out_dir)
        done: Dict[str, PredictionRecord] = {}
        if not cp_path.exists():
            return done
        with open(cp_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        rec = PredictionRecord(**{k: d[k] for k in PredictionRecord.__dataclass_fields__})
                        done[rec.id] = rec
                    except Exception:
                        pass
        return done

    @staticmethod
    def _append_checkpoint(out_dir: Path, rec: PredictionRecord) -> None:
        cp_path = EvalRunner._checkpoint_path(out_dir)
        with open(cp_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(self, items: Optional[List[Dict]] = None, dataset: str = "canonical") -> Dict[str, Any]:
        """Run evaluation, with optional resume.

        Args:
            items: Pre-loaded items list (skips loading if provided).
            dataset: "canonical" (load from canonical selector) or a file path.

        Returns:
            dict with keys: run_id, output_dir, metrics, predictions_path, manifest_path.
        """
        cfg = self.config

        # --- Resolve run_id and output directory -------------------------
        if cfg.resume_run_id:
            # Resuming an existing run
            run_id = cfg.resume_run_id
            out_dir = Path(cfg.output_dir) / run_id
            if not out_dir.exists():
                raise FileNotFoundError(
                    f"Cannot resume: run directory not found: {out_dir}"
                )
        else:
            run_id = cfg.run_id or _make_run_id(cfg.provider.name)
            out_dir = Path(cfg.output_dir) / run_id
            out_dir.mkdir(parents=True, exist_ok=True)

        # --- Load items --------------------------------------------------
        if items is None:
            if dataset == "canonical":
                items = _load_canonical_items(self.canonical_selector)
            else:
                items = _load_subset_items(dataset)

        # Optional sampling (applied before resume filtering)
        if cfg.max_items and cfg.max_items < len(items):
            rng = random.Random(cfg.seed)
            items = rng.sample(items, cfg.max_items)

        total_planned = len(items)

        # --- Resume: skip already-completed items ------------------------
        already_done: Dict[str, PredictionRecord] = {}
        if cfg.resume_run_id:
            already_done = self._load_checkpoint(out_dir)
            if already_done:
                print(
                    f"[resume] Found {len(already_done)} completed items in checkpoint; "
                    f"skipping them.",
                    file=sys.stderr,
                )
        done_ids = set(already_done.keys())
        remaining = [it for it in items if it.get("id", it.get("item_id", "")) not in done_ids]

        # --- Run evaluation (sequential or parallel) --------------------
        new_records: List[PredictionRecord] = []
        _lock = threading.Lock()

        try:
            from tqdm import tqdm as _tqdm_cls
        except ImportError:
            _tqdm_cls = None

        n_already = len(already_done)
        n_todo = len(remaining)

        _bar_valid = [n_already]
        _bar_invalid = [0]

        def _make_bar(total: int, initial: int = 0):
            if _tqdm_cls is None:
                return None
            bar = _tqdm_cls(
                total=total,
                initial=initial,
                desc=f"eval [{cfg.provider.name}]",
                unit="q",
                dynamic_ncols=True,
            )
            bar.set_postfix(valid=_bar_valid[0], invalid=_bar_invalid[0])
            return bar

        flush_counter = [0]  # mutable counter for checkpoint flushing

        def _on_record(rec: PredictionRecord, bar) -> None:
            """Callback after each item completes; updates bar and checkpoint."""
            with _lock:
                new_records.append(rec)
                # Append to checkpoint for resume capability
                self._append_checkpoint(out_dir, rec)
                flush_counter[0] += 1
                if bar is not None:
                    all_so_far = list(already_done.values()) + new_records
                    _bar_valid[0] = sum(1 for r in all_so_far if r.outcome != "INVALID")
                    _bar_invalid[0] = sum(1 for r in all_so_far if r.outcome == "INVALID")
                    bar.set_postfix(valid=_bar_valid[0], invalid=_bar_invalid[0])
                    bar.update(1)

        bar = _make_bar(total_planned, initial=n_already)

        if cfg.workers <= 1:
            for item in remaining:
                rec = _evaluate_item(item, cfg.provider, cfg.retries, cfg.strict_parsing)
                _on_record(rec, bar)
        else:
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                futures = {
                    pool.submit(
                        _evaluate_item, item, cfg.provider, cfg.retries, cfg.strict_parsing
                    ): item
                    for item in remaining
                }
                for future in as_completed(futures):
                    rec = future.result()
                    _on_record(rec, bar)

        if bar is not None:
            bar.close()

        # Combine resumed + new records (preserve original item order)
        id_to_record: Dict[str, PredictionRecord] = {**already_done}
        for rec in new_records:
            id_to_record[rec.id] = rec
        records = [id_to_record.get(
            it.get("id", it.get("item_id", "")), None
        ) for it in items]
        records = [r for r in records if r is not None]

        # --- Compute metrics ---------------------------------------------
        metrics = compute_metrics(records)

        # --- Write predictions.jsonl (final, complete) -------------------
        preds_path = out_dir / "predictions.jsonl"
        with open(preds_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(asdict(rec)) + "\n")

        # Remove checkpoint once predictions are safely written
        cp = self._checkpoint_path(out_dir)
        if cp.exists():
            cp.unlink(missing_ok=True)

        # --- Write metrics.json ------------------------------------------
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        # --- Write summary.md --------------------------------------------
        summary_path = out_dir / "summary.md"
        _write_summary_md(summary_path, run_id, metrics, cfg)

        # --- Write run_manifest.json -------------------------------------
        try:
            global_sha = _dataset_global_sha256(self.canonical_selector)
        except Exception:
            global_sha = "unavailable"

        manifest = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "provider": cfg.provider.name,
            "prompt_version": get_prompt_version(),
            "prompt_hash": get_prompt_hash(),
            "dataset_global_sha256": global_sha,
            "canonical_selector": self.canonical_selector,
            "total_items_evaluated": total_planned,
            "max_items": cfg.max_items,
            "seed": cfg.seed,
            "retries": cfg.retries,
            "strict_parsing": cfg.strict_parsing,
            "workers": cfg.workers,
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "metrics_summary": {
                k: metrics[k]
                for k in ["coverage", "accuracy_valid", "effective_accuracy", "balanced_accuracy", "mcc"]
                if k in metrics
            },
            "resumed_from": cfg.resume_run_id,
        }
        manifest_path = out_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return {
            "run_id": run_id,
            "output_dir": str(out_dir),
            "metrics": metrics,
            "predictions_path": str(preds_path),
            "manifest_path": str(manifest_path),
        }


def _write_summary_md(path: Path, run_id: str, metrics: Dict, cfg: RunConfig) -> None:
    m = metrics
    lines = [
        f"# Evaluation Summary: {run_id}",
        "",
        f"**Provider:** {cfg.provider.name}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total items | {m.get('total', 0)} |",
        f"| Valid responses | {m.get('valid', 0)} |",
        f"| Invalid responses | {m.get('invalid', 0)} |",
        f"| Coverage | {m.get('coverage', 0):.4f} |",
        f"| Accuracy (valid) | {m.get('accuracy_valid', 0):.4f} |",
        f"| Effective accuracy | {m.get('effective_accuracy', 0):.4f} |",
        f"| Balanced accuracy | {m.get('balanced_accuracy', 0):.4f} |",
        f"| MCC | {m.get('mcc', 0):.4f} |",
        "",
        "## Per-Family Accuracy",
        "",
        "| Family | Total | Coverage | Accuracy |",
        "|--------|-------|----------|----------|",
    ]
    for fam, fstats in sorted(m.get("per_family", {}).items()):
        lines.append(
            f"| {fam} | {fstats['total']} | {fstats['coverage']:.3f} | {fstats['accuracy_valid']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")
