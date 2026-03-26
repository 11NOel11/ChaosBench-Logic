#!/usr/bin/env python3
"""Run prompt-variant robustness suite with CARE-v3 repair evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.eval.run import EvalRunner, RunConfig
from chaosbench.eval.providers import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    GroqProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from chaosbench.repair.engine import (
    compute_axiom_violation_rate,
    load_selector_index,
    read_jsonl,
    repair_records,
    write_jsonl,
)
from chaosbench.repair.types import RepairConfig

DEFAULT_MODELS = [
    ("anthropic", "claude-sonnet-4-6"),
    ("openai", "gpt-4o"),
    ("gemini", "gemini-2.5-flash"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("deepseek", "deepseek-chat"),
]


@dataclass
class EvalSpec:
    provider: str
    model: str
    prompt_variant: str

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}::{self.prompt_variant}"


def write_csv(
    path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_cells(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_model_specs(raw: str) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    if not raw.strip():
        return specs
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid model spec '{token}', expected provider:model")
        provider, model = token.split(":", 1)
        specs.append((provider.strip(), model.strip()))
    return specs


def parse_variants(raw: str) -> List[str]:
    out: List[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out


def provider_instance(provider: str, model: str, temperature: float, max_tokens: int):
    p = provider.lower().strip()
    if p == "anthropic":
        return AnthropicProvider(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
    if p == "openai":
        return OpenAIProvider(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
    if p == "gemini":
        effective_max_tokens = max(64, int(max_tokens))
        return GeminiProvider(
            model=model, temperature=temperature, max_tokens=effective_max_tokens
        )
    if p == "groq":
        return GroqProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if p == "openrouter":
        return OpenRouterProvider(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
    if p == "deepseek":
        return DeepSeekProvider(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
    raise ValueError(f"Unsupported provider for prompt suite: {provider}")


def metrics_from_records(
    records: Sequence[Dict[str, Any]], label_key: str
) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    total = len(records)
    valid = 0
    for row in records:
        gt = row.get("ground_truth")
        pred = row.get(label_key)
        if gt not in {"TRUE", "FALSE"}:
            continue
        if pred not in {"TRUE", "FALSE"}:
            continue
        valid += 1
        if gt == "TRUE" and pred == "TRUE":
            tp += 1
        elif gt == "FALSE" and pred == "FALSE":
            tn += 1
        elif gt == "FALSE" and pred == "TRUE":
            fp += 1
        elif gt == "TRUE" and pred == "FALSE":
            fn += 1

    coverage = (valid / total) if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    ba = (tpr + tnr) / 2.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "total": float(total),
        "valid": float(valid),
        "coverage": float(coverage),
        "mcc": float(mcc),
        "balanced_accuracy": float(ba),
        "tpr": float(tpr),
        "tnr": float(tnr),
    }


def sign_test_two_sided(values: Sequence[float]) -> Dict[str, float]:
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    n = positives + negatives
    if n == 0:
        return {"n": 0.0, "positives": 0.0, "negatives": 0.0, "p_value": 1.0}
    tail = min(positives, negatives)
    cumulative = 0.0
    for i in range(tail + 1):
        cumulative += math.comb(n, i) * (0.5**n)
    p_value = min(1.0, 2.0 * cumulative)
    return {
        "n": float(n),
        "positives": float(positives),
        "negatives": float(negatives),
        "p_value": float(p_value),
    }


def load_frozen_config(path: Path) -> RepairConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    return RepairConfig(
        name=str(cfg["name"]),
        gate_families=tuple(cfg["gate_families"]) if cfg.get("gate_families") else None,
        extractor_strategy=str(cfg["extractor_strategy"]),
        polarity_mode=str(cfg["polarity_mode"]),
        leave_invalid_unchanged=bool(cfg["leave_invalid_unchanged"]),
        enable_group_consistency=bool(cfg["enable_group_consistency"]),
        seed=int(cfg.get("seed", 42)),
    )


def safe_model_name(provider: str, model: str) -> str:
    return f"{provider}_{model}".replace("/", "_").replace(":", "-")


def safe_variant_name(variant: str) -> str:
    return variant.replace("/", "_").replace(":", "-")


def deterministic_run_id(
    spec: EvalSpec,
    subset_path: Path,
    seed: int,
    max_items: int,
) -> str:
    token = "|".join(
        [
            spec.provider,
            spec.model,
            spec.prompt_variant,
            str(subset_path.resolve()),
            str(seed),
            str(max_items),
        ]
    )
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]
    return (
        f"pv_{safe_model_name(spec.provider, spec.model)}"
        f"_{safe_variant_name(spec.prompt_variant)}_{digest}"
    )


def normalize_cell_row(row: Dict[str, Any]) -> Dict[str, Any]:
    numeric_float_fields = [
        "mcc_pre",
        "mcc_post",
        "delta_mcc",
        "balanced_accuracy_pre",
        "balanced_accuracy_post",
        "delta_balanced_accuracy",
        "row_flip_rate",
        "axiom_violation_rate_pre",
        "axiom_violation_rate_post",
        "axiom_violation_reduction",
        "axiom_violations_pre",
        "axiom_violations_post",
    ]
    normalized = dict(row)
    normalized["provider"] = str(row.get("provider", ""))
    normalized["model"] = str(row.get("model", ""))
    normalized["prompt_variant"] = str(row.get("prompt_variant", ""))
    normalized["status"] = str(row.get("status", ""))
    normalized["run_id"] = str(row.get("run_id", ""))
    normalized["run_dir"] = str(row.get("run_dir", ""))
    normalized["repaired_dir"] = str(row.get("repaired_dir", ""))
    normalized["prompt_hash"] = str(row.get("prompt_hash", ""))
    normalized["error"] = str(row.get("error", ""))
    for field in numeric_float_fields:
        normalized[field] = as_float(row.get(field), default=0.0)
    return normalized


def write_progress(
    out_dir: Path,
    planned_specs: Sequence[EvalSpec],
    cell_map: Dict[str, Dict[str, Any]],
    started_utc: str,
    suite_start_s: float,
    last_cell: str,
) -> None:
    planned = len(planned_specs)
    completed = len(cell_map)
    ok_count = sum(1 for row in cell_map.values() if row.get("status") == "ok")
    error_count = sum(1 for row in cell_map.values() if row.get("status") != "ok")
    elapsed = max(0.0, time.monotonic() - suite_start_s)
    avg_cell = (elapsed / completed) if completed else 0.0
    remaining = max(0, planned - completed)
    eta = (avg_cell * remaining) if completed else 0.0
    progress = {
        "started_utc": started_utc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "planned_cells": planned,
        "completed_cells": completed,
        "remaining_cells": remaining,
        "ok_cells": ok_count,
        "error_cells": error_count,
        "elapsed_seconds": elapsed,
        "avg_seconds_per_cell": avg_cell,
        "eta_seconds": eta,
        "last_cell": last_cell,
    }
    (out_dir / "prompt_variant_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ordered_rows(
    planned_specs: Sequence[EvalSpec],
    cell_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in planned_specs:
        row = cell_map.get(spec.key)
        if row is not None:
            rows.append(row)
    return rows


def write_cell_csv(
    out_dir: Path,
    planned_specs: Sequence[EvalSpec],
    cell_map: Dict[str, Dict[str, Any]],
) -> None:
    rows = ordered_rows(planned_specs, cell_map)
    write_csv(
        out_dir / "prompt_variant_cells.csv",
        rows,
        fieldnames=[
            "provider",
            "model",
            "prompt_variant",
            "status",
            "run_id",
            "run_dir",
            "repaired_dir",
            "prompt_hash",
            "mcc_pre",
            "mcc_post",
            "delta_mcc",
            "balanced_accuracy_pre",
            "balanced_accuracy_post",
            "delta_balanced_accuracy",
            "row_flip_rate",
            "axiom_violation_rate_pre",
            "axiom_violation_rate_post",
            "axiom_violation_reduction",
            "axiom_violations_pre",
            "axiom_violations_post",
            "error",
        ],
    )


def run_eval_once(
    spec: EvalSpec,
    subset_path: Path,
    runs_dir: Path,
    run_id: str,
    seed: int,
    workers: int,
    retries: int,
    temperature: float,
    max_tokens: int,
    max_usd: float,
    max_items: int,
) -> Tuple[bool, Dict[str, Any]]:
    os.environ["CHAOSBENCH_PROMPT_VARIANT"] = spec.prompt_variant
    provider = provider_instance(spec.provider, spec.model, temperature, max_tokens)

    out_dir = runs_dir / run_id
    preds_path = out_dir / "predictions.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    metrics_path = out_dir / "metrics.json"
    checkpoint_path = out_dir / ".eval_checkpoint.jsonl"
    if preds_path.exists() and manifest_path.exists() and metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
        return True, {
            "run_id": run_id,
            "output_dir": str(out_dir),
            "metrics": metrics,
            "predictions_path": str(preds_path),
            "manifest_path": str(manifest_path),
            "resumed": False,
            "reused_complete": True,
        }

    resume_run_id = run_id if checkpoint_path.exists() else None
    assigned_run_id = None if resume_run_id else run_id

    cfg = RunConfig(
        provider=provider,
        output_dir=str(runs_dir),
        max_items=max_items if max_items > 0 else None,
        seed=seed,
        workers=workers,
        retries=retries,
        strict_parsing=True,
        max_usd=max_usd,
        run_id=assigned_run_id,
        resume_run_id=resume_run_id,
    )

    runner = EvalRunner(cfg)
    try:
        result = runner.run(dataset=str(subset_path))
    except Exception as exc:
        return False, {"error": str(exc)}
    result["resumed"] = bool(resume_run_id)
    result["reused_complete"] = False
    return True, result


def summarize_hypotheses(
    cell_rows: List[Dict[str, Any]],
    positive_cell_threshold: float,
    alpha: float,
    max_spread_threshold: float,
) -> Dict[str, Any]:
    successful = [row for row in cell_rows if row["status"] == "ok"]
    deltas = [float(row["delta_mcc"]) for row in successful]
    sign = sign_test_two_sided(deltas)

    n_cells = len(successful)
    n_pos = sum(1 for value in deltas if value > 0)
    pos_rate = (n_pos / n_cells) if n_cells else 0.0
    h4_pass = pos_rate >= positive_cell_threshold and float(sign["p_value"]) <= alpha

    by_model: Dict[str, List[float]] = {}
    for row in successful:
        key = f"{row['provider']}/{row['model']}"
        by_model.setdefault(key, []).append(float(row["delta_mcc"]))

    model_summary: List[Dict[str, Any]] = []
    all_model_means_positive = True
    all_model_spread_ok = True
    for key in sorted(by_model.keys()):
        values = by_model[key]
        mean_delta = statistics.mean(values)
        spread = max(values) - min(values)
        std_delta = statistics.pstdev(values) if len(values) > 1 else 0.0
        if mean_delta <= 0.0:
            all_model_means_positive = False
        if spread > max_spread_threshold:
            all_model_spread_ok = False
        provider, model = key.split("/", 1)
        model_summary.append(
            {
                "provider": provider,
                "model": model,
                "n_variants": float(len(values)),
                "mean_delta_mcc": float(mean_delta),
                "std_delta_mcc": float(std_delta),
                "min_delta_mcc": float(min(values)),
                "max_delta_mcc": float(max(values)),
                "spread_delta_mcc": float(spread),
            }
        )

    return {
        "h4_cell_positive_rate": {
            "n_cells": float(n_cells),
            "n_positive": float(n_pos),
            "positive_rate": float(pos_rate),
            "sign_test": sign,
            "threshold": float(positive_cell_threshold),
            "alpha": float(alpha),
            "status": "PASS" if h4_pass else "FAIL",
        },
        "h5_model_mean_positive": {
            "status": "PASS" if all_model_means_positive else "FAIL",
            "n_models": float(len(model_summary)),
        },
        "h6_model_spread_bounded": {
            "status": "PASS" if all_model_spread_ok else "FAIL",
            "max_spread_threshold": float(max_spread_threshold),
            "n_models": float(len(model_summary)),
        },
        "model_summary": model_summary,
    }


def build_report(
    out_path: Path,
    cell_rows: List[Dict[str, Any]],
    hypothesis: Dict[str, Any],
) -> None:
    lines: List[str] = [
        "# Prompt Variant Robustness Results",
        "",
        "## Hypotheses",
        "",
        "| Hypothesis | Status | Detail |",
        "|---|---|---|",
    ]

    h4 = hypothesis["h4_cell_positive_rate"]
    h5 = hypothesis["h5_model_mean_positive"]
    h6 = hypothesis["h6_model_spread_bounded"]

    lines.append(
        f"| H4: Cell-level positive rate | {h4['status']} | "
        f"{int(h4['n_positive'])}/{int(h4['n_cells'])} positive, "
        f"rate={h4['positive_rate']:.3f}, p={h4['sign_test']['p_value']:.6f} |"
    )
    lines.append(
        f"| H5: Model mean delta > 0 | {h5['status']} | models={int(h5['n_models'])} |"
    )
    lines.append(
        f"| H6: Variant spread bounded | {h6['status']} | "
        f"max spread threshold={h6['max_spread_threshold']:.3f} |"
    )

    lines.extend(
        [
            "",
            "## Cell results",
            "",
            "| Provider | Model | Prompt variant | MCC pre | MCC post | Delta MCC | Delta BA | Status |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        cell_rows,
        key=lambda item: (item["provider"], item["model"], item["prompt_variant"]),
    ):
        if row["status"] != "ok":
            lines.append(
                f"| {row['provider']} | {row['model']} | {row['prompt_variant']} | - | - | - | - | error |"
            )
            continue
        lines.append(
            f"| {row['provider']} | {row['model']} | {row['prompt_variant']} | "
            f"{row['mcc_pre']:.4f} | {row['mcc_post']:.4f} | {row['delta_mcc']:+.4f} | "
            f"{row['delta_balanced_accuracy']:+.4f} | ok |"
        )

    lines.extend(
        [
            "",
            "## Model summary",
            "",
            "| Provider | Model | Mean delta MCC | Std delta MCC | Min | Max | Spread |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in hypothesis["model_summary"]:
        lines.append(
            f"| {row['provider']} | {row['model']} | {row['mean_delta_mcc']:+.4f} | {row['std_delta_mcc']:.4f} | "
            f"{row['min_delta_mcc']:+.4f} | {row['max_delta_mcc']:+.4f} | {row['spread_delta_mcc']:.4f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prompt variant robustness suite")
    parser.add_argument(
        "--repair-dir",
        default=str(
            PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "repair_v3"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "prompt_variants"
        ),
    )
    parser.add_argument(
        "--subset",
        default=str(PROJECT_ROOT / "data" / "subsets" / "api_balanced_1k.jsonl"),
    )
    parser.add_argument("--selector", default="data/canonical_v2_files.json")
    parser.add_argument("--variants", default="v1,v1_compact,v1_logiccheck")
    parser.add_argument(
        "--models",
        default=",".join(f"{provider}:{model}" for provider, model in DEFAULT_MODELS),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--positive-cell-threshold", type=float, default=0.75)
    parser.add_argument("--max-spread-threshold", type=float, default=0.05)
    parser.add_argument(
        "--post-m5-controller-mode",
        choices=["none", "static", "online"],
        default="none",
        help="Optionally run M5/M6 transfer after prompt-variant suite",
    )
    parser.add_argument(
        "--post-m5-policy-json",
        default=None,
        help="Policy JSON for post-suite M5 transfer (defaults to <repair-dir>/m5_instance/m5_policy.json)",
    )
    parser.add_argument(
        "--post-m5-provider-thresholds-json",
        default=None,
        help="Optional provider threshold map for post-suite M5 transfer",
    )
    parser.add_argument(
        "--post-m5-out-dir",
        default=None,
        help="Output dir for post-suite M5 transfer",
    )
    parser.add_argument("--post-m5-default-split", default="heldout")
    args = parser.parse_args()

    repair_dir = Path(args.repair_dir)
    out_dir = Path(args.out_dir)
    subset_path = Path(args.subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    repair_runs_dir = out_dir / "repair_runs"
    repair_runs_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants)
    model_specs = parse_model_specs(args.models)
    frozen = load_frozen_config(repair_dir / "frozen_config.json")
    id_to_meta = load_selector_index(
        selector_path=Path(args.selector), project_root=PROJECT_ROOT
    )

    planned_specs = [
        EvalSpec(provider=provider, model=model, prompt_variant=variant)
        for provider, model in model_specs
        for variant in variants
    ]

    existing_rows = load_existing_cells(out_dir / "prompt_variant_cells.csv")
    cell_map: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        normalized = normalize_cell_row(row)
        key = (
            f"{normalized['provider']}/{normalized['model']}"
            f"::{normalized['prompt_variant']}"
        )
        cell_map[key] = normalized

    started_utc = datetime.now(timezone.utc).isoformat()
    suite_start_s = time.monotonic()
    for spec in planned_specs:
        expected_run_id = deterministic_run_id(
            spec=spec,
            subset_path=subset_path,
            seed=args.seed,
            max_items=args.max_items,
        )
        existing = cell_map.get(spec.key)
        if (
            existing is not None
            and existing.get("status") == "ok"
            and str(existing.get("run_id", "")) == expected_run_id
        ):
            print(f"[skip] {spec.key} already complete")
            continue

        n_done = len(cell_map)
        n_planned = len(planned_specs)
        print(f"[{n_done + 1}/{n_planned}] running {spec.key}")
        cell_start_s = time.monotonic()
        run_id = expected_run_id
        ok, result = run_eval_once(
            spec=spec,
            subset_path=subset_path,
            runs_dir=runs_dir,
            run_id=run_id,
            seed=args.seed,
            workers=args.workers,
            retries=args.retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_usd=args.max_usd,
            max_items=args.max_items,
        )

        if not ok:
            row = normalize_cell_row(
                {
                    "provider": spec.provider,
                    "model": spec.model,
                    "prompt_variant": spec.prompt_variant,
                    "status": "error",
                    "error": str(result.get("error", "unknown_error")),
                    "run_id": run_id,
                }
            )
            cell_map[spec.key] = row
            write_cell_csv(
                out_dir=out_dir, planned_specs=planned_specs, cell_map=cell_map
            )
            write_progress(
                out_dir=out_dir,
                planned_specs=planned_specs,
                cell_map=cell_map,
                started_utc=started_utc,
                suite_start_s=suite_start_s,
                last_cell=spec.key,
            )
            elapsed = max(0.0, time.monotonic() - cell_start_s)
            print(f"[error] {spec.key} in {elapsed:.1f}s: {row['error']}")
            continue

        run_id = str(result["run_id"])
        run_dir = Path(result["output_dir"])
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        records = read_jsonl(run_dir / "predictions.jsonl")
        repaired = repair_records(records=records, id_to_meta=id_to_meta, config=frozen)

        pre = metrics_from_records(records, "parsed_label")
        post = metrics_from_records(repaired.records, "repaired_label")

        pre_v_count, pre_v_rate = compute_axiom_violation_rate(
            records=records,
            label_key="parsed_label",
            id_to_meta=id_to_meta,
            config=frozen,
            gate_families=None,
        )
        post_v_count, post_v_rate = compute_axiom_violation_rate(
            records=repaired.records,
            label_key="repaired_label",
            id_to_meta=id_to_meta,
            config=frozen,
            gate_families=None,
        )

        out_repair_dir = repair_runs_dir / run_id
        out_repair_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_repair_dir / "repaired_predictions.jsonl", repaired.records)
        (out_repair_dir / "repair_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "provider": spec.provider,
                    "model": spec.model,
                    "prompt_variant": spec.prompt_variant,
                    "constraint_hash": repaired.constraint_hash,
                    "config": frozen.to_dict(),
                    "stats": repaired.stats.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        delta_mcc = post["mcc"] - pre["mcc"]
        delta_ba = post["balanced_accuracy"] - pre["balanced_accuracy"]

        row = normalize_cell_row(
            {
                "provider": spec.provider,
                "model": spec.model,
                "prompt_variant": spec.prompt_variant,
                "status": "ok",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "repaired_dir": str(out_repair_dir),
                "prompt_hash": str(manifest.get("prompt_hash", "")),
                "mcc_pre": pre["mcc"],
                "mcc_post": post["mcc"],
                "delta_mcc": delta_mcc,
                "balanced_accuracy_pre": pre["balanced_accuracy"],
                "balanced_accuracy_post": post["balanced_accuracy"],
                "delta_balanced_accuracy": delta_ba,
                "row_flip_rate": (
                    repaired.stats.row_flips / repaired.stats.valid_records
                    if repaired.stats.valid_records
                    else 0.0
                ),
                "axiom_violation_rate_pre": pre_v_rate,
                "axiom_violation_rate_post": post_v_rate,
                "axiom_violation_reduction": pre_v_rate - post_v_rate,
                "axiom_violations_pre": float(pre_v_count),
                "axiom_violations_post": float(post_v_count),
            }
        )
        cell_map[spec.key] = row
        write_cell_csv(out_dir=out_dir, planned_specs=planned_specs, cell_map=cell_map)
        write_progress(
            out_dir=out_dir,
            planned_specs=planned_specs,
            cell_map=cell_map,
            started_utc=started_utc,
            suite_start_s=suite_start_s,
            last_cell=spec.key,
        )
        elapsed = max(0.0, time.monotonic() - cell_start_s)
        remaining = max(0, len(planned_specs) - len(cell_map))
        avg = (time.monotonic() - suite_start_s) / max(1, len(cell_map))
        eta = avg * remaining
        print(
            f"[done] {spec.key} in {elapsed:.1f}s "
            f"(delta_mcc={row['delta_mcc']:+.4f}, eta={eta / 60.0:.1f}m)"
        )

    cell_rows = ordered_rows(planned_specs, cell_map)

    write_cell_csv(out_dir=out_dir, planned_specs=planned_specs, cell_map=cell_map)

    hypothesis = summarize_hypotheses(
        cell_rows=cell_rows,
        positive_cell_threshold=args.positive_cell_threshold,
        alpha=args.alpha,
        max_spread_threshold=args.max_spread_threshold,
    )

    write_csv(
        out_dir / "prompt_variant_by_model.csv",
        hypothesis["model_summary"],
        fieldnames=[
            "provider",
            "model",
            "n_variants",
            "mean_delta_mcc",
            "std_delta_mcc",
            "min_delta_mcc",
            "max_delta_mcc",
            "spread_delta_mcc",
        ],
    )

    post_m5_result: Dict[str, Any] | None = None
    if args.post_m5_controller_mode != "none":
        policy_path = (
            Path(args.post_m5_policy_json)
            if args.post_m5_policy_json
            else Path(args.repair_dir) / "m5_instance" / "m5_policy.json"
        )
        if not policy_path.exists():
            raise FileNotFoundError(
                f"Post-suite M5 policy not found at {policy_path}. "
                "Provide --post-m5-policy-json explicitly."
            )

        post_m5_out_dir = (
            Path(args.post_m5_out_dir)
            if args.post_m5_out_dir
            else out_dir / f"m5_post_{args.post_m5_controller_mode}"
        )
        command = [
            sys.executable,
            "scripts/run_m5_instance_guardrail.py",
            "--repair-dir",
            str(out_dir),
            "--policy-json",
            str(policy_path),
            "--controller-mode",
            str(args.post_m5_controller_mode),
            "--default-split",
            str(args.post_m5_default_split),
            "--out-dir",
            str(post_m5_out_dir),
        ]
        if args.post_m5_provider_thresholds_json:
            command.extend(
                [
                    "--provider-thresholds-json",
                    str(args.post_m5_provider_thresholds_json),
                ]
            )
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        post_m5_result = {
            "controller_mode": str(args.post_m5_controller_mode),
            "policy_json": str(policy_path),
            "provider_thresholds_json": (
                str(args.post_m5_provider_thresholds_json)
                if args.post_m5_provider_thresholds_json
                else None
            ),
            "default_split": str(args.post_m5_default_split),
            "out_dir": str(post_m5_out_dir),
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "subset": str(subset_path),
        "n_models": len(model_specs),
        "n_variants": len(variants),
        "planned_cells": len(planned_specs),
        "successful_cells": sum(1 for row in cell_rows if row["status"] == "ok"),
        "failed_cells": sum(1 for row in cell_rows if row["status"] != "ok"),
        "variants": variants,
        "models": [f"{provider}:{model}" for provider, model in model_specs],
        "hypotheses": hypothesis,
        "post_m5_result": post_m5_result,
    }
    (out_dir / "prompt_variant_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    build_report(
        out_path=out_dir / "PROMPT_VARIANT_RESULTS.md",
        cell_rows=cell_rows,
        hypothesis=hypothesis,
    )

    write_progress(
        out_dir=out_dir,
        planned_specs=planned_specs,
        cell_map=cell_map,
        started_utc=started_utc,
        suite_start_s=suite_start_s,
        last_cell="suite_complete",
    )

    print("Prompt variant suite complete")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
