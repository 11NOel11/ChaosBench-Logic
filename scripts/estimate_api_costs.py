"""CLI tool to estimate API costs for a ChaosBench evaluation run.

Usage:
    python scripts/estimate_api_costs.py \\
      --subset data/subsets/subset_1k_armored.jsonl \\
      --provider anthropic \\
      --model claude-sonnet-4-6 \\
      --pricing_config configs/pricing/anthropic.yaml \\
      --assumed_output_tokens 2 \\
      --budget_usd 50.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script from the repo root
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chaosbench.eval.prompts import build_prompt


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file, using PyYAML if available."""
    try:
        import yaml  # type: ignore

        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        pass

    # Fallback: minimal hand-rolled parser for the simple flat/nested structure
    # used in configs/pricing/*.yaml.
    result: Dict[str, Any] = {}
    current_model: Optional[str] = None
    models: Dict[str, Dict[str, float]] = {}
    in_models = False

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Remove inline comments
            if " #" in stripped:
                stripped = stripped[: stripped.index(" #")].rstrip()
            if ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")

            if indent == 0:
                if key == "models":
                    in_models = True
                    current_model = None
                    continue
                else:
                    in_models = False
                    result[key] = val
            elif indent == 2 and in_models:
                current_model = key
                models[current_model] = {}
            elif indent == 4 and in_models and current_model:
                try:
                    models[current_model][key] = float(val)
                except ValueError:
                    models[current_model][key] = val

    result["models"] = models
    return result


def _load_subset(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _estimate_tokens(text: str) -> int:
    """Estimate token count: ceil(chars / 4). Conservative lower bound."""
    return max(1, math.ceil(len(text) / 4))


def estimate_cost(
    subset_path: str,
    provider: str,
    model: str,
    pricing_config: str,
    assumed_output_tokens: int = 2,
    budget_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute cost estimate for running a subset through a model.

    Args:
        subset_path: Path to the JSONL subset file.
        provider: Provider name (for metadata; must match pricing_config provider).
        model: Model identifier, must exist in the pricing_config models section.
        pricing_config: Path to a pricing YAML file.
        assumed_output_tokens: Assumed number of output tokens per item.
        budget_usd: Optional budget cap in USD.

    Returns:
        Dict with cost estimate fields (see JSON schema in plan).

    Raises:
        KeyError: If the model is not found in the pricing config.
        FileNotFoundError: If the subset or pricing config file does not exist.
    """
    items = _load_subset(subset_path)
    n_items = len(items)

    # Build prompts and estimate input tokens
    per_item_tokens: List[int] = []
    for item in items:
        question = item.get("question", item.get("prompt", ""))
        prompt = build_prompt(question)
        per_item_tokens.append(_estimate_tokens(prompt))

    avg_input_toks = sum(per_item_tokens) / n_items if n_items else 0
    total_input_toks = sum(per_item_tokens)
    upper_bound_input_toks = math.ceil(total_input_toks * 1.25)

    total_output_toks = n_items * assumed_output_tokens

    # Load pricing
    pricing = _load_yaml(pricing_config)
    models_section = pricing.get("models", {})
    if model not in models_section:
        available = list(models_section.keys())
        raise KeyError(
            f"Model '{model}' not found in {pricing_config}. "
            f"Available models: {available}"
        )
    model_prices = models_section[model]
    input_price_per_token = float(model_prices["input_per_1m_tokens"]) / 1_000_000
    output_price_per_token = float(model_prices["output_per_1m_tokens"]) / 1_000_000

    estimated_cost = (
        total_input_toks * input_price_per_token
        + total_output_toks * output_price_per_token
    )
    upper_bound_cost = (
        upper_bound_input_toks * input_price_per_token
        + total_output_toks * output_price_per_token
    )

    items_feasible: Optional[int] = None
    if budget_usd is not None:
        # Compute how many items fit within budget at average cost per item
        avg_cost_per_item = estimated_cost / n_items if n_items else 0
        if avg_cost_per_item > 0:
            items_feasible = min(n_items, int(budget_usd / avg_cost_per_item))
        else:
            items_feasible = n_items

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "subset_path": subset_path,
        "n_items": n_items,
        "estimated_input_tokens_per_item": round(avg_input_toks),
        "estimated_total_input_tokens": total_input_toks,
        "upper_bound_total_input_tokens": upper_bound_input_toks,
        "assumed_output_tokens_per_item": assumed_output_tokens,
        "estimated_total_output_tokens": total_output_toks,
        "estimated_cost_usd": round(estimated_cost, 4),
        "upper_bound_cost_usd": round(upper_bound_cost, 4),
        "budget_usd": budget_usd,
        "items_feasible_under_budget": items_feasible,
        "pricing_source": pricing_config,
    }


def _write_cost_md(out_dir: Path, result: Dict[str, Any]) -> Path:
    md_lines = [
        "# API Cost Estimate",
        "",
        f"**Generated:** {result['timestamp']}",
        f"**Provider:** {result['provider']}",
        f"**Model:** {result['model']}",
        f"**Subset:** {result['subset_path']}",
        f"**Pricing source:** {result['pricing_source']}",
        "",
        "> ⚠️ PLACEHOLDER PRICES — verify at provider pricing page before use.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Items | {result['n_items']:,} |",
        f"| Est. input tokens/item | {result['estimated_input_tokens_per_item']:,} |",
        f"| Est. total input tokens | {result['estimated_total_input_tokens']:,} |",
        f"| Upper-bound input tokens (×1.25) | {result['upper_bound_total_input_tokens']:,} |",
        f"| Assumed output tokens/item | {result['assumed_output_tokens_per_item']} |",
        f"| Est. total output tokens | {result['estimated_total_output_tokens']:,} |",
        f"| **Estimated cost (USD)** | **${result['estimated_cost_usd']:.4f}** |",
        f"| Upper-bound cost (USD) | ${result['upper_bound_cost_usd']:.4f} |",
    ]
    if result.get("budget_usd") is not None:
        md_lines += [
            f"| Budget (USD) | ${result['budget_usd']:.2f} |",
            f"| Items feasible under budget | {result['items_feasible_under_budget']:,} |",
        ]
    md_path = out_dir / "cost_estimate.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    return md_path


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Estimate API costs for a ChaosBench evaluation run."
    )
    parser.add_argument("--subset", required=True, help="Path to JSONL subset file")
    parser.add_argument("--provider", required=True, help="Provider name (openai/anthropic/gemini)")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--pricing_config", required=True, help="Path to pricing YAML file")
    parser.add_argument(
        "--assumed_output_tokens", type=int, default=2, help="Assumed output tokens per item"
    )
    parser.add_argument("--budget_usd", type=float, default=None, help="Optional budget cap (USD)")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: artifacts/api_costs/<timestamp>)",
    )
    args = parser.parse_args(argv)

    result = estimate_cost(
        subset_path=args.subset,
        provider=args.provider,
        model=args.model,
        pricing_config=args.pricing_config,
        assumed_output_tokens=args.assumed_output_tokens,
        budget_usd=args.budget_usd,
    )

    # Write output artifacts
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("artifacts") / "api_costs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cost_estimate.json"
    json_path.write_text(json.dumps(result, indent=2))

    md_path = _write_cost_md(out_dir, result)

    print(f"\nCost estimate for {result['provider']}/{result['model']} on {result['n_items']:,} items:")
    print(f"  Estimated cost:    ${result['estimated_cost_usd']:.4f} USD")
    print(f"  Upper-bound cost:  ${result['upper_bound_cost_usd']:.4f} USD")
    if result.get("budget_usd") is not None:
        feasible = result["items_feasible_under_budget"]
        print(f"  Budget:            ${result['budget_usd']:.2f} USD")
        status = "OK" if feasible >= result["n_items"] else f"OVER (only {feasible:,} items feasible)"
        print(f"  Budget status:     {status}")
    print(f"\nOutputs written to: {out_dir}/")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return result


if __name__ == "__main__":
    main()
