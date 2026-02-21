# ChaosBench-Logic v2 — Run Protocol

**Applies to:** All official evaluation runs (OSS and proprietary)
**Last updated:** 2026-02

---

## 1. Core Invariants

Every official run MUST satisfy these invariants for cross-model comparability:

| Invariant | Required Value | Purpose |
|-----------|---------------|---------|
| `prompt_hash` | Must match canonical hash | Same question framing for all models |
| `shuffle_seed` | `None` (canonical order) | Reproducible ordering |
| `strict_parsing` | `True` | Consistent label extraction |
| `dataset_global_sha256` | Must match freeze SHA | Same dataset version |
| `temperature` | `0.0` | Deterministic outputs |

To check the canonical prompt hash:
```bash
python -c "from chaosbench.eval.prompts import get_prompt_hash; print(get_prompt_hash())"
```

---

## 2. OSS Runs (Ollama)

### Prerequisites

```bash
# Confirm Ollama is running and model is pulled
ollama list
ollama pull qwen2.5:32b
```

### Standard Eval Command

```bash
chaosbench eval \
  --provider ollama \
  --model qwen2.5:32b \
  --dataset canonical \
  --workers 4 \
  --retries 1
```

### Subset Run

```bash
chaosbench eval \
  --provider ollama \
  --model qwen2.5:32b \
  --dataset data/subsets/subset_5k_armored.jsonl
```

### Resume Interrupted Run

```bash
chaosbench eval --resume runs/<run_id>
```

---

## 3. Proprietary Runs (API)

### Additional Rules (beyond core invariants)

1. **`max_tokens=16`** — Set in the provider constructor. Prevents CoT output.
2. **`truncate_pred_text=200`** — Truncate stored prediction text. Belt-and-suspenders.
3. **`strict_suffix=True`** — Appends `"\n\nReturn exactly one token: TRUE or FALSE. No explanation."` to every prompt.
4. **No raw responses stored** — Token usage stored in `raw` (not written to predictions.jsonl).

### Pre-Run Cost Estimation (mandatory)

```bash
python scripts/estimate_api_costs.py \
  --subset data/subsets/subset_1k_armored.jsonl \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --pricing_config configs/pricing/anthropic.yaml \
  --budget_usd 50.0
```

Always review the output before executing. Set `--max-usd` in the eval command to match your approved budget.

### Phased Execution

Use the launch script for safe phased execution:

```bash
# Dry-run first (always)
python scripts/launch_proprietary_runs.py \
  --provider anthropic --model claude-sonnet-4-6 --phase P1 --dry-run

# Execute after reviewing dry-run output
python scripts/launch_proprietary_runs.py \
  --provider anthropic --model claude-sonnet-4-6 --phase P1 --execute \
  --budget-usd 20.0
```

See `docs/PROPRIETARY_EXECUTION_PLAN.md` for full phase ladder and go/no-go criteria.

---

## 4. Budget Controls

### Fields in RunConfig

| Field | Default | Description |
|-------|---------|-------------|
| `max_usd` | `None` | Soft stop when estimated spend exceeds this |
| `dry_run_cost` | `False` | Return cost estimate without running eval |
| `truncate_pred_text` | `0` | Truncate pred_text to N chars (0 = disabled) |
| `cost_per_input_token` | `0.0` | USD/token for runtime spend tracking |
| `cost_per_output_token` | `0.0` | USD/token for runtime spend tracking |
| `assumed_output_tokens` | `2` | Assumed output tokens/item for estimation |

### Programmatic Dry-Run

```python
from chaosbench.eval.run import EvalRunner, RunConfig
from chaosbench.eval.providers.anthropic import AnthropicProvider

cfg = RunConfig(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    dry_run_cost=True,
    cost_per_input_token=3.0 / 1_000_000,
    cost_per_output_token=15.0 / 1_000_000,
    max_usd=20.0,
)
result = EvalRunner(cfg).run(dataset="data/subsets/subset_1k_armored.jsonl")
# result = {"dry_run": True, "estimated_cost_usd": ..., "budget_status": "OK/OVER", ...}
```

---

## 5. Run Manifest Fields

Every run writes `run_manifest.json` with these fields relevant to reproducibility:

```json
{
  "prompt_hash": "<16-char hex>",
  "prompt_version": "v1",
  "dataset_global_sha256": "<64-char hex>",
  "order_mode": "canonical",
  "shuffle_seed": null,
  "budget": {
    "max_usd": 20.0,
    "truncate_pred_text": 200,
    "phase": "P1",
    "dry_run_cost": false
  }
}
```

---

## 6. Quality Gates

After each run, verify:

```bash
# Check coverage and MCC
cat runs/<run_id>/metrics.json | python -m json.tool

# Check invalid rate
python -c "
import json
m = json.load(open('runs/<run_id>/metrics.json'))
print(f\"Coverage: {m['coverage']:.4f}\")
print(f\"MCC: {m['mcc']:.4f}\")
print(f\"Invalid rate: {m['invalid_rate']:.4f}\")
"
```

Minimum thresholds for publication:

| Metric | Threshold |
|--------|-----------|
| coverage | ≥ 0.97 |
| invalid_rate | ≤ 0.05 |
| MCC | reported (no minimum for publication) |

---

## 7. Pricing Config Maintenance

Pricing files live in `configs/pricing/{provider}.yaml`. These are **placeholder estimates** — always verify against the provider's current pricing page before running expensive evaluations.

```bash
# Verify current Anthropic pricing
# https://www.anthropic.com/pricing

# Verify current OpenAI pricing
# https://openai.com/api/pricing

# Verify current Google pricing
# https://ai.google.dev/pricing
```

Update the `updated:` field in each YAML after verifying.

---

*For proprietary-specific guidance, see `docs/PROPRIETARY_EXECUTION_PLAN.md`.*
*For dataset specification, see `docs/V2_SPEC.md`.*
