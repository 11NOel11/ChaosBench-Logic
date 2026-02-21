# ChaosBench-Logic v2 — Proprietary Model Execution Plan

**Status:** Draft — ready for execution after API key procurement
**Last updated:** 2026-02

---

## 1. Overview & Hard Rules

This document governs the evaluation of proprietary models (OpenAI, Anthropic, Google) on ChaosBench-Logic v2. The OSS baseline (Qwen2.5-32B: MCC=0.478) sets the comparison floor.

### Hard Rules

1. **No CoT storage.** `truncate_pred_text=200` is mandatory for all proprietary runs. Long chain-of-thought reasoning must not be stored in predictions.jsonl.
2. **`max_tokens=16`.** Set at the provider level. Physically prevents CoT output.
3. **Same prompt, same shuffle.** Use the exact same `prompt_hash` and `shuffle_seed=None` (canonical order) as OSS baselines for fair comparison.
4. **Pre-estimate costs.** Always run `estimate_api_costs.py` before executing a phase. Never spend beyond the per-phase `--max-usd` limit.
5. **Strict parsing only.** `strict_parsing=True` for all runs. Do not use lenient parsing for proprietary comparisons.
6. **One model per run.** Do not batch multiple models in a single run invocation.
7. **Publish only completed runs.** Runs with coverage < 97% require a ⚠️ caveat before publishing.

---

## 2. Model Shortlist

### Tier 1 — Primary comparison (full canonical run planned)

| Provider | Model ID | Notes |
|----------|----------|-------|
| Anthropic | `claude-sonnet-4-6` | Primary Anthropic model |
| OpenAI | `gpt-4o` | Primary OpenAI model |
| Google | `gemini-2.0-flash` | Primary Google model |

### Tier 2 — Cost-efficient baselines

| Provider | Model ID | Notes |
|----------|----------|-------|
| Anthropic | `claude-haiku-4-5` | Fast, low-cost alternative |
| OpenAI | `gpt-4o-mini` | Fast, low-cost alternative |
| Google | `gemini-1.5-flash` | Fast, low-cost alternative |

### Tier 3 — Frontier (budget permitting)

| Provider | Model ID | Notes |
|----------|----------|-------|
| Anthropic | `claude-opus-4-6` | Highest capability, highest cost |
| OpenAI | `o1` / `o3-mini` | Reasoning models — CoT suppression critical |

---

## 3. Phased Evaluation Ladder

### Phase Definitions

| Phase | Subset | N | Purpose | Default Max USD |
|-------|--------|---|---------|----------------|
| P0 | `api_balanced_100.jsonl` | 100 | Sanity check: confirm provider works, parsing rate, no errors | $2 |
| P1 | `subset_1k_armored.jsonl` | 1,000 | First signal: MCC estimate, invalid rate, output format | $20 |
| P2 | `subset_5k_armored.jsonl` | 5,000 | Main comparison: full armored metrics, per-family breakdown | $100 |
| P3 | canonical (full) | 40,886 | Paper-quality full run | $700 |

### Go / No-Go Criteria

After each phase, check:

1. **invalid_rate ≤ 0.02** (2%). Exceeding this → ABORT, do not advance.
2. **MCC ≥ 0.35** on P1/P2. Below this → NO-GO, do not advance to next phase.
3. **|pred_true_pct − 0.495| ≤ 0.15**. Outside this → BIAS FLAG, review prompt.

---

## 4. Exact Commands

### Setup Environment Variables

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# Google Gemini
export GEMINI_API_KEY=AIza...
```

### Phase P0 — Sanity Check (100 items)

```bash
# 1. Estimate cost
python scripts/estimate_api_costs.py \
  --subset data/subsets/api_balanced_100.jsonl \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --pricing_config configs/pricing/anthropic.yaml

# 2. Run eval (dry-run first to confirm command)
python scripts/launch_proprietary_runs.py \
  --provider anthropic --model claude-sonnet-4-6 --phase P0 --dry-run

# 3. Execute
python scripts/launch_proprietary_runs.py \
  --provider anthropic --model claude-sonnet-4-6 --phase P0 --execute \
  --budget-usd 2.0
```

### Phase P1 — 1k Armored

```bash
python scripts/estimate_api_costs.py \
  --subset data/subsets/subset_1k_armored.jsonl \
  --provider anthropic --model claude-sonnet-4-6 \
  --pricing_config configs/pricing/anthropic.yaml \
  --budget_usd 20.0

python scripts/launch_proprietary_runs.py \
  --provider anthropic --model claude-sonnet-4-6 --phase P1 --execute \
  --budget-usd 20.0 --stop-mcc-threshold 0.35
```

### Phase P2 — 5k Armored

```bash
python scripts/estimate_api_costs.py \
  --subset data/subsets/subset_5k_armored.jsonl \
  --provider anthropic --model claude-sonnet-4-6 \
  --pricing_config configs/pricing/anthropic.yaml \
  --budget_usd 100.0

python scripts/launch_proprietary_runs.py \
  --provider anthropic --model claude-sonnet-4-6 --phase P2 --execute \
  --budget-usd 100.0
```

### Phase P3 — Full Canonical

```bash
# Only after P2 passes go/no-go
chaosbench eval \
  --provider anthropic --model claude-sonnet-4-6 \
  --dataset canonical \
  --phase P3 \
  --truncate-pred-text 200 \
  --strict-parsing \
  --max-usd 700.0
```

### All Providers — Batch Dry-Run

```bash
for PROVIDER in openai anthropic gemini; do
  for MODEL in gpt-4o claude-sonnet-4-6 gemini-2.0-flash; do
    python scripts/launch_proprietary_runs.py \
      --provider $PROVIDER --model $MODEL --phase P1 --dry-run
  done
done
```

---

## 5. Cost Estimation Tables

### Anthropic (claude-sonnet-4-6, $3.00/$15.00 per 1M tokens)

| Subset | N | Est. Input Tokens | Est. Cost | Upper Bound |
|--------|---|-------------------|-----------|-------------|
| api_balanced_100 | 100 | ~32,000 | ~$0.10 | ~$0.13 |
| subset_1k_armored | 1,000 | ~320,000 | ~$0.99 | ~$1.24 |
| subset_5k_armored | 5,000 | ~1,600,000 | ~$4.93 | ~$6.16 |
| full_canonical | 40,886 | ~13,083,520 | ~$40.35 | ~$50.44 |

> ⚠️ These are rough estimates (4 chars ≈ 1 token). Always pre-run `estimate_api_costs.py`.

### OpenAI (gpt-4o, $2.50/$10.00 per 1M tokens)

| Subset | N | Est. Cost | Upper Bound |
|--------|---|-----------|-------------|
| api_balanced_100 | 100 | ~$0.08 | ~$0.10 |
| subset_1k_armored | 1,000 | ~$0.82 | ~$1.03 |
| subset_5k_armored | 5,000 | ~$4.11 | ~$5.14 |
| full_canonical | 40,886 | ~$33.63 | ~$42.04 |

### Google (gemini-2.0-flash, $0.10/$0.40 per 1M tokens)

| Subset | N | Est. Cost | Upper Bound |
|--------|---|-----------|-------------|
| api_balanced_100 | 100 | ~$0.003 | ~$0.004 |
| subset_1k_armored | 1,000 | ~$0.033 | ~$0.041 |
| subset_5k_armored | 5,000 | ~$0.164 | ~$0.205 |
| full_canonical | 40,886 | ~$1.34 | ~$1.68 |

---

## 6. Failure Handling & Retry Strategy

### Provider Errors

- **4xx errors**: Non-retryable. Check API key, model name, and request format.
- **5xx errors**: Retried up to 2 times with exponential backoff (1s, 2s).
- **Timeout**: Increase `--timeout` flag (default 60s). Consider rate limiting.

### Run Interruption

Runs checkpoint every 200 items. Resume with:
```bash
chaosbench eval --resume runs/<interrupted_run_id>
```

### High Invalid Rate

If `invalid_rate > 0.02`:
1. Inspect `predictions.jsonl` for raw outputs.
2. Check if `strict_suffix` is appended (required for all proprietary providers).
3. Verify `max_tokens=16` is respected by the provider.
4. Consider enabling lenient parsing mode for diagnostics only (not for official metrics).

### Budget Exhaustion

If `--max-usd` is reached mid-run, the runner stops and writes a checkpoint. The partial run can be resumed with a higher budget, or the partial results can be analyzed as-is (with appropriate coverage caveats).

---

## 7. Risk & Confound Checklist

Before publishing any proprietary run:

- [ ] **Prompt drift**: Verify `prompt_hash` matches OSS baselines.
- [ ] **Label balance**: `pred_true_pct` within [0.345, 0.645] (±0.15 from 0.495).
- [ ] **Coverage**: ≥ 97%. If 97–99.5%, add ⚠️ caveat.
- [ ] **Invalid rate**: ≤ 2% total, ≤ 5% per task family.
- [ ] **CoT contamination**: `truncate_pred_text=200` was active. No full CoT in predictions.jsonl.
- [ ] **API version pinning**: Record model version returned in raw metadata if available.
- [ ] **Ordering**: `order_mode=canonical` (no shuffle) for cross-model comparison.
- [ ] **Temperature=0**: All runs use `temperature=0.0` for reproducibility.
- [ ] **No data leakage**: Questions are synthetic; no overlap with known training data.
- [ ] **Date stamp**: Record run date (model providers update weights without versioning).

---

## 8. Paper Reporting Guidelines

### Metrics to Report

For each model × subset, report in the paper:
- MCC (primary metric)
- Balanced accuracy
- Coverage (valid response rate)
- Per-family MCC breakdown (where N ≥ 100 per family)

### Comparison Rules

- Compare proprietary models on `full_canonical` against OSS models on `full_canonical` only.
- Do not mix subset sizes in the main comparison table.
- Include OSS baselines in the same table as proprietary for direct comparison.
- Report upper and lower bounds where re-running is infeasible.

### Required Caveats

- Proprietary models use `truncate_pred_text=200`; stored pred_text may be truncated.
- API-based evaluation subject to model version drift; date of run must be cited.
- Costs and token counts are estimates; actual billing may differ.

### Figures

- Figure 1: MCC vs. model size / compute (scatter or line) — include all models.
- Figure 2: Per-family MCC heatmap — show gap between OSS and proprietary.
- Figure 3: Cost-effectiveness frontier (MCC vs. cost-per-1k-items).

---

## 9. Pre-Publish Checklist

Before calling `chaosbench publish-run`:

- [ ] `coverage ≥ 0.97`
- [ ] `invalid_rate ≤ 0.05` (hard limit for publication)
- [ ] `prompt_hash` matches canonical hash from `chaosbench eval --show-prompt-hash`
- [ ] `run_manifest.json` contains `truncate_pred_text: 200` and `phase` field
- [ ] Predictions file written atomically (checkpoint removed)
- [ ] Dataset SHA256 in manifest matches current freeze SHA
- [ ] Run reviewed by at least one team member before public indexing
- [ ] Cost recorded and within approved budget

---

*For OSS run protocol, see `docs/RUN_PROTOCOL.md`. For dataset specification, see `docs/V2_SPEC.md`.*
