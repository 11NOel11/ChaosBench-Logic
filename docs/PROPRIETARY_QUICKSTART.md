# ChaosBench-Logic v2 — Proprietary Model Quickstart

**Last updated:** 2026-02-21
**Tested with:** uv ≥ 0.1.0

---

## 1. API Key Setup

### 1.1 Get Your API Keys

#### **Anthropic (claude-sonnet-4-6)**
1. Visit: https://console.anthropic.com
2. Sign in / create account
3. Go to **Settings** → **API keys**
4. Click "Create Key"
5. Copy and save (you won't see it again)

#### **OpenAI (gpt-4o)**
1. Visit: https://platform.openai.com
2. Sign in / create account
3. Go to **API keys** → **Create new secret key**
4. Copy and save

#### **Google Gemini (gemini-2.0-flash)**
1. Visit: https://ai.google.dev/
2. Sign in with Google account
3. Go to **API Keys** (left sidebar)
4. Click "Create API Key"
5. Copy and save

### 1.2 Export Keys (one-time setup)

Add to your **~/.zshrc** or **~/.bash_profile**:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-v0-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google Gemini
export GEMINI_API_KEY="AIza..."
```

Then reload:
```bash
source ~/.zshrc
```

Verify keys are set:
```bash
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
echo $GEMINI_API_KEY
```

All three should print non-empty strings (first 10 chars visible).

---

## 2. Sequential Evaluation Run (Full Pipeline)

### Choose Your Model

Pick **one** to start:
- **Anthropic**: `claude-sonnet-4-6` (primary) or `claude-haiku-4-5` (budget)
- **OpenAI**: `gpt-4o` (primary) or `gpt-4o-mini` (budget)
- **Google**: `gemini-2.0-flash` (primary) or `gemini-1.5-flash` (budget)

### Example: Anthropic claude-sonnet-4-6

**Set provider & model for convenience:**

```bash
export EVAL_PROVIDER="anthropic"
export EVAL_MODEL="claude-sonnet-4-6"
export WORK_DIR="/Users/noel.thomas/chaos-logic-bench"
cd $WORK_DIR
```

---

## 3. Phase P0 — Sanity Check (100 items, ~$0.10)

**Objective:** Confirm provider works, check parsing, estimate actual costs.

### Step 1: Estimate cost

```bash
uv run python scripts/estimate_api_costs.py \
  --subset data/subsets/api_balanced_100.jsonl \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --pricing_config configs/pricing/${EVAL_PROVIDER}.yaml
```

**Expected output:**
```
Estimated total cost: $0.09–$0.12
Estimated tokens: ~30,000 input
```

### Step 2: Dry-run (no API calls)

```bash
uv run python scripts/launch_proprietary_runs.py \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --phase P0 \
  --dry-run
```

**Expected output:**
```
[DRY-RUN] Would execute 100 items
[DRY-RUN] Estimated cost: $0.09
Ready to execute. Re-run with --execute flag.
```

### Step 3: Execute P0

```bash
uv run python scripts/launch_proprietary_runs.py \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --phase P0 \
  --execute \
  --budget-usd 2.0
```

**Expected output:**
```
[P0] Running 100 items...
[P0] Item 1/100 ✓
...
[P0] Item 100/100 ✓
[P0] Coverage: 100%
[P0] Invalid rate: 0.00%
[P0] MCC: 0.42
[P0] Results saved to: runs/anthropic_claude-sonnet-4-6_p0_20260221_120000/
```

### ✅ Go/No-Go Check (P0)

```bash
# Inspect results
jq '.model, .coverage, .invalid_rate, .mcc' \
  runs/anthropic_claude-sonnet-4-6_p0_*/run_manifest.json
```

**Proceed to P1 if:**
- ✅ `coverage ≥ 0.97` (97%)
- ✅ `invalid_rate ≤ 0.02` (2%)
- ✅ No API errors

If any fail → **STOP**, debug, and retry P0.

---

## 4. Phase P1 — 1k Armored (1,000 items, ~$1.00)

**Objective:** Get first signal on MCC, check per-family metrics.

### Step 1: Estimate cost

```bash
uv run python scripts/estimate_api_costs.py \
  --subset data/subsets/subset_1k_armored.jsonl \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --pricing_config configs/pricing/${EVAL_PROVIDER}.yaml
```

### Step 2: Execute P1

```bash
uv run python scripts/launch_proprietary_runs.py \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --phase P1 \
  --execute \
  --budget-usd 20.0 \
  --stop-mcc-threshold 0.35
```

**Expected output:**
```
[P1] Running 1,000 items...
[P1] Item 200/1000 ✓ (coverage: 100%, MCC so far: 0.38)
...
[P1] Item 1000/1000 ✓
[P1] Coverage: 99.8%
[P1] Invalid rate: 0.01%
[P1] MCC: 0.44
[P1] Per-family MCC:
  - atomic: 0.45
  - multi_hop: 0.42
  ...
[P1] Results saved to: runs/anthropic_claude-sonnet-4-6_p1_20260221_130000/
```

### ✅ Go/No-Go Check (P1)

```bash
jq '.mcc, .coverage, .invalid_rate, .label_balance' \
  runs/anthropic_claude-sonnet-4-6_p1_*/run_manifest.json
```

**Proceed to P2 if:**
- ✅ `MCC ≥ 0.35`
- ✅ `coverage ≥ 0.97`
- ✅ `invalid_rate ≤ 0.02`
- ✅ `pred_true_pct` within [0.345, 0.645]

If `MCC < 0.35` → **NO-GO to P2**, but you can still run P2 for diagnostic only.

---

## 5. Phase P2 — 5k Armored (5,000 items, ~$5.00)

**Objective:** Validate on larger subset, get full per-family breakdown.

### Step 1: Estimate cost

```bash
uv run python scripts/estimate_api_costs.py \
  --subset data/subsets/subset_5k_armored.jsonl \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --pricing_config configs/pricing/${EVAL_PROVIDER}.yaml
```

### Step 2: Execute P2

```bash
uv run python scripts/launch_proprietary_runs.py \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --phase P2 \
  --execute \
  --budget-usd 100.0
```

**Expected time:** ~10–20 minutes (depends on API latency)

### ✅ Go/No-Go Check (P2)

```bash
jq '.mcc, .coverage, .invalid_rate, .label_balance, .per_family_mcc' \
  runs/anthropic_claude-sonnet-4-6_p2_*/run_manifest.json | head -20
```

**Proceed to P3 if:**
- ✅ `MCC ≥ 0.35`
- ✅ `coverage ≥ 0.97`
- ✅ `invalid_rate ≤ 0.02`
- ✅ All 10 families have N ≥ 100 and MCC ≤ 0.70

If any fail → **NO-GO to P3**, analyze why and consider retrying P1/P2.

---

## 6. Phase P3 — Full Canonical (40,886 items, ~$40)

**Objective:** Paper-quality full run. Only proceed if P2 passes all checks.

### Step 1: Estimate cost (final sanity check)

```bash
uv run python scripts/estimate_api_costs.py \
  --subset data/canonical_v2_files.json \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --pricing_config configs/pricing/${EVAL_PROVIDER}.yaml
```

### Step 2: Execute P3

```bash
uv run python scripts/launch_proprietary_runs.py \
  --provider $EVAL_PROVIDER \
  --model $EVAL_MODEL \
  --phase P3 \
  --execute \
  --budget-usd 700.0
```

**Expected time:** ~60–120 minutes

**Real-time progress:**
```bash
# In another terminal, monitor progress
tail -f runs/anthropic_claude-sonnet-4-6_p3_*/progress.log
```

### ✅ Final Validation (P3)

```bash
jq '.mcc, .coverage, .invalid_rate, .label_balance' \
  runs/anthropic_claude-sonnet-4-6_p3_*/run_manifest.json
```

**Publish if:**
- ✅ `coverage ≥ 0.97`
- ✅ `invalid_rate ≤ 0.05` (hard limit for publication)

---

## 7. Publishing Results

After P3 completes and passes validation:

```bash
# Extract run ID from latest P3 run
RUN_ID=$(ls -1rt runs/anthropic_claude-sonnet-4-6_p3_* | tail -1 | xargs basename)

# Publish
uv run python -m chaosbench.publish.run \
  --run-id "$RUN_ID" \
  --tag OFFICIAL_V2 \
  --model-name "claude-sonnet-4-6" \
  --provider "anthropic"
```

**Verify publication:**
```bash
curl -s "https://chaosbench-public.s3.amazonaws.com/results/$RUN_ID/run_manifest.json" | jq .
```

---

## 8. Multi-Provider Sequential Execution

To run **all three** providers in sequence:

```bash
#!/bin/bash
set -e

PROVIDERS=(
  "anthropic:claude-sonnet-4-6"
  "openai:gpt-4o"
  "gemini:gemini-2.0-flash"
)

for PROVIDER_MODEL in "${PROVIDERS[@]}"; do
  IFS=':' read -r PROVIDER MODEL <<< "$PROVIDER_MODEL"
  
  echo "=========================================="
  echo "Starting $PROVIDER / $MODEL"
  echo "=========================================="
  
  # P0
  uv run python scripts/launch_proprietary_runs.py \
    --provider "$PROVIDER" --model "$MODEL" --phase P0 \
    --execute --budget-usd 2.0
  
  # P1
  uv run python scripts/launch_proprietary_runs.py \
    --provider "$PROVIDER" --model "$MODEL" --phase P1 \
    --execute --budget-usd 20.0
  
  # P2
  uv run python scripts/launch_proprietary_runs.py \
    --provider "$PROVIDER" --model "$MODEL" --phase P2 \
    --execute --budget-usd 100.0
  
  # P3 (only if P2 passed)
  uv run python scripts/launch_proprietary_runs.py \
    --provider "$PROVIDER" --model "$MODEL" --phase P3 \
    --execute --budget-usd 700.0
  
  echo "✓ $PROVIDER / $MODEL complete"
done

echo "All providers done!"
```

Save as `scripts/batch_proprietary_eval.sh` and run:
```bash
chmod +x scripts/batch_proprietary_eval.sh
uv run bash scripts/batch_proprietary_eval.sh
```

---

## 9. Troubleshooting

### "API key not found"
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY
# If empty, run:
source ~/.zshrc
# Then retry command
```

### "Timeout after 60s"
Increase timeout in the command:
```bash
uv run python scripts/launch_proprietary_runs.py \
  ... --timeout 120
```

### "Budget exceeded"
The run stopped mid-execution. Results are saved in `runs/<run_id>/`. You can resume with a higher budget:
```bash
uv run python scripts/launch_proprietary_runs.py \
  --provider $EVAL_PROVIDER --model $EVAL_MODEL \
  --phase P2 --resume \
  --budget-usd 200.0
```

### "Invalid rate > 2%"
Check raw outputs:
```bash
tail -20 runs/<run_id>/predictions.jsonl | jq .pred_text
```

If prediction is empty or "TIMEOUT", the provider is rate-limited. Increase delay:
```bash
# Pass a per-request delay (in seconds)
--delay-per-request 1.0
```

---

## 10. Quick Reference

| Phase | Subset | N | Est. Cost | Max Budget | Go/No-Go |
|-------|--------|---|-----------|-----------|----------|
| P0 | 100-item sanity | 100 | ~$0.10 | $2 | coverage ≥ 97%, invalid ≤ 2% |
| P1 | 1k armored | 1,000 | ~$1.00 | $20 | MCC ≥ 0.35, coverage ≥ 97% |
| P2 | 5k armored | 5,000 | ~$5.00 | $100 | MCC ≥ 0.35, coverage ≥ 97% |
| P3 | Full canonical | 40,886 | ~$40 | $700 | coverage ≥ 97%, invalid ≤ 5% |

---

*For full details, see `docs/PROPRIETARY_EXECUTION_PLAN.md`.*
