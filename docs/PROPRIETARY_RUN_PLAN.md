# Proprietary Model Run Plan — ChaosBench-Logic v2

_Status: PLAN ONLY — no runs executed. Document version: 2026-02-21._

---

## 0. Overview

This document specifies the protocol for evaluating proprietary (API-based) language models
on ChaosBench-Logic v2. It establishes:

- Which models to evaluate first
- Dataset subsets to use and scaling strategy
- Run protocol (prompt, parsing, storage)
- Comparison safety rules
- Cost budgeting

**Hard rule**: All proprietary runs must produce `OFFICIAL_V2` artifacts using the same
protocol as OSS runs. Never compare v1 results with v2 results.

---

## 1. Model Selection

Evaluate **one model per provider** in the initial pass to establish cross-provider baselines.
Choose the strongest general-purpose model available for each provider.

| Priority | Provider | Model (initial) | API endpoint | Notes |
|----------|----------|-----------------|--------------|-------|
| 1 | Anthropic | claude-sonnet-4-6 (latest Sonnet) | Anthropic API | Good cost/capability balance |
| 2 | OpenAI | gpt-4o (2024-08 or later) | OpenAI API | Strongest available |
| 3 | Google | gemini-2.0-flash (or gemini-1.5-pro) | Google AI Studio | Flash for cost; Pro if competitive |
| 4 | Mistral AI | mistral-large-2 | Mistral API | Compare against local mistral:7b |
| 5 | Cohere | command-r-plus | Cohere API | Optional; add only if above 4 are run |

Start with providers 1–3 only. Add 4–5 after initial analysis if budget allows.

---

## 2. Dataset Subsets

### Phase 1: 1k Armored Subset (Initial Probe)

Use `data/subsets/api_balanced_1k.jsonl` (N=1,000, subset SHA16: `1ac9f1af107b626a`).

- Budget indicator: ~$5–15 per model depending on provider pricing
- Decision gate: if MCC_micro ≥ 0.50, proceed to Phase 2
- Expected outcomes: proprietary models should significantly outperform OSS 7–9B baselines

### Phase 2: 5k Armored Subset (If Phase 1 competitive)

Use `data/subsets/armored_5k.jsonl` (or equivalent canonical-armored subset, N≈5,000).

- Budget indicator: ~$25–75 per model
- Decision gate: if MCC_micro ≥ 0.60, consider full canonical run

### Phase 3: Full Canonical (Only if competitive with strong OSS)

Use `data/canonical_v2_files.json` (N=40,886).

- Budget indicator: ~$200–600 per model at gpt-4o pricing
- Only proceed if model's Phase 2 MCC significantly exceeds Qwen2.5-14B (0.503)

---

## 3. Run Protocol

### 3.1 Prompt

Use the same prompt as all v2 OSS runs:
- `prompt_version: v1`
- `prompt_hash: 5881f664c444e3d3`

Do NOT modify the prompt between OSS and proprietary runs. If a model requires a system
prompt, use the default ChaosBench system prompt (see `chaosbench/eval/prompts.py`).

### 3.2 Execution

```bash
# Phase 1 — 1k subset
chaosbench eval \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --subset data/subsets/api_balanced_1k.jsonl \
  --workers 4 \
  --retries 2 \
  --strict-parsing

# Phase 2 — 5k subset (if MCC ≥ 0.50)
chaosbench eval \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --subset data/subsets/armored_5k.jsonl \
  --workers 8 \
  --retries 2 \
  --strict-parsing
```

**Workers**: Use 4–8 workers per provider (respect rate limits).
**Retries**: Use `--retries 2` for API runs to handle transient failures.
**Strict parsing**: Always `--strict-parsing`. Do NOT use lenient mode for official runs.

### 3.3 Output Verification

After each run:
1. Verify `coverage ≥ 0.99` in `run_manifest.json`
2. Verify `dataset_global_sha256` matches freeze SHA
3. Check `invalid_rate` — if > 2%, investigate before publishing
4. Run `python scripts/build_results_pack.py` to verify metrics

---

## 4. Storage and Publishing

### 4.1 Run artifacts

All runs are stored in `runs/` (gitignored). Run directory name: `<run_id>` (auto-generated).

Publish only the lightweight artifacts:
```bash
chaosbench publish-run --run runs/<run_id>
```

For 1k and 5k runs, compress predictions:
```bash
chaosbench publish-run --run runs/<run_id> --compress-predictions
```

Do NOT publish full predictions for full canonical runs (> 40k lines).

### 4.2 Chain-of-thought storage

**Safety rule**: If the provider's Terms of Service prohibit storing chain-of-thought outputs,
do NOT log `pred_text` for CoT variants. Set a flag in the run manifest:
```json
"cot_storage_suppressed": true,
"cot_suppression_reason": "provider_tos"
```

For standard zero-shot binary-output runs (TRUE/FALSE), storage is safe as `pred_text`
contains only "TRUE" or "FALSE" (not a full CoT trace).

### 4.3 Run manifest requirements

Each official run manifest MUST include:
- `provider`: e.g., `anthropic/claude-sonnet-4-6`
- `prompt_hash`: must match `5881f664c444e3d3`
- `dataset_global_sha256`: must match `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279`
- `canonical_selector`: `data/canonical_v2_files.json`
- `strict_parsing`: `true`
- `git_commit`: current HEAD at run time

---

## 5. Cost Budgeting

Rough estimates (prices as of 2026-02):

| Phase | N | Tokens/q est. | Anthropic Sonnet | OpenAI GPT-4o | Google Flash |
|-------|---|---------------|-----------------|---------------|--------------|
| 1k | 1,000 | ~300 in / 5 out | ~$0.90 | ~$1.25 | ~$0.15 |
| 5k | 5,000 | ~300 in / 5 out | ~$4.50 | ~$6.25 | ~$0.75 |
| Full | 40,886 | ~300 in / 5 out | ~$37 | ~$51 | ~$6 |

**Strategy**: Start with Google Flash (cheapest) for Phase 1 to calibrate before spending on
Anthropic/OpenAI. Only commit to Phase 3 if Phase 2 MCC exceeds 0.55.

---

## 6. Comparison Protocol

### What can be compared

✅ Compare OFFICIAL_V2 runs only
✅ Compare models at the same subset size when possible (prefer 5k vs 5k, or full vs full)
✅ Cross-subset comparison (1k vs 5k) is acceptable for the SAME model, with explicit caveat
✅ OSS models vs. proprietary models on the same subset

### What must NOT be compared

❌ Never compare v1 results (claude3_zeroshot, gpt4_zeroshot, etc.) with v2 results
❌ Never compare PARTIAL_V2 runs with OFFICIAL_V2 runs in the same table
❌ Never report mock run results in any published table
❌ Never compare runs with different prompt_hash values without explicit delta analysis

### Reporting format

In paper tables, always specify:
- Model name and version
- N (items evaluated)
- Subset used (1k / 5k / full canonical)
- Date of run (to allow future reproducibility checks)

---

## 7. If a Proprietary Model is Below OSS Baselines

If a proprietary model (e.g., Gemini Flash) underperforms Qwen2.5-14B (MCC=0.503) on the 5k subset:

1. Verify the run is not anomalous (check per-family breakdown, invalid rate)
2. Check if the model requires a different output format (some models add preamble)
3. Consider a lenient-parsing post-hoc analysis to quantify format vs. reasoning errors
4. DO NOT re-run with a modified prompt as the "official" result — report original run,
   note the prompt sensitivity as a finding

---

## 8. Checklist Before Publishing Proprietary Results

- [ ] Run manifest SHA matches freeze: `cfcfcc739988…`
- [ ] Coverage ≥ 0.99
- [ ] Invalid rate < 2%
- [ ] Strict parsing confirmed
- [ ] `python scripts/build_results_pack.py` shows ✅ metric verification
- [ ] Compared ONLY against other OFFICIAL_V2 runs
- [ ] No CoT text stored if provider TOS prohibits
- [ ] `chaosbench publish-run` executed and receipt present
- [ ] `published_results/runs/README.md` updated
