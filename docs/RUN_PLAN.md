# ChaosBench-Logic v2 — Evaluation Run Plan

## 1. Overview

This document specifies the canonical evaluation protocol for producing paper-quality results on ChaosBench-Logic v2.2.

| Item | Value |
|------|-------|
| Dataset version | v2.2.0 |
| Canonical selector | `data/canonical_v2_files.json` |
| Standard subsets | `data/subsets/subset_1k_armored.jsonl`, `data/subsets/subset_5k_armored.jsonl` |
| Family diagnostics | `data/subsets/subset_family_suites/<family>.jsonl` |
| Prompt version | see `chaosbench/eval/prompts.py` |
| Metrics | balanced_accuracy (primary), MCC, coverage |

**Eval protocol**: Every run uses `--seed 42`, deterministic provider settings (temperature=0), and the armored subset as input. All outputs go to `runs/<run_id>/`. Runs are published with `chaosbench publish-run` before analysis.

---

## 2. Priority Tiers

| Tier | Purpose | Dataset |
|------|---------|---------|
| **P0** | Sanity smoke — verify pipeline works end-to-end | `subset_1k_armored`, `--max-items 50` |
| **P1** | 5k armored baselines — compare all models head-to-head | `subset_5k_armored` |
| **P2** | Full canonical — top 2–3 models only, for paper | canonical (all ~25k items) |
| **P3** | Large model — qwen2.5:32b if 32 GB VRAM available | canonical |
| **P4** | Family diagnostics — per-family deep dive | `subset_family_suites/<family>` |

---

## 3. Commands

### P0 — Sanity smoke
```bash
chaosbench eval --provider mock --subset data/subsets/subset_1k_armored.jsonl --max-items 50
```

### P1 — 5k armored baselines

```bash
# qwen2.5:14b  (14B → workers=2)
chaosbench eval --provider ollama --model qwen2.5:14b \
    --subset data/subsets/subset_5k_armored.jsonl --workers 2 --seed 42

# gemma2:9b  (9B → workers=4)
chaosbench eval --provider ollama --model gemma2:9b \
    --subset data/subsets/subset_5k_armored.jsonl --workers 4 --seed 42

# mistral:7b  (7B → workers=4, one retry)
chaosbench eval --provider ollama --model mistral:7b \
    --subset data/subsets/subset_5k_armored.jsonl --workers 4 --seed 42 --retries 1

# deepseek-r1:14b  (reasoning model, longer output)
chaosbench eval --provider ollama --model deepseek-r1:14b \
    --subset data/subsets/subset_5k_armored.jsonl --workers 2 --seed 42 --max-tokens 256
```

### P2 — Full canonical (top models)

```bash
chaosbench eval --provider ollama --model qwen2.5:14b --dataset canonical --workers 2 --seed 42
chaosbench eval --provider ollama --model gemma2:9b   --dataset canonical --workers 4 --seed 42
```

### P3 — qwen2.5:32b (if 32 GB VRAM available)

```bash
chaosbench eval --provider ollama --model qwen2.5:32b --dataset canonical --workers 1 --seed 42
```

### P4 — Family diagnostics

```bash
for family in adversarial atomic consistency_paraphrase cross_indicator \
    extended_systems fol_inference indicator_diagnostics multi_hop \
    perturbation_robustness regime_transition; do
  chaosbench eval --provider ollama --model qwen2.5:14b \
    --subset data/subsets/subset_family_suites/${family}.jsonl --workers 2 --seed 42
done
```

### Post-run (run after every completed run)

```bash
chaosbench publish-run --run runs/<run_id>
chaosbench analyze-runs --runs-dir published_results/runs --out-dir artifacts/runs_audit
```

---

## 4. Acceptance Criteria

Every paper-quality run must satisfy all of the following before being included in analysis:

| Criterion | Check |
|-----------|-------|
| SHA match | `run_manifest.json.dataset_global_sha256` matches current `chaosbench freeze` output |
| Prompt hash | `run_manifest.json.prompt_hash` matches `get_prompt_hash()` for this codebase commit |
| Zero integrity errors | `metrics.json.invalid_rate` ≤ 0.05 (coverage ≥ 0.95) |
| Coverage reported | `metrics.json.coverage` present and > 0 |
| Published | `chaosbench publish-run` executed; artifact in `published_results/runs/` |

Verification command:
```bash
python scripts/freeze_v2_dataset.py   # re-compute SHA → compare with manifest
```

---

## 5. Interpreting Results in a Paper

### Headline metrics

Report **balanced_accuracy** and **MCC** as the two primary metrics:

- `balanced_accuracy = (TPR + TNR) / 2` — insensitive to class imbalance; use as headline accuracy.
- `MCC` (Matthews Correlation Coefficient) — a single number summarising all four confusion matrix cells. Ranges –1 to +1; random = 0; chance on balanced data ≈ 0.

Do **not** use raw accuracy on the full dataset as the headline — it is dominated by the majority class.

### Confidence intervals for small N

For subset runs (N ≤ 1,000), report **Wilson score confidence intervals** at 95%:

```python
from scipy.stats import norm
z = norm.ppf(0.975)   # 1.96
p = balanced_accuracy
n = valid_count
lo = (p + z**2/(2*n) - z * ((p*(1-p)/n + z**2/(4*n**2))**0.5)) / (1 + z**2/n)
hi = (p + z**2/(2*n) + z * ((p*(1-p)/n + z**2/(4*n**2))**0.5)) / (1 + z**2/n)
```

### Bias note (TPR vs TNR)

Always report **TPR** and **TNR** separately alongside balanced_accuracy to surface directional bias (e.g. a model that always predicts TRUE achieves TPR=1, TNR=0, bal_acc=0.5, MCC=0).

### `regime_transition` caveat

The `regime_transition` family has only 68 items in the canonical dataset and up to 8 items in the 5k armored subset. Per-family results for this family have wide confidence intervals; annotate accordingly in tables.

---

## 6. Commit Boundaries

| PR / Commit | Contents |
|-------------|----------|
| `fix(hygiene)` | `git rm --cached` for artifacts/ and runs/ only; no code changes |
| `feat(data)` | `cli.py` subset wrapper + generated subset files + `tests/test_armored_subsets.py` |
| `feat(eval)` | `run.py` shuffle fields + `cli.py` `--shuffle-seed` + `tests/test_eval_shuffle.py` |
| `feat(quality)` | `pre_freeze_check.py` HARD degeneracy check + `docs/DATASET.md` note |
| `docs` | `docs/RUN_PLAN.md` + `artifacts/repo_cleanup/final_push_readiness.md` |

**Rule**: never commit `runs/`, `artifacts/paper_assets/`, or `artifacts/runs_audit/` — these are gitignored generated outputs.
