# ChaosBench-Logic — Runs Storage Policy

_Last updated: 2026-02-20_

---

## 1. What Is an "Official" Run?

A run is **official** when all of the following hold:

| Criterion | Required value |
|-----------|---------------|
| `canonical_selector` | `data/canonical_v2_files.json` |
| `dataset_global_sha256` | matches `artifacts/freeze/v2_freeze_manifest.json → global_sha256` |
| `prompt_hash` | matches current `chaosbench/eval/prompts.py → get_prompt_hash()` |
| `max_items` | `null` (full dataset) **or** a documented subset size |
| `git_commit` | traceable to a tagged commit on `master` |

Runs that fail any criterion are labeled **exploratory** and must not appear
in paper tables or leaderboard entries.

> **Note on SHA matching**: `dataset_global_sha256` is now computed by
> `chaosbench.data.hashing.dataset_global_sha256` (formula:
> `sha256(path:file_sha:line_count\n)` over sorted files). Both the eval
> runner and the freeze script use this module, so their hashes agree.

---

## 2. Where Runs Are Stored

### 2a. Live run data (gitignored)

```
runs/
  <run_id>/
    run_manifest.json     — metadata + SHA + git commit
    predictions.jsonl     — per-item model outputs (~18 MB for full run)
    metrics.json          — aggregate metrics
    summary.md            — markdown summary
    .eval_checkpoint.jsonl — resume checkpoint (deleted on completion)
```

- `runs/` is in `.gitignore` and is never committed directly.
- The full-scale llama3.1:8b run under `runs/llama31_8b_full/` was committed
  during initial development; it should be `git rm --cached` before the next PR.

### 2b. Published artifacts (tracked, lightweight)

```
published_results/
  runs/
    <run_id>/
      run_manifest.json
      metrics.json
      summary.md
      publish_receipt.json
      [predictions_subset.jsonl.gz]  — subset runs only
    README.md             — auto-generated index
```

Only published artifacts are committed to the repository.
They contain no raw model completions (for full runs) to keep the repo lightweight.

---

## 3. Publishing a Run

Use the `chaosbench publish-run` command:

```bash
# Publish a full run (no predictions)
chaosbench publish-run --run runs/20260220T104105Z_ollama_llama3.1:8b

# Publish a 1k subset run with compressed predictions
chaosbench publish-run \
    --run runs/20260219T193151Z_ollama_qwen2.5:7b \
    --compress-predictions

# Publish to a custom destination
chaosbench publish-run \
    --run runs/20260220T104105Z_ollama_llama3.1:8b \
    --out published_results/runs/llama31_8b_canonical_v2 \
    --force
```

After publishing, `published_results/runs/README.md` is automatically regenerated.

---

## 4. Dataset Fingerprinting

The canonical dataset fingerprint is computed by
**`chaosbench.data.hashing.dataset_global_sha256`**:

```python
from chaosbench.data.hashing import dataset_global_sha256
sha = dataset_global_sha256(Path("data/canonical_v2_files.json"))
```

Formula:
```
sha256(
    concat over sorted(canonical_files) of:
        "<rel_path>:<file_sha256>:<line_count>\n"
)
```

The `:line_count` component binds the hash to the exact number of dataset
rows, not just raw bytes.  This formula is used by **both**:
- `scripts/freeze_v2_dataset.py` (produces `artifacts/freeze/v2_freeze_manifest.json`)
- `chaosbench/eval/run.py` (stored in `run_manifest.json → dataset_global_sha256`)

---

## 5. Resume / Checkpointing

The eval runner writes a per-item checkpoint file
(`.eval_checkpoint.jsonl`) in the run directory every `_CHECKPOINT_INTERVAL`
items.  To resume an interrupted run:

```bash
chaosbench eval \
    --provider ollama \
    --model llama3.1:8b \
    --dataset canonical \
    --resume 20260220T104105Z_ollama_llama3.1:8b
```

On completion the checkpoint file is deleted.

---

## 6. Worker Auto-Selection

If `--workers` is not specified (defaults to 1), the CLI automatically selects
a safe parallelism level based on model size:

| Model parameter count | Default workers |
|-----------------------|----------------|
| ≤ 8B  | 4 |
| 14B – 13B | 2 |
| ≥ 30B | 2 |

Override with `--workers N` at any time.

---

## 7. Audit and Validation

After every official run:

```bash
python scripts/analyze_runs.py --runs_dir runs --out_dir artifacts/runs_audit
# or
chaosbench analyze-runs
```

This produces `artifacts/runs_audit/RUNS_AUDIT.md` with:
- SHA reconciliation verdict (OFFICIAL / EXPLORATORY)
- Bias & confusion matrix (LABEL-BIASED / OK)
- Per-family metrics with Wilson CIs for N < 100
- What-could-have-gone-wrong checklist

Paper tables are written to `artifacts/paper_assets/`.

---

## 8. Commit Checklist for a New Official Run

1. `chaosbench eval ...` — run evaluation
2. Verify `artifacts/runs_audit/RUNS_AUDIT.md` shows SHA = OFFICIAL
3. `chaosbench publish-run --run runs/<id>` — publish lightweight artifacts
4. `git add published_results/runs/<id>/` — stage artifacts
5. `git commit -m "results(<model>): add official v2 canonical run"`
6. Open PR; link to `RUNS_AUDIT.md` section in PR description
