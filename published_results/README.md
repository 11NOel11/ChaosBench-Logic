# Published Results — ChaosBench-Logic

This directory contains curated, lightweight evaluation artifacts suitable for git tracking.

## v2 Runs (OFFICIAL)

Published v2 evaluation runs are under **`runs/`** (see `runs/README.md` for index).
These runs used the frozen canonical dataset (40,886 questions, freeze SHA `cfcfcc739988…`).

| Run | Provider | N | Bal_acc | MCC | Notes |
|-----|----------|---|---------|-----|-------|
| [20260220T104105Z_ollama_llama3.1:8b](runs/20260220T104105Z_ollama_llama3.1:8b/) | ollama/llama3.1:8b | 40,886 | 0.5988 | 0.2400 | Full canonical |
| [20260220T132439Z_ollama_gemma2:9b](runs/20260220T132439Z_ollama_gemma2:9b/) | ollama/gemma2:9b | 3,828 | 0.6750 | 0.3504 | 5k subset |
| [20260220T132658Z_ollama_mistral:7b](runs/20260220T132658Z_ollama_mistral:7b/) | ollama/mistral:7b | 3,828 | 0.6385 | 0.2778 | 5k subset |
| [20260220T132350Z_ollama_qwen2.5:14b](runs/20260220T132350Z_ollama_qwen2.5:14b/) | ollama/qwen2.5:14b | 3,828 | 0.7472 | 0.5036 | 5k subset |
| [20260219T193425Z_ollama_qwen2.5:14b](runs/20260219T193425Z_ollama_qwen2.5:14b/) | ollama/qwen2.5:14b | 1,000 | 0.6977 | 0.4016 | 1k subset |
| [20260219T193151Z_ollama_qwen2.5:7b](runs/20260219T193151Z_ollama_qwen2.5:7b/) | ollama/qwen2.5:7b | 1,000 | 0.6243 | 0.2680 | 1k subset |
| [20260219T193320Z_ollama_llama3.1:8b](runs/20260219T193320Z_ollama_llama3.1:8b/) | ollama/llama3.1:8b | 1,000 | 0.5978 | 0.2497 | 1k subset |

> **Primary metrics**: `balanced_accuracy` and `MCC`. See `docs/RUNS_POLICY.md`.

## v1 Results (ARCHIVED — NOT COMPARABLE TO v2)

The v1 results (GPT-4, Claude-3, Gemini, LLaMA-3 on the 621-item v1 dataset) have been
moved to **`archive_v1/`**. They MUST NOT be combined with v2 results in any table or figure.
See `archive_v1/README.md` for explanation.

## Artifact Structure

Each published run contains:
- `run_manifest.json` — run metadata and dataset SHA256
- `metrics.json` — aggregate metrics (coverage, balanced_accuracy, MCC, per_family)
- `summary.md` — human-readable summary table
- `publish_receipt.json` — publish provenance
