# Published Runs — ChaosBench-Logic v2

Auto-generated index. Edit via `chaosbench publish-run`.

| Run ID | Provider | N | Acc_valid | Bal_acc | MCC | Date |
|--------|----------|---|-----------|---------|-----|------|
| `20260219T193151Z_ollama_qwen2.5:7b` | ollama/qwen2.5:7b | 1,000 | 0.6250 | 0.6243 | 0.2680 | 2026-02-19 |
| `20260219T193320Z_ollama_llama3.1:8b` | ollama/llama3.1:8b | 1,000 | 0.5990 | 0.5978 | 0.2497 | 2026-02-19 |
| `20260219T193425Z_ollama_qwen2.5:14b` | ollama/qwen2.5:14b | 1,000 | 0.6980 | 0.6977 | 0.4016 | 2026-02-19 |
| `20260220T104105Z_ollama_llama3.1:8b` | ollama/llama3.1:8b | 40,886 | 0.6017 | 0.5988 | 0.2400 | 2026-02-20 |
| `20260220T132350Z_ollama_qwen2.5:14b` | ollama/qwen2.5:14b | 3,828 | 0.7471 | 0.7472 | 0.5036 | 2026-02-20 |
| `20260220T132439Z_ollama_gemma2:9b` | ollama/gemma2:9b | 3,828 | 0.6750 | 0.6750 | 0.3504 | 2026-02-20 |
| `20260220T132658Z_ollama_mistral:7b` | ollama/mistral:7b | 3,828 | 0.6382 | 0.6385 | 0.2778 | 2026-02-20 |

## Artifact Contents

Each run directory contains:
- `run_manifest.json` — run metadata, dataset SHA256, config
- `metrics.json` — aggregate metrics (coverage, accuracy, MCC, per-family)
- `summary.md` — markdown summary table
- `publish_receipt.json` — publish provenance
- `predictions_subset.jsonl.gz` — compressed predictions (subset runs only)

See `docs/RUNS_POLICY.md` for the full policy.
