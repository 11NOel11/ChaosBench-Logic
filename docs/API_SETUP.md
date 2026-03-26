# API Setup

This guide covers provider credentials for running evaluations in
ChaosBench-Logic v2.

## Scope

- Preferred evaluation path: `chaosbench eval` (provider-based runtime).
- Legacy compatibility path: `scripts/run_benchmark.py` (older model alias runner).

## Provider Environment Variables

Set only the keys you need for the providers you plan to run.

| Provider | CLI Flag | Example Model | Required Environment Variable |
|----------|----------|---------------|-------------------------------|
| OpenAI | `--provider openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `--provider anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Gemini | `--provider gemini` | `gemini-2.0-flash` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| DeepSeek | `--provider deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| OpenRouter | `--provider openrouter` | `google/gemini-2.5-flash` | `OPENROUTER_API_KEY` |
| Groq | `--provider groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Ollama (local) | `--provider ollama` | `qwen2.5:7b` | none |

Legacy runner only (`scripts/run_benchmark.py`) may also require `HF_API_KEY`
for Hugging Face-hosted model aliases (`llama3`, `mixtral`, `openhermes`).

## Recommended Setup

1. Copy environment template:

```bash
cp .env.example .env
```

2. Fill in required keys in `.env`.

3. Export variables into your current shell before running evaluations:

```bash
set -a
source .env
set +a
```

## Smoke Tests

Start with a zero-cost local sanity check:

```bash
uv run chaosbench eval --provider mock --dataset canonical --max-items 5
```

Then test one provider at a time:

```bash
uv run chaosbench eval --provider openai --model gpt-4o-mini --dataset canonical --max-items 5
uv run chaosbench eval --provider anthropic --model claude-sonnet-4-6 --dataset canonical --max-items 5
uv run chaosbench eval --provider gemini --model gemini-2.0-flash --dataset canonical --max-items 5
```

Optional providers:

```bash
uv run chaosbench eval --provider deepseek --model deepseek-chat --dataset canonical --max-items 5
uv run chaosbench eval --provider openrouter --model google/gemini-2.5-flash --dataset canonical --max-items 5
uv run chaosbench eval --provider groq --model llama-3.3-70b-versatile --dataset canonical --max-items 5
uv run chaosbench eval --provider ollama --model qwen2.5:7b --dataset canonical --max-items 5
```

## Troubleshooting

- Missing key error: verify variable name and shell export state.
- 401/403 errors: key is invalid, expired, or lacks required access.
- 429 errors: reduce concurrency (`--workers 1` or `--workers 2`) and retry.
- Empty/invalid outputs: keep `--temperature 0.0`, use strict parsing defaults.
- Ollama connection error: ensure local server is running (`ollama serve`).

## Security Practices

- Never commit `.env`.
- Use distinct keys per environment/project.
- Rotate provider keys periodically.
- Monitor provider dashboards for usage and spend.

## Related Docs

- `docs/CONTRIBUTING.md`
- `docs/EVAL_PROTOCOL.md`
- `README.md`
