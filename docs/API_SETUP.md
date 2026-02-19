# API Keys Setup Guide

This guide explains how to obtain and configure API keys for ChaosBench-Logic.

## Required API Keys

ChaosBench-Logic supports multiple LLM models across 4 providers. You only need keys for the models you want to test.

**Models with published results:** GPT-4, Claude-3.5, Gemini-2.5, LLaMA-3 70B
**Additional supported models (code only):** Mixtral, OpenHermes

| Provider | Models | Required For |
|----------|--------|--------------|
| OpenAI | GPT-4 | `--model gpt4` |
| Anthropic | Claude-3.5 | `--model claude3` |
| Google | Gemini-2.5 | `--model gemini` |
| HuggingFace | LLaMA-3, Mixtral, OpenHermes | `--model llama3`, `mixtral`, `openhermes` |

---

## Step-by-Step Setup

### 1. Create Environment File

```bash
cp .env.example .env
```

This creates a `.env` file that will store your API keys. **Never commit this file to Git!**

### 2. Obtain API Keys

#### OpenAI (GPT-4)

1. Go to: https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

**Cost**: Variable (depends on API pricing, prompt length, and model version)

#### Anthropic (Claude-3.5)

1. Go to: https://console.anthropic.com/settings/keys
2. Sign up or log in
3. Click "Create Key"
4. Copy the key (starts with `sk-ant-`)
5. Add to `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
   ```

**Cost**: Variable (depends on API pricing)

#### Google (Gemini-2.5)

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key
5. Add to `.env`:
   ```
   GOOGLE_API_KEY=your-actual-key-here
   ```

**Cost**: Free tier available (check Google AI Studio for current limits)

#### HuggingFace (LLaMA-3, Mixtral, OpenHermes)

1. Go to: https://huggingface.co/settings/tokens
2. Sign up or log in
3. Click "New token"
4. Select "Read" access
5. Copy the token (starts with `hf_`)
6. Add to `.env`:
   ```
   HF_API_KEY=hf_your-actual-key-here
   ```

**Important**:
- HuggingFace requires **credits** for inference API
- Free tier: Limited requests
- LLaMA-3 70B requires significant credits (check current pricing)
- Add credits at: https://huggingface.co/settings/billing

---

## Alternative: Export Environment Variables

Instead of using `.env` file, you can export environment variables:

### macOS/Linux:
```bash
export OPENAI_API_KEY="sk-your-key"
export ANTHROPIC_API_KEY="sk-ant-your-key"
export GOOGLE_API_KEY="your-key"
export HF_API_KEY="hf_your-key"
```

### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-your-key"
$env:ANTHROPIC_API_KEY="sk-ant-your-key"
$env:GOOGLE_API_KEY="your-key"
$env:HF_API_KEY="hf_your-key"
```

---

## Verify Setup

Test that your API keys are working:

```bash
# Test GPT-4
python -c "import os; from clients import OpenAIClient; c=OpenAIClient('gpt4'); print(c.call('Hello'))"

# Test Claude-3.5
python -c "import os; from clients import ClaudeClient; c=ClaudeClient(); print(c.call('Hello'))"

# Test Gemini
python -c "import os; from clients import GeminiClient; c=GeminiClient(); print(c.call('Hello'))"

# Test LLaMA-3
python -c "import os; from clients import HFaceClient; c=HFaceClient('llama3'); print(c.call('Hello'))"
```

If any test fails, check:
1. API key is correct in `.env`
2. API key has proper permissions
3. Account has available credits (for paid APIs)

---

## Security Best Practices

### ✅ DO:
- Keep `.env` file in `.gitignore` (already configured)
- Use separate API keys for different projects
- Rotate keys regularly
- Monitor API usage and costs
- Use read-only keys when possible

### ❌ DON'T:
- Commit API keys to Git
- Share keys publicly
- Hardcode keys in source files
- Use production keys for testing
- Share `.env` file

---

## Troubleshooting

### "API key not found" error

**Solution**: Ensure `.env` file exists and contains your keys:
```bash
cat .env  # Check file contents
```

### "Invalid API key" error

**Solution**: 
1. Verify key is copied correctly (no extra spaces)
2. Check key hasn't expired
3. Ensure account is active and has credits

### "Rate limit exceeded" error

**Solution**:
- Reduce `--workers` count: `--workers 2`
- Wait a few minutes and retry
- Check API usage dashboard
- Upgrade to higher tier if needed

### HuggingFace "402 Payment Required" error

**Solution**:
1. Go to: https://huggingface.co/settings/billing
2. Add credits to your account
3. Wait 5-10 minutes for activation
4. Retry evaluation

---

## Cost Considerations

API costs for running the full benchmark vary significantly based on dataset size and configuration:
- **Current API pricing** (changes frequently - check provider websites)
- **Prompt mode** (chain-of-thought generates longer outputs than zero-shot)
- **Model version** (different versions have different pricing tiers)
- **Worker configuration** (parallel workers may hit rate limits, increasing retries)

**Estimation approach:**
1. Check current pricing at provider dashboards
2. Estimate ~100-200 tokens input per question (varies by prompt mode)
3. Estimate ~50-500 tokens output (zeroshot: ~50, CoT: ~200-500)
4. Multiply by number of questions (621 for v1 baseline, 41,507 for full v2)

**Cost-saving tips:**
- Start with free tiers (Gemini) or cheaper models
- Run zeroshot first (faster, cheaper) before CoT
- Use `--workers 2` for expensive models to avoid rate limit retries
- Monitor API dashboards during evaluation

---

## Questions?

- **Can't get a specific API key?** You can still run other models
- **Cost concerns?** Start with Gemini (cheapest) or use free tiers
- **Academic use?** Check if providers offer research credits
- **Issues?** Open an issue on GitHub

---

## Next Steps

Once your API keys are configured:

1. **Run a test**: `python run_benchmark.py --model gpt4 --mode zeroshot`
2. **Check results**: `cat results/gpt4_zeroshot/summary.json`
3. **Run full benchmark**: `python run_benchmark.py --model all --mode both`

See [README.md](README.md) for usage instructions and [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
