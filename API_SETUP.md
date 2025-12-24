# API Keys Setup Guide

This guide explains how to obtain and configure API keys for ChaosBench-Logic.

## Required API Keys

ChaosBench-Logic supports 6 LLM models across 4 providers. You only need keys for the models you want to test.

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

**Cost**: ~$0.50-2.00 per full benchmark run (621 items)

#### Anthropic (Claude-3.5)

1. Go to: https://console.anthropic.com/settings/keys
2. Sign up or log in
3. Click "Create Key"
4. Copy the key (starts with `sk-ant-`)
5. Add to `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
   ```

**Cost**: ~$0.30-1.50 per full benchmark run

#### Google (Gemini-2.5)

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key
5. Add to `.env`:
   ```
   GOOGLE_API_KEY=your-actual-key-here
   ```

**Cost**: Free tier available, ~$0.10-0.50 per run

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
- LLaMA-3 70B is **slow and expensive** (~$2-5 per run)
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

## Cost Estimates

Full benchmark (621 items) estimated costs:

| Model | Zeroshot | Chain-of-Thought | Total (both) |
|-------|----------|------------------|--------------|
| GPT-4 | $0.50 | $1.50 | $2.00 |
| Claude-3.5 | $0.30 | $1.00 | $1.30 |
| Gemini-2.5 | $0.10 | $0.40 | $0.50 |
| LLaMA-3 70B | $2.00 | $4.00 | $6.00 |
| Mixtral | $0.50 | $1.50 | $2.00 |
| OpenHermes | $0.30 | $1.00 | $1.30 |

**Total for all models**: ~$12-15

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
