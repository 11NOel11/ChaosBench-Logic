# Contributing to ChaosBench-Logic

Thank you for your interest in contributing to ChaosBench-Logic! This document provides setup instructions and contribution guidelines.

## Table of Contents
- [Quick Start (Recommended: uv)](#quick-start-recommended-uv)
- [Alternative Setup Methods](#alternative-setup-methods)
- [Running Evaluations](#running-evaluations)
- [Contributing Code](#contributing-code)
- [Adding New Models](#adding-new-models)
- [Reporting Issues](#reporting-issues)

---

## Quick Start (Recommended: uv)

We recommend using **[uv](https://docs.astral.sh/uv/)** by Astral - a blazing fast Rust-based Python package manager.

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip
pip install uv
```

### 2. Clone the Repository

```bash
git clone https://github.com/11NOel11/ChaosBench-Logic.git
cd ChaosBench-Logic
```

### 3. Setup Environment

```bash
# Create virtual environment and install dependencies (automatically!)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install runtime dependencies from pyproject.toml
uv sync

# For development (includes pytest, pytest-cov):
uv sync --all-groups
```

**💡 Quick uv Commands:**
```bash
# Add a new package
uv add package-name

# Run Python scripts with uv (without activating venv)
uv run python scripts/run_benchmark.py --model gpt4 --mode zeroshot

# Update dependencies
uv pip install --upgrade package-name

# Lock dependencies
uv pip freeze > requirements.txt
```

### 4. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your favorite editor
```

See [API_SETUP.md](API_SETUP.md) for detailed instructions on obtaining API keys.

### 5. Run Your First Evaluation

```bash
# Test with a single model
uv run python scripts/run_benchmark.py --model gpt4 --mode zeroshot

# Or run all models
uv run python scripts/run_benchmark.py --model all --mode zeroshot
```

---

## Alternative Setup Methods

### Option 1: Python venv (Standard Library)

```bash
# Clone repository
git clone https://github.com/11NOel11/ChaosBench-Logic.git
cd ChaosBench-Logic

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Edit .env with your keys
```

### Option 2: Conda

```bash
# Clone repository
git clone https://github.com/11NOel11/ChaosBench-Logic.git
cd ChaosBench-Logic

# Create conda environment
conda create -n chaosbench python=3.11
conda activate chaosbench

# Install dependencies
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Edit .env with your keys
```

### Option 3: pip (Global Install - Not Recommended)

```bash
# Clone repository
git clone https://github.com/11NOel11/ChaosBench-Logic.git
cd ChaosBench-Logic

# Install dependencies globally
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Edit .env with your keys
```

---

## Running Evaluations

### Basic Usage

```bash
# Single model, single mode
uv run python scripts/run_benchmark.py --model gpt4 --mode zeroshot

# Single model, both modes (zeroshot + chain-of-thought)
uv run python scripts/run_benchmark.py --model claude3 --mode both

# All models, single mode
uv run python scripts/run_benchmark.py --model all --mode zeroshot
```

### Advanced Options

```bash
# Control parallelism (useful for rate limits)
uv run python scripts/run_benchmark.py --model llama3 --mode zeroshot --workers 2

# Clear checkpoints and restart from scratch
uv run python scripts/run_benchmark.py --model gemini --mode cot --clear-checkpoints

# Enable detailed debug output
uv run python scripts/run_benchmark.py --model mixtral --mode zeroshot --debug
```

### Supported Models

| Model | ID | Provider | Recommended Workers |
|-------|-----|----------|-------------------|
| GPT-4 | `gpt4` | OpenAI | 4-5 |
| Claude-3.5 | `claude3` | Anthropic | 4 |
| Gemini-2.5 | `gemini` | Google | 6-8 |
| LLaMA-3 70B | `llama3` | HuggingFace | 2 (rate limits) |
| Mixtral | `mixtral` | HuggingFace | 2-4 |
| OpenHermes | `openhermes` | HuggingFace | 2-4 |

### Performance Notes

⚠️ **Performance varies significantly across models and depends on:**
- API provider infrastructure and current load
- Network latency and connectivity
- Worker configuration (`--workers` parameter)
- Prompt mode (CoT generates longer responses than zeroshot)
- Rate limiting and retry logic

**Recommendations for stable evaluation:**
- **LLaMA-3 70B**: Use `--workers 2` (higher counts may hit rate limits)
- **GPT-4, Claude-3.5, Gemini**: Can use higher worker counts (4-8)
- **Mixtral, OpenHermes**: Start with `--workers 2-4`

**Timing is environment-dependent** and not reported in published results to avoid misleading comparisons across different infrastructure setups.

---

## Contributing Code

### Setting Up for Development

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/ChaosBench-Logic.git
   cd ChaosBench-Logic
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test thoroughly
5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** on GitHub

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular
- Test your changes before submitting

### Testing Your Changes

```bash
# Run a quick test with a single model
uv run python scripts/run_benchmark.py --model gpt4 --mode zeroshot

# Check results are generated correctly
cat results/gpt4_zeroshot/summary.json
```

---

## Adding New Models

To add support for a new LLM:

1. **Add adapter class** in `chaosbench/models/adapters/`:
   ```python
   class YourModelClient(Client):
       def __init__(self):
           # Initialize API client
           
       def call(self, prompt: str, **kwargs) -> str:
           # Implement API call
           # Return model response as string
   ```

2. **Register in adapter factory** in `chaosbench/models/adapters/__init__.py`:
   ```python
   def make_model_client(config: ModelConfig) -> Client:
       if config.name == "yourmodel":
           return YourModelClient()
       # ... existing models
   ```

3. **Add to legacy runner aliases** in `scripts/run_benchmark.py` (if needed):
   ```python
   SUPPORTED_MODELS = ["gpt4", "claude3", "gemini", "llama3", 
                       "mixtral", "openhermes", "yourmodel"]
   ```

4. **Test your model**:
   ```bash
   uv run python scripts/run_benchmark.py --model yourmodel --mode zeroshot
   ```

5. **Submit a pull request** with:
   - Model implementation
   - Documentation in README.md
   - Example results

---

## Reporting Issues

Found a bug? Have a feature request?

1. **Check existing issues** on GitHub
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Relevant logs or error messages

---

## Project Structure

```
ChaosBench-Logic/
├── chaosbench/             # Core package (eval, logic, tasks, repair)
├── scripts/                # Build/eval/audit operational scripts
├── data/                   # Canonical dataset and manifests
├── systems/                # Dynamical system definitions
├── published_results/      # Published run artifacts
├── docs/                   # Canonical documentation
├── .env.example            # API key template
├── pyproject.toml          # uv/pip package config
├── requirements.txt        # pip compatibility dependencies
└── README.md               # Main documentation
```

---

## Questions?

- **Documentation**: See [README.md](README.md)
- **API Setup**: See [API_SETUP.md](API_SETUP.md)
- **Results**: See [RESULTS.md](RESULTS.md)
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions

Thank you for contributing! 🚀
