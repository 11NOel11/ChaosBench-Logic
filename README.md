<div align="center">

# ChaosBench-Logic

### A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![GitHub Stars](https://img.shields.io/github/stars/11NOel11/chaos-logic-bench?style=social)](https://github.com/11NOel11/chaos-logic-bench)

[**Paper**](https://arxiv.org/) | [**Dataset**](data/) | [**Results**](RESULTS.md) | [**Documentation**](API_SETUP.md)

</div>

---

## 📋 Abstract

**ChaosBench-Logic** is a comprehensive benchmark designed to evaluate the reasoning capabilities of Large Language Models (LLMs) in the context of chaotic and non-chaotic dynamical systems. The benchmark tests models' abilities across multiple dimensions of complex reasoning: logical inference, symbolic manipulation, multi-hop reasoning, cross-system comparison, and counterfactual analysis. We evaluate **6 state-of-the-art LLMs** on **621 carefully curated questions** spanning **30 dynamical systems** from physics, chemistry, biology, and mathematics.

Our findings reveal that while modern LLMs achieve impressive accuracy (up to 91.6%), they exhibit varying strengths across different reasoning tasks, with notable challenges in compositional reasoning and certain types of logical implications.

---

## 🎯 Key Features

<div align="center">

| Feature | Description |
|---------|-------------|
| **📊 621 Questions** | Spanning 7 difficulty levels from atomic facts to counterfactual reasoning |
| **🔬 30 Systems** | Lorenz-63, double pendulum, Brusselator, FitzHugh-Nagumo, logistic map, and more |
| **🧠 6 LLMs** | GPT-4, Claude-3.5, Gemini-2.5, LLaMA-3 70B, Mixtral, OpenHermes |
| **🎲 11 Predicates** | Stability, chaos, bifurcations, periodicity, sensitivity, and more |
| **🔄 2 Modes** | Zero-shot and chain-of-thought reasoning |
| **📈 Rich Metrics** | Overall accuracy, dialogue accuracy, task-specific breakdowns, bias analysis |

</div>

---

## 📊 Main Results

<div align="center">

### Performance Summary

| Rank | Model | Mode | Overall Acc | Dialogue Acc | Throughput* | Valid |
|:----:|-------|:----:|:-----------:|:------------:|:-----------:|:-----:|
| 🥇 | **LLaMA-3 70B** | Zero-shot | **91.6%** | **75.5%** | 1.2 items/s | 620/621 |
| 🥈 | **GPT-4** | CoT | **90.2%** | **73.7%** | ~10 items/s | 621/621 |
| 🥉 | **GPT-4** | Zero-shot | **90.0%** | **72.8%** | ~15 items/s | 621/621 |
| 4 | **LLaMA-3 70B** | CoT | **89.5%** | **65.3%** | 0.2 items/s | 620/621 |
| 5 | **Claude-3.5** | Zero-shot | **88.2%** | **68.3%** | ~12 items/s | 621/621 |
| 6 | **Gemini-2.5** | Zero-shot | **87.9%** | **67.6%** | ~18 items/s | 620/621 |

*Throughput measured with 2 parallel workers (practical deployment scenario)

</div>

**Key Findings:**
- 🏆 **LLaMA-3 70B** achieves highest overall accuracy and best dialogue consistency
- ⚡ **Gemini-2.5** offers best speed-to-accuracy ratio
- 🎯 **GPT-4** shows strong chain-of-thought reasoning capabilities
- 📈 All models achieve >85% accuracy, demonstrating strong logical reasoning capabilities

> **Note:** LLaMA-3 70B requires longer inference time (~8 min for zero-shot, ~55 min for CoT) compared to other models (~2-5 min per run).

See [**RESULTS.md**](RESULTS.md) for comprehensive analysis and task-specific breakdowns.

---

## 🚀 Quick Start

### Installation

We recommend using **[uv](https://docs.astral.sh/uv/)** (a fast Rust-based Python package manager):

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/11NOel11/chaos-logic-bench.git
cd chaos-logic-bench

# Setup environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

<details>
<summary><b>Alternative: Using pip or conda</b></summary>

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Using conda:**
```bash
conda create -n chaosbench python=3.11
conda activate chaosbench
pip install -r requirements.txt
```
</details>

### Configuration

```bash
# Setup API keys
cp .env.example .env
nano .env  # Add your API keys
```

See [**API_SETUP.md**](API_SETUP.md) for detailed instructions on obtaining API keys from OpenAI, Anthropic, Google, and HuggingFace.

### Running Evaluations

```bash
# Evaluate a single model
python run_benchmark.py --model gpt4 --mode zeroshot

# Run all models
python run_benchmark.py --model all --mode zeroshot

# Chain-of-thought reasoning
python run_benchmark.py --model claude3 --mode cot

# Custom worker count (for rate limiting)
python run_benchmark.py --model llama3 --mode zeroshot --workers 2
```

---

## 🧪 Testing & Quality Assurance

ChaosBench-Logic includes a comprehensive pytest test suite with **122 test cases** ensuring correctness of evaluation logic, FOL violation detection, and answer normalization.

### Running Tests

```bash
# Install pytest (in virtual environment)
source .venv/bin/activate
pip install pytest

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_normalization.py -v

# Run with coverage report
pytest --cov=eval_chaosbench --cov-report=html
```

### Test Coverage

<div align="center">

| Test Suite | Test Cases | Coverage |
|------------|:----------:|----------|
| **Answer Normalization** | 48 | 8-step CoT parsing cascade, TRUE/FALSE variants, edge cases |
| **FOL Rules & Ontology** | 46 | Axiom definitions, system loading, predicate extraction, violation detection |
| **Summary Metrics** | 21 | Accuracy, dialogue metrics, contradiction rate, FOL violations, bias error |
| **Integration Smoke Tests** | 7 | End-to-end evaluation flow, metric aggregation |
| **Total** | **122** | Comprehensive coverage of core evaluation logic |

</div>

### Key Quality Improvements (Phase 2)

**1. First-Order Logic (FOL) Violation Detection**

ChaosBench-Logic now tracks **logical consistency** in addition to simple contradictions:

- **`contradiction_rate`** (binary): Did the model contradict itself by giving both YES and NO for the same (system, task) pair?
- **`avg_violations_per_dialogue`** (granular): How many formal logic axioms were violated?

**Example:** If a model says "Chaotic=YES" but "Deterministic=NO", this violates the axiom **Chaotic → Deterministic** and counts as 1 FOL violation. The model may not contradict itself (give both YES and NO), but it still violates logical consistency.

**2. Improved Chain-of-Thought Parsing**

The `normalize_label()` function uses an 8-step cascade to robustly extract YES/NO from diverse model outputs:
- Standard `FINAL_ANSWER:` markers
- Chain-of-thought conclusions without explicit markers
- Revision patterns ("yes... actually no")
- TRUE/FALSE variants
- Markdown formatting

See `tests/test_normalization.py` for 48 test cases documenting all supported formats.

**3. Bug Fixes**

- **bias_error computation**: Fixed to correctly compute error rate per bias family (was previously undefined)
- **Dialogue grouping**: Single questions now correctly treated as length-1 dialogues for FOL violation checking

### Test Organization

```
tests/
├── __init__.py                    # Test suite documentation
├── test_normalization.py          # Answer extraction (48 tests)
├── test_fol_rules.py              # FOL logic (46 tests)
├── test_summary_metrics.py        # Metric computation (21 tests)
└── test_integration_smoke.py      # Integration tests (7 tests)
```

All tests use **synthetic data** and do not require API keys or external network access.

---

## 📂 Repository Structure

```
chaos-logic-bench/
├── 📄 run_benchmark.py        # Main evaluation runner
├── 📄 eval_chaosbench.py      # Core evaluation framework
├── 📄 clients.py              # LLM API client implementations
├── 📁 data/                   # Benchmark dataset (621 questions)
│   ├── batch1_atomic_implication.jsonl
│   ├── batch2_multiHop_crossSystem.jsonl
│   ├── batch3_pde_chem_bio.jsonl
│   ├── batch4_maps_advanced.jsonl
│   ├── batch5_counterfactual_high_difficulty.jsonl
│   ├── batch6_deep_bias_probes.jsonl
│   └── batch7_multiturn_advanced.jsonl
├── 📁 systems/                # 30 dynamical system definitions
│   ├── lorenz63.json, rossler.json, double_pendulum.json
│   ├── brusselator.json, fitzhugh_nagumo.json, ...
│   └── [27 more systems]
├── 📁 tests/                  # Pytest test suite (122 test cases)
│   ├── test_normalization.py  # Answer extraction tests (48)
│   ├── test_fol_rules.py      # FOL violation tests (46)
│   ├── test_summary_metrics.py # Metrics computation tests (21)
│   └── test_integration_smoke.py # Integration tests (7)
├── 📁 results/                # Evaluation outputs (auto-generated)
├── 📄 .env.example            # API key template
├── 📄 pyproject.toml          # Package configuration
└── 📚 Documentation
    ├── README.md              # This file
    ├── RESULTS.md             # Detailed evaluation results
    ├── API_SETUP.md           # API key setup guide
    └── CONTRIBUTING.md        # Contribution guidelines
```

---

## 🧪 Benchmark Design

### Task Categories

<div align="center">

| Category | Questions | Description |
|----------|:---------:|-------------|
| **Atomic Facts** | 109 | Basic properties: stability, chaos, dimension, periodicity |
| **Implications** | 93 | Logical consequences: if A then B |
| **Multi-hop Reasoning** | 98 | Chained logical inference across multiple facts |
| **Cross-system Comparison** | 87 | Relative properties between different systems |
| **PDE/Chem/Bio** | 76 | Domain-specific technical reasoning |
| **Counterfactual** | 68 | "What if" scenarios with parameter modifications |
| **Multi-turn Dialogue** | 90 | Contextual Q&A sequences |
| **Total** | **621** | Comprehensive reasoning evaluation |

</div>

### Dynamical Systems Coverage

Our benchmark spans diverse systems across multiple domains:

- **Classical Chaos**: Lorenz-63, Rössler, double pendulum, Duffing oscillator
- **Chemical Systems**: Brusselator, Oregonator
- **Biological Models**: FitzHugh-Nagumo, Hindmarsh-Rose, Lotka-Volterra
- **Maps**: Logistic map, Hénon map, standard map, Arnold cat map, baker's map
- **PDEs**: Kuramoto-Sivashinsky, sine-Gordon
- **Others**: Van der Pol, Chua circuit, Mackey-Glass, Lorenz-96

### Evaluation Metrics

Each run generates comprehensive analytics:
- ✅ **Overall Accuracy** - Correct predictions across all tasks
- 💬 **Dialogue Accuracy** - Multi-turn conversation consistency
- 📊 **Task-specific Accuracy** - Per-category performance breakdowns
- ⚖️ **Bias Analysis** - Response distribution patterns
- ⏱️ **Execution Metrics** - Throughput, latency, API success rates
- 📈 **Visual Analytics** - Heatmaps, error distributions, confusion matrices

Results are exported in **JSON**, **CSV**, and **PNG** formats for downstream analysis.

---

## 🔬 Supported Models

<div align="center">

| Model | Provider | Version | Speed | Cost/Run | Notes |
|-------|----------|---------|-------|----------|-------|
| GPT-4 | OpenAI | gpt-4-turbo | ⚡⚡⚡ | ~$2.00 | Fast, accurate |
| Claude-3.5 | Anthropic | claude-3-5-sonnet | ⚡⚡⚡ | ~$1.30 | Strong reasoning |
| Gemini-2.5 | Google | gemini-2.5-flash | ⚡⚡⚡⚡ | ~$0.50 | Fastest, cost-effective |
| LLaMA-3 70B | HuggingFace | Meta-Llama-3-70B | ⚡ | ~$6.00 | Best accuracy, slower |
| Mixtral | HuggingFace | Mixtral-8x7B | ⚡⚡ | ~$2.00 | Medium speed |
| OpenHermes | HuggingFace | OpenHermes-2.5 | ⚡⚡ | ~$1.30 | Medium speed |

</div>

---

## 📝 Citation

If you use ChaosBench-Logic in your research, please cite:

```bibtex
@article{chaosbench2024,
  title={ChaosBench-Logic: Benchmarking Large Language Models on Complex Reasoning about Dynamical Systems},
  author={ChaosBench Team},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024},
  url={https://github.com/11NOel11/chaos-logic-bench}
}
```

---

## 🤝 Contributing

We welcome contributions from the community! Areas for contribution:

- 🆕 Adding support for new LLM models
- 📊 Expanding the dataset with new questions or systems
- 🔧 Improving evaluation metrics and analysis tools
- 📖 Enhancing documentation
- 🐛 Bug fixes and performance improvements

See [**CONTRIBUTING.md**](CONTRIBUTING.md) for detailed guidelines on:
- Environment setup (uv, conda, pip, venv)
- Adding new models
- Code style and testing
- Pull request workflow

---

## 📚 Documentation

- **[API_SETUP.md](API_SETUP.md)** - Comprehensive guide for obtaining and configuring API keys
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development setup and contribution guidelines  
- **[RESULTS.md](RESULTS.md)** - Complete evaluation results with detailed analysis
- **[LICENSE](LICENSE)** - MIT License

---

## ❓ FAQ & Troubleshooting

<details>
<summary><b>API key errors?</b></summary>

See [API_SETUP.md#troubleshooting](API_SETUP.md#troubleshooting) for common API key issues and solutions.
</details>

<details>
<summary><b>Rate limit issues?</b></summary>

Reduce parallel workers: `python run_benchmark.py --model llama3 --mode zeroshot --workers 2`
</details>

<details>
<summary><b>Why is LLaMA-3 so slow?</b></summary>

LLaMA-3 70B has longer inference latency (~8-55 min vs 2-5 min for other models). This is expected for the model size and HuggingFace inference API. The accuracy gains may justify the tradeoff for research purposes.
</details>

<details>
<summary><b>Can I add my own models?</b></summary>

Yes! See [CONTRIBUTING.md#adding-new-models](CONTRIBUTING.md#adding-new-models) for step-by-step instructions.
</details>

<details>
<summary><b>Missing dependencies?</b></summary>

Reinstall: `pip install -r requirements.txt` or `uv pip install -e .`
</details>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Free for academic and commercial use with attribution.

---

## 🙏 Acknowledgments

This work builds upon:
- **LLM APIs**: [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Google AI](https://ai.google.dev/), [HuggingFace](https://huggingface.co/)
- **Dynamical Systems Theory**: Research from the chaos theory and nonlinear dynamics community
- **Benchmark Design**: Inspired by existing LLM reasoning benchmarks

Special thanks to the open-source community for tools and libraries that made this work possible.

---

<div align="center">

### 🌟 Star us on GitHub if you find this useful!

**Maintained by the ChaosBench Team**

[Report Bug](https://github.com/11NOel11/chaos-logic-bench/issues) · [Request Feature](https://github.com/11NOel11/chaos-logic-bench/issues) · [Discussions](https://github.com/11NOel11/chaos-logic-bench/discussions)

</div>
