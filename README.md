<div align="center">

# ChaosBench-Logic

### A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-blue.svg)](LICENSE_DATA)
[![Tests](https://github.com/11NOel11/ChaosBench-Logic/actions/workflows/ci.yml/badge.svg)](https://github.com/11NOel11/ChaosBench-Logic/actions)
[![GitHub Stars](https://img.shields.io/github/stars/11NOel11/ChaosBench-Logic?style=social)](https://github.com/11NOel11/ChaosBench-Logic)

[**Dataset Card**](docs/DATASET.md) | [**Ontology**](docs/ONTOLOGY.md) | [**Results**](docs/RESULTS.md) | [**API Setup**](docs/API_SETUP.md) | [**Contributing**](docs/CONTRIBUTING.md)

</div>

---

## 📋 Abstract

**ChaosBench-Logic** is a comprehensive benchmark designed to evaluate the reasoning capabilities of Large Language Models (LLMs) in the context of chaotic and non-chaotic dynamical systems. The benchmark tests models' abilities across multiple dimensions of complex reasoning: logical inference, symbolic manipulation, multi-hop reasoning, cross-system comparison, and counterfactual analysis. We evaluate **6 state-of-the-art LLMs** on **621 carefully curated questions** spanning **30 dynamical systems** from physics, chemistry, biology, and mathematics.

Our findings reveal that while modern LLMs achieve impressive accuracy (up to 91.6%), they exhibit varying strengths across different reasoning tasks, with notable challenges in compositional reasoning and certain types of logical implications.

---

## 📊 Dataset Statistics

<div align="center">

| Metric | Count | Details |
|--------|-------|---------|
| **Total Questions** | 621 | Unique IDs: q0001 to q0621 |
| **Task Types** | 17 | Atomic, multi-hop, counterfactual, multi-turn, bias, cross-system, etc. |
| **Systems Used** | 27 | Chaotic, periodic, quasi-periodic, and stochastic systems |
| **Systems Defined** | 30 | 3 additional systems available for extension |
| **Multi-turn Dialogues** | 49 | Average 4.1 turns per dialogue (3-6 turns) |
| **Predicates per System** | 11 | Chaotic, Deterministic, Periodic, StrangeAttr, PosLyap, etc. |
| **Ground Truth Labels** | YES/NO, TRUE/FALSE | Plus special DISAPPEAR label for counterfactuals |

</div>

**Note:** 27 systems are actively used in the dataset; 3 systems (Chua circuit, damped oscillator, double pendulum) are defined but reserved for future batches.

See [**DATASET.md**](docs/DATASET.md) for complete schema documentation and [**ONTOLOGY.md**](docs/ONTOLOGY.md) for predicate definitions and FOL axioms.

---

## 🎯 Key Features

<div align="center">

| Feature | Description |
|---------|-------------|
| **📊 621 Questions** | 17 task types across 7 high-level categories of reasoning complexity |
| **🔬 27 Systems** | Lorenz-63, Brusselator, FitzHugh-Nagumo, logistic map, and more (30 defined) |
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

See [**RESULTS.md**](docs/RESULTS.md) for comprehensive analysis and task-specific breakdowns.

---

## 🚀 Quick Start

### Installation

We recommend using **[uv](https://docs.astral.sh/uv/)** (a fast Rust-based Python package manager):

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/11NOel11/ChaosBench-Logic.git
cd ChaosBench-Logic

# Setup environment with uv (creates virtualenv and installs dependencies)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .

# Install dev dependencies (pytest, etc.)
uv pip install -r requirements.txt
uv pip install --group dev
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

See [**docs/API_SETUP.md**](docs/API_SETUP.md) for detailed instructions on obtaining API keys from OpenAI, Anthropic, Google, and HuggingFace.

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
ChaosBench-Logic/
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
├── 📁 docs/                   # Documentation
│   ├── DATASET.md             # Dataset card and schema
│   ├── ONTOLOGY.md            # Predicate definitions and FOL axioms
│   ├── RESULTS.md             # Detailed evaluation results
│   ├── API_SETUP.md           # API key setup guide
│   └── CONTRIBUTING.md        # Contribution guidelines
├── 📁 scripts/                # Utility scripts
│   ├── dataset_stats.py       # Compute dataset statistics
│   ├── validate_repo_claims.py # Validate documentation claims
│   └── aggregate_results.py   # Generate results tables
├── 📄 README.md               # This file
├── 📄 .env.example            # API key template
├── 📄 pyproject.toml          # Package configuration
├── 📄 LICENSE                 # MIT License (code)
└── 📄 LICENSE_DATA            # CC BY 4.0 (dataset)
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

## 📝 How to Cite

If you use ChaosBench-Logic in your research, please cite:

```bibtex
@software{chaosbench2024,
  title={ChaosBench-Logic: A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems},
  author={Thomas, Noel},
  year={2024},
  url={https://github.com/11NOel11/ChaosBench-Logic},
  institution={Mohamed bin Zayed University of Artificial Intelligence}
}
```

You can also use the [CITATION.cff](CITATION.cff) file for automatic citation generation in GitHub.

---

## 🤝 Contributing

We welcome contributions from the community! Areas for contribution:

- 🆕 Adding support for new LLM models
- 📊 Expanding the dataset with new questions or systems
- 🔧 Improving evaluation metrics and analysis tools
- 📖 Enhancing documentation
- 🐛 Bug fixes and performance improvements

See [**docs/CONTRIBUTING.md**](docs/CONTRIBUTING.md) for detailed guidelines on:
- Environment setup (uv, conda, pip, venv)
- Adding new models
- Code style and testing
- Pull request workflow

---

## 📚 Documentation

- **[docs/DATASET.md](docs/DATASET.md)** - Complete dataset card with schema, statistics, and construction methodology
- **[docs/ONTOLOGY.md](docs/ONTOLOGY.md)** - Predicate definitions and first-order logic axioms
- **[docs/RESULTS.md](docs/RESULTS.md)** - Complete evaluation results with detailed analysis
- **[docs/API_SETUP.md](docs/API_SETUP.md)** - Comprehensive guide for obtaining and configuring API keys
- **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Development setup and contribution guidelines
- **[LICENSE](LICENSE)** - MIT License (code)
- **[LICENSE_DATA](LICENSE_DATA)** - CC BY 4.0 (dataset)

---

## ❓ FAQ & Troubleshooting

<details>
<summary><b>API key errors?</b></summary>

See [docs/API_SETUP.md#troubleshooting](docs/API_SETUP.md#troubleshooting) for common API key issues and solutions.
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

Yes! See [docs/CONTRIBUTING.md#adding-new-models](docs/CONTRIBUTING.md#adding-new-models) for step-by-step instructions.
</details>

<details>
<summary><b>Missing dependencies?</b></summary>

Reinstall: `pip install -r requirements.txt` or `uv pip install -e .`
</details>

---

## 📄 License

**Code:** This project's code is licensed under the [MIT License](LICENSE) - free for academic and commercial use with attribution.

**Dataset:** The benchmark dataset (data/ and systems/ directories) is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE_DATA). You are free to share and adapt the data with proper attribution.

When using this benchmark, please cite using the format in the [How to Cite](#-how-to-cite) section above.

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

**Author:** Noel Thomas (Mohamed bin Zayed University of Artificial Intelligence)

[Report Bug](https://github.com/11NOel11/ChaosBench-Logic/issues) · [Request Feature](https://github.com/11NOel11/ChaosBench-Logic/issues) · [Discussions](https://github.com/11NOel11/ChaosBench-Logic/discussions)

</div>
