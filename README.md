<div align="center">

# ChaosBench-Logic

### A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-blue.svg)](LICENSE_DATA)
[![Tests](https://github.com/11NOel11/ChaosBench-Logic/actions/workflows/ci.yml/badge.svg)](https://github.com/11NOel11/ChaosBench-Logic/actions)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-ChaosBench--Logic-orange.svg)](https://huggingface.co/datasets/11NOel11/ChaosBench-Logic)
[![GitHub Stars](https://img.shields.io/github/stars/11NOel11/ChaosBench-Logic?style=social)](https://github.com/11NOel11/ChaosBench-Logic)

[**🤗 HuggingFace Dataset**](https://huggingface.co/datasets/11NOel11/ChaosBench-Logic) | [**Dataset Card**](docs/DATASET.md) | [**Ontology**](docs/ONTOLOGY.md) | [**Results**](docs/RESULTS.md) | [**API Setup**](docs/API_SETUP.md) | [**Contributing**](docs/CONTRIBUTING.md)

</div>

---

## 📋 Abstract

**ChaosBench-Logic** is a comprehensive benchmark designed to evaluate the reasoning capabilities of Large Language Models (LLMs) in the context of chaotic and non-chaotic dynamical systems. The benchmark tests models' abilities across multiple dimensions of complex reasoning: logical inference, symbolic manipulation, multi-hop reasoning, cross-system comparison, and counterfactual analysis. We evaluate **multiple state-of-the-art LLMs** on **621 carefully curated questions** spanning **30 dynamical systems** (27 actively used, 3 reserved for extension) from physics, chemistry, biology, and mathematics.

Our findings reveal that while modern LLMs achieve impressive accuracy (up to 94.0%), they exhibit varying strengths across different reasoning tasks, with notable challenges in compositional reasoning and certain types of logical implications.

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
| **🧠 Multiple LLMs** | Evaluated: GPT-4, Claude-3.5, Gemini-2.5, LLaMA-3 70B • Supported: Mixtral, OpenHermes |
| **🎲 11 Predicates** | Stability, chaos, bifurcations, periodicity, sensitivity, and more |
| **🔄 2 Modes** | Zero-shot and chain-of-thought reasoning |
| **📈 Rich Metrics** | Overall accuracy, dialogue accuracy, task-specific breakdowns, FOL violations, bias analysis |

</div>

---

## 📊 Main Results

<div align="center">

### Performance Summary

| Rank | Model | Mode | Overall Acc | Dialogue Acc | Coverage |
|:----:|-------|:----:|:-----------:|:------------:|:--------:|
| 🥇 | **GPT-4** | Zero-shot | **94.0%** | **69.4%** | 620/621 |
| 🥈 | **Gemini-2.5** | Zero-shot | **91.9%** | **71.4%** | 620/621 |
| 🥈 | **Claude-3.5** | Zero-shot | **91.6%** | **67.3%** | 620/621 |
| 🥈 | **LLaMA-3 70B** | Zero-shot | **91.6%** | **75.5%** | 620/621 |
| 4 | **LLaMA-3 70B** | CoT | **89.5%** | **65.3%** | 620/621 |
| 5 | **GPT-4** | CoT | **88.2%** | **53.1%** | 620/621 |

</div>

> **Coverage Note:** 620/621 indicates 1 dataset item lacks a ground truth label and is excluded from accuracy calculations. See `published_results/*/run_meta.json` (`num_items_no_gold: 1`) for details.

**Key Findings:**
- 🏆 **GPT-4 Zero-shot** achieves highest overall accuracy (94.0%)
- 💬 **LLaMA-3 70B Zero-shot** shows best dialogue consistency (75.5%)
- 🎯 Multiple models achieve >91% accuracy, demonstrating strong logical reasoning capabilities
- ⚠️ Chain-of-thought prompting shows mixed results (degraded for both GPT-4 and LLaMA-3)

> **Note:** Throughput and timing metrics are environment-dependent and not reported. Worker counts varied by model (2-8 workers). See `published_results/*/run_meta.json` for deployment configuration details.

See [**RESULTS.md**](docs/RESULTS.md) for comprehensive analysis and task-specific breakdowns.

---

## 🚀 Quick Start

### Installation

We recommend using **[uv](https://docs.astral.sh/uv/)** (a fast Rust-based Python package manager):

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/11NOel11/ChaosBench-Logic.git
cd ChaosBench-Logic

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install runtime dependencies from pyproject.toml
uv sync

# For development (includes pytest, pytest-cov):
uv sync --all-groups
```

**Why uv?** ~10-100x faster than pip, automatic virtualenv management, lockfile support (uv.lock), and respects pyproject.toml dependency groups.

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

### Loading Dataset from HuggingFace

The ChaosBench-Logic dataset is available on HuggingFace for easy loading with the `datasets` library:

```python
from datasets import load_dataset

# Load single-turn questions (420 questions)
single_turn = load_dataset("11NOel11/ChaosBench-Logic", "single_turn")

# Load multi-turn dialogues (201 dialogue turns across 49 dialogues)
multi_turn = load_dataset("11NOel11/ChaosBench-Logic", "multi_turn")

# Access test split
print(f"Single-turn: {len(single_turn['test'])} questions")
print(f"Multi-turn: {len(multi_turn['test'])} turns")
print(f"Total benchmark: {len(single_turn['test']) + len(multi_turn['test'])} questions")

# Example: Load first question
first_question = single_turn['test'][0]
print(first_question['question'])
print(f"Ground truth: {first_question['ground_truth']}")
```

**Why two configurations?** Single-turn and multi-turn questions have different schemas:
- **Single-turn** (6 fields): `id`, `system_id`, `type`, `question`, `ground_truth`, `template`
- **Multi-turn** (8 fields): Adds `dialogue_id` and `turn` for dialogue tracking

**Note on `system_id`:** 159 questions have `null` system_id by design. These are general ontology/implication questions (e.g., "If a system is chaotic, must it be deterministic?") that test reasoning about FOL axioms rather than properties of specific systems.

See the [HuggingFace Dataset Card](https://huggingface.co/datasets/11NOel11/ChaosBench-Logic) for complete documentation.

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

ChaosBench-Logic includes a comprehensive pytest test suite ensuring correctness of evaluation logic, FOL violation detection, and answer normalization.

### Running Tests

```bash
# Ensure dev dependencies are installed
uv sync --all-groups

# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_normalization.py -v

# Run with coverage report
uv run pytest --cov=eval_chaosbench --cov-report=html
```

**Note:** Using `uv run` automatically activates the virtualenv and runs the command, so you don't need to manually activate.

### Test Coverage

<div align="center">

| Test Suite | Coverage |
|------------|----------|
| **Answer Normalization** | 8-step CoT parsing cascade, TRUE/FALSE variants, edge cases |
| **FOL Rules & Ontology** | Axiom definitions, system loading, predicate extraction, violation detection |
| **Summary Metrics** | Accuracy, dialogue metrics, contradiction rate, FOL violations, bias error |
| **Integration Smoke Tests** | End-to-end evaluation flow, metric aggregation |

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

See `tests/test_normalization.py` for comprehensive test cases documenting all supported formats.

**3. Bug Fixes**

- **bias_error computation**: Fixed to correctly compute error rate per bias family (was previously undefined)
- **Dialogue grouping**: Single questions now correctly treated as length-1 dialogues for FOL violation checking

### Test Organization

```
tests/
├── __init__.py                    # Test suite documentation
├── test_normalization.py          # Answer extraction tests
├── test_fol_rules.py              # FOL logic tests
├── test_summary_metrics.py        # Metric computation tests
└── test_integration_smoke.py      # Integration tests
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
├── 📁 tests/                  # Pytest test suite
│   ├── test_normalization.py  # Answer extraction tests
│   ├── test_fol_rules.py      # FOL violation tests
│   ├── test_summary_metrics.py # Metrics computation tests
│   └── test_integration_smoke.py # Integration tests
├── 📁 published_results/      # Published evaluation artifacts (tracked)
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

### Published Results

The `published_results/` directory contains **minimal, verifiable artifacts** used to generate the results tables:

- `summary.json` - Overall accuracy and per-task-type breakdown
- `run_meta.json` - Run configuration metadata
- `accuracy_by_task.csv` - Task-type accuracy table
- `metrics_overview.csv` - Summary metrics

**What's NOT included:** Raw model completions, debug logs, and checkpoint files are intentionally excluded to keep the repository lightweight (see `published_results/README.md` for details).

**To verify results:**
```bash
# View summary for any configuration
cat published_results/gpt4_zeroshot/summary.json | python3 -m json.tool

# Aggregate all results into markdown table
python scripts/aggregate_results.py
```

---

## 🧪 Benchmark Design

### Task Type Distribution

<div align="center">

| Task Type | Count | Description |
|-----------|:-----:|-------------|
| **multi_turn** | 213 | Contextual Q&A sequences (49 dialogues, avg 4.1 turns) |
| **bias** | 114 | Common misconceptions about chaos and dynamical systems |
| **atomic** | 76 | Basic properties: stability, chaos, dimension, periodicity |
| **counterfactual** | 76 | "What if" scenarios with parameter modifications |
| **hard** | 35 | Domain-specific technical reasoning (PDEs, chemistry, biology) |
| **multi_hop** | 34 | Chained logical inference across multiple facts |
| **cross_system** | 26 | Relative properties between different systems |
| **Other (11 types)** | 47 | implication, cf, cf_chain, validity, analogy, adversarial, trap, structural, fallacy, compositional |
| **Total** | **621** | 17 distinct task types across 7 high-level reasoning categories |

</div>

See [**DATASET.md**](docs/DATASET.md) for complete task type breakdown and statistics.

### Dynamical Systems Coverage

**27 systems actively used in dataset:**

- **Classical Chaos**: Lorenz-63, Lorenz-84, Lorenz-96, Rössler, Duffing (chaotic), Chen system
- **Chemical Systems**: Brusselator, Oregonator
- **Biological Models**: FitzHugh-Nagumo, Hindmarsh-Rose, Lotka-Volterra, Mackey-Glass
- **Maps**: Logistic (r=4.0, r=2.8), Hénon, Ikeda, Standard, Arnold cat, Baker's, Circle (quasiperiodic)
- **PDEs**: Kuramoto-Sivashinsky, Sine-Gordon
- **Neural Models**: Rikitake dynamo
- **Oscillators**: Van der Pol (vdp), Simple harmonic (shm), Damped driven pendulum (non-chaotic)
- **Stochastic**: Ornstein-Uhlenbeck process

**3 systems defined but reserved for extension:**
- Chua circuit, Damped oscillator, Double pendulum

See [**DATASET.md**](docs/DATASET.md) for system usage statistics and [**ONTOLOGY.md**](docs/ONTOLOGY.md) for complete system definitions.

### Evaluation Metrics

Each run generates comprehensive analytics:
- ✅ **Overall Accuracy** - Correct predictions across all tasks
- 💬 **Dialogue Accuracy** - Multi-turn conversation consistency
- 📊 **Task-specific Accuracy** - Per-category performance breakdowns
- ⚖️ **Bias Analysis** - Response distribution patterns
- 🔧 **Execution Metadata** - Worker configuration, coverage, API success rates
- 📈 **Visual Analytics** - Heatmaps, error distributions, confusion matrices

Results are exported in **JSON**, **CSV**, and **PNG** formats for downstream analysis.

---

## 🔬 Supported Models

The codebase supports the following LLM providers via API:

<div align="center">

| Model ID | Provider | API Model Name | Evaluated |
|----------|----------|----------------|:---------:|
| `gpt4` | OpenAI | gpt-4-turbo | ✅ Yes |
| `claude3` | Anthropic | claude-3-5-sonnet-20241022 | ✅ Yes |
| `gemini` | Google | gemini-2.5-flash-preview-0514 | ✅ Yes |
| `llama3` | HuggingFace | Meta-Llama-3-70B-Instruct | ✅ Yes |
| `mixtral` | HuggingFace | Mixtral-8x7B-Instruct-v0.1 | ⚠️ Code only |
| `openhermes` | HuggingFace | teknium/OpenHermes-2.5-Mistral-7B | ⚠️ Code only |

</div>

**Notes:**
- ✅ **Evaluated**: Results available in `published_results/` directory
- ⚠️ **Code only**: Client implementation exists but no evaluation results included
- **Speed/Cost**: Environment-dependent; not reported to avoid misleading comparisons
- **Workers**: Configured per model (2-8 workers) based on rate limits

To add new models, see [**CONTRIBUTING.md**](docs/CONTRIBUTING.md#adding-new-models).

---

## 📝 How to Cite

If you use ChaosBench-Logic in your research, please cite:

```bibtex
@software{chaosbench2025,
  title={ChaosBench-Logic: A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems},
  author={Thomas, Noel},
  year={2025},
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
<summary><b>Performance and timing?</b></summary>

Execution timing is environment-dependent and varies based on API provider, network conditions, and worker count. We do not report timing metrics to avoid misleading comparisons. Worker configuration details are available in `published_results/*/run_meta.json`.
</details>

<details>
<summary><b>Can I add my own models?</b></summary>

Yes! See [docs/CONTRIBUTING.md#adding-new-models](docs/CONTRIBUTING.md#adding-new-models) for step-by-step instructions.
</details>

<details>
<summary><b>Missing dependencies?</b></summary>

Reinstall: `pip install -r requirements.txt` or `uv sync --all-groups`
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
