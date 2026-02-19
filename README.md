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

**ChaosBench-Logic** is a comprehensive benchmark designed to evaluate the reasoning capabilities of Large Language Models (LLMs) in the context of chaotic and non-chaotic dynamical systems. The benchmark tests models' abilities across multiple dimensions of complex reasoning: logical inference, symbolic manipulation, multi-hop reasoning, indicator diagnostics, regime transitions, and FOL consistency.

**Version 2** (default) contains **40,886 questions** across **10 task families** and **15 predicates**, spanning **30 core systems** (manually curated) and **135 dysts-imported systems** (165 total) from physics, chemistry, biology, and mathematics. The dataset includes novel task types for chaos indicator interpretation (0-1 test, permutation entropy, MEGNO), bifurcation reasoning, perturbation robustness, and multi-indicator cross-validation.

**Version 1** (621 questions, archived) established baseline performance. v2 scales coverage 60x+ to test generalization to unseen systems, robustness under perturbation, and complex multi-hop reasoning. Total dataset: 41,507 questions (v1 archived in `data/archive/v1/`, v2 in `data/`).

---

## 📊 Dataset Statistics (v2)

<div align="center">

| Metric | Count | Details |
|--------|-------|---------|
| **v2 Questions** | 40,886 | 10 task families (default evaluation target) |
| **v1 Questions (archived)** | 621 | 7 batches in `data/archive/v1/` |
| **Total** | 41,507 | v1 + v2 |
| **Task Families** | 10 | Atomic, multi-hop, indicator diagnostics, regime transitions, FOL inference, cross-indicator, extended systems, adversarial, consistency, perturbation robustness |
| **Core Systems** | 30 | Manually curated chaotic, periodic, quasi-periodic, stochastic systems |
| **Extended Systems** | 135 | Imported from dysts library for generalization testing |
| **Predicates** | 15 | Chaotic, Deterministic, Periodic, StrangeAttractor, PositiveLyapunov, Bounded, Conservative, Dissipative, Stochastic, Stable, SensitiveToIC, and 4 additional |
| **Ground Truth Labels** | TRUE/FALSE | Binary classification (v2), YES/NO (v1 archived) |

</div>

**Dataset Composition (v2 - default):**

| Task Family | Questions |
|-------------|----------:|
| atomic | 25,000 |
| consistency_paraphrase | 4,139 |
| multi_hop | 6,000 |
| perturbation_robustness | 1,994 |
| adversarial | 1,285 |
| indicator_diagnostics | 530 |
| fol_inference | 1,758 |
| regime_transition | 68 |
| cross_indicator | 67 |
| extended_systems | 45 |
| **v2 Total** | **40,886** |

All v2 data files follow the naming convention `v22_*.jsonl` in `data/`.

**Archived v1 (data/archive/v1/):**
- 621 questions: Original baseline tasks on 30 core systems

See [**DATASET.md**](docs/DATASET.md) for complete schema documentation and [**ONTOLOGY.md**](docs/ONTOLOGY.md) for predicate definitions and FOL axioms.

---

## 🎯 Key Features

<div align="center">

| Feature | Description |
|---------|-------------|
| **📊 40,886 v2 Questions** | 10 task families across `v22_*.jsonl` files (default evaluation) |
| **📦 621 v1 Archived** | Original baseline in `data/archive/v1/` |
| **🔬 165 Systems** | 30 core + 135 dysts-imported (Lorenz, Rössler, Brusselator, logistic, etc.) |
| **🧠 Multiple LLMs** | Evaluated: GPT-4, Claude-3.5, Gemini-2.5, LLaMA-3 70B • Supported: Mixtral, OpenHermes |
| **🎲 15 Predicates** | Stability, chaos, bifurcations, periodicity, sensitivity, and more |
| **🔄 2 Modes** | Zero-shot and chain-of-thought reasoning |
| **📈 Rich Metrics** | Overall accuracy, dialogue accuracy, task-specific breakdowns, FOL violations, bias analysis |

</div>

---

## 📊 Baseline Results (v1: 621 questions)

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

> **Dataset Version Note:** These results are from v1 (archived in `data/archive/v1/`, 621 questions). v1 established strong baseline performance on core reasoning tasks. **v2 (default)** adds 40,886 questions with challenging extensions for indicator interpretation, regime transitions, robustness testing, and generalization to 135 dysts systems. Full v2 evaluation pending.

**Key Findings (v1 Baseline):**
- 🏆 **GPT-4 Zero-shot** achieves highest overall accuracy (94.0%)
- 💬 **LLaMA-3 70B Zero-shot** shows best dialogue consistency (75.5%)
- 🎯 Multiple models achieve >91% accuracy on core reasoning tasks
- ⚠️ Chain-of-thought prompting shows mixed results (degraded for both GPT-4 and LLaMA-3)

See [**RESULTS.md**](docs/RESULTS.md) for comprehensive analysis and task-specific breakdowns.

---

## 📦 Paper Assets & Reproducibility

The freeze artifact, leaderboard, and analysis files are in `artifacts/paper_assets/` (gitignored; regenerate locally):

| Asset | Command | Output |
|-------|---------|--------|
| Dataset freeze | `python scripts/freeze_v2_dataset.py` | `artifacts/freeze/v2_freeze_manifest.json` |
| Locked 1k subset | `python scripts/make_api_subset.py --data_dir data/ --out_path data/subsets/api_balanced_1k.jsonl --size 1000 --balance --seed 42` | `data/subsets/api_balanced_1k.jsonl` |
| Local baseline | `chaosbench eval --provider ollama --model qwen2.5:7b --subset data/subsets/api_balanced_1k.jsonl` | `runs/<run_id>/` |
| Failure analysis | `python scripts/analyze_run_failures.py --run-dir runs/<run_id>` | `artifacts/paper_assets/failure_analysis/` |

**Dataset SHA256**: `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279` (40,886 questions)

See [docs/RELEASE_NOTES_V2.md](docs/RELEASE_NOTES_V2.md) for full release details.

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

# Load full v2 dataset (41,507 questions)
dataset = load_dataset("11NOel11/ChaosBench-Logic")

# Access test split
print(f"Total questions: {len(dataset['test'])} questions")

# Filter by task family
indicator_questions = [q for q in dataset['test'] if q['type'] == 'indicator_diagnostic']
print(f"Indicator diagnostic questions: {len(indicator_questions)}")

# Example: Load first question
first_question = single_turn['test'][0]
print(first_question['question'])
print(f"Ground truth: {first_question['ground_truth']}")
```

**Schema:** All questions use a unified format with 6 fields:
- `id` (string): Unique question identifier
- `question` (string): Natural language question text
- `ground_truth` (string): Binary answer (`"TRUE"` or `"FALSE"`)
- `type` (string): Task family identifier
- `system_id` (string, nullable): System identifier (null for ontology questions)
- `template` (string): Template version label (e.g., "V2")

**Note on `system_id`:** Ontology/FOL questions have `null` system_id by design. These test reasoning about logical axioms (e.g., "If a system is chaotic, must it be deterministic?") rather than properties of specific systems.

See the [HuggingFace Dataset Card](https://huggingface.co/datasets/11NOel11/ChaosBench-Logic) for complete documentation.

### Running Evaluations

> **Evaluation is closed-book**: models receive only the natural language question, with no access to system equations, time series, or numerical solvers.

```bash
# Evaluate a single model
python run_benchmark.py --model gpt4 --mode zeroshot

# Run all models
python run_benchmark.py --model all --mode zeroshot

# Chain-of-thought reasoning
python run_benchmark.py --model claude3 --mode cot

# Custom worker count (for rate limiting)
python run_benchmark.py --model llama3 --mode zeroshot --workers 2

# Distributed sharded run (example: shard 1/4)
python run_benchmark.py --model gpt4 --mode zeroshot --num-shards 4 --shard-index 0

# Merge shard outputs into canonical run directory
python scripts/merge_sharded_runs.py --model gpt4 --mode zeroshot --results-dir results

# Each run writes reproducibility manifests to runs/ and results/*/run_manifest.json
```

### Config-Driven Dataset Builds

```bash
# Build v2 dataset with default generation recipe
python scripts/build_v2_dataset.py --config configs/generation/v2_default.yaml

# Validate release gates (strict mode enables contamination check)
python scripts/validate_v2.py --strict --max-duplicate-questions 200
```

### Cluster-Scale Evaluations (SLURM)

```bash
# Dry-run script generation for 6 models x 2 modes, each split into 4 shards
python scripts/run_cluster_eval.py --dry-run --base-dir results --num-shards 4

# Submit jobs to SLURM
python scripts/run_cluster_eval.py --submit --base-dir results --num-shards 4
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
├── 📁 data/                   # Benchmark dataset (41,507 questions)
│   ├── v22_atomic.jsonl (25,000 questions)
│   ├── v22_consistency_paraphrase.jsonl (4,139 questions)
│   ├── v22_multi_hop.jsonl (6,000 questions)
│   ├── v22_perturbation_robustness.jsonl (1,994 questions)
│   ├── v22_adversarial.jsonl (1,285 questions)
│   ├── v22_indicator_diagnostics.jsonl (530 questions)
│   ├── v22_fol_inference.jsonl (1,758 questions)
│   ├── v22_regime_transition.jsonl (68 questions)
│   ├── v22_cross_indicator.jsonl (67 questions)
│   ├── v22_extended_systems.jsonl (45 questions)
│   ├── archive/v1/             # Archived v1 batches (621 questions)
│   └── v2_manifest.json  # Dataset generation manifest
├── 📁 systems/                # 166 dynamical system definitions
│   ├── *.json                # 30 core systems (manually curated)
│   ├── dysts/                # 135 systems imported from dysts library
│   └── indicators/           # Precomputed indicator values (0-1 test, PE, MEGNO)
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

### Task Family Distribution (v2)

<div align="center">

| Task Family | Questions | Description |
|-------------|----------:|-------------|
| **atomic** | 25,000 | Basic single-predicate queries across all systems |
| **multi_hop** | 6,000 | Chained logical inference (2-6 steps) |
| **consistency_paraphrase** | 4,139 | Linguistic variations testing answer stability |
| **perturbation_robustness** | 1,994 | Stability under parameter perturbations |
| **adversarial** | 1,285 | Common misconceptions and edge cases |
| **fol_inference** | 1,758 | First-order logic reasoning from premises |
| **indicator_diagnostics** | 530 | Chaos indicator interpretation (0-1 test, permutation entropy, MEGNO) |
| **regime_transition** | 68 | Bifurcation and parameter-dependent behavior |
| **cross_indicator** | 67 | Reasoning across multiple chaos indicators |
| **extended_systems** | 45 | Questions on underrepresented systems |
| **v2 Total** | **40,886** | 10 task families |
| **v1 (archived)** | 621 | Original baseline in `data/archive/v1/` |
| **Grand Total** | **41,507** | All versions combined |

</div>

See [**DATASET.md**](docs/DATASET.md) for complete task type breakdown and statistics.

### Dynamical Systems Coverage

**30 core systems (manually curated):**

- **Classical Chaos**: Lorenz-63, Lorenz-84, Lorenz-96, Rössler, Duffing (chaotic), Chen system
- **Chemical Systems**: Brusselator, Oregonator
- **Biological Models**: FitzHugh-Nagumo, Hindmarsh-Rose, Lotka-Volterra, Mackey-Glass
- **Maps**: Logistic (r=4.0, r=2.8), Hénon, Ikeda, Standard, Arnold cat, Baker's, Circle (quasiperiodic)
- **PDEs**: Kuramoto-Sivashinsky, Sine-Gordon
- **Neural Models**: Rikitake dynamo
- **Oscillators**: Van der Pol, Simple harmonic, Damped driven pendulum, Chua circuit, Double pendulum, Damped oscillator
- **Stochastic**: Ornstein-Uhlenbeck process

**135 extended systems (from dysts library):**
- Additional chaotic ODEs, rare system classes, expanded bifurcation scenarios
- Used in extended_systems task family and for heldout generalization testing
- Full list: `systems/dysts/*.json`

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
- **[docs/QUALITY_STANDARD.md](docs/QUALITY_STANDARD.md)** - Quality gates and validation criteria
- **[docs/ONTOLOGY_V2_EXTENSION.md](docs/ONTOLOGY_V2_EXTENSION.md)** - Extended predicate definitions for v2
- **[docs/SCALING_ROADMAP.md](docs/SCALING_ROADMAP.md)** - Major rescale plan and milestones
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
