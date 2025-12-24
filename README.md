# ChaosBench-Logic

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ChaosBench-Logic** is a comprehensive benchmark for evaluating Large Language Models (LLMs) on complex reasoning tasks involving chaotic and non-chaotic dynamical systems. It tests models' ability to reason about stability, bifurcations, chaos detection, and multi-hop logical inference across 30 diverse systems from physics, chemistry, biology, and mathematics.

## 🎯 Key Features

- **30 Dynamical Systems**: Lorenz-63, double pendulum, Brusselator, FitzHugh-Nagumo, logistic map, and more
- **621 Questions**: Spanning 7 difficulty levels from atomic facts to counterfactual reasoning
- **11 Logical Predicates**: Tests stability, chaos, bifurcations, periodicity, sensitivity, and cross-system comparisons
- **6 LLM Models**: GPT-4, Claude-3.5, Gemini-2.5, LLaMA-3, Mixtral, OpenHermes
- **2 Evaluation Modes**: Zero-shot and chain-of-thought reasoning
- **Comprehensive Metrics**: Overall accuracy, dialogue accuracy, task-specific breakdowns, bias analysis

## 📊 Quick Results

| Model | Overall Acc | Dialogue Acc | Valid Responses | Throughput* |
|-------|-------------|--------------|-----------------|-------------|
| **LLaMA-3 (zeroshot)** | **91.6%** | **75.5%** | 620/621 | 1.2 items/s ⚠️ |
| GPT-4 (cot) | 90.2% | 73.7% | 621/621 | ~10 items/s |
| GPT-4 (zeroshot) | 90.0% | 72.8% | 621/621 | ~15 items/s |
| LLaMA-3 (cot) | 89.5% | 65.3% | 620/621 | 0.2 items/s ⚠️ |
| Claude-3.5 (zeroshot) | 88.2% | 68.3% | 621/621 | ~12 items/s |
| Gemini-2.5 (zeroshot) | 87.9% | 67.6% | 620/621 | ~18 items/s |

*LLaMA-3 throughput measured with 2 parallel workers (practical real-world speed)

⚠️ **Note**: LLaMA-3 70B takes longer than other models (~8 minutes for zeroshot, ~55 minutes for chain-of-thought).

See [RESULTS.md](RESULTS.md) for detailed analysis.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- API keys for the models you want to test (see [API_SETUP.md](API_SETUP.md))

### Installation (Recommended: uv)

We recommend using **[uv](https://docs.astral.sh/uv/)** - a fast Rust-based Python package manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/chaos-logic-bench.git
cd chaos-logic-bench

# Setup environment (automatic!)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Alternative Installation Methods

<details>
<summary><b>Using pip</b></summary>

```bash
git clone https://github.com/yourusername/chaos-logic-bench.git
cd chaos-logic-bench
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Using conda</b></summary>

```bash
git clone https://github.com/yourusername/chaos-logic-bench.git
cd chaos-logic-bench
conda create -n chaosbench python=3.11
conda activate chaosbench
pip install -r requirements.txt
```
</details>

### Configure API Keys

```bash
# Copy template and add your keys
cp .env.example .env
nano .env  # Edit with your API keys
```

See [API_SETUP.md](API_SETUP.md) for detailed instructions on obtaining API keys.

### Run Your First Evaluation

```bash
# Single model evaluation
python run_benchmark.py --model gpt4 --mode zeroshot

# Run all models
python run_benchmark.py --model all --mode zeroshot

# Both modes (zeroshot + chain-of-thought)
python run_benchmark.py --model claude3 --mode both
```

---

## 📖 Usage

### Basic Commands

```bash
# Run specific model and mode
python run_benchmark.py --model gpt4 --mode zeroshot

# Run with custom worker count (useful for rate limiting)
python run_benchmark.py --model llama3 --mode zeroshot --workers 2

# Clear checkpoints and restart from scratch
python run_benchmark.py --model gemini --mode cot --clear-checkpoints
```

### Supported Models

| Model ID | Model Name | Provider | Speed | Cost (per run) |
|----------|------------|----------|-------|----------------|
| `gpt4` | GPT-4 | OpenAI | Fast | ~$2.00 |
| `claude3` | Claude-3.5 | Anthropic | Fast | ~$1.30 |
| `gemini` | Gemini-2.5 | Google | Fast | ~$0.50 |
| `llama3` | LLaMA-3 70B | HuggingFace | **Slow** ⚠️ | ~$6.00 |
| `mixtral` | Mixtral | HuggingFace | Medium | ~$2.00 |
| `openhermes` | OpenHermes | HuggingFace | Medium | ~$1.30 |

### Evaluation Modes

- **`zeroshot`**: Direct question answering without examples
- **`cot`**: Chain-of-thought reasoning with step-by-step explanations
- **`both`**: Run both modes sequentially

---

## 📂 Repository Structure

```
chaos-logic-bench/
├── run_benchmark.py        # 🚀 Main evaluation runner
├── eval_chaosbench.py      # Core evaluation framework
├── clients.py              # LLM API client implementations
├── data/                   # 📊 Benchmark dataset (621 questions, 7 batches)
│   ├── batch1_atomic_implication.jsonl
│   ├── batch2_multiHop_crossSystem.jsonl
│   └── ...
├── systems/                # ⚙️ Dynamical system definitions (30 systems)
│   ├── lorenz63.json
│   ├── double_pendulum.json
│   └── ...
├── results/                # 📈 Evaluation outputs (auto-generated)
│   ├── gpt4_zeroshot/
│   ├── claude3_zeroshot/
│   └── ...
├── .env.example            # API key template
├── pyproject.toml          # uv/pip package configuration
├── requirements.txt        # pip dependencies
├── API_SETUP.md            # 🔑 Detailed API key setup guide
├── CONTRIBUTING.md         # 🤝 Contribution guidelines
└── RESULTS.md              # 📊 Comprehensive evaluation results
```

---

## 🧪 Evaluation Tasks

ChaosBench-Logic tests 7 categories of reasoning:

1. **Atomic Facts** (109 questions): Basic properties like stability, chaos, dimension
2. **Implications** (93 questions): Logical consequences (if A then B)
3. **Multi-hop Reasoning** (98 questions): Chained logical inference across facts
4. **Cross-system Comparison** (87 questions): Relative properties between systems
5. **PDE/Chemistry/Biology** (76 questions): Domain-specific technical reasoning
6. **Counterfactual** (68 questions): "What if" parameter modifications
7. **Multi-turn Dialogue** (90 questions): Contextual Q&A sequences

**Total**: 621 questions across 30 dynamical systems

---

## 📈 Metrics

Each evaluation generates:

- **Overall Accuracy**: Correct predictions across all tasks
- **Dialogue Accuracy**: Multi-turn conversation performance  
- **Task-specific Accuracy**: Per-category breakdowns
- **Bias Analysis**: Response distribution (yes/no tendencies)
- **Response Validity**: API success rate
- **Execution Time**: Speed benchmarks
- **Visual Analytics**: Accuracy heatmaps, error distributions

Results are saved in JSON, CSV, and PNG formats for easy analysis.

---

## 🤝 Contributing

We welcome contributions! Whether you want to:

- Add support for new LLM models
- Improve evaluation metrics
- Add new dynamical systems or questions
- Fix bugs or improve documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Environment setup (uv, conda, pip, venv)
- Code style and testing
- Pull request process
- Adding new models

---

## 📚 Documentation

- **[API_SETUP.md](API_SETUP.md)**: Detailed guide for obtaining and configuring API keys
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Development setup and contribution guidelines
- **[RESULTS.md](RESULTS.md)**: Comprehensive evaluation results and analysis
- **[LICENSE](LICENSE)**: MIT License details

---

## 📝 Citation

If you use ChaosBench-Logic in your research, please cite:

```bibtex
@article{chaosbench2024,
  title={ChaosBench-Logic: Benchmarking LLMs on Complex Reasoning about Dynamical Systems},
  author={ChaosBench Team},
  year={2024},
  journal={arXiv preprint},
  url={https://github.com/yourusername/chaos-logic-bench}
}
```

---

## ❓ Troubleshooting

**API key errors?** → See [API_SETUP.md](API_SETUP.md#troubleshooting)

**Rate limit issues?** → Reduce workers: `--workers 2`

**LLaMA-3 too slow?** → This is expected (see performance notes above)

**Missing dependencies?** → Reinstall: `pip install -r requirements.txt`

**Questions?** → Open an issue on GitHub

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [OpenAI API](https://openai.com/)
- [Anthropic Claude](https://anthropic.com/)
- [Google Gemini](https://ai.google.dev/)
- [HuggingFace Inference API](https://huggingface.co/)

Special thanks to the dynamical systems research community for inspiration.

---

**Made with ❤️ for advancing LLM reasoning on complex scientific problems**
