# Dataset Card: ChaosBench-Logic

## Dataset Summary

ChaosBench-Logic is a benchmark dataset for evaluating Large Language Model reasoning capabilities on dynamical systems and chaos theory. The dataset tests logical inference, symbolic manipulation, multi-hop reasoning, indicator diagnostics, regime transitions, and FOL consistency through binary classification questions.

**Version 2.0.0** (default) consists of **40,886 questions** spanning 30 core manually-curated dynamical systems and 135 systems imported from the dysts library (165 total). Questions are organized into 10 task families, testing diverse reasoning capabilities from basic atomic queries to complex multi-indicator cross-validation.

**Version 1** (621 questions, archived in `data/archive/v1/`) established baseline performance metrics. Total dataset: 41,507 questions.

## Dataset Description

- **Repository**: https://github.com/11NOel11/ChaosBench-Logic
- **HuggingFace Dataset**: https://huggingface.co/datasets/11NOel11/ChaosBench-Logic
- **Version**: 2.0.0
- **License**: MIT (code), CC BY 4.0 (dataset)

### Dataset Size (v2 - default)

| Family | Questions | Description |
|--------|-----------|-------------|
| atomic | 25,000 | Single predicate queries about system properties |
| multi_hop | 6,000 | Chained logical inference (2-6 reasoning steps) |
| consistency_paraphrase | 4,139 | Linguistic variations testing answer consistency |
| perturbation_robustness | 1,994 | Minor perturbations to phrasing |
| adversarial | 1,285 | Common misconceptions and edge cases |
| fol_inference | 1,758 | First-order logic reasoning from premises |
| indicator_diagnostics | 530 | Interpretation of chaos indicators |
| regime_transition | 68 | Bifurcation and parameter-dependent behavior |
| cross_indicator | 67 | Reasoning across multiple chaos indicators |
| extended_systems | 45 | Questions on underrepresented systems |
| **v2 Total** | **40,886** | |

**Archived v1** (data/archive/v1/): 621 questions (batches 1-7, original baseline)
**Combined Total**: 41,507 questions

### Task Families

The dataset includes 10 task families testing different reasoning capabilities:

1. **atomic**: Single predicate queries about system properties
2. **multi_hop**: Chained logical inference (2-4 reasoning steps)
3. **indicator_diagnostics**: Interpretation of chaos indicators (K-test, permutation entropy, MEGNO)
4. **regime_transition**: Bifurcation and parameter-dependent behavior changes
5. **fol_inference**: First-order logic reasoning from premises
6. **cross_indicator**: Reasoning across multiple chaos indicators
7. **extended_systems**: Questions on underrepresented systems
8. **consistency_paraphrase**: Linguistic variations testing answer consistency
9. **adversarial**: Common misconceptions and edge cases
10. **perturbation_robustness**: Minor perturbations to phrasing

### System Coverage

**30 Core Systems (manually curated):**
- Classical chaos: Lorenz-63, Lorenz-84, Lorenz-96, Rössler, Duffing, Chen system
- Chemical systems: Brusselator, Oregonator
- Biological models: FitzHugh-Nagumo, Hindmarsh-Rose, Lotka-Volterra, Mackey-Glass
- Discrete maps: Logistic (multiple parameters), Hénon, Ikeda, Standard, Arnold cat, Baker's, Circle
- PDEs: Kuramoto-Sivashinsky, Sine-Gordon
- Neural models: Rikitake dynamo
- Oscillators: Van der Pol, Simple harmonic, Damped driven pendulum, Chua circuit, Double pendulum, Damped oscillator
- Stochastic: Ornstein-Uhlenbeck process

**135 Extended Systems (from dysts):**
- Additional chaotic ODEs imported from dysts library
- Used for extended_systems task family
- Enables testing generalization to unseen systems

## Languages

English only.

## Dataset Structure

### Data Format

Questions are stored in JSONL format:

```json
{
  "id": "ind_direct_0001",
  "question": "The 0-1 test gives K=0.99 for Arnold's cat map. Is this system chaotic?",
  "ground_truth": "TRUE",
  "type": "indicator_diagnostic",
  "system_id": "arnold_cat_map",
  "template": "V2"
}
```

### Data Fields

- **id** (string): Unique question identifier
- **question** (string): Natural language question text
- **ground_truth** (string): Binary answer - `"TRUE"` or `"FALSE"`
- **type** (string): Task family identifier
- **system_id** (string, nullable): System identifier, `null` for ontology questions
- **template** (string): Template version label

### Data Organization

Questions are organized into 10 task families:

- **v22_atomic.jsonl**: Single predicate queries about system properties
- **v22_multi_hop.jsonl**: Chained logical inference (2-4 reasoning steps)
- **v22_consistency_paraphrase.jsonl**: Linguistic variations testing answer consistency
- **v22_perturbation_robustness.jsonl**: Minor perturbations to phrasing
- **v22_adversarial.jsonl**: Common misconceptions and edge cases
- **v22_indicator_diagnostics.jsonl**: Interpretation of chaos indicators
- **v22_fol_inference.jsonl**: First-order logic reasoning from premises
- **v22_regime_transition.jsonl**: Bifurcation and parameter-dependent behavior
- **v22_cross_indicator.jsonl**: Reasoning across multiple chaos indicators
- **v22_extended_systems.jsonl**: Questions on underrepresented systems

All data is available in `data/v22_*.jsonl` files (10 canonical files).

## Dataset Creation

### Curation Rationale

ChaosBench-Logic was created to evaluate LLM reasoning on scientific domains requiring symbolic manipulation, logical inference, and mathematical understanding. Dynamical systems provide a rich testbed because:

1. Well-defined ground truth through mathematical analysis
2. Diverse reasoning types (deductive, inductive, counterfactual)
3. Requires both domain knowledge and logical reasoning
4. Contains common misconceptions suitable for adversarial testing

### Source Data

#### Core Systems

30 systems were manually curated from classical dynamical systems literature, including:
- Textbook examples (Strogatz, Wiggins)
- Landmark papers (Lorenz 1963, Rössler 1976)
- Standard benchmarks in chaos theory

Each system includes verified ground truth for 15 predicates based on mathematical analysis.

#### Extended Systems

135 systems imported from the dysts library (Gilpin et al., 2021) with provenance tracking. Systems were selected for:
- Diversity in system classes (ODEs, maps, PDEs)
- Coverage of chaotic and non-chaotic regimes
- Representation of different scientific domains

#### Chaos Indicators

Three chaos indicators are computed for each system:
- **Zero-One K Test**: Gottwald & Melbourne (2004)
- **Permutation Entropy**: Bandt & Pompe (2002)
- **MEGNO**: Cincotta & Simó (2000)

Indicator thresholds were empirically validated on the benchmark systems. See `docs/INDICATOR_THRESHOLDS.md` for methodology.

### Annotations

Ground truth labels are derived from:
1. Mathematical analysis of system equations
2. Literature values for standard systems
3. Numerical computation of chaos indicators
4. Logical inference from FOL axioms

All annotations are deterministic given the system definition and parameters.

### Personal and Sensitive Information

The dataset contains no personal or sensitive information. All content is mathematical and scientific in nature.

## Considerations for Using the Data

### Social Impact

This benchmark is designed for scientific and educational purposes. Potential impacts:

**Positive:**
- Advances LLM capabilities in scientific reasoning
- Provides standardized evaluation for mathematical reasoning
- Helps identify gaps in model understanding

**Neutral/Limited:**
- Domain-specific (dynamical systems) with limited direct social impact
- Requires technical background to interpret results

### Discussion of Biases

**Dataset Biases:**
- English language only
- Western scientific perspective and notation
- Overrepresentation of classical systems from 1960s-1980s literature
- Binary questions only (no open-ended or numerical answers)

**Mitigation:**
- Diverse system selection across scientific domains
- Adversarial questions targeting common misconceptions
- Consistency checks via paraphrase variants

### Other Known Limitations

1. **Binary Classification Only**: Questions require TRUE/FALSE answers. Does not test numerical prediction, equation derivation, or open-ended explanation.

2. **Static Dataset**: System parameters are fixed. Does not test continuous parameter exploration or bifurcation diagram construction.

3. **Text-Only**: No visual representations (phase portraits, time series plots). LLMs must reason from equations and descriptions.

4. **English Only**: Limits evaluation to English-capable models.

5. **Snapshot of Knowledge**: Systems and questions reflect current (2026) understanding of dynamical systems theory.

6. **Indicator Limitations**: Chaos indicators have known failure modes and numerical issues (see `docs/INDICATOR_COMPUTATION.md`).

## Additional Information

### Dataset Curators

Noel Thomas, Mohamed bin Zayed University of Artificial Intelligence

### Licensing Information

- **Code**: MIT License
- **Dataset**: Creative Commons Attribution 4.0 International (CC BY 4.0)

Users are free to share and adapt with proper attribution.

### Citation Information

```bibtex
@software{chaosbench2025,
  title={ChaosBench-Logic: A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems},
  author={Thomas, Noel},
  year={2025},
  url={https://github.com/11NOel11/ChaosBench-Logic},
  institution={Mohamed bin Zayed University of Artificial Intelligence}
}
```

### Contributions

Contributions are welcome. See `docs/CONTRIBUTING.md` for guidelines.

Areas for contribution:
- Additional system definitions
- New task families
- Translations to other languages
- Extended evaluation metrics

### Contact

- GitHub Issues: https://github.com/11NOel11/ChaosBench-Logic/issues
- GitHub Discussions: https://github.com/11NOel11/ChaosBench-Logic/discussions

### Acknowledgments

This work builds upon:
- **dysts library**: William Gilpin (2021) - https://github.com/williamgilpin/dysts
- **Chaos theory literature**: Lorenz, Rössler, Strogatz, Wiggins, and many others
- **LLM APIs**: OpenAI, Anthropic, Google, HuggingFace

### References

1. Gottwald, G. A., & Melbourne, I. (2004). A new test for chaos in deterministic systems. Proceedings of the Royal Society A, 460(2042), 603-611.

2. Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. Physical Review Letters, 88(17), 174102.

3. Cincotta, P. M., & Simó, C. (2000). Simple tools to study global dynamics in non-axisymmetric galactic potentials-I. Astronomy and Astrophysics Supplement Series, 147(2), 205-228.

4. Gilpin, W. (2021). dysts: A Python library for dynamical systems. https://github.com/williamgilpin/dysts
