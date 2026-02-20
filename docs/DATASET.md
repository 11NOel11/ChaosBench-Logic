# ChaosBench-Logic Dataset Card

## Dataset Description

**ChaosBench-Logic** is a comprehensive benchmark dataset for evaluating Large Language Models (LLMs) on complex reasoning tasks about dynamical systems. The dataset tests models across multiple dimensions: logical inference, symbolic manipulation, multi-hop reasoning, indicator diagnostics, regime transitions, and FOL consistency.

### Key Statistics (v2 - default)

| Metric | Value |
|--------|-------|
| **v2 Questions** | 40,886 (default evaluation target) |
| **v1 Questions (archived)** | 621 (in `data/archive/v1/`) |
| **Total** | 41,507 |
| **Question ID Format** | Various (e.g., ind_direct_0001 for indicators, fol_0001 for FOL, etc.) |
| **Core Systems** | 30 manually curated |
| **Extended Systems** | 135 from dysts library |
| **Total Systems** | 165 |
| **Unique Predicates** | 15 logical predicates per system |
| **Data Format** | JSONL (JSON Lines) |
| **v2 Files** | 10 (v22_*.jsonl) |
| **v1 Batch Files (archived)** | 7 (batches 1-7) |

### Data Split (v2 - default)

The v2 dataset uses 10 canonical files:

| File | Items | Primary Task Types |
|------|-------|-------------------|
| v22_atomic.jsonl | 25,000 | atomic |
| v22_multi_hop.jsonl | 6,000 | multi_hop |
| v22_consistency_paraphrase.jsonl | 4,139 | consistency_paraphrase |
| v22_perturbation_robustness.jsonl | 1,994 | perturbation |
| v22_adversarial.jsonl | 1,285 | adversarial_misleading, adversarial_nearmiss | <!-- adversarial_misleading: all FALSE by design (tests LLM resistance to misleading phrasing) -->
| v22_fol_inference.jsonl | 1,758 | fol_inference |
| v22_indicator_diagnostics.jsonl | 530 | indicator_diagnostics |
| v22_regime_transition.jsonl | 68 | regime_transition |
| v22_cross_indicator.jsonl | 67 | cross_indicator |
| v22_extended_systems.jsonl | 45 | extended_systems |

**v1 Archived** (data/archive/v1/, 621 questions):
- Batches 1-7: Original baseline tasks on 30 core systems

---

## Schema Definition

### Required Fields

Every question in the dataset has these fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique question identifier | `"q0001"` |
| `question` | string | Natural language question text | `"Is the Lorenz-63 system chaotic?"` |
| `ground_truth` | string | Ground truth answer label | `"TRUE"`, `"FALSE"` (v2); `"YES"`, `"NO"` (v1 archived) |
| `type` | string | Task type / reasoning category | `"atomic"`, `"multi_hop"`, `"bias"`, etc. |
| `template` | string | Template ID used for generation | `"A1"`, `"B_chain_2"`, `"C1"`, etc. |

### Optional Fields

Some questions have additional context:

| Field | Type | Description | When Present |
|-------|------|-------------|--------------|
| `system_id` | string | Dynamical system identifier | Present for most items - absent for ontology/FOL questions |
| `dialogue_id` | string | Multi-turn dialogue identifier | Only in batch7 (201 items) |
| `turn` | integer | Turn number within dialogue (1-indexed) | Present when `dialogue_id` exists |

**Note on Missing `system_id`:** 159 questions (25.6%) are system-agnostic bias probes designed to test general reasoning about chaos theory concepts without reference to a specific system. This is intentional.

---

## Ground Truth Labels

The dataset uses multiple label formats to test model robustness:

| Label | Count | Percentage | Usage |
|-------|-------|------------|-------|
| `NO` | 199 | 32.0% | Negative answers (zero-shot format) |
| `TRUE` | 190 | 30.6% | Affirmative answers (boolean format) |
| `YES` | 128 | 20.6% | Affirmative answers (zero-shot format) |
| `FALSE` | 103 | 16.6% | Negative answers (boolean format) |
| `DISAPPEAR` | 1 | 0.2% | Special counterfactual case |

**Design Choice:** We intentionally mix `YES/NO` and `TRUE/FALSE` formats to test whether models can handle both answer conventions. The evaluation pipeline normalizes both to a common format.

**Special Label:** `DISAPPEAR` is used for one counterfactual question where the correct answer is that a phenomenon (chaotic attractor) would cease to exist under the hypothetical conditions.

---

## Task Taxonomy

The dataset contains **17 distinct task types** organized into 7 high-level reasoning categories:

### Task Type Distribution

| Type | Count | Category | Description |
|------|-------|----------|-------------|
| `multi_turn` | 213 | Multi-turn Dialogue | Questions requiring context from previous turns |
| `bias` | 114 | Bias & Misconceptions | Probing common misconceptions about chaos |
| `atomic` | 76 | Atomic Facts | Single-hop questions about system properties |
| `counterfactual` | 76 | Counterfactual Reasoning | "What if" scenarios with parameter changes |
| `hard` | 35 | Domain-Specific Technical | Requires deep technical knowledge |
| `multi_hop` | 34 | Multi-hop Reasoning | Chaining multiple logical implications |
| `cross_system` | 26 | Cross-System Comparison | Comparing properties across systems |
| `cf` | 16 | Counterfactual (short) | Short counterfactual questions |
| `implication` | 6 | Logical Implications | Formal logic rules (A → B) |
| `cf_chain` | 5 | Counterfactual Chain | Multi-step counterfactual reasoning |
| `validity` | 5 | Logical Validity | Testing formal logic validity |
| `analogy` | 4 | Analogical Reasoning | Cross-domain analogies |
| `adversarial` | 3 | Adversarial | Designed to trick models |
| `trap` | 3 | Trap Questions | Questions with misleading framing |
| `structural` | 2 | Structural Reasoning | About system structure/topology |
| `fallacy` | 2 | Logical Fallacies | Identifying invalid reasoning |
| `compositional` | 1 | Compositional | Complex multi-component question |

### Mapping to 7 High-Level Categories

The 17 types group into these categories:

1. **Atomic Facts** → `atomic` (76 items)
2. **Logical Implications & Validity** → `implication`, `validity` (11 items)
3. **Multi-hop Reasoning** → `multi_hop`, `cf_chain` (39 items)
4. **Cross-System Comparison** → `cross_system`, `analogy` (30 items)
5. **Domain-Specific Technical** → `hard`, `cf`, `structural`, `compositional` (54 items)
6. **Counterfactual Analysis** → `counterfactual` (76 items)
7. **Multi-turn Dialogue** → `multi_turn` (213 items)
8. **Bias & Adversarial** → `bias`, `adversarial`, `trap`, `fallacy` (122 items)

**Note:** The original paper mentions "7 difficulty levels" which refers to these high-level categories. The dataset uses 17 fine-grained types for more precise analysis.

---

## Dynamical Systems Coverage

### Systems Used in Dataset (27)

The dataset includes questions about 27 dynamical systems:

| System | Count | System | Count |
|--------|-------|--------|-------|
| lorenz63 | 41 | kuramoto_sivashinsky | 28 |
| lorenz96 | 26 | hindmarsh_rose | 24 |
| oregonator | 23 | sine_gordon | 23 |
| standard_map | 23 | arnold_cat_map | 22 |
| brusselator | 22 | stochastic_ou | 21 |
| fitzhugh_nagumo | 20 | logistic_r4 | 19 |
| circle_map_quasiperiodic | 18 | henon | 32 |
| ikeda_map | 17 | shm | 17 |
| bakers_map | 15 | logistic_r2_8 | 15 |
| rossler | 15 | vdp | 11 |
| chen_system | 8 | duffing_chaotic | 6 |
| lorenz84 | 5 | damped_driven_pendulum_nonchaotic | 3 |
| lotka_volterra | 3 | rikitake_dynamo | 3 |
| mackey_glass | 2 | | |

### Systems Defined But Unused (3)

Three systems have complete ontology definitions in `systems/*.json` but no questions in the current dataset:

1. **chua_circuit** - Chua's circuit (canonical electronic chaotic oscillator)
2. **damped_oscillator** - Simple damped harmonic oscillator (non-chaotic baseline)
3. **double_pendulum** - Double pendulum system (classically chaotic mechanical system)

These systems are reserved for future dataset expansions or can be used by researchers extending the benchmark.

---

## Data Format

### File Structure

Each batch is a JSONL (JSON Lines) file where each line is a valid JSON object representing one question.

### Example: Atomic Question

```json
{
    "id": "q0001",
    "system_id": "lorenz63",
    "type": "atomic",
    "question": "Is the Lorenz-63 system chaotic?",
    "ground_truth": "TRUE",
    "template": "A1"
}
```

### Example: Multi-hop Question

```json
{
    "id": "q0051",
    "system_id": "logistic_r4",
    "type": "multi_hop",
    "question": "The logistic map at r = 4 has a positive Lyapunov exponent. Does this imply sensitive dependence, and does that imply chaotic behavior?",
    "ground_truth": "YES",
    "template": "B_chain_3"
}
```

### Example: Dialogue Turn

```json
{
    "id": "q0421",
    "dialogue_id": "d001",
    "turn": 1,
    "system_id": "lorenz63",
    "type": "multi_turn",
    "question": "Is the Lorenz-63 system deterministic?",
    "ground_truth": "TRUE",
    "template": "C1"
}
```

### Example: Bias Question (No System)

```json
{
    "id": "q0331",
    "type": "bias",
    "question": "Does chaotic behavior imply that a system must be random?",
    "ground_truth": "NO",
    "template": "D_random"
}
```

---

## System Ontology

Each of the 30 systems is defined in `systems/{system_id}.json` with:

- **System metadata:** Name, category, equations, parameters
- **Descriptions:** Simple and detailed explanations
- **Truth assignment:** Boolean values for all 15 predicates

See [ONTOLOGY.md](ONTOLOGY.md) for details on the 15 predicates and logical axioms.

---

## Data Quality

### Validation Checks

✅ **No duplicate IDs:** All 41,507 IDs are unique across 10 v2 files + 7 v1 archive files
✅ **Structured IDs:** IDs follow task-specific patterns (q0001-q0621 for v1, ind_direct_0001 for indicators, etc.)  
✅ **All questions have ground truth:** 0 missing labels  
✅ **Well-formed JSON:** All JSONL files parse correctly  
✅ **Consistent schema:** All required fields present in every item  

### Known Limitations

- **Mixed label formats:** YES/NO vs TRUE/FALSE (intentional design choice)
- **System-agnostic questions:** 25.6% lack `system_id` (bias questions)
- **Unbalanced types:** Some task types have very few examples (compositional: 1, structural: 2)

---

## Usage

### Loading the Dataset

```python
import json
from pathlib import Path

all_items = []
for f in sorted(Path("data").glob("v22_*.jsonl")):
    with open(f) as fh:
        for line in fh:
            if line.strip():
                all_items.append(json.loads(line))
print(f"Loaded {len(all_items)} questions")  # 40,886
```

### Filtering by Type

```python
# Get only atomic questions
atomic_questions = [item for item in all_items if item["type"] == "atomic"]

# Get questions for a specific system
lorenz_questions = [item for item in all_items if item.get("system_id") == "lorenz63"]

# Get all dialogues
dialogues = {}
for item in all_items:
    if "dialogue_id" in item:
        if item["dialogue_id"] not in dialogues:
            dialogues[item["dialogue_id"]] = []
        dialogues[item["dialogue_id"]].append(item)
```

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@software{thomas2025chaosbench,
  author = {Thomas, Noel},
  title = {ChaosBench-Logic: A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems},
  year = {2025},
  url = {https://github.com/11NOel11/ChaosBench-Logic}
}
```

---

## License

The ChaosBench-Logic dataset is licensed under [**Creative Commons Attribution 4.0 International (CC BY 4.0)**](LICENSE_DATA).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made

The code for evaluation and data processing is licensed under the MIT License (see [LICENSE](LICENSE)).

---

## Contact

For questions, issues, or contributions, please:
- Open an issue on [GitHub](https://github.com/11NOel11/ChaosBench-Logic/issues)
- Contact: Noel Thomas (MBZUAI)

---

## Changelog

### Version 2.0.0 (2026-02-18)
- Scaled to 40,886 questions across 10 task families, 165 systems, 15 predicates. Quality gates enforced. Canonical v22_*.jsonl naming.

### Version 2.1.0 (2026-02-17)
- Intermediate scaling with 11 batches (archived to data/archive/v21_intermediate/)
- New task families: indicator diagnostics, regime transitions, FOL inference, cross-indicator, adversarial, consistency paraphrases, extended systems
- 30 core systems + 136 dysts-imported systems (166 total)
- Empirically validated indicator thresholds

### Version 1.0.0 (2025-01-01)
- Initial public release
- 621 questions (batches 1-7) across 27 dynamical systems
- 7 batches covering 17 task types
- 49 multi-turn dialogues
