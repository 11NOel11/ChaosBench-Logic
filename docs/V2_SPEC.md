# ChaosBench-Logic v2 Specification

This document defines the v2 dataset format, schema, and generation protocol for ChaosBench-Logic.

## Overview

ChaosBench-Logic v2 is a comprehensive benchmark for evaluating LLM reasoning on dynamical systems. The dataset consists of approximately 25,000 binary classification questions across 30 core systems and ~100 dysts-imported systems.

## Dataset Schema

### Question Format

Questions are stored in JSONL format with the following schema:

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

**Field Descriptions:**

- **id** (string, required): Unique identifier for the question. Format: `{prefix}_{counter}` where prefix indicates task family.
- **question** (string, required): Natural language question text.
- **ground_truth** (string, required): Binary answer. Values: `"TRUE"` or `"FALSE"`.
- **type** (string, required): Task family identifier (see Task Families section).
- **system_id** (string, nullable): Identifier of the dynamical system. May be `null` for ontology questions.
- **template** (string, required): Template version label (e.g., `"V2"`).

### System Format

System definitions are stored as individual JSON files in `systems/` directory:

```json
{
  "system_id": "lorenz63",
  "name": "Lorenz-63 System",
  "category": "ode",
  "equations": [
    "dx/dt = sigma * (y - x)",
    "dy/dt = x * (rho - z) - y",
    "dz/dt = x * y - beta * z"
  ],
  "parameters": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.667
  },
  "dimension": 3,
  "truth_assignment": {
    "Chaotic": true,
    "Deterministic": true,
    "Periodic": false,
    "StrangeAttractor": true,
    "PositiveLyapunov": true,
    "Bounded": true,
    "Conservative": false,
    "Dissipative": true,
    "Stochastic": false,
    "Stable": false,
    "SensitiveToIC": true
  },
  "provenance": null
}
```

**Field Descriptions:**

- **system_id** (string, required): Unique identifier matching filename.
- **name** (string, required): Human-readable name.
- **category** (string, required): System type. Values: `"ode"`, `"map"`, `"pde"`, `"sde"`.
- **equations** (list[string], required): Mathematical equations defining the system.
- **parameters** (dict, required): Default parameter values used for computations.
- **dimension** (integer, required): State space dimension.
- **truth_assignment** (dict, required): Ground truth for 11 predicates (see Ontology).
- **provenance** (dict, nullable): For dysts-imported systems only (see Provenance section).

### Dysts-Imported Systems

Systems imported from the dysts library include provenance metadata:

```json
{
  "system_id": "dysts_aizawa",
  "name": "Aizawa",
  "category": "ode",
  "equations": ["..."],
  "parameters": {"..."},
  "dimension": 3,
  "truth_assignment": {"..."},
  "provenance": {
    "source": "dysts",
    "dysts_name": "Aizawa",
    "import_date": "2026-02-15",
    "dysts_url": "https://github.com/williamgilpin/dysts"
  }
}
```

Approximately 100 dysts systems are included to expand coverage of chaotic ODEs.

## Data Splits

The v2 dataset uses a 5-way split protocol for comprehensive evaluation:

### Split Definitions

1. **core** (~40% of data)
   - Questions from 30 manually curated benchmark systems
   - All task families represented
   - Primary evaluation split

2. **robustness** (~20% of data)
   - Paraphrase variants of core questions
   - Adversarial rephrasing
   - Tests consistency under linguistic variation

3. **heldout_systems** (~15% of data)
   - Questions from dysts-imported systems
   - Tests generalization to unseen systems
   - Same task families as core

4. **heldout_templates** (~15% of data)
   - Novel question templates not in core
   - Tests compositional generalization
   - Same systems as core

5. **hard** (~10% of data)
   - Multi-hop chains (≥3 reasoning steps)
   - Cross-indicator reasoning
   - FOL inference with ≥3 premises
   - Counterfactual chains

### Split Assignment

Split assignment is deterministic based on:
- System ID (for system-level splits)
- Question template hash (for template splits)
- Task complexity metadata (for difficulty splits)

Split labels are tracked in dataset manifests.

## Task Families

The v2 dataset includes 10 distinct task families:

### Core Task Families

1. **atomic**
   - Single predicate queries
   - Example: "Is the Lorenz-63 system chaotic?"
   - Count: ~2,000 questions

2. **multi_hop**
   - Chained logical inference (2-3 steps)
   - Example: "If system A is chaotic, and chaotic implies deterministic, is A deterministic?"
   - Count: ~1,500 questions

3. **indicator_diagnostics**
   - Interpretation of chaos indicators (K-test, PE, MEGNO)
   - Example: "The MEGNO value is 5.2. Is this system chaotic?"
   - Count: ~550 questions

4. **regime_transition**
   - Bifurcation and parameter-dependent behavior
   - Example: "Does the Duffing oscillator transition to chaos as gamma increases?"
   - Count: ~68 questions

5. **fol_inference**
   - First-order logic reasoning from premises
   - Example: "Given: Chaotic → Deterministic, A is Chaotic. What can we infer?"
   - Count: ~121 questions

6. **cross_indicator**
   - Reasoning across multiple indicators
   - Example: "If K=0.8 and PE=0.6, but MEGNO=1.5, is the system chaotic?"
   - Count: ~70 questions

### Robustness Task Families

7. **extended_systems**
   - Questions on underrepresented systems
   - Ensures balanced system coverage
   - Count: ~45 questions

8. **consistency_paraphrase**
   - Linguistic variations of core questions
   - Tests answer consistency
   - Count: ~300 questions

9. **adversarial**
   - Common misconceptions and edge cases
   - Example: "All periodic systems are stable. True or false?"
   - Count: ~104 questions

10. **perturbation_robustness**
    - Minor perturbations to question phrasing
    - Tests semantic parsing robustness
    - Included in evaluation pipeline

## Determinism and Reproducibility

### Generation Seeds

All stochastic generation steps use fixed seeds:
- Primary seed: 42 (set in config)
- Indicator computation seed: 42
- Question sampling seed: derived from primary seed

### Configuration Hash

Each dataset build computes a configuration hash including:
- Seed values
- Template versions
- System file hashes (SHA-256)
- Generation parameters

This hash ensures bit-for-bit reproducibility.

### Manifest Format

Each dataset build produces a manifest:

```json
{
  "version": "2.0.0",
  "timestamp": "2026-02-16T14:45:35.912946+00:00",
  "batches": {
    "batch8_indicator_diagnostics": {
      "count": 550,
      "sha256": "e87c378672e30677c5b8e56d37a1f46fcd4040959cebfc6f70f33d98331692ee"
    }
  },
  "total_new_questions": 1258,
  "total_existing_questions": 621,
  "grand_total": 1879
}
```

## Generation Pipeline

### Build Process

1. **System Loading**
   - Load 30 core systems from `systems/*.json`
   - Validate schema and truth assignments
   - Load ~100 dysts systems from `systems/dysts/*.json`

2. **Indicator Computation**
   - Compute chaos indicators for all systems
   - Store results in `systems/indicators/`
   - Apply validation thresholds (see `docs/INDICATOR_THRESHOLDS.md`)

3. **Question Generation**
   - Execute task-specific generators
   - Apply templates and linguistic variations
   - Validate schema compliance

4. **Batch Writing**
   - Write JSONL files with deterministic ordering
   - Compute SHA-256 hashes
   - Generate manifest

5. **Validation**
   - Run `scripts/validate_v2.py`
   - Check duplicate questions
   - Verify ground truth consistency

### Configuration

Generation is controlled via YAML config files in `configs/generation/`:

**Example: `v2_default.yaml`**

```yaml
seed: 42
template: V2

indicators:
  seed: 42

adversarial:
  n_per_type: 50
  drop_unknown: true

consistency:
  batch2_take: 50
  paraphrase_variants: 3
```

### Build Command

```bash
python scripts/build_v2_dataset.py --config configs/generation/v2_default.yaml
```

## Provenance Requirements

### Dysts-Imported Systems

All systems imported from dysts must include provenance:

- **source**: Always `"dysts"`
- **dysts_name**: Original name in dysts library
- **import_date**: Date of import (ISO 8601)
- **dysts_url**: Repository URL

### Manual Systems

Core 30 systems have `provenance: null` indicating manual curation.

## Validation Gates

### Pre-Release Validation

Before release, run:

```bash
python scripts/validate_v2.py --strict --max-duplicate-questions 200
```

This checks:
- Schema compliance
- Duplicate question detection (threshold: 200 allowed)
- System reference integrity
- Ground truth consistency
- Manifest hash verification

### Continuous Integration

CI runs validation on every commit:
- Schema validation
- JSON parsing
- System file counts
- Manifest integrity

## Version History

- **v2.0.0** (February 2026): Initial v2 release with ~25k questions
- **v1.0.0** (2025): Original 621-question benchmark

## References

- System definitions: `systems/` directory
- Indicator computation: `docs/INDICATOR_COMPUTATION.md`
- Threshold validation: `docs/INDICATOR_THRESHOLDS.md`
- Evaluation protocol: `docs/EVAL_PROTOCOL.md`
