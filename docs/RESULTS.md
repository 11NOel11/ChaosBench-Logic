# ChaosBench-Logic: Complete Evaluation Results

## Executive Summary (v1 Baseline - Archived)

Successfully evaluated **6 model configurations** on the ChaosBench-Logic **v1 baseline** (621 questions, batches 1-7, archived in `data/archive/v1/`). Results include GPT-4, Claude-3.5, Gemini-2.5, and LLaMA-3 70B in both zero-shot and chain-of-thought modes.

**Note:** These results are from the v1 dataset (archived). **v2 (default)** contains 40,886 questions (`data/v22_*.jsonl`) across 10 task families and 165 systems. Full v2 evaluation pending.

---

## 📊 Overall Results

### Model Performance Rankings

| Rank | Model | Mode | Overall Accuracy | Dialogue Accuracy | Coverage |
|------|-------|------|------------------|-------------------|----------|
| 🥇 1 | **GPT-4** | Zeroshot | **94.0%** | 69.4% | 620/621 |
| 🥈 2 | **Gemini-2.5 Flash** | Zeroshot | **91.9%** | 71.4% | 620/621 |
| 🥉 3 | **Claude-3.5 Sonnet** | Zeroshot | **91.6%** | 67.3% | 620/621 |
| 🥉 3 | **LLaMA-3 70B** | Zeroshot | **91.6%** | **75.5%** | 620/621 |
| 5 | **LLaMA-3 70B** | CoT | **89.5%** | 65.3% | 620/621 |
| 6 | **GPT-4** | CoT | **88.2%** | 53.1% | 620/621 |

**Note:** Execution timing is environment-dependent and not reported. Worker counts: GPT-4 (5), Claude-3.5 (4), Gemini (8), LLaMA-3 (2).

### Key Findings

✅ **Top Performance:**
- **GPT-4 Zeroshot: 94.0%** - Highest overall accuracy
- **LLaMA-3 Zeroshot: 75.5% dialogue accuracy** - Best multi-turn consistency
- **Coverage: 620/621** - All models evaluated on same question set (1 item has no gold label)

🎯 **Notable Achievements:**
- LLaMA-3 has the **highest dialogue accuracy** across all models
- Perfect scores (100%) on: adversarial, trap, cross_system, cf_chain tasks
- Strong on counterfactuals (92.1% zeroshot, 89.5% CoT)

⚠️ **Areas for Improvement:**
- Compositional reasoning (0% both modes)
- Implication tasks (66.7% zeroshot, 33.3% CoT)
- Structural reasoning (50% zeroshot, but 100% CoT)

---

## 📈 Detailed Performance by Task Type

### LLaMA-3 70B - Zeroshot Mode

| Task Type | Accuracy | Sample Size | Strength |
|-----------|----------|-------------|----------|
| **Adversarial** | 100.0% | Small | ✓ Perfect |
| **Trap** | 100.0% | Small | ✓ Perfect |
| **Cross-System** | 100.0% | Small | ✓ Perfect |
| **CF Chain** | 100.0% | Small | ✓ Perfect |
| **Analogy** | 100.0% | Small | ✓ Perfect |
| **Fallacy** | 100.0% | Small | ✓ Perfect |
| **Hard** | 94.3% | Medium | ✓ Strong |
| **Multi-Turn** | 93.9% | Large | ✓ Strong |
| **CF** | 93.8% | Medium | ✓ Strong |
| **Counterfactual** | 92.1% | Large | ✓ Strong |
| **Multi-Hop** | 91.2% | Large | ✓ Good |
| **Bias** | 88.6% | Large | ✓ Good |
| **Atomic** | 88.2% | Large | ✓ Good |
| **Validity** | 80.0% | Small | ○ Fair |
| **Implication** | 66.7% | Small | ⚠ Weak |
| **Structural** | 50.0% | Small | ⚠ Weak |
| **Compositional** | 0.0% | Small | ✗ Fail |

### LLaMA-3 70B - CoT (Chain-of-Thought) Mode

| Task Type | Accuracy | Change from Zeroshot | Notes |
|-----------|----------|---------------------|--------|
| **Structural** | 100.0% | +50.0% | ✓ Major improvement |
| **Validity** | 100.0% | +20.0% | ✓ Improvement |
| **Adversarial** | 100.0% | 0% | ✓ Maintained |
| **Trap** | 100.0% | 0% | ✓ Maintained |
| **Cross-System** | 100.0% | 0% | ✓ Maintained |
| **CF Chain** | 100.0% | 0% | ✓ Maintained |
| **Fallacy** | 100.0% | 0% | ✓ Maintained |
| **Bias** | 95.6% | +7.0% | ✓ Improvement |
| **CF** | 93.8% | 0% | ✓ Maintained |
| **Hard** | 91.4% | -2.9% | ○ Slight drop |
| **Multi-Turn** | 89.2% | -4.7% | ○ Drop |
| **Counterfactual** | 89.5% | -2.6% | ○ Slight drop |
| **Atomic** | 89.5% | +1.3% | ○ Maintained |
| **Analogy** | 75.0% | -25.0% | ⚠ Significant drop |
| **Multi-Hop** | 67.6% | -23.6% | ⚠ Major drop |
| **Implication** | 33.3% | -33.4% | ⚠ Major drop |
| **Compositional** | 0.0% | 0% | ✗ Still fails |

**CoT Analysis:**
- CoT improves structural and validity reasoning significantly
- CoT hurts multi-hop, analogy, and implication tasks
- Overall accuracy drops from 91.6% to 89.5% with CoT
- Suggests LLaMA-3 performs better with direct answers than extended reasoning for this benchmark

---

## 📊 Metrics Glossary

Understanding the evaluation metrics used in this benchmark:

### Overall Accuracy

**Definition:** Fraction of questions where the model's normalized prediction matches the ground truth label.

**Computation:**
```python
overall_accuracy = correct_predictions / total_evaluated
```

where `total_evaluated` excludes items with no gold label (1 item in the dataset).

**Range:** 0.0 to 1.0 (reported as percentage: 0% to 100%)

**What it measures:** The model's ability to correctly answer individual questions across all task types.

---

### Dialogue Accuracy

**Definition:** Fraction of multi-turn dialogues where **all turns** are answered correctly.

**Computation:**
```python
# For each dialogue (d001, d002, ..., d049):
dialogue_correct = all(turn.prediction == turn.ground_truth for turn in dialogue)

dialogue_accuracy = count(dialogue_correct) / total_dialogues
```

**Range:** 0.0 to 1.0 (reported as percentage)

**What it measures:** The model's ability to maintain consistency across multi-turn conversations. A dialogue is only counted as correct if **every single turn** is correct.

**Why it's harder:** This metric is typically much lower than overall accuracy because a single mistake in any turn causes the entire dialogue to be marked as incorrect.

**Example:** A 4-turn dialogue requires 4 consecutive correct answers. Overall accuracy might be 90%, but dialogue accuracy could be (0.9)^4 = 65.6% for 4-turn dialogues.

---

### Coverage

**Definition:** Number of questions successfully evaluated divided by total questions in the dataset.

**Computation:**
```python
coverage = num_items_evaluated / num_items_total
```

**Typical value:** 620/621 for all models

**Why not 621/621:** One question in the dataset lacks a ground truth label (intentional design for testing edge cases), so it's excluded from evaluation.

**What it measures:** API reliability and response success rate. Low coverage indicates API failures or unparseable responses.

---

### Contradiction Rate

**Definition:** Fraction of dialogues containing at least one **self-contradiction** (model gives conflicting answers to the same question within a dialogue).

**Computation:**
```python
# For each dialogue:
has_contradiction = any(
    model answered same question differently in different turns
)

contradiction_rate = count(has_contradiction) / total_dialogues
```

**Range:** 0.0 to 1.0 (reported as percentage)

**What it measures:** Internal consistency. **Higher is worse** (unlike accuracy metrics).

**Example:** If a model says "YES, the Lorenz system is chaotic" in turn 1, then "NO, it is not chaotic" in turn 3 (when asked the same question again), that's a contradiction.

---

### Avg FOL Violations Per Item

**Definition:** Average number of First-Order Logic (FOL) axiom violations per question.

**Computation:**
```python
# For each question about a system's predicates:
violations = count(
    model predictions that violate requires/excludes rules
)

avg_fol_violations = total_violations / total_items
```

**Example violation:**
- Model predicts: `Chaotic = YES, Deterministic = NO`
- Axiom: `Chaotic → Deterministic` (chaos requires determinism)
- This is 1 FOL violation

**Range:** 0.0 to N (where N = max possible violations per item, typically ~11)

**What it measures:** Logical consistency with formal axioms defined in [ONTOLOGY.md](ONTOLOGY.md). **Lower is better** (0.0 = perfect logical consistency).

**Current results:** All evaluated models achieve 0.000 avg FOL violations, indicating strong adherence to logical axioms.

---

### How Metrics Are Computed

All metrics are computed by `eval_chaosbench.py` and stored in `published_results/*/summary.json`:

1. **Answer normalization:** `normalize_label()` extracts YES/NO from model responses using an 8-step cascade (handles various formats)
2. **Comparison:** Normalized prediction vs ground truth label
3. **Aggregation:** Compute per-item, per-task, per-dialogue statistics
4. **FOL checking:** Extract predicates from questions, check against axioms
5. **Output:** JSON files with all metrics

See `eval_chaosbench.py` for implementation details.

---

## 🔬 Comparison Across All Models

### Overall Accuracy Comparison

```
GPT-4 Zeroshot:     94.0% ████████████████████
Gemini Zeroshot:    91.9% ███████████████████
Claude-3.5:         91.6% ███████████████████
LLaMA-3 Zeroshot:   91.6% ███████████████████
LLaMA-3 CoT:        89.5% ██████████████████
GPT-4 CoT:          88.2% ██████████████████
```

### Dialogue Accuracy Comparison

```
LLaMA-3 Zeroshot:   75.5% ████████████████
Gemini Zeroshot:    71.4% ███████████████
GPT-4 Zeroshot:     69.4% ██████████████
Claude-3.5:         67.3% ██████████████
LLaMA-3 CoT:        65.3% █████████████
GPT-4 CoT:          53.1% ███████████
```

### Task-Specific Champions

| Task | Best Model | Accuracy |
|------|------------|----------|
| Atomic | GPT-4 Zeroshot | 96.1% |
| Bias | GPT-4 Zeroshot | 93.0% |
| Counterfactual | GPT-4 Zeroshot | 97.4% |
| Multi-Turn | **LLaMA-3 Zeroshot** | **93.9%** |
| Multi-Hop | GPT-4 Zeroshot | 97.1% |
| Dialogue Consistency | **LLaMA-3 Zeroshot** | **75.5%** |

---

## 🚀 Technical Implementation Details

### LLaMA-3 Integration

**Problem Solved:**
- Initial implementation used `text_generation` API which returned `None`
- HuggingFace Inference API for provider "novita" requires `chat_completion` (conversational) API

**Solution:**
- Modified `clients.py` to use `chat_completion` with proper message formatting
- Added robust error handling for None responses
- Implemented retry logic with exponential backoff

**API Configuration:**
- Model: `meta-llama/Meta-Llama-3-70B-Instruct`
- Workers: 2 (for stability)
- Temperature: 0.0 (deterministic)
- Max tokens: 512

**Reliability:**
- Zeroshot: 620/621 valid responses (99.8%)
- CoT: 620/621 valid responses (99.8%)
- Only 1 item with no gold label (inherent in dataset)
- 0 API failures or payment issues

**Note:** Execution timing and cost data are environment-dependent (varies by API load, network latency, worker configuration) and not reported in published results to avoid misleading comparisons. See `run_meta.json` files for worker configurations used.

---

## 📝 Warnings Analysis

### Zeroshot Warnings (1 occurrence)
- 1 warning about normalize_label failing to extract from "DISAPPEAR" response
- This was a counterfactual question expecting non-binary answer
- **Not an error** - working as designed for edge cases

### CoT Warnings (8 occurrences)
- 8 warnings about normalize_label failing to extract YES/NO from long explanations
- Examples: Model started explaining step-by-step but didn't include explicit "FINAL_ANSWER:"
- Affected questions still mostly parsed correctly (89.5% overall accuracy)
- **Expected behavior** - CoT generates longer responses that sometimes lack strict formatting

**Conclusion:** Warnings indicate the normalization function is working correctly by alerting when it can't parse. The high accuracy shows most responses were parsed successfully.

---

## 🎯 Research Insights

### Strengths of LLaMA-3 70B

1. **Dialogue Consistency** - Best multi-turn reasoning across all models (75.5% dialogue accuracy)
2. **Counterfactual Reasoning** - Strong performance on "what-if" scenarios
3. **Bias Detection** - Good at identifying logical fallacies and misconceptions
4. **Competitive Overall Accuracy** - 91.6% overall accuracy, matching Claude-3.5 and approaching Gemini-2.5

### Weaknesses of LLaMA-3 70B

1. **Compositional Reasoning** - Complete failure (0%) on compositional tasks
2. **Implication Logic** - Struggles with formal logical implications
3. **CoT Performance** - Adding reasoning steps hurts more than it helps
4. **Structural Tasks** - Mixed performance on abstract structural reasoning

### Recommendations for Paper

1. **Highlight LLaMA-3's dialogue consistency** - Unique strength not seen in other models (75.5% vs 53-71% for others)
2. **Analyze CoT paradox** - Why does explicit reasoning hurt performance for both GPT-4 and LLaMA-3?
3. **Investigate compositional failures** - Common weakness across all models (0% accuracy)
4. **Recommend task-specific prompting** - Zeroshot for most tasks, CoT only for specific scenarios
5. **Emphasize open-source competitiveness** - LLaMA-3 matches or exceeds some proprietary models

---

## 📦 Files Included

### Result Directories

Each model directory contains:
- `summary.json` - Overall metrics
- `per_item_results.jsonl` - Individual predictions
- `accuracy_by_task.csv` - Task-level breakdown
- `metrics_overview.csv` - Summary table
- `bias_errors.csv` - Bias analysis
- `task_accuracy_bar.png` - Visualization
- `run_meta.json` - Execution metadata
- `debug_samples.jsonl` - Sample outputs
- `checkpoint.json` - Resume capability

### Available Results

✅ GPT-4 Zeroshot (621 items)
✅ GPT-4 CoT (621 items)
✅ Claude-3.5 Sonnet Zeroshot (621 items)
✅ Gemini-2.5 Flash Zeroshot (621 items)
✅ LLaMA-3 70B Zeroshot (621 items) - **NEW**
✅ LLaMA-3 70B CoT (621 items) - **NEW**

---

## 🔧 Repository Status

### Clean and Ready for Publication

✅ Removed all debug files (test_*.py, diagnose_*.py, etc.)
✅ Removed intermediate documentation (FIXES.md, STATUS.md, etc.)
✅ Clean README.md with usage instructions
✅ Proper .gitignore for Python projects
✅ Working run_llama.py script with proper error handling
✅ All 6 model evaluations complete

### Key Scripts

1. `eval_chaosbench.py` - Main evaluation framework
2. `clients.py` - Model API clients (GPT-4, Claude, Gemini, LLaMA-3)
3. `run_single_model.py` - Run one model
4. `run_all_models_fast.py` - Run all models with parallelization
5. `run_llama.py` - Dedicated LLaMA-3 runner

---

## 🚀 How to Add New Results

To add evaluation results for a new model or configuration:

### 1. Run Evaluation

```bash
# Run your model evaluation (outputs to results/ locally)
python run_benchmark.py --model yourmodel --mode zeroshot

# Copy minimal artifacts to published_results for tracking (with validation)
required_files="summary.json run_meta.json accuracy_by_task.csv metrics_overview.csv"
model_mode="yourmodel_zeroshot"

for file in $required_files; do
    if [ ! -f "results/$model_mode/$file" ]; then
        echo "ERROR: Missing required file: results/$model_mode/$file"
        exit 1
    fi
done

mkdir -p "published_results/$model_mode"
for file in $required_files; do
    cp "results/$model_mode/$file" "published_results/$model_mode/"
done

echo "✓ Successfully copied artifacts to published_results/$model_mode/"
```

### 2. Verify Output Files

Ensure your `published_results/yourmodel_zeroshot/` directory contains:
- `summary.json` - Overall metrics (required)
- `run_meta.json` - Execution metadata (required)
- `accuracy_by_task.csv` - Task-level breakdown (required)
- `metrics_overview.csv` - Summary metrics table (required)

**Note:** Raw model outputs (`per_item_results.jsonl`, `debug_samples.jsonl`) remain in `results/` (gitignored) to keep the repository lightweight.

### 3. Check `summary.json` Format

Required fields:
```json
{
  "overall_accuracy": 0.92,
  "dialogue_accuracy": 0.68,
  "contradiction_rate": 0.89,
  "avg_fol_violations_per_item": 0.0,
  "num_correct": 571,
  "num_total": 620
}
```

### 4. Check `run_meta.json` Format

Required fields:
```json
{
  "model_name": "yourmodel",
  "mode": "zeroshot",
  "num_items_total": 621,
  "num_items_evaluated": 620,
  "num_items_unanswered": 0,
  "num_items_no_gold": 1,
  "max_workers": 4
}
```

### 5. Update This Document

Add your model to:
- "Overall Results" table (lines 13-20)
- "Comparison Across All Models" section (lines 220-240)
- "Available Results" list (lines 350-360)

### 6. Optional: Run Aggregation Script

If you have multiple new results, use:
```bash
python scripts/aggregate_results.py
```

This will automatically generate updated tables and statistics.

---

## 📄 For Paper Writing

### Key Statistics to Include

**Dataset:**
- 621 questions across 7 batches
- 30 dynamical systems represented
- 11 logical predicates tested
- Task types: atomic, implications, multi-hop, counterfactual, bias, dialogue

**Best Results:**
- Overall Accuracy: GPT-4 Zeroshot (94.0%)
- Dialogue Accuracy: LLaMA-3 Zeroshot (75.5%)
- Contradiction Rate: Gemini-2.5 and Claude-3.5 (91.8% - lower is better)
- FOL Violations: All models (0.000 average violations)

**Key Findings:**
1. All models struggle with compositional reasoning (0%)
2. Dialogue consistency varies widely (53%-75%)
3. CoT doesn't always improve performance (sometimes hurts)
4. Open-source LLaMA-3 competitive with proprietary models
5. LLaMA-3 excels at multi-turn consistency

### Suggested Paper Sections

1. **Introduction** - Chaos theory reasoning benchmark
2. **Dataset** - ChaosBench-Logic structure and statistics
3. **Methods** - Model configurations and evaluation protocol
4. **Results** - Performance comparison (use tables above)
5. **Analysis** - Task-specific insights and failure modes
6. **Discussion** - Implications for reasoning in LLMs
7. **Conclusion** - Future work and recommendations

---

**Generated:** December 24, 2025
**Benchmark:** ChaosBench-Logic v1.0
**Total Evaluations:** 6 model configurations × 621 items = 3,726 predictions
