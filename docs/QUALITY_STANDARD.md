# ChaosBench-Logic Quality Standard

**Version:** 2.0
**Last Updated:** 2026-02-18
**Scope:** v2 dataset (v1 archived, not subject to these standards)

---

## Overview

This document defines explicit acceptance criteria for the ChaosBench-Logic benchmark dataset. All criteria must be satisfied before freezing an API subset for publication.

---

## 1. Duplicate Control

### 1.1 Exact Duplicates (Accidental)

**Definition:** Questions with identical (normalized_text, system_id, ground_truth) tuples that are NOT part of an intentional group (e.g., paraphrase sets).

**Acceptance Criteria:**
- **Threshold:** 0 accidental duplicates
- **Enforcement:** HARD FAIL (blocking)
- **Detection:** Normalize via `_normalize_text()` (lowercase, strip punctuation), compute uniqueness key `{family}:{system_id}:{norm_text}:{ground_truth}`

**Rationale:** Accidental duplicates indicate generation bugs, inflate metrics, and waste evaluation budget.

---

### 1.2 Exact Duplicates (Intentional)

**Definition:** Questions that are part of designed groups (e.g., consistency paraphrase sets, perturbation variants).

**Acceptance Criteria:**
- **Threshold:** Allowed if `group_id` is assigned
- **Enforcement:** SOFT (report only)
- **Validation:** All members of a group must share the same `group_id`

**Rationale:** Intentional groups test model consistency and sensitivity - duplicates within groups are by design.

---

### 1.3 Near-Duplicates

**Definition:** Question pairs with high text similarity (Jaccard ≥ 0.85 on normalized tokens) but not exact matches.

**Acceptance Criteria:**
- **Threshold (overall):** ≤ 5% of total pairs flagged as near-duplicates
- **Threshold (per family):**
  - `consistency_paraphrase`: ≤ 30% (by design includes paraphrases)
  - `perturbation`: ≤ 20% (by design includes variants)
  - All other families: ≤ 10%
- **Enforcement:** SOFT (report + investigate)

**Rationale:** Some near-duplicates are intentional (paraphrases, perturbations). High rates in other families indicate template homogeneity or generation artifacts.

**Action on Threshold Breach:**
1. Identify top offending templates/operators
2. Audit for semantic distinctness
3. If templates are synonym-based (not structure-based), enhance diversity
4. If unavoidable, document and use group-aware metrics

---

## 2. Label Leakage

### 2.1 Forbidden Tokens

**Definition:** Ground truth values appearing verbatim in question text.

**Forbidden Patterns:**
- `ground_truth`, `answer_is`, `correct answer`
- `the answer is [TRUE|FALSE]`
- `(TRUE)`, `(FALSE)` annotations

**Acceptance Criteria:**
- **Threshold:** 0 leaks
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** Label leakage enables trivial pattern matching, invalidating the benchmark.

---

### 2.2 Implicit Leakage

**Definition:** Question phrasing that strongly implies the answer.

**Examples:**
- "Is it TRUE that X?" (implies answer)
- "X is chaotic, correct?" (leading question)

**Acceptance Criteria:**
- **Threshold:** Manual audit of samples (automated detection infeasible)
- **Enforcement:** SOFT (review samples per family)

**Rationale:** Hard to automate, but critical for validity. Requires expert review.

---

## 3. Class Balance

### 3.1 Overall Balance

**Acceptance Criteria:**
- **Threshold:** 40-60% TRUE (equivalently, 40-60% FALSE)
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** Severe imbalance (>60% majority class) makes metrics misleading. Models can achieve high accuracy by predicting majority class.

---

### 3.2 Per-Family Balance

**Acceptance Criteria:**
- **Threshold (standard families):** 30-70% TRUE
  - Applies to: `atomic`, `consistency_paraphrase`, `perturbation`, `adversarial_misleading`, `cross_indicator`, `regime_transition`, `extended_systems`
- **Threshold (logic-heavy families):** 20-80% TRUE
  - Applies to: `multi_hop`, `fol_inference`
  - Rationale: Logic structure may naturally skew (e.g., more FALSE from excludes chains)
- **Exemptions (specialized families):**
  - `indicator_diagnostic`: No threshold (many systems lack indicators → FALSE)
  - `adversarial_nearmiss`: No threshold (near-misses are typically TRUE)

**Enforcement:** SOFT (report + justify)

**Action on Threshold Breach:**
1. Diagnose root cause (generation bias, ontology structure)
2. Attempt low-risk fixes (per-system parity, template adjustment)
3. If infeasible without distorting semantics, document and report balanced_accuracy/MCC prominently

---

### 3.3 Per-Split Balance

**Acceptance Criteria:**
- **Threshold:** 35-65% TRUE in each split (`core`, `robustness`, `hard`, `heldout_systems`, `heldout_templates`)
- **Enforcement:** SOFT (report + justify)

**Rationale:** Splits should have comparable difficulty distributions. Severe imbalance may indicate leakage or sampling bias.

---

## 4. Determinism

### 4.1 Reproducible Builds

**Definition:** Identical config + seed yields identical dataset (bitwise).

**Acceptance Criteria:**
- **Threshold:** 100% reproducibility (SHA-256 hashes match)
- **Enforcement:** HARD FAIL (blocking)
- **Validation:** Run build twice with same config/seed, compare file hashes

**Rationale:** Determinism is essential for:
- Versioning (dataset v2.0 must be immutable)
- Debugging (trace generation errors)
- Reproducibility (other researchers can regenerate)

**Implementation Requirements:**
- All RNGs seeded explicitly
- File iteration order deterministic (sorted)
- No non-deterministic data structures (e.g., Python dict iteration pre-3.7)

---

### 4.2 Manifest Stability

**Definition:** Manifest regenerates identically for same dataset.

**Acceptance Criteria:**
- **Threshold:** Manifest fields are stable (config_hash, dataset_hash, counts)
- **Enforcement:** SOFT (validate in CI)

**Rationale:** Manifest is the source of truth for dataset version. Must not drift.

---

## 5. Split Integrity

### 5.1 No Cross-Split Question Leakage

**Definition:** No question appears in multiple splits.

**Acceptance Criteria:**
- **Threshold:** 0 item_id collisions across splits
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** Violates train/test separation, invalidates heldout evaluation.

---

### 5.2 No Normalized Text Leakage

**Definition:** No questions with identical normalized text appear across disjoint splits (e.g., `core` vs `heldout_templates`).

**Acceptance Criteria:**
- **Threshold:** 0 normalized text collisions between non-robustness splits
- **Exception:** `robustness` split intentionally contains variants of `core` questions
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** Even with different IDs, identical normalized text leaks information.

**Validation:**
```python
core_texts = {normalize(q['question']) for q in core_split}
heldout_texts = {normalize(q['question']) for q in heldout_split}
assert len(core_texts & heldout_texts) == 0
```

---

### 5.3 No System Leakage

**Definition:** Heldout systems do not appear in core/robustness splits.

**Acceptance Criteria:**
- **Threshold:** 0 system_id collisions between core/robustness and heldout_systems
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** System-level generalization requires strict separation.

---

### 5.4 No Template Leakage

**Definition:** Heldout templates (hash-based) do not appear in core split.

**Acceptance Criteria:**
- **Threshold:** 0 template_hash collisions between core and heldout_templates
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** Template-level generalization requires unseen question structures.

**Implementation:**
```python
def compute_template_hash(question_text, system_id):
    # Replace system name with placeholder, hash remainder
    template = re.sub(r'\b' + re.escape(system_name) + r'\b', '<SYSTEM>', question_text)
    return hashlib.sha256(template.encode()).hexdigest()[:16]
```

---

### 5.5 No Group Leakage

**Definition:** Questions with the same `group_id` do not span disjoint splits.

**Acceptance Criteria:**
- **Threshold:** 0 group_id collisions between non-robustness splits
- **Exception:** Groups may span core+robustness (for consistency testing)
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** Groups test sensitivity - must not leak across heldout boundaries.

---

## 6. Non-Degeneracy

### 6.1 Minimum Minority Class Representation

**Definition:** Both TRUE and FALSE must be sufficiently represented in every split.

**Acceptance Criteria:**
- **Threshold:** ≥ 10% minority class in every split
- **Enforcement:** HARD FAIL (blocking)

**Rationale:** Splits with <10% minority class are degenerate - models can ignore the minority class.

**Example:**
- If split has 95% TRUE, 5% FALSE → FAIL (minority class too small)

---

### 6.2 Minimum Questions Per Family (in Full Dataset)

**Acceptance Criteria:**
- **Threshold:** ≥ 50 questions per family (in combined v1+v2)
- **Enforcement:** SOFT (report + justify)

**Rationale:** Families with <50 questions have high variance in metrics, insufficient for reliable evaluation.

**Exemptions:** New experimental families (clearly marked) may have <50.

---

## 7. System Coverage

### 7.1 Minimum Coverage

**Definition:** Percentage of systems with at least N questions.

**Acceptance Criteria:**
- **Threshold:** ≥ 90% of real systems (excluding synthetic "vs" systems) have ≥ 50 questions
- **Enforcement:** SOFT (report)

**Rationale:** Ensures benchmark tests diverse systems, not just a handful.

**Current Status:** 98.8% systems have 100+ questions (exceeds threshold).

---

### 7.2 Coverage Tail

**Definition:** Percentage of systems with very few questions (<10).

**Acceptance Criteria:**
- **Threshold:** ≤ 10% of real systems have <10 questions
- **Enforcement:** SOFT (report)

**Rationale:** Long tail of under-represented systems indicates uneven generation.

---

## 8. Metadata Completeness

### 8.1 Required Fields Per Question

**Acceptance Criteria:**
- **Required fields:** `id`, `question`, `ground_truth`, `type`, `system_id`, `template`
- **Optional fields:** `group_id`, `metadata`, `predicates`
- **Enforcement:** HARD FAIL (blocking) if required fields missing

---

### 8.2 Manifest Completeness

**Acceptance Criteria:**
- **Required manifest fields:**
  - `schema_version` (e.g., "2.0")
  - `dataset_hash` (SHA-256 of concatenated file hashes)
  - `config_hash` (SHA-256 of generation config)
  - `seed` (RNG seed used)
  - `total_questions` (count)
  - `families` (list with counts)
  - `splits` (dict with counts and hashes)
  - `dedupe_stats` (exact duplicates removed, by family)
  - `near_duplicate_summary` (overall rate, per-family rates)
  - `label_distribution` (TRUE/FALSE counts per family)
- **Enforcement:** SOFT (report + enhance)

---

## 9. Quality Gate Execution

### 9.1 Pre-Freeze Check

**Definition:** Comprehensive QA pipeline that validates all criteria.

**Acceptance Criteria:**
- **Command:** `python scripts/pre_freeze_check.py`
- **Exit code:** 0 if all HARD FAIL criteria pass, non-zero otherwise
- **Outputs:**
  - `reports/pre_freeze/summary.md` (human-readable)
  - `reports/pre_freeze/summary.json` (machine-readable)
- **Runtime:** ≤ 10 minutes on full dataset (for CI feasibility)

---

### 9.2 CI Integration

**Acceptance Criteria:**
- **CI smoke test:** `pre_freeze_check.py --smoke` runs on every PR (≤ 2 min)
- **CI full test:** `pre_freeze_check.py` runs nightly or on release branches

**Enforcement:** HARD FAIL (blocking) - PRs cannot merge if smoke test fails.

---

## 10. Thresholds Rationale

### Why 0 Accidental Duplicates?
- Any accidental duplicate indicates a generation bug
- Duplicates inflate dataset size without adding information
- Waste evaluation compute and budget

### Why 40-60% Overall Balance?
- Majority class >60% enables trivial baselines (always predict majority)
- Balanced datasets ensure metrics reflect true understanding, not class priors

### Why 30-70% Per-Family Balance?
- Allows some natural skew from logic structure (e.g., FOL excludes chains)
- Stricter than overall (40-60%) would force artificial balancing that distorts semantics

### Why ≥10% Minority Class?
- Below 10%, class becomes negligible in metrics
- Models can ignore minority class with minimal accuracy penalty

### Why ≤5% Near-Duplicates Overall?
- Near-duplicates reduce effective dataset diversity
- >5% suggests template homogeneity or generation artifacts
- Paraphrase/perturbation families exempt (by design)

### Why 100% Determinism?
- Dataset must be immutable for versioning (v2.0 means specific bitwise artifact)
- Enables reproducible research
- Essential for debugging generation issues

---

## 11. Exception Handling

### When to Grant Exceptions

Exceptions to SOFT thresholds may be granted if:
1. **Justified by design:** Family is inherently imbalanced (e.g., `indicator_diagnostic`)
2. **Minimal impact:** Affects <5% of dataset, well-documented
3. **Infeasible to fix:** Would require ontology redesign or semantic distortion

**Approval Process:**
1. Document exception in `docs/EXCEPTIONS.md`
2. Include quantitative impact analysis
3. Propose mitigation (e.g., report MCC in addition to accuracy)
4. Get sign-off from 2+ maintainers

### When Exceptions Are NOT Allowed

- **HARD FAIL criteria:** No exceptions (accidental duplicates, split leakage, label leakage)
- **Determinism:** No exceptions (dataset must be reproducible)

---

## 12. Continuous Monitoring

### Post-Release Validation

Even after freeze, monitor:
- **External evaluations:** Do reported accuracies align with expectations?
- **Duplicate reports:** Users flagging near-duplicates?
- **Leakage reports:** Any post-hoc discoveries?

**Action Plan:**
1. If critical issue found (e.g., split leakage), issue dataset v2.1 with fix
2. If minor issue (e.g., near-duplicate in obscure family), document in errata

---

## 13. Summary Checklist

Before freezing API subset, verify:

- [ ] **Accidental duplicates = 0** (HARD FAIL)
- [ ] **Overall balance 40-60% TRUE** (HARD FAIL)
- [ ] **No label leakage** (HARD FAIL)
- [ ] **Determinism verified** (HARD FAIL)
- [ ] **Split integrity proven** (HARD FAIL)
- [ ] **Non-degeneracy confirmed** (≥10% minority class) (HARD FAIL)
- [ ] **Near-duplicate rates documented** (SOFT)
- [ ] **Per-family balance justified** (SOFT)
- [ ] **Manifest complete** (SOFT)
- [ ] **pre_freeze_check.py passes** (HARD FAIL)

**Sign-off:** All HARD FAIL criteria must pass. SOFT criteria must be either met or documented with justification.

---

**Revision History:**
- v2.0 (2026-02-18): Initial comprehensive standard for v2 dataset
