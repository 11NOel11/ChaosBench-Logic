# ChaosBench-Logic: Camera-Ready Submission (Final)

This document is the final camera-ready text package for the ICLR 2026 LLM Reasoning Workshop submission.

## Paper Metadata

- Title: ChaosBench-Logic: A Benchmark for Evaluating Large Language Models on Complex Reasoning about Dynamical Systems
- Release: v2.0.0 (official v2 release)
- Repository: https://github.com/11NOel11/ChaosBench-Logic
- Dataset: https://huggingface.co/datasets/11NOel11/ChaosBench-Logic

## Final Abstract

We present ChaosBench-Logic, a benchmark for evaluating large language model reasoning on chaotic and non-chaotic dynamical systems. The benchmark targets scientific reasoning behaviors beyond surface pattern matching, including logical inference, multi-hop deduction, indicator interpretation, adversarial robustness, and consistency under paraphrase and perturbation. The official v2 release contains 40,886 questions across 10 task families, 165 systems (30 manually curated core systems and 135 systems imported from dysts), 27 ontology predicates, and 78 directed axiom edges. Questions are designed for closed-book evaluation: models receive natural language prompts only, without direct access to equations, simulators, or numerical traces. We provide a reproducible data release with canonical file lists, per-file hashes, and a global dataset checksum, plus standardized evaluation scripts and reporting utilities. ChaosBench-Logic is intended as a stress test for reliable scientific reasoning, helping researchers measure both accuracy and logical consistency in model behavior.

## Main Contributions

1. A large-scale reasoning benchmark focused on dynamical systems and chaos-related concepts.
2. A structured ontology-driven question design that supports atomic, compositional, and adversarial reasoning tests.
3. A release-ready reproducibility pipeline with manifest-backed canonical files and verification artifacts.

## Reproducibility Statement

- Canonical dataset files: `data/v22_*.jsonl`.
- Canonical selector file: `data/canonical_v2_files.json`.
- Manifest and per-file hashes: `data/v2_manifest.json`.
- Global dataset SHA256: `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279`.

### Verification Commands

```bash
uv sync --all-groups
uv run python scripts/validate_v2.py --strict --max-duplicate-questions 200
uv run python scripts/freeze_v2_dataset.py
uv run pytest -q
```

## Artifact Availability Text (Camera-Ready)

Use the following wording in the final PDF and OpenReview fields:

```text
Code and reproducibility assets are available at https://github.com/11NOel11/ChaosBench-Logic.
The official ChaosBench-Logic v2.0.0 dataset release is available at https://huggingface.co/datasets/11NOel11/ChaosBench-Logic.
```

## Notes for Final Submission

- Cite the release as v2.0.0.
- Keep all reported dataset counts aligned with the official v2 numbers in the manifest.
- Use `docs/CLAIM_EVIDENCE_MATRIX.md` to map paper claims to repository artifacts before final upload.
