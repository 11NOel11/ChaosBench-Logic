# ChaosBench-Logic

Official v2.0.0 release of ChaosBench-Logic, a benchmark for evaluating LLM reasoning on dynamical systems.

## v2 Snapshot

- 40,886 v2 questions (default)
- 10 task families
- 165 systems (30 core + 135 dysts)
- 27 ontology predicates
- 621 archived v1 questions in `data/archive/v1/`

## Core Links

- Repository: https://github.com/11NOel11/ChaosBench-Logic
- Dataset (Hugging Face): https://huggingface.co/datasets/11NOel11/ChaosBench-Logic
- Dataset docs: `docs/DATASET.md`
- Evaluation protocol: `docs/EVAL_PROTOCOL.md`
- Ontology: `docs/ONTOLOGY.md`
- Camera-ready package: `docs/CAMERA_READY_SUBMISSION.md`

## Quick Start

```bash
uv sync --all-groups
uv run pytest -q
uv run python run_benchmark.py --model gpt4 --mode zeroshot
```

## Dataset Identity

- Canonical files: `data/v22_*.jsonl`
- Canonical selector: `data/canonical_v2_files.json`
- Manifest: `data/v2_manifest.json`
- Global SHA256: `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279`

## Citation

Use `CITATION.cff`.

## License

- Code: MIT (`LICENSE`)
- Dataset: CC BY 4.0 (`LICENSE_DATA`)
