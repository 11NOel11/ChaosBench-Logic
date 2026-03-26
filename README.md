<div align="center">

# ChaosBench-Logic v2

<p>
  <img src="https://img.shields.io/badge/tests-863%20passed-2ea44f" alt="Tests">
  <a href="./pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/11NOel11/ChaosBench-Logic/tree/v2.0.0"><img src="https://img.shields.io/badge/release-v2.0.0-b5651d" alt="Release"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="License"></a>
</p>

<p><strong>Official v2.0.0 release for evaluating LLM logical reasoning on dynamical systems.</strong></p>

<p>
  <a href="https://huggingface.co/datasets/11NOel11/ChaosBench-Logic">Dataset</a> |
  <a href="./published_results/README.md">Published Runs</a> |
  <a href="./docs/CAMERA_READY_SUBMISSION.md">Camera-Ready</a> |
  <a href="./CITATION.cff">Citation</a>
</p>

</div>

## Workshop Acceptance

- v1 accepted at AAAI 2026 BridgeLM Reasoning Workshop.
- v2 accepted at ICLR 2026 LLM Reasoning Workshop.

## Release Snapshot

- Official release: `v2.0.0`
- Canonical v2 questions: `40,886`
- Archived v1 questions: `621`
- Total questions: `41,507`
- Task families: `10`
- Systems: `165` (`30` core + `135` dysts)
- Ontology size: `27` predicates with axiom constraints

## Repository Layout

- `chaosbench/` - core package (data, tasks, eval, repair)
- `data/` - canonical v2 files (`v22_*.jsonl`), manifest, selectors, archive
- `systems/` - system metadata for core and dysts imports
- `scripts/` - build, validation, analysis, and release tooling
- `published_results/runs/` - published evaluation runs and manifests
- `docs/` - dataset, protocol, ontology, and camera-ready documentation

## Quick Start

```bash
uv sync --all-groups
uv run pytest -q
uv run chaosbench eval --provider mock --subset data/ci_smoke/smoke.jsonl
```

## Validate v2 and Runs

```bash
# Validate canonical v2 dataset
uv run python scripts/validate_v2.py --strict --max-duplicate-questions 200
uv run chaosbench freeze

# Validate published runs and generate audit tables
uv run chaosbench analyze-runs --runs-dir published_results/runs --out-dir artifacts/runs_audit
uv run python scripts/analyze_runs.py --published_dir published_results/runs --out_dir artifacts/runs_audit --paper_assets_dir artifacts/paper_assets
```

## Dataset Identity

- Canonical files: `data/v22_*.jsonl`
- Canonical selector: `data/canonical_v2_files.json`
- Manifest: `data/v2_manifest.json`
- Global SHA256: `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279`

## Documentation

- `docs/DATASET.md` - dataset schema, counts, hashes
- `docs/EVAL_PROTOCOL.md` - metrics and evaluation protocol
- `docs/ONTOLOGY.md` - ontology and predicate constraints
- `docs/CAMERA_READY_REPO.md` - camera-ready repo bundle workflow

## License

- Code: MIT (`LICENSE`)
- Dataset: CC BY 4.0 (`LICENSE_DATA`)
