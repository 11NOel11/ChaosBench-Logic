<div align="center">

# ChaosBench-Logic v2

<p>
  <img src="https://img.shields.io/badge/tests-863%20passed-2ea44f" alt="Tests">
  <a href="./pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/11NOel11/ChaosBench-Logic/tree/v2.0.0"><img src="https://img.shields.io/badge/release-v2.0.0-b5651d" alt="Release"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="License"></a>
</p>

**Can LLMs reason about chaotic dynamical systems — or just pattern-match the leaderboard?**

</div>

ChaosBench-Logic tests LLM logical reasoning on dynamical systems (chaos, bifurcations,
attractors, Lyapunov exponents) with TRUE/FALSE questions grounded in a formal ontology — so
every answer is checkable against axioms, not vibes.

[Dataset](https://huggingface.co/datasets/11NOel11/ChaosBench-Logic) ·
[Published runs](./published_results/README.md) ·
[Docs](#documentation) ·
[Citation](./CITATION.cff)

- **Accepted:** v1 — AAAI 2026 BridgeLM workshop · v2 — ICLR 2026 LLM Reasoning workshop

## Release snapshot (v2.0.0)

- **40,886** canonical questions (+ 621 archived v1 = **41,507** total)
- **10** task families · **165** systems (30 core + 135 dysts)
- **27**-predicate ontology with axiom constraints

## Quick start

```bash
uv sync --all-groups
uv run pytest -q
uv run chaosbench eval --provider mock --dataset canonical --max-items 50   # no API key needed
```

## Validate the dataset and runs

```bash
uv run python scripts/validate_v2.py --strict --max-duplicate-questions 200
uv run chaosbench freeze
uv run chaosbench analyze-runs --runs-dir published_results/runs --out-dir artifacts/runs_audit
```

## Layout

```
chaosbench/          core package (data, tasks, eval, repair)
data/                canonical v2 files (v22_*.jsonl), manifest, selectors, archive
systems/             system metadata (core + dysts imports)
scripts/             build, validation, analysis, release tooling
published_results/   evaluation runs + manifests
docs/                dataset, protocol, ontology, camera-ready
```

## Dataset

- Canonical files: `data/v22_*.jsonl` (selector `data/canonical_v2_files.json`, manifest `data/v2_manifest.json`)
- SHA256: `cfcfcc739988ad99c38d47dd171ff39f67df3ddca7d8d452e8c77b30f14e7279`
- Schema, counts, hashes: [docs/DATASET.md](./docs/DATASET.md)
- Hugging Face: [11NOel11/ChaosBench-Logic](https://huggingface.co/datasets/11NOel11/ChaosBench-Logic)

## Documentation

- [DATASET.md](./docs/DATASET.md) — dataset schema, counts, hashes
- [EVAL_PROTOCOL.md](./docs/EVAL_PROTOCOL.md) — metrics and evaluation protocol
- [ONTOLOGY.md](./docs/ONTOLOGY.md) — ontology and predicate constraints
- [CAMERA_READY_REPO.md](./docs/CAMERA_READY_REPO.md) — camera-ready bundle workflow

## License

Code MIT (`LICENSE`) · dataset CC BY 4.0 (`LICENSE_DATA`).
