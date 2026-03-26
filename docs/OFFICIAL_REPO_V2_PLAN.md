# ChaosBench-Logic v2 Official Repository Plan

This plan defines the target public layout for the official `v2.0.0` repository.

## Goals

- Keep root minimal and stable for external users.
- Make canonical dataset identity and reproducibility discoverable in one click.
- Separate source assets from generated artifacts.
- Preserve historical material without polluting top-level navigation.

## Target Root Layout

Keep only these top-level docs and metadata files:

- `README.md` (minimal quick-start + release identity)
- `DATASET_CARD.md` (HF-compatible narrative card)
- `CITATION.cff`
- `SECURITY.md`
- `LICENSE` and `LICENSE_DATA`

All additional prose docs live under `docs/`.

## Canonical Content Layout

- `data/`
  - canonical dataset files: `v22_*.jsonl`
  - release identity files: `v2_manifest.json`, `canonical_v2_files.json`
  - historical data in `data/archive/`
- `systems/`
  - core and `systems/dysts/` definitions
- `chaosbench/`
  - importable package and evaluation stack
- `scripts/`
  - generation, validation, analysis, and figure-build utilities
- `docs/`
  - canonical documentation
  - `docs/archive/` for legacy notes/changelogs

## Publication Surface (What Users Should See First)

1. `README.md` -> quick start + links
2. `docs/DATASET.md` -> schema, counts, hashes
3. `docs/EVAL_PROTOCOL.md` -> metrics and methodology
4. `docs/RELEASE_NOTES_V2.md` -> release deltas and known limits

## Artifact Policy

- Track only reproducible source inputs in git.
- Keep large/generated outputs under `artifacts/`, `workspace/`, or `runs/` and out of canonical root docs.
- Publish release-facing artifacts via:
  - GitHub repository: `https://github.com/11NOel11/ChaosBench-Logic`
  - Hugging Face dataset: `https://huggingface.co/datasets/11NOel11/ChaosBench-Logic`

## Versioning and Release Rules

- Official v2 line is `v2.0.0`.
- Keep `pyproject.toml`, `chaosbench/__init__.py`, `CITATION.cff`, and `data/v2_manifest.json` version fields aligned.
- Use `data/v2_manifest.json` as source of truth for dataset counts and checksums.

## Maintenance Checklist (Ongoing)

- Run strict validation and tests via `uv` before release updates.
- Keep `docs/README.md` index in sync with canonical docs.
- Move deprecated operational notes into `docs/archive/` instead of root.

## Camera-Ready Build Artifact

Generate a curated snapshot and zip bundle with:

```bash
uv run python scripts/build_camera_ready_repo.py --force
```

Outputs:

- `workspace/camera_ready_repo_v2.0.0/`
- `workspace/camera_ready_repo_v2.0.0.zip`
