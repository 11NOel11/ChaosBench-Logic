# Camera-Ready Repo Build

This document defines how to build the camera-ready repository snapshot for the
ICLR 2026 workshop release.

## Command

```bash
uv run python scripts/build_camera_ready_repo.py --force
```

## Output

- Curated directory: `workspace/camera_ready_repo_v2.0.0/`
- Zip archive: `workspace/camera_ready_repo_v2.0.0.zip`
- Manifest: `workspace/camera_ready_repo_v2.0.0/CAMERA_READY_MANIFEST.json`
- Bundle notes: `workspace/camera_ready_repo_v2.0.0/README_CAMERA_READY_BUNDLE.md`

## Included Content

- Minimal root release surface (`README.md`, `DATASET_CARD.md`, `CITATION.cff`,
  `SECURITY.md`, licenses)
- Canonical v2 dataset files and manifests
- Source code, configs, scripts, tests
- Canonical docs under `docs/`

## Excluded Content

- Local/generated directories (`artifacts/`, `runs/`, `workspace/`, `results/`,
  logs, caches)

## Public Links

- GitHub: `https://github.com/11NOel11/ChaosBench-Logic`
- Hugging Face: `https://huggingface.co/datasets/11NOel11/ChaosBench-Logic`
