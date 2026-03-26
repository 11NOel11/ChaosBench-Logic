# Camera-Ready Repo Status

Status: ready.

## Built Artifacts

- Curated snapshot: `workspace/camera_ready_repo_v2.0.0/`
- Zip bundle: `workspace/camera_ready_repo_v2.0.0.zip`
- Bundle manifest: `workspace/camera_ready_repo_v2.0.0/CAMERA_READY_MANIFEST.json`
- Bundle notes: `workspace/camera_ready_repo_v2.0.0/README_CAMERA_READY_BUNDLE.md`

## Build Command

```bash
uv run python scripts/build_camera_ready_repo.py --force
```

## Validation Commands

```bash
uv run python scripts/repo_hygiene.py
uv run pytest -q tests/test_repo_hygiene.py tests/test_batch_consistency.py
```

## Paper Links Included

- GitHub: `https://github.com/11NOel11/ChaosBench-Logic`
- Hugging Face: `https://huggingface.co/datasets/11NOel11/ChaosBench-Logic`
