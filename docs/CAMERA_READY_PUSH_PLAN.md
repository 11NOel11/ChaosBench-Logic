# Camera-Ready Push Plan (v2.0.0)

This plan defines exactly what should be pushed for the official camera-ready
ChaosBench-Logic v2.0.0 release.

## Push Scope A (Recommended): Official Camera-Ready Release

Push only release-facing cleanup and metadata alignment:

- Root cleanup
  - `README.md`
  - `DATASET_CARD.md`
  - `CITATION.cff`
  - delete `CHANGELOG.md`
  - delete `RELEASE_CHECKLIST.md`
  - add `docs/archive/CHANGELOG_LEGACY.md`
- Version and manifest alignment
  - `pyproject.toml`
  - `chaosbench/__init__.py`
  - `data/v2_manifest.json`
- Canonical docs and camera-ready docs
  - `docs/DATASET.md`
  - `docs/ONTOLOGY.md`
  - `docs/RELEASE_NOTES_V2.md`
  - `docs/RUN_PLAN.md`
  - `docs/V2_SPEC.md`
  - `docs/FUTURE_WORK.md`
  - `docs/SCALING_ROADMAP.md`
  - `docs/REPO_POLICY.md`
  - `docs/README.md`
  - `docs/CAMERA_READY_SUBMISSION.md`
  - `docs/CAMERA_READY_LINKS.md`
  - `docs/CAMERA_READY_REPO.md`
  - `docs/CAMERA_READY_REPO_STATUS.md`
  - `docs/CAMERA_READY_PUSH_PLAN.md`
  - `docs/CLAIM_EVIDENCE_MATRIX.md`
  - `docs/OFFICIAL_REPO_V2_PLAN.md`
- CI/hygiene/test wording and repo-bundle builder
  - `.github/workflows/ci.yml`
  - `.gitignore`
  - `scripts/repo_hygiene.py`
  - `scripts/build_v2_dataset.py`
  - `scripts/build_camera_ready_repo.py`
  - `tests/test_batch_consistency.py`
  - `tests/test_fol_rules.py`
  - `tests/test_dysts_import.py`
  - `chaosbench/logic/axioms.py`
  - `chaosbench/tasks/multi_hop.py`
  - `chaosbench/tasks/fol_inference.py`

## Push Scope B (Separate Branch): Ongoing Repair/Survey Work

Do not include repair/survey WIP in the official camera-ready release commit.
Keep these changes in a separate branch/PR:

- `chaosbench/repair/`
- `scripts/run_m*_*.py`, `scripts/*repair*`, `scripts/*m6*`
- `tests/test_repair_*.py`, `tests/test_m5_*.py`, `tests/test_online_controller.py`
- Other unrelated local edits (for example `chaosbench/eval/prompts.py`,
  `chaosbench/eval/providers/gemini.py`, `scripts/verify_figures.py`, `uv.lock`)

## Root Cleanliness Target

After Scope A, root should keep a minimal public surface:

- `README.md`
- `DATASET_CARD.md`
- `CITATION.cff`
- `SECURITY.md`
- `LICENSE`, `LICENSE_DATA`

All additional operational docs should live under `docs/`.
