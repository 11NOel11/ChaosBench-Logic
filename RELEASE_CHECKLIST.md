# Release Checklist

This document outlines the complete release process for ChaosBench-Logic.

## Pre-Release

### Code Quality

- [ ] All tests pass locally: `uv run pytest -v`
- [ ] CI pipeline is green on master branch
- [ ] Code coverage ≥ 80% for core evaluation logic
- [ ] No placeholder URLs in documentation (check for "yourusername")
- [ ] License files exist: `LICENSE` and `LICENSE_DATA`

### Dataset Validation

- [ ] Run strict validation: `python scripts/validate_v2.py --strict --max-duplicate-questions 200`
- [ ] Verify dataset manifest integrity
- [ ] Check all 30 system JSON files parse correctly
- [ ] Confirm batch file counts match manifest
- [ ] Validate SHA-256 hashes for all batches

### Dataset Quality Gates

- [ ] No duplicate questions beyond threshold (200 allowed)
- [ ] All system references resolve correctly
- [ ] Ground truth consistency checks pass
- [ ] Schema compliance for all questions
- [ ] Split assignments are valid and complete

### Documentation

- [ ] README.md statistics match dataset
- [ ] DATASET_CARD.md is complete and accurate
- [ ] V2_SPEC.md reflects current schema
- [ ] EVAL_PROTOCOL.md documents all metrics
- [ ] INDICATOR_THRESHOLDS.md and INDICATOR_COMPUTATION.md are current
- [ ] All internal links work
- [ ] Citation information is correct

### Reproducibility

- [ ] Configuration files are versioned: `configs/generation/*.yaml`
- [ ] Generation script produces identical output with same seed
- [ ] Manifest includes all necessary hashes
- [ ] Environment requirements are documented: `pyproject.toml`

## Release Process

### Version Tagging

- [ ] Update version in `chaosbench/__init__.py` to `2.1.0`
- [ ] Update version in `pyproject.toml` to `2.1.0`
- [ ] Commit version bump: `git commit -m "Bump version to 2.1.0"`
- [ ] Create annotated tag: `git tag -a v2.1.0 -m "Release v2.1.0: Extended dataset and validation"`
- [ ] Push tag: `git push origin v2.1.0`

### GitHub Release

- [ ] Create GitHub release from tag v2.1.0
- [ ] Release title: "ChaosBench-Logic v2.1.0"
- [ ] Release notes include:
  - Summary of changes
  - Dataset statistics
  - Breaking changes (if any)
  - Migration guide (if needed)
  - Known issues
- [ ] Attach artifacts (optional):
  - Compiled dataset archive
  - Pre-computed indicator data

### HuggingFace Upload

- [ ] Ensure HuggingFace dataset repo exists: `11NOel11/ChaosBench-Logic`
- [ ] Update dataset card with v2.1.0 information
- [ ] Upload all batch files to HuggingFace
- [ ] Upload manifest: `v2_manifest.json`
- [ ] Upload system definitions (if changed)
- [ ] Tag release on HuggingFace: `v2.1.0`
- [ ] Verify dataset loads correctly:
  ```python
  from datasets import load_dataset
  ds = load_dataset("11NOel11/ChaosBench-Logic", "single_turn")
  assert len(ds['test']) > 0
  ```

### Package Distribution

- [ ] Build package: `uv build`
- [ ] Test package installation in clean environment
- [ ] Verify imports work: `import chaosbench; print(chaosbench.__version__)`
- [ ] (Optional) Publish to PyPI if public release desired

### Release Announcement

- [ ] Update repository README badges (if versions changed)
- [ ] Post to GitHub Discussions announcing release
- [ ] Update any external documentation or wikis

## Post-Release

### Monitoring

- [ ] Monitor GitHub Issues for bug reports
- [ ] Check HuggingFace downloads and usage
- [ ] Verify CI remains green on tagged release
- [ ] Monitor for API compatibility issues with LLM providers

### Leaderboard Update

- [ ] If leaderboard exists, add baseline results for v2.1.0
- [ ] Document any performance changes from previous version
- [ ] Update evaluation results in `published_results/` if needed

### Documentation Updates

- [ ] Archive v2.0.0 documentation if major changes
- [ ] Update quickstart guides with any new features
- [ ] Add migration guide if breaking changes exist

### Community Engagement

- [ ] Respond to issues and questions promptly
- [ ] Review and merge community pull requests
- [ ] Acknowledge contributors in release notes

## Hotfix Process

If critical issues are discovered post-release:

### Immediate Response

- [ ] Assess severity and impact
- [ ] Create hotfix branch from release tag: `git checkout -b hotfix-v2.1.1 v2.1.0`
- [ ] Implement minimal fix
- [ ] Add regression test

### Hotfix Release

- [ ] Update version to v2.1.1
- [ ] Create new tag: `git tag -a v2.1.1 -m "Hotfix: [brief description]"`
- [ ] Push tag and create GitHub release
- [ ] Update HuggingFace if dataset affected
- [ ] Announce hotfix in GitHub Issues

### Documentation

- [ ] Update CHANGELOG.md with hotfix details
- [ ] Document issue in known issues section
- [ ] Add troubleshooting entry if user-facing

## Rollback Procedure

If release must be rolled back:

- [ ] Mark GitHub release as "Pre-release"
- [ ] Add warning to release notes
- [ ] Do NOT delete tags (breaks reproducibility)
- [ ] Revert HuggingFace to previous working version
- [ ] Post announcement explaining rollback
- [ ] Document issue for future prevention

## Version Numbering

ChaosBench-Logic uses semantic versioning:

- **Major (X.0.0)**: Breaking changes, incompatible API
- **Minor (2.X.0)**: New features, backward-compatible
- **Patch (2.1.X)**: Bug fixes, no new features

Current release: **v2.1.0** (minor release with extended dataset and validation)

## Release Notes Template

```markdown
## ChaosBench-Logic v2.1.0

**Release Date:** YYYY-MM-DD

### Summary

Brief overview of release focus and major changes.

### Dataset Changes

- Extended dataset to ~25,000 questions
- Added ~100 dysts-imported systems
- Improved indicator threshold validation

### New Features

- List new features
- With brief descriptions

### Bug Fixes

- List bug fixes
- With issue numbers if applicable

### Breaking Changes

- List any breaking changes
- With migration instructions

### Known Issues

- List known issues
- With workarounds if available

### Contributors

Thanks to all contributors who made this release possible.
```

## Contact

For release-related questions, contact the maintainer via GitHub Issues.
