# Contributing to ChaosBench-Logic v2

This guide defines the preferred workflow for contributions to the official
`v2.0.0` repository line.

## Principles

- Keep the repository reproducible and release-ready.
- Prefer small, focused pull requests.
- Keep root clean and move operational notes into `docs/`.
- Never commit generated artifacts outside `published_results/`.

## Development Setup

Use `uv` as the default environment manager.

```bash
git clone https://github.com/11NOel11/ChaosBench-Logic.git
cd ChaosBench-Logic
uv sync --all-groups
```

Optional API key setup for provider-backed evaluation:

```bash
cp .env.example .env
```

See `docs/API_SETUP.md` for provider key details.

## Core Commands

```bash
# Full test suite
uv run pytest -q

# Repo hygiene check
uv run python scripts/repo_hygiene.py

# Canonical v2 validation
uv run python scripts/validate_v2.py --strict --max-duplicate-questions 200

# Smoke eval via unified CLI
uv run chaosbench eval --provider mock --dataset canonical --max-items 50
```

## Recommended Contribution Flow

1. Create a feature branch from the current working branch.
2. Make focused changes in one area (data, eval, docs, or tooling).
3. Run relevant checks locally.
4. Commit with a clear message (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
5. Open a PR with a concise rationale and verification notes.

## Pull Request Checklist

- Tests pass for changed behavior.
- Documentation is updated when behavior or interfaces change.
- No generated files added under `artifacts/`, `runs/`, `results/`, `workspace/`, or `tmp/`.
- Canonical dataset identity (`data/v2_manifest.json`) remains consistent unless intentionally changed.
- Changes follow `docs/REPO_POLICY.md`.

## Code Areas

- `chaosbench/` - package code (logic, tasks, eval, repair)
- `scripts/` - operational scripts and release utilities
- `data/` - canonical dataset files and manifests
- `systems/` - system metadata and truth assignments
- `tests/` - automated checks
- `docs/` - canonical documentation

## Adding or Updating Model Support

For provider-based CLI evaluation (`chaosbench eval`):

1. Add/update provider implementation in `chaosbench/eval/providers/`.
2. Register exports in `chaosbench/eval/providers/__init__.py`.
3. Add tests under `tests/` for failure handling and parsing behavior.
4. Update docs where user-facing behavior changes.

For legacy compatibility runner behavior, update `scripts/run_benchmark.py` only if needed.

## Dataset and Runs Policy

- Canonical v2 files: `data/v22_*.jsonl`.
- Canonical manifest: `data/v2_manifest.json`.
- Published lightweight run artifacts: `published_results/runs/`.
- Do not mix archived v1 and v2 in comparative claims unless explicitly stated.

## Security Reporting

For security issues, follow `SECURITY.md` and use GitHub Security Advisories.

## Questions

- Open an issue for bugs and feature requests.
- Link the relevant file paths and reproduction commands.
