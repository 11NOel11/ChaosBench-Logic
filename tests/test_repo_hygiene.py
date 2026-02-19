"""Repository hygiene tests.

Fails if:
- artifacts/ is not gitignored
- reports/ or runs/ are tracked in git
- Unexpected *SUMMARY*.md files exist at repo root
- /tmp paths appear in tracked docs

See docs/REPO_POLICY.md for the full policy.
"""

import fnmatch
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


def _read_gitignore_patterns():
    gi = PROJECT_ROOT / ".gitignore"
    if not gi.exists():
        return []
    return [
        l.strip()
        for l in gi.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


def _dir_is_gitignored(dirname: str, patterns) -> bool:
    for pat in patterns:
        pat_clean = pat.rstrip("/")
        if pat_clean == dirname:
            return True
        if fnmatch.fnmatch(dirname, pat_clean):
            return True
    return False


def _git_ls_files(*paths) -> list:
    """Return files tracked in git under the given paths."""
    result = subprocess.run(
        ["git", "ls-files", "--", *paths],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return [l.strip() for l in result.stdout.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGitignore:
    def test_artifacts_is_gitignored(self):
        """artifacts/ must be gitignored so generated outputs are never tracked."""
        patterns = _read_gitignore_patterns()
        assert _dir_is_gitignored("artifacts", patterns), (
            "'artifacts/' is not in .gitignore. "
            "Add 'artifacts/' to .gitignore to prevent tracking generated outputs."
        )

    def test_reports_is_gitignored(self):
        """reports/ must be gitignored."""
        patterns = _read_gitignore_patterns()
        assert _dir_is_gitignored("reports", patterns), (
            "'reports/' is not in .gitignore."
        )

    def test_runs_is_gitignored(self):
        """runs/ must be gitignored."""
        patterns = _read_gitignore_patterns()
        assert _dir_is_gitignored("runs", patterns), (
            "'runs/' is not in .gitignore."
        )

    def test_workspace_is_gitignored(self):
        """workspace/ must be gitignored."""
        patterns = _read_gitignore_patterns()
        assert _dir_is_gitignored("workspace", patterns), (
            "'workspace/' is not in .gitignore."
        )

    def test_scratch_is_gitignored(self):
        """scratch/ must be gitignored."""
        patterns = _read_gitignore_patterns()
        assert _dir_is_gitignored("scratch", patterns), (
            "'scratch/' is not in .gitignore."
        )

    def test_tmp_is_gitignored(self):
        """tmp/ must be gitignored."""
        patterns = _read_gitignore_patterns()
        assert _dir_is_gitignored("tmp", patterns), (
            "'tmp/' is not in .gitignore."
        )

    def test_data_manifest_not_ignored(self):
        """data/v2_manifest.json must NOT be gitignored (it is canonical)."""
        # If gitignored, it won't appear in ls-files
        tracked = _git_ls_files("data/v2_manifest.json")
        # It's OK if it's untracked (new repo) but it must not be explicitly ignored
        # We check it's not in a deny-list pattern
        patterns = _read_gitignore_patterns()
        manifest_name = "data/v2_manifest.json"
        for pat in patterns:
            if pat.startswith("!"):
                continue  # negation pattern — allows the file
            if fnmatch.fnmatch(manifest_name, pat):
                # Check there's no negation pattern that overrides
                negated = any(
                    p.startswith("!") and fnmatch.fnmatch(manifest_name, p[1:])
                    for p in patterns
                )
                if not negated:
                    pytest.fail(
                        f"data/v2_manifest.json is gitignored by pattern '{pat}' "
                        "but it is a canonical file and must be tracked."
                    )


class TestTrackedFiles:
    def test_reports_not_tracked(self):
        """reports/ directory must not have any tracked files in git."""
        tracked = _git_ls_files("reports/")
        assert not tracked, (
            f"Found tracked files in reports/: {tracked[:5]}. "
            "Run: git rm --cached <file> for each."
        )

    def test_runs_not_tracked(self):
        """runs/ directory must not have any tracked files in git."""
        tracked = _git_ls_files("runs/")
        assert not tracked, (
            f"Found tracked files in runs/: {tracked[:5]}. "
            "Run: git rm --cached <file> for each."
        )

    def test_artifacts_not_tracked(self):
        """artifacts/ directory must not have any tracked files in git."""
        tracked = _git_ls_files("artifacts/")
        assert not tracked, (
            f"Found tracked files in artifacts/: {tracked[:5]}. "
            "Run: git rm --cached <file> for each."
        )

    def test_workspace_not_tracked(self):
        """workspace/ directory must not have any tracked files in git."""
        tracked = _git_ls_files("workspace/")
        assert not tracked, (
            f"Found tracked files in workspace/: {tracked[:5]}."
        )


class TestRootCleanup:
    def test_no_summary_md_at_root(self):
        """No *SUMMARY*.md files allowed at repo root."""
        violations = []
        for item in PROJECT_ROOT.iterdir():
            if item.is_file() and "SUMMARY" in item.name.upper() and item.suffix.lower() == ".md":
                violations.append(item.name)
        assert not violations, (
            f"Found *SUMMARY*.md at repo root: {violations}. "
            "Move to docs/archive/ or delete."
        )

    def test_no_report_md_at_root(self):
        """No *REPORT*.md files allowed at repo root."""
        violations = []
        for item in PROJECT_ROOT.iterdir():
            if (item.is_file()
                    and "REPORT" in item.name.upper()
                    and item.suffix.lower() == ".md"):
                violations.append(item.name)
        assert not violations, (
            f"Found *REPORT*.md at repo root: {violations}. "
            "Move to docs/archive/ or delete."
        )

    def test_no_implementation_md_at_root(self):
        """No IMPLEMENTATION*.md files allowed at repo root."""
        violations = []
        for item in PROJECT_ROOT.iterdir():
            if (item.is_file()
                    and item.name.upper().startswith("IMPLEMENTATION")
                    and item.suffix.lower() == ".md"):
                violations.append(item.name)
        assert not violations, (
            f"Found IMPLEMENTATION*.md at repo root: {violations}. "
            "Move to docs/archive/ or delete."
        )

    def test_no_log_files_at_root(self):
        """No .log or .out files at repo root."""
        violations = []
        for item in PROJECT_ROOT.iterdir():
            if item.is_file() and item.suffix.lower() in {".log", ".out"}:
                violations.append(item.name)
        assert not violations, (
            f"Found log/out files at repo root: {violations}."
        )


class TestDocContent:
    def test_no_tmp_paths_in_tracked_docs(self):
        """Tracked documentation must not contain /tmp/ paths."""
        tmp_pattern = re.compile(r"/tmp/[^\s\"')\]>]+")
        doc_paths = list((PROJECT_ROOT / "docs").rglob("*.md"))
        doc_paths += [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "DATASET_CARD.md",
        ]

        violations = []
        for doc in doc_paths:
            if not doc.exists():
                continue
            # Skip archive files (might have historical /tmp references)
            if "archive" in doc.parts:
                continue
            try:
                text = doc.read_text(encoding="utf-8", errors="ignore")
                matches = tmp_pattern.findall(text)
                if matches:
                    rel = str(doc.relative_to(PROJECT_ROOT))
                    violations.append(f"{rel}: {matches[:3]}")
            except Exception:
                pass

        assert not violations, (
            "Found /tmp/ paths in tracked docs:\n" +
            "\n".join(f"  {v}" for v in violations)
        )

    def test_canonical_docs_exist(self):
        """Key canonical documentation files must exist."""
        required = [
            "docs/README.md",
            "docs/DATASET.md",
            "docs/EVAL_PROTOCOL.md",
            "docs/ONTOLOGY.md",
            "docs/QUALITY_STANDARD.md",
            "docs/REPO_POLICY.md",
            "README.md",
            "DATASET_CARD.md",
        ]
        missing = [f for f in required if not (PROJECT_ROOT / f).exists()]
        assert not missing, f"Missing canonical docs: {missing}"

    def test_repo_policy_exists(self):
        """docs/REPO_POLICY.md must exist and be non-empty."""
        policy = PROJECT_ROOT / "docs" / "REPO_POLICY.md"
        assert policy.exists(), "docs/REPO_POLICY.md does not exist"
        assert policy.stat().st_size > 100, "docs/REPO_POLICY.md appears empty"
