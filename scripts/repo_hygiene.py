#!/usr/bin/env python3
"""Repository hygiene checker for ChaosBench-Logic.

Enforces the rules defined in docs/REPO_POLICY.md.

Usage:
    python scripts/repo_hygiene.py              # dry-run: print what would be flagged
    python scripts/repo_hygiene.py --apply      # apply safe moves (no deletes)
    python scripts/repo_hygiene.py --delete     # also delete explicitly flagged files
    python scripts/repo_hygiene.py --print-policy  # print the policy rules

Exit Codes:
    0  No violations found (or --print-policy)
    1  Violations found (or --apply/--delete found issues)
    2  Script error

Actions are logged to artifacts/repo_cleanup/actions.log
"""

import argparse
import fnmatch
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

# Files / patterns that MUST NOT appear at the repo root
ROOT_BANNED_PATTERNS = [
    "*SUMMARY*.md",
    "*REPORT*.md",
    "IMPLEMENTATION*.md",
    "AUDIT_REPORT_*.md",
    "*.log",
    "*.out",
    "*.tmp",
    "*.temp",
]

# Directories that must be gitignored (not tracked)
MUST_BE_GITIGNORED = [
    "artifacts",
    "reports",
    "runs",
    "workspace",
    "scratch",
    "tmp",
]

# Patterns that must never be tracked in git (checked against index)
NEVER_TRACK_PATTERNS = [
    "*.log",
    "*.out",
    "reports/*",
    "runs/*",
    "artifacts/*",
    "workspace/*",
    "scratch/*",
    "tmp/*",
    "data/backup_*",
]

# Canonical docs that MUST stay in place
CANONICAL_DOCS = [
    "README.md",
    "DATASET_CARD.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "docs/README.md",
    "docs/DATASET.md",
    "docs/EVAL_PROTOCOL.md",
    "docs/QUALITY_STANDARD.md",
    "docs/ONTOLOGY.md",
    "docs/REPO_POLICY.md",
]

POLICY_TEXT = """
Repository Hygiene Policy
=========================

Source: docs/REPO_POLICY.md

1. TRACKED (must be in git)
   - Source code: chaosbench/**/*.py, scripts/*.py, tests/*.py
   - Canonical dataset: data/v22_*.jsonl, data/v2_manifest.json
   - Archived batches: data/archive/**/*.jsonl
   - CI smoke data: data/ci_smoke/
   - Configs: configs/**/*.yaml, pyproject.toml, uv.lock
   - System definitions: systems/**/*.json
   - Canonical docs: docs/*.md, README.md, DATASET_CARD.md, etc.
   - Published results: published_results/

2. NEVER TRACKED (gitignored)
   - artifacts/       — all generated outputs
   - reports/         — pipeline reports
   - runs/            — evaluation run outputs
   - workspace/       — local working materials
   - scratch/         — scratch work
   - tmp/             — temporary files
   - figures/         — generated figures
   - results/         — local evaluation results
   - data/backup_*/   — backup snapshots
   - *.log, *.out     — log files

3. ROOT DIRECTORY RULES
   - No *SUMMARY*.md files
   - No *REPORT*.md files
   - No IMPLEMENTATION*.md files
   - No *.log, *.out, *.tmp files

4. CONTENT RULES
   - No /tmp paths in tracked docs
   - No absolute local paths in tracked docs

5. ARCHIVE RULES
   - Historical AI-generated summaries → docs/archive/ (tracked)
   - Generated outputs → artifacts/ (gitignored)
"""


# ---------------------------------------------------------------------------
# Violation dataclass
# ---------------------------------------------------------------------------

class Violation:
    def __init__(self, severity: str, path: str, rule: str, suggestion: str = ""):
        self.severity = severity  # ERROR or WARN
        self.path = path
        self.rule = rule
        self.suggestion = suggestion

    def __str__(self) -> str:
        s = f"[{self.severity}] {self.path}: {self.rule}"
        if self.suggestion:
            s += f"\n         Suggestion: {self.suggestion}"
        return s


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _read_gitignore_patterns(root: Path) -> List[str]:
    gi_path = root / ".gitignore"
    if not gi_path.exists():
        return []
    lines = gi_path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def _is_in_gitignore(directory: str, patterns: List[str]) -> bool:
    """Check if a directory name is covered by any gitignore pattern."""
    for pat in patterns:
        # Simple check: exact match or trailing-slash match
        pat_clean = pat.rstrip("/")
        if pat_clean == directory or pat_clean == f"{directory}/":
            return True
        if fnmatch.fnmatch(directory, pat_clean):
            return True
    return False


def check_root_banned_files(root: Path) -> List[Violation]:
    """Check for banned file patterns in repo root."""
    violations = []
    for item in root.iterdir():
        if not item.is_file():
            continue
        name = item.name
        for pat in ROOT_BANNED_PATTERNS:
            if fnmatch.fnmatch(name, pat):
                violations.append(Violation(
                    "ERROR",
                    str(item.relative_to(root)),
                    f"Banned pattern '{pat}' at repo root",
                    "Move to docs/archive/ or artifacts/repo_cleanup/",
                ))
                break
    return violations


def check_gitignored_dirs(root: Path) -> List[Violation]:
    """Check that required directories are gitignored."""
    violations = []
    patterns = _read_gitignore_patterns(root)
    for dirname in MUST_BE_GITIGNORED:
        if not _is_in_gitignore(dirname, patterns):
            violations.append(Violation(
                "ERROR",
                dirname + "/",
                f"Directory '{dirname}' is not gitignored",
                f"Add '{dirname}/' to .gitignore",
            ))
    return violations


def check_tracked_reports_or_runs(root: Path) -> List[Violation]:
    """Check that reports/ and runs/ are not tracked in git (not in git index)."""
    violations = []
    try:
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--", "reports/", "runs/", "artifacts/"],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        tracked = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        for f in tracked:
            violations.append(Violation(
                "ERROR",
                f,
                "File in reports/, runs/, or artifacts/ is tracked in git",
                "Run: git rm --cached " + f,
            ))
    except Exception as e:
        violations.append(Violation("WARN", "git", f"Could not check git index: {e}"))
    return violations


def check_summary_files_at_root(root: Path) -> List[Violation]:
    """Check for *SUMMARY*.md files at repo root."""
    violations = []
    for item in root.iterdir():
        if not item.is_file():
            continue
        name = item.name.upper()
        if "SUMMARY" in name and name.endswith(".MD"):
            violations.append(Violation(
                "ERROR",
                item.name,
                "AI-generated SUMMARY markdown at repo root",
                "Move to docs/archive/ or delete if obsolete",
            ))
    return violations


def check_tmp_paths_in_docs(root: Path) -> List[Violation]:
    """Check for /tmp paths in tracked documentation files."""
    violations = []
    doc_dirs = ["docs", "."]
    checked_extensions = {".md", ".rst", ".txt"}
    tmp_pattern = re.compile(r"/tmp/[^\s\"']+")

    for doc_dir in doc_dirs:
        scan_root = root / doc_dir if doc_dir != "." else root
        if not scan_root.is_dir():
            continue
        depth = 0 if doc_dir == "." else 2
        for dirpath, dirnames, filenames in os.walk(scan_root):
            # Skip non-doc dirs at root level
            if doc_dir == ".":
                dirnames[:] = [
                    d for d in dirnames
                    if d not in {
                        ".git", ".venv", ".pytest_cache", "__pycache__",
                        "artifacts", "reports", "runs", "workspace",
                        "chaosbench", "scripts", "tests", "data", "systems",
                        "configs", "infra", "published_results", "figures",
                    }
                ]
            fpath = Path(dirpath)
            for fname in filenames:
                if Path(fname).suffix not in checked_extensions:
                    continue
                full = fpath / fname
                try:
                    text = full.read_text(encoding="utf-8", errors="ignore")
                    matches = tmp_pattern.findall(text)
                    if matches:
                        rel = str(full.relative_to(root))
                        violations.append(Violation(
                            "WARN",
                            rel,
                            f"/tmp path(s) found in document: {matches[:3]}",
                            "Replace with relative paths or remove",
                        ))
                except Exception:
                    pass

    return violations


def check_results_zip_tracked(root: Path) -> List[Violation]:
    """Check if results.zip or similar archive is tracked."""
    violations = []
    try:
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--", "*.zip", "*.tar.gz", "*.tar", "*.rar"],
            capture_output=True, text=True, cwd=str(root),
        )
        tracked = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        for f in tracked:
            violations.append(Violation(
                "WARN",
                f,
                "Archive file is tracked in git",
                "Add to .gitignore and run: git rm --cached " + f,
            ))
    except Exception:
        pass
    return violations


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_checks(root: Path) -> List[Violation]:
    all_violations: List[Violation] = []
    all_violations.extend(check_root_banned_files(root))
    all_violations.extend(check_gitignored_dirs(root))
    all_violations.extend(check_tracked_reports_or_runs(root))
    all_violations.extend(check_summary_files_at_root(root))
    all_violations.extend(check_tmp_paths_in_docs(root))
    all_violations.extend(check_results_zip_tracked(root))
    return all_violations


def _log_action(log_path: Path, action: str, details: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {action}: {details}\n")


def apply_safe_moves(root: Path, dry_run: bool = True) -> List[str]:
    """Apply safe moves: root-level SUMMARY/REPORT files → docs/archive/."""
    actions = []
    archive_dir = root / "docs" / "archive"
    log_path = root / "artifacts" / "repo_cleanup" / "actions.log"

    for item in root.iterdir():
        if not item.is_file():
            continue
        name = item.name.upper()
        is_banned = False
        for pat in ROOT_BANNED_PATTERNS:
            if fnmatch.fnmatch(item.name, pat):
                is_banned = True
                break

        if is_banned and item.suffix.lower() == ".md":
            dest = archive_dir / item.name
            msg = f"MOVE {item.relative_to(root)} -> docs/archive/{item.name}"
            actions.append(msg)
            if not dry_run:
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))
                _log_action(log_path, "MOVE", f"{item} -> {dest}")

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ChaosBench-Logic repository hygiene checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply safe moves (default: dry-run only)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Also delete explicitly flagged files (requires --apply)",
    )
    parser.add_argument(
        "--print-policy",
        action="store_true",
        help="Print the policy rules and exit",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.print_policy:
        print(POLICY_TEXT)
        return 0

    root = args.root.resolve()
    print(f"ChaosBench-Logic Repo Hygiene Checker")
    print(f"Root: {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    # --- Run checks ---
    violations = run_checks(root)

    errors = [v for v in violations if v.severity == "ERROR"]
    warns = [v for v in violations if v.severity == "WARN"]

    if not violations:
        print("OK: No violations found.")
    else:
        if errors:
            print(f"ERRORS ({len(errors)}):")
            for v in errors:
                print(f"  {v}")
            print()
        if warns:
            print(f"WARNINGS ({len(warns)}):")
            for v in warns:
                print(f"  {v}")
            print()

    # --- Apply safe moves ---
    if args.apply or not violations:
        actions = apply_safe_moves(root, dry_run=not args.apply)
        if actions:
            label = "Applied" if args.apply else "Would apply"
            print(f"{label} moves:")
            for a in actions:
                print(f"  {a}")
        elif args.apply:
            print("No safe moves to apply.")

    # Log summary
    log_path = root / "artifacts" / "repo_cleanup" / "actions.log"
    if args.apply:
        _log_action(log_path, "SCAN", f"{len(errors)} errors, {len(warns)} warnings")

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
