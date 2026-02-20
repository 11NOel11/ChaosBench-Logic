"""chaosbench/data/hashing.py — Canonical dataset fingerprinting.

Single source-of-truth for computing dataset SHA256 hashes.
Used by BOTH freeze_v2_dataset.py and chaosbench/eval/run.py so that
the stored dataset_global_sha256 in a run manifest always matches the
freeze manifest's global_sha256.

Hash formula (authoritative):
    global_sha256 = sha256(
        concat over sorted(files) of:
            "<rel_path>:<file_sha256>:<line_count>\\n"
    )

The :count component is included because it binds the hash to the
specific number of data rows, not just the byte content.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "sha256_file",
    "dataset_global_sha256",
    "verify_against_freeze_manifest",
]


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file (chunked, memory-efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def dataset_global_sha256(
    selector_path: Path,
    project_root: Optional[Path] = None,
) -> str:
    """Compute the canonical global SHA256 for a dataset selector.

    This is the AUTHORITATIVE hash used by both freeze_v2_dataset.py and
    the eval runner.  Formula:

        sha256(sorted per-file strings of "<path>:<file_sha256>:<count>\\n")

    Args:
        selector_path: Path to a JSON file with a "files" list of relative paths.
        project_root: Root of the project (files are resolved relative to this).
                      Defaults to the parent of the selector's parent (data/../).

    Returns:
        64-character lowercase hex SHA256 string.
    """
    import json

    if project_root is None:
        # selector_path is typically data/canonical_v2_files.json;
        # project_root is data/../ = repo root
        project_root = selector_path.parent.parent

    sel = json.loads(selector_path.read_text(encoding="utf-8"))
    files: List[str] = sel["files"]

    global_h = hashlib.sha256()
    for rel_path in sorted(files):
        fpath = project_root / rel_path
        file_sha = sha256_file(fpath)
        count = _count_lines(fpath)
        global_h.update(f"{rel_path}:{file_sha}:{count}\n".encode("utf-8"))
    return global_h.hexdigest()


def verify_against_freeze_manifest(
    selector_path: Path,
    freeze_manifest_path: Path,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Verify current dataset files against a freeze manifest.

    Returns a dict with:
        - "global_sha_match": bool
        - "per_file_mismatches": list of filenames with hash mismatch
        - "computed_sha": the freshly-computed global SHA
        - "manifest_sha": the SHA stored in the freeze manifest
    """
    import json

    manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    manifest_global = manifest.get("global_sha256", "")

    if project_root is None:
        project_root = selector_path.parent.parent

    computed_global = dataset_global_sha256(selector_path, project_root)

    freeze_file_shas: Dict[str, str] = {
        item["path"]: item["sha256"]
        for item in manifest.get("canonical_files", [])
    }

    sel = json.loads(selector_path.read_text(encoding="utf-8"))
    mismatches = []
    for rel_path in sorted(sel["files"]):
        fpath = project_root / rel_path
        if fpath.exists():
            computed = sha256_file(fpath)
            if computed != freeze_file_shas.get(rel_path, ""):
                mismatches.append(rel_path)

    return {
        "global_sha_match": computed_global == manifest_global,
        "per_file_mismatches": mismatches,
        "computed_sha": computed_global,
        "manifest_sha": manifest_global,
    }
