"""
tests/test_hashing.py — Tests for chaosbench.data.hashing (unified SHA module).

Verifies:
- sha256_file produces consistent, correct hashes
- dataset_global_sha256 formula matches freeze_v2_dataset.py formula
- dataset_global_sha256 now matches freeze manifest (SHA consistency fixed)
- verify_against_freeze_manifest works correctly
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from chaosbench.data.hashing import (
    sha256_file,
    dataset_global_sha256,
    verify_against_freeze_manifest,
)

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------

class TestSha256File:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        h = sha256_file(f)
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected

    def test_known_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world\n")
        h = sha256_file(f)
        expected = hashlib.sha256(b"hello world\n").hexdigest()
        assert h == expected

    def test_deterministic(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02\x03" * 1000)
        assert sha256_file(f) == sha256_file(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert sha256_file(f1) != sha256_file(f2)


# ---------------------------------------------------------------------------
# dataset_global_sha256 — formula consistency
# ---------------------------------------------------------------------------

def _make_fake_selector(tmp_path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    """Create a fake project root with data files and a selector JSON.

    Args:
        files: mapping rel_path -> content
    Returns:
        (project_root, selector_path)
    """
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "data").mkdir()

    file_list = []
    for rel_path, content in files.items():
        fpath = project_root / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
        file_list.append(rel_path)

    sel = {"files": file_list}
    sel_path = project_root / "data" / "canonical_v2_files.json"
    sel_path.write_text(json.dumps(sel))
    return project_root, sel_path


def _freeze_formula_sha(project_root: Path, file_list: list[str]) -> str:
    """Replicate freeze_v2_dataset.py global hash formula exactly."""
    global_h = hashlib.sha256()
    for rel_path in sorted(file_list):
        fpath = project_root / rel_path
        h = hashlib.sha256()
        with open(fpath, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        file_sha = h.hexdigest()
        count = sum(1 for line in fpath.open() if line.strip())
        global_h.update(f"{rel_path}:{file_sha}:{count}\n".encode("utf-8"))
    return global_h.hexdigest()


class TestDatasetGlobalSha256:
    def test_matches_freeze_formula_single_file(self, tmp_path):
        """dataset_global_sha256 must match freeze_v2_dataset.py formula."""
        files = {"data/v22_atomic.jsonl": '{"id": "a1", "ground_truth": "TRUE"}\n'}
        root, sel = _make_fake_selector(tmp_path, files)

        computed = dataset_global_sha256(sel, root)
        expected = _freeze_formula_sha(root, list(files.keys()))
        assert computed == expected

    def test_matches_freeze_formula_multi_file(self, tmp_path):
        """Must match across multiple files in sorted order."""
        files = {
            "data/v22_atomic.jsonl": '{"id": "a1"}\n{"id": "a2"}\n',
            "data/v22_fol.jsonl": '{"id": "f1"}\n',
            "data/v22_multi.jsonl": '{"id": "m1"}\n{"id": "m2"}\n{"id": "m3"}\n',
        }
        root, sel = _make_fake_selector(tmp_path, files)

        computed = dataset_global_sha256(sel, root)
        expected = _freeze_formula_sha(root, list(files.keys()))
        assert computed == expected

    def test_order_independence(self, tmp_path):
        """Result must be the same regardless of selector list order."""
        files = {
            "data/b.jsonl": '{"id": "b1"}\n',
            "data/a.jsonl": '{"id": "a1"}\n',
        }
        root1, sel1 = _make_fake_selector(tmp_path / "r1", files)
        # Reverse order in selector
        files2 = dict(reversed(list(files.items())))
        root2, sel2 = _make_fake_selector(tmp_path / "r2", files2)

        sha1 = dataset_global_sha256(sel1, root1)
        sha2 = dataset_global_sha256(sel2, root2)
        assert sha1 == sha2

    def test_content_change_changes_sha(self, tmp_path):
        files = {"data/v22_test.jsonl": '{"id": "x1"}\n'}
        root, sel = _make_fake_selector(tmp_path, files)
        sha_before = dataset_global_sha256(sel, root)
        # Modify file
        (root / "data" / "v22_test.jsonl").write_text('{"id": "x1_modified"}\n')
        sha_after = dataset_global_sha256(sel, root)
        assert sha_before != sha_after

    def test_count_included_in_hash(self, tmp_path):
        """Adding a line to a file should change the global SHA (count changes)."""
        files = {"data/v22_test.jsonl": '{"id": "x1"}\n'}
        root, sel = _make_fake_selector(tmp_path, files)
        sha_before = dataset_global_sha256(sel, root)
        # Add a line — same structure, different count
        (root / "data" / "v22_test.jsonl").write_text(
            '{"id": "x1"}\n{"id": "x2"}\n'
        )
        sha_after = dataset_global_sha256(sel, root)
        assert sha_before != sha_after


class TestFreezeShaMachtesEvalSha:
    """High-value integration test: eval runner SHA must equal freeze SHA."""

    def test_canonical_sha_matches_freeze_manifest(self):
        """The current canonical dataset must match the frozen SHA.

        This test ensures that chaosbench.data.hashing.dataset_global_sha256
        produces the same value as artifacts/freeze/v2_freeze_manifest.json.

        If this test fails, either:
        - The dataset files changed after the freeze (data integrity issue)
        - The hashing formula diverged (tooling bug)
        """
        freeze_manifest = PROJECT_ROOT / "artifacts" / "freeze" / "v2_freeze_manifest.json"
        canonical_sel = PROJECT_ROOT / "data" / "canonical_v2_files.json"

        if not freeze_manifest.exists():
            pytest.skip("Freeze manifest not present (not in CI tree)")
        if not canonical_sel.exists():
            pytest.skip("Canonical selector not present")

        freeze_data = json.loads(freeze_manifest.read_text())
        freeze_sha = freeze_data.get("global_sha256", "")

        # Check all canonical files exist before computing
        sel = json.loads(canonical_sel.read_text())
        missing = [f for f in sel["files"] if not (PROJECT_ROOT / f).exists()]
        if missing:
            pytest.skip(f"Some canonical files missing (expected in CI): {missing[:3]}")

        computed = dataset_global_sha256(canonical_sel, PROJECT_ROOT)
        assert computed == freeze_sha, (
            f"dataset_global_sha256 does not match freeze manifest.\n"
            f"  computed: {computed}\n"
            f"  freeze:   {freeze_sha}\n"
            f"This means either the dataset files changed or the hash formula diverged."
        )


# ---------------------------------------------------------------------------
# verify_against_freeze_manifest
# ---------------------------------------------------------------------------

class TestVerifyAgainstFreezeManifest:
    def _make_freeze_manifest(self, project_root: Path, files: dict[str, str]) -> Path:
        file_records = []
        global_h = hashlib.sha256()
        for rel_path in sorted(files.keys()):
            fpath = project_root / rel_path
            h = hashlib.sha256()
            with open(fpath, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            file_sha = h.hexdigest()
            count = sum(1 for line in fpath.open() if line.strip())
            global_h.update(f"{rel_path}:{file_sha}:{count}\n".encode("utf-8"))
            file_records.append({"path": rel_path, "sha256": file_sha, "count": count})
        global_sha = global_h.hexdigest()

        manifest = {
            "global_sha256": global_sha,
            "canonical_files": file_records,
        }
        freeze_path = project_root / "artifacts" / "freeze" / "v2_freeze_manifest.json"
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(json.dumps(manifest))
        return freeze_path

    def test_match(self, tmp_path):
        files = {"data/v22_test.jsonl": '{"id": "x1"}\n'}
        root, sel = _make_fake_selector(tmp_path, files)
        freeze_path = self._make_freeze_manifest(root, files)

        result = verify_against_freeze_manifest(sel, freeze_path, root)
        assert result["global_sha_match"] is True
        assert result["per_file_mismatches"] == []

    def test_mismatch_after_modification(self, tmp_path):
        files = {"data/v22_test.jsonl": '{"id": "x1"}\n'}
        root, sel = _make_fake_selector(tmp_path, files)
        freeze_path = self._make_freeze_manifest(root, files)

        # Modify the file after freezing
        (root / "data" / "v22_test.jsonl").write_text('{"id": "MODIFIED"}\n')

        result = verify_against_freeze_manifest(sel, freeze_path, root)
        assert result["global_sha_match"] is False
        assert "data/v22_test.jsonl" in result["per_file_mismatches"]
