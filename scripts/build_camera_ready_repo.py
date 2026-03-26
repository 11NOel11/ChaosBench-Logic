#!/usr/bin/env python3
"""Build a camera-ready repository bundle for ChaosBench-Logic v2.0.0.

This script copies a curated set of source files into a clean output directory
and generates a zip archive suitable for camera-ready artifact packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "workspace" / "camera_ready_repo_v2.0.0"


INCLUDE_PATTERNS = [
    "README.md",
    "DATASET_CARD.md",
    "CITATION.cff",
    "SECURITY.md",
    "LICENSE",
    "LICENSE_DATA",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "chaosbench/**/*.py",
    "configs/**/*.yaml",
    "data/v22_*.jsonl",
    "data/v2_manifest.json",
    "data/canonical_v2_files.json",
    "data/archive/v1/**/*.jsonl",
    "docs/**/*.md",
    "scripts/**/*.py",
    "scripts/**/*.md",
    "scripts/**/*.sh",
    "systems/**/*.json",
    "tests/**/*.py",
]


EXCLUDE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "artifacts",
    "coverage",
    "logs",
    "results",
    "results_ci_local",
    "runs",
    "tmp",
    "workspace",
}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts)


def collect_files(root: Path) -> list[Path]:
    selected: set[Path] = set()
    for pattern in INCLUDE_PATTERNS:
        for matched in root.glob(pattern):
            if not matched.is_file():
                continue
            rel = matched.relative_to(root)
            if _is_excluded(rel):
                continue
            selected.add(rel)
    return sorted(selected)


def copy_files(root: Path, files: list[Path], output_dir: Path) -> None:
    for rel in files:
        src = root / rel
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_manifest(
    root: Path, files: list[Path], output_dir: Path, zip_path: Path
) -> None:
    records = []
    total_bytes = 0
    for rel in files:
        copied = output_dir / rel
        size = copied.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": rel.as_posix(),
                "size_bytes": size,
                "sha256": _sha256(copied),
            }
        )

    manifest = {
        "name": "ChaosBench-Logic camera-ready repository bundle",
        "release_version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "output_dir": str(output_dir),
        "zip_path": str(zip_path),
        "file_count": len(records),
        "total_size_bytes": total_bytes,
        "files": records,
    }

    manifest_path = output_dir / "CAMERA_READY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_bundle_notes(output_dir: Path, zip_path: Path) -> None:
    notes = [
        "# Camera-Ready Repo Bundle",
        "",
        "This directory is a curated camera-ready snapshot for ChaosBench-Logic v2.0.0.",
        "",
        "## Included surfaces",
        "",
        "- Root release docs and metadata",
        "- Canonical dataset files and manifest",
        "- Source code, configs, tests, and scripts",
        "- Canonical documentation under docs/",
        "",
        "## Generated artifacts",
        "",
        f"- Zip archive: `{zip_path.name}`",
        "- Manifest: `CAMERA_READY_MANIFEST.json`",
        "",
        "## Public links",
        "",
        "- GitHub: https://github.com/11NOel11/ChaosBench-Logic",
        "- Dataset: https://huggingface.co/datasets/11NOel11/ChaosBench-Logic",
    ]
    (output_dir / "README_CAMERA_READY_BUNDLE.md").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )


def build_zip(output_dir: Path, zip_path: Path) -> None:
    bundle_root = output_dir.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(output_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(output_dir)
            arcname = Path(bundle_root) / rel
            zf.write(file_path, arcname.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the curated camera-ready bundle.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Path to zip archive (defaults to <output-dir>.zip).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    zip_path = args.zip_path or (output_dir.parent / f"{output_dir.name}.zip")

    if output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Output directory already exists: {output_dir} (use --force to overwrite)"
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(PROJECT_ROOT)
    copy_files(PROJECT_ROOT, files, output_dir)
    write_bundle_notes(output_dir, zip_path)
    build_manifest(PROJECT_ROOT, files, output_dir, zip_path)
    build_zip(output_dir, zip_path)

    print(f"Camera-ready repo built: {output_dir}")
    print(f"Zip archive: {zip_path}")
    print(f"Files copied: {len(files)}")


if __name__ == "__main__":
    main()
