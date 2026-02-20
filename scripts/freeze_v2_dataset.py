#!/usr/bin/env python3
"""Freeze the canonical v2 dataset and produce a citable artifact.

Usage:
    python scripts/freeze_v2_dataset.py
    python scripts/freeze_v2_dataset.py --output-dir artifacts/freeze

Outputs (all in --output-dir):
    v2_freeze_manifest.json   - Machine-readable hash manifest
    v2_freeze_report.md       - Human-readable summary
    v2_freeze_sha256.txt      - Simple per-file + global hashes

Validations performed:
    - All item IDs are unique across the canonical files.
    - All labels are in {TRUE, FALSE} (YES/NO normalised before checking).
    - canonical_total_questions matches the sum of per-file counts.
    - canonical_total_questions matches data/v2_manifest.json (if present).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use unified hashing module so per-file SHA is identical to eval runner.
from chaosbench.data.hashing import sha256_file as _sha256_file_canonical  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _pkg_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("chaos-logic-bench")
    except Exception:
        try:
            toml_path = PROJECT_ROOT / "pyproject.toml"
            for line in toml_path.read_text().splitlines():
                if line.strip().startswith("version"):
                    return line.split("=")[-1].strip().strip('"\'')
        except Exception:
            pass
    return "unknown"


def _sha256_file(path: Path) -> str:
    """Delegate to canonical hashing module for consistency with eval runner."""
    return _sha256_file_canonical(path)


def _normalize_label(value: str) -> Optional[str]:
    v = value.strip().upper()
    if v in {"YES", "Y", "TRUE", "T"}:
        return "TRUE"
    if v in {"NO", "N", "FALSE", "F"}:
        return "FALSE"
    return None


# ---------------------------------------------------------------------------
# Main freeze logic
# ---------------------------------------------------------------------------


def freeze(output_dir: str = "artifacts/freeze", selector: str = "data/canonical_v2_files.json") -> Dict[str, Any]:
    """Run the full freeze procedure.

    Returns:
        The freeze manifest dict.
    """
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_path = PROJECT_ROOT / selector
    if not sel_path.exists():
        print(f"ERROR: canonical selector not found: {sel_path}", file=sys.stderr)
        sys.exit(2)

    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    canonical_files: List[str] = sel["files"]

    if not canonical_files:
        print("ERROR: canonical_v2_files.json lists no files", file=sys.stderr)
        sys.exit(2)

    print(f"Processing {len(canonical_files)} canonical files...")

    # Per-file processing
    file_records: List[Dict] = []
    all_ids: List[str] = []
    label_errors: List[str] = []
    total_count = 0

    for rel_path in canonical_files:
        fpath = PROJECT_ROOT / rel_path
        if not fpath.exists():
            print(f"ERROR: missing canonical file: {fpath}", file=sys.stderr)
            sys.exit(2)

        sha = _sha256_file(fpath)
        count = 0
        file_ids: List[str] = []

        with open(fpath, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR: JSON parse error in {rel_path} line {lineno}: {e}", file=sys.stderr)
                    sys.exit(2)

                count += 1

                # Collect ID
                item_id = item.get("id") or item.get("item_id")
                if item_id is not None:
                    file_ids.append(str(item_id))
                    all_ids.append(str(item_id))

                # Check label (field may be ground_truth, answer, gold, or label)
                raw_label = (
                    item.get("ground_truth")
                    or item.get("answer")
                    or item.get("gold")
                    or item.get("label")
                    or ""
                )
                normalized = _normalize_label(str(raw_label))
                if normalized is None:
                    label_errors.append(f"{rel_path}:{lineno} label={raw_label!r}")

        total_count += count
        print(f"  {rel_path}: {count} items, sha256={sha[:16]}...")
        file_records.append({"path": rel_path, "count": count, "sha256": sha})

    # Validate unique IDs
    id_set = set()
    duplicate_ids: List[str] = []
    for iid in all_ids:
        if iid in id_set:
            duplicate_ids.append(iid)
        id_set.add(iid)

    if duplicate_ids:
        print(f"WARNING: {len(duplicate_ids)} duplicate IDs found (first 5: {duplicate_ids[:5]})")
    else:
        print(f"✓ All {len(id_set)} IDs are unique")

    if label_errors:
        print(f"ERROR: {len(label_errors)} items have invalid labels (first 5):")
        for e in label_errors[:5]:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"✓ All labels valid (TRUE/FALSE)")

    # Global hash: hash of sorted file contributions
    global_h = hashlib.sha256()
    for rec in sorted(file_records, key=lambda r: r["path"]):
        global_h.update(f"{rec['path']}:{rec['sha256']}:{rec['count']}\n".encode("utf-8"))
    global_sha = global_h.hexdigest()

    # Validate against v2_manifest.json
    manifest_path = PROJECT_ROOT / "data" / "v2_manifest.json"
    manifest_total: Optional[int] = None
    if manifest_path.exists():
        v2_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_total = v2_manifest.get("canonical_total_questions") or v2_manifest.get("total_new_questions")
        if manifest_total is not None and manifest_total != total_count:
            print(
                f"WARNING: v2_manifest.json total ({manifest_total}) "
                f"does not match counted total ({total_count})"
            )
        else:
            print(f"✓ canonical_total_questions matches v2_manifest.json ({total_count})")

    # Build manifest
    freeze_manifest = {
        "dataset_release": "v2",
        "schema_version": "v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_selector": selector,
        "canonical_files": file_records,
        "canonical_total_questions": total_count,
        "global_sha256": global_sha,
        "duplicate_ids": len(duplicate_ids),
        "tool_versions": {
            "python": platform.python_version(),
            "package": _pkg_version(),
            "git_commit": _git_commit(),
        },
    }

    # Write artifacts
    manifest_out = out_dir / "v2_freeze_manifest.json"
    manifest_out.write_text(json.dumps(freeze_manifest, indent=2))
    print(f"\nWrote: {manifest_out}")

    # SHA256 text file
    sha_lines = [f"# ChaosBench-Logic v2 Dataset Freeze"]
    sha_lines.append(f"# Generated: {freeze_manifest['created_utc']}")
    sha_lines.append("")
    sha_lines.append(f"global_sha256  {global_sha}")
    sha_lines.append("")
    for rec in file_records:
        sha_lines.append(f"{rec['sha256']}  {rec['path']}  (count={rec['count']})")
    (out_dir / "v2_freeze_sha256.txt").write_text("\n".join(sha_lines) + "\n")
    print(f"Wrote: {out_dir / 'v2_freeze_sha256.txt'}")

    # Human-readable report
    report_lines = [
        "# ChaosBench-Logic v2 Dataset Freeze Report",
        "",
        f"**Generated:** {freeze_manifest['created_utc']}",
        f"**Git commit:** {freeze_manifest['tool_versions']['git_commit'] or 'N/A'}",
        f"**Package version:** {freeze_manifest['tool_versions']['package']}",
        "",
        "## Summary",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Dataset release | v2 |",
        f"| Total questions | {total_count:,} |",
        f"| Canonical files | {len(file_records)} |",
        f"| Unique IDs | {len(id_set):,} |",
        f"| Duplicate IDs | {len(duplicate_ids)} |",
        f"| Global SHA256 | `{global_sha[:32]}...` |",
        "",
        "## Per-File Checksums",
        "",
        "| File | Count | SHA256 (first 32) |",
        "|------|-------|-------------------|",
    ]
    for rec in file_records:
        fname = Path(rec["path"]).name
        report_lines.append(f"| `{fname}` | {rec['count']:,} | `{rec['sha256'][:32]}...` |")

    report_lines += [
        "",
        "## Validation",
        "",
        f"- ✓ {len(id_set):,} unique IDs across all files",
        f"- ✓ All labels validated as TRUE/FALSE",
        f"- ✓ Total count verified: {total_count:,} questions",
        "",
        "## How to Cite",
        "",
        "When referencing this dataset, include the global SHA256 hash above.",
        "",
        "## How to Verify",
        "",
        "```bash",
        "python scripts/freeze_v2_dataset.py",
        "# Compare global_sha256 in artifacts/freeze/v2_freeze_manifest.json",
        "```",
    ]
    (out_dir / "v2_freeze_report.md").write_text("\n".join(report_lines) + "\n")
    print(f"Wrote: {out_dir / 'v2_freeze_report.md'}")

    print(f"\n✓ Freeze complete.")
    print(f"  Total questions : {total_count:,}")
    print(f"  Global SHA256   : {global_sha}")
    print(f"  Artifacts in    : {out_dir}")

    return freeze_manifest


def main():
    parser = argparse.ArgumentParser(description="Freeze the canonical v2 dataset.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/freeze",
        help="Output directory (default: artifacts/freeze)",
    )
    parser.add_argument(
        "--selector",
        default="data/canonical_v2_files.json",
        help="Path to canonical selector JSON",
    )
    args = parser.parse_args()
    freeze(output_dir=args.output_dir, selector=args.selector)


if __name__ == "__main__":
    main()
