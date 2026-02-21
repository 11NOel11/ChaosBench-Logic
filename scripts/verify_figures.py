#!/usr/bin/env python3
"""scripts/verify_figures.py — Figure QA checks for ChaosBench-Logic v2 results pack.

Checks:
  1. All expected files exist (.pdf and .png)
  2. File sizes above minimum threshold (> 5 KB for PNG, > 2 KB for PDF)
  3. PDFs are valid (lightweight structural check — no rendering required)
  4. PNGs are valid (header check)
  5. Warns if any expected figure is missing
  6. Summarizes pass/fail

Exit code: 0 if all pass, 1 if any failure.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "results_pack" / "figures"

EXPECTED_FIGURES = [
    "mcc_bar",
    "family_heatmap",
    "bias_plot",
    "latency_plot",
    "family_grouped_bar",
]

MIN_PNG_BYTES = 5_000   # 5 KB
MIN_PDF_BYTES = 2_000   # 2 KB


def check_png(path: Path) -> tuple[bool, str]:
    """Check that file is a valid PNG (header check)."""
    try:
        data = path.read_bytes()
        if len(data) < 8:
            return False, f"file too small ({len(data)} bytes)"
        if data[:8] != b'\x89PNG\r\n\x1a\n':
            return False, "invalid PNG header"
        if len(data) < MIN_PNG_BYTES:
            return False, f"suspiciously small ({len(data)} bytes < {MIN_PNG_BYTES})"
        return True, f"OK ({len(data):,} bytes)"
    except Exception as e:
        return False, str(e)


def check_pdf(path: Path) -> tuple[bool, str]:
    """Lightweight PDF validity check — verify %PDF header and %%EOF."""
    try:
        data = path.read_bytes()
        if len(data) < 8:
            return False, f"file too small ({len(data)} bytes)"
        if not data[:4] == b'%PDF':
            return False, "missing %PDF header"
        if len(data) < MIN_PDF_BYTES:
            return False, f"suspiciously small ({len(data)} bytes < {MIN_PDF_BYTES})"
        # Check for %%EOF near end
        tail = data[-512:] if len(data) > 512 else data
        if b'%%EOF' not in tail:
            return False, "missing %%EOF marker"
        return True, f"OK ({len(data):,} bytes)"
    except Exception as e:
        return False, str(e)


def main() -> None:
    print("[verify_figures] Checking figures in:", FIGURES_DIR)
    print()

    if not FIGURES_DIR.exists():
        print(f"ERROR: Figures directory does not exist: {FIGURES_DIR}")
        print("  Run: python scripts/generate_results_figures.py")
        sys.exit(1)

    failures = []
    warnings = []

    for stem in EXPECTED_FIGURES:
        for ext, checker in [(".png", check_png), (".pdf", check_pdf)]:
            fpath = FIGURES_DIR / f"{stem}{ext}"
            if not fpath.exists():
                msg = f"MISSING: {fpath.name}"
                print(f"  ❌ {msg}")
                failures.append(msg)
            else:
                ok, detail = checker(fpath)
                if ok:
                    print(f"  ✅ {fpath.name}: {detail}")
                else:
                    msg = f"INVALID {fpath.name}: {detail}"
                    print(f"  ❌ {msg}")
                    failures.append(msg)

    # Check FIGURE_INDEX.md
    idx = FIGURES_DIR / "FIGURE_INDEX.md"
    if not idx.exists():
        warnings.append("FIGURE_INDEX.md missing")
        print(f"  ⚠️  FIGURE_INDEX.md missing")
    else:
        print(f"  ✅ FIGURE_INDEX.md present")

    print()
    print(f"Summary: {len(EXPECTED_FIGURES)*2} expected files | "
          f"{len(failures)} failures | {len(warnings)} warnings")

    if failures:
        print("\nFailed checks:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✅ All figure QA checks passed.")


if __name__ == "__main__":
    main()
