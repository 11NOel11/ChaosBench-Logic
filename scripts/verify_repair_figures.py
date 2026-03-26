#!/usr/bin/env python3
"""Validate CARE-v3 figure artifacts for basic publishability constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = (
    PROJECT_ROOT / "workspace" / "deep_survey_2026-03-01" / "repair_v3" / "figures"
)

REQUIRED_STEMS = (
    "repair_boundary_scatter",
    "repair_violation_vs_delta",
    "repair_delta_by_family",
    "repair_flip_breakdown",
)


def check_pdf_header(path: Path) -> bool:
    data = path.read_bytes()
    return data.startswith(b"%PDF")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CARE-v3 figure files")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--min-bytes", type=int, default=5_000)
    parser.add_argument("--min-width", type=int, default=900)
    parser.add_argument("--min-height", type=int, default=550)
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    failures = []

    if not figures_dir.exists():
        failures.append(f"figures directory missing: {figures_dir}")
    else:
        for stem in REQUIRED_STEMS:
            png_path = figures_dir / f"{stem}.png"
            pdf_path = figures_dir / f"{stem}.pdf"

            if not png_path.exists():
                failures.append(f"missing png: {png_path}")
            if not pdf_path.exists():
                failures.append(f"missing pdf: {pdf_path}")

            if png_path.exists() and png_path.stat().st_size < args.min_bytes:
                failures.append(f"png too small: {png_path}")
            if pdf_path.exists() and pdf_path.stat().st_size < args.min_bytes:
                failures.append(f"pdf too small: {pdf_path}")

            if png_path.exists():
                image = mpimg.imread(png_path)
                if image.ndim < 2:
                    failures.append(f"png unreadable: {png_path}")
                else:
                    height, width = image.shape[0], image.shape[1]
                    if width < args.min_width or height < args.min_height:
                        failures.append(
                            f"png dimensions too small: {png_path} ({width}x{height})"
                        )

            if pdf_path.exists() and not check_pdf_header(pdf_path):
                failures.append(f"pdf header invalid: {pdf_path}")

        manifest_path = figures_dir / "figures_manifest.json"
        if not manifest_path.exists():
            failures.append(f"missing figures_manifest.json: {manifest_path}")
        else:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            for stem in REQUIRED_STEMS:
                if stem not in metadata:
                    failures.append(f"missing metadata entry for {stem}")
                    continue
                entry = metadata[stem]
                if int(entry.get("label_max_len", 0)) > 26:
                    failures.append(
                        f"label_max_len too large for {stem}: {entry.get('label_max_len')}"
                    )
                if stem in {"repair_delta_by_family", "repair_flip_breakdown"}:
                    if int(entry.get("x_tick_rotation", 0)) < 20:
                        failures.append(
                            f"x_tick_rotation too small for {stem}: {entry.get('x_tick_rotation')}"
                        )

    if failures:
        print("Figure verification FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Figure verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
