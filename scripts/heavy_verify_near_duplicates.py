#!/usr/bin/env python3
"""Near-duplicate detection for ChaosBench-Logic v2 canonical dataset.

Computes Jaccard similarity on word unigram+bigram sets between pairs of
questions to detect unintentional near-duplicates across splits.

Usage:
    python scripts/heavy_verify_near_duplicates.py [--data-dir data/] [--output-dir artifacts/heavy_verify/]
    python scripts/heavy_verify_near_duplicates.py --full  # 100k pair sample

Exit Codes:
    0  All checks passed (no genuine near-dups crossing heldout boundary)
    1  Hard-fail: genuine near-dups found crossing heldout_systems boundary
    2  Script error

Outputs:
    artifacts/heavy_verify/near_duplicates_report.md
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chaosbench.data.splits import assign_split_v22, HELDOUT_SYSTEM_IDS


# ---------------------------------------------------------------------------
# Canonical file loader
# ---------------------------------------------------------------------------

def _load_canonical_files(data_dir: Path) -> List[str]:
    """Load canonical file names from data/canonical_v2_files.json."""
    selector_path = data_dir.parent / "data" / "canonical_v2_files.json"
    if not selector_path.exists():
        selector_path = data_dir / "canonical_v2_files.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    return [Path(f).name for f in selector["files"]]


def load_all_records(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all canonical records."""
    records = []
    for fname in _load_canonical_files(data_dir):
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        for line in fpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# N-gram fingerprinting
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase + split into word tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def ngram_set(text: str) -> Set[str]:
    """Compute unigram + bigram set from question text."""
    tokens = _tokenize(text)
    unigrams = set(tokens)
    bigrams = {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}
    return unigrams | bigrams


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Intentional near-dup detection
# ---------------------------------------------------------------------------

def _mask_system_name(text: str, system_id: str) -> str:
    """Replace system name occurrences in text with a placeholder."""
    # Strip dysts_ prefix and underscores to get a base name for matching
    base = system_id.replace("dysts_", "").replace("_", "")
    text_masked = re.sub(re.escape(base), "<SYS>", text, flags=re.IGNORECASE)
    # Also strip the original system_id form
    text_masked = re.sub(re.escape(system_id.replace("_", " ")), "<SYS>", text_masked, flags=re.IGNORECASE)
    return text_masked


def _is_intentional_pair(
    id_a: str,
    id_b: str,
    record_a: Optional[Dict[str, Any]] = None,
    record_b: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check if a pair of IDs/records are intentionally similar.

    Intentional cases:
    1. Both are paraphrase variants of the same base: atomic_{N}_para_{M1} / atomic_{N}_para_{M2}
    2. Both are perturbation variants with same base number: perturb_*_{N} pattern
    3. Same template on different systems: questions differ only by system name reference
       (template-based generation produces structurally identical questions for each system)
    """
    # Paraphrase group: same base prefix
    para_a = re.match(r"^(atomic_\d+)_para_\d+$", id_a)
    para_b = re.match(r"^(atomic_\d+)_para_\d+$", id_b)
    if para_a and para_b and para_a.group(1) == para_b.group(1):
        return True

    # Perturbation variants with same base number
    perturb_a = re.match(r"^perturb_\w+_(\d+)$", id_a)
    perturb_b = re.match(r"^perturb_\w+_(\d+)$", id_b)
    if perturb_a and perturb_b and perturb_a.group(1) == perturb_b.group(1):
        return True

    # Same template on different systems: mask system names, then check if Jaccard > 0.95
    # This identifies template-parallel questions (same structure, different system references)
    if record_a is not None and record_b is not None:
        sys_a = record_a.get("system_id", "")
        sys_b = record_b.get("system_id", "")
        if sys_a != sys_b and record_a.get("type") == record_b.get("type"):
            q_a = _mask_system_name(record_a.get("question", ""), sys_a)
            q_b = _mask_system_name(record_b.get("question", ""), sys_b)
            masked_sim = jaccard(ngram_set(q_a), ngram_set(q_b))
            if masked_sim >= 0.95:
                return True

    return False


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def run_near_duplicate_check(
    records: List[Dict[str, Any]],
    n_pairs: int,
    seed: int,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample pairs and compute Jaccard similarity.

    Returns:
        (genuine_near_dups, cross_split_near_dups)
        - genuine: non-intentional pairs with Jaccard >= threshold
        - cross_split: genuine near-dups where one is heldout_systems and other is core/hard
    """
    rng = random.Random(seed)
    n = len(records)

    # Precompute ngram sets
    print(f"  Computing n-gram sets for {n:,} records...")
    ngrams = [ngram_set(r.get("question", "")) for r in records]

    # Assign splits
    print(f"  Assigning splits...")
    splits = [assign_split_v22(r) for r in records]

    # Sample pairs
    print(f"  Sampling {n_pairs:,} pairs (seed={seed})...")
    indices = list(range(n))
    pairs = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 10
    while len(pairs) < n_pairs and attempts < max_attempts:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            attempts += 1
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append(key)
        attempts += 1

    print(f"  Checking {len(pairs):,} pairs for Jaccard >= {threshold}...")
    genuine_near_dups = []
    cross_split_near_dups = []

    for i, j in pairs:
        sim = jaccard(ngrams[i], ngrams[j])
        if sim >= threshold:
            id_i = records[i].get("id", f"idx_{i}")
            id_j = records[j].get("id", f"idx_{j}")

            # Check if intentional (pass records for template-parallel detection)
            if _is_intentional_pair(id_i, id_j, records[i], records[j]):
                continue

            entry = {
                "id_a": id_i,
                "id_b": id_j,
                "jaccard": round(sim, 4),
                "split_a": splits[i],
                "split_b": splits[j],
                "type_a": records[i].get("type", ""),
                "type_b": records[j].get("type", ""),
                "system_a": records[i].get("system_id", ""),
                "system_b": records[j].get("system_id", ""),
            }
            genuine_near_dups.append(entry)

            # Cross-split: heldout_systems vs core/hard
            split_set = {splits[i], splits[j]}
            if "heldout_systems" in split_set and split_set - {"heldout_systems"}:
                cross_split_near_dups.append(entry)

    return genuine_near_dups, cross_split_near_dups


def write_report(
    output_dir: Path,
    records_count: int,
    n_pairs_checked: int,
    seed: int,
    threshold: float,
    genuine_near_dups: List[Dict[str, Any]],
    cross_split_near_dups: List[Dict[str, Any]],
    passed: bool,
) -> None:
    """Write near-duplicate report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "near_duplicates_report.md"

    lines = [
        "# Near-Duplicate Detection Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Records checked:** {records_count:,}",
        f"**Pairs sampled:** {n_pairs_checked:,} (seed={seed})",
        f"**Jaccard threshold:** {threshold}",
        f"**Overall status:** {'✅ PASSED' if passed else '❌ FAILED'}",
        "",
        "---",
        "",
        "## Results",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Genuine near-dups (non-intentional, Jaccard ≥ {threshold}) | {len(genuine_near_dups)} |",
        f"| Cross-split near-dups (heldout_systems boundary) | {len(cross_split_near_dups)} |",
        "",
    ]

    if genuine_near_dups:
        lines += [
            "## Genuine Near-Duplicates (sample, up to 20)",
            "",
            "| ID A | ID B | Jaccard | Split A | Split B | Type A | Type B |",
            "|------|------|---------|---------|---------|--------|--------|",
        ]
        for entry in genuine_near_dups[:20]:
            lines.append(
                f"| {entry['id_a']} | {entry['id_b']} | {entry['jaccard']:.3f} | "
                f"{entry['split_a']} | {entry['split_b']} | {entry['type_a']} | {entry['type_b']} |"
            )
        lines.append("")

    if cross_split_near_dups:
        lines += [
            "## CRITICAL: Cross-Heldout-Split Near-Duplicates",
            "",
            "These pairs cross the heldout_systems boundary and represent potential data leakage:",
            "",
            "| ID A | ID B | Jaccard | Split A | Split B |",
            "|------|------|---------|---------|---------|",
        ]
        for entry in cross_split_near_dups[:20]:
            lines.append(
                f"| {entry['id_a']} | {entry['id_b']} | {entry['jaccard']:.3f} | "
                f"{entry['split_a']} | {entry['split_b']} |"
            )
        lines.append("")

    lines += [
        "## Intentional Near-Duplicate Patterns (Excluded)",
        "",
        "The following are intentionally similar and excluded from checks:",
        "- `consistency_paraphrase` variants: `atomic_{N}_para_{M}` pairs sharing the same base ID",
        "- Perturbation variants: `perturb_*_{N}` pairs sharing the same base number",
        "- Template-parallel questions: same template on different systems (masked Jaccard ≥ 0.95)",
        "",
        "## Hard-Fail Condition",
        "",
        "Genuine near-dups crossing the heldout_systems boundary: **0 allowed**.",
        f"Found: **{len(cross_split_near_dups)}**",
        "",
        f"**Result: {'PASS ✅' if passed else 'FAIL ❌'}**",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Near-duplicate detection for v2 dataset")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--output-dir", default="artifacts/heavy_verify/", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--full", action="store_true", help="Use 100k pair sample (vs 50k CI mode)")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir

    n_pairs = 100_000 if args.full else 50_000

    print("=" * 60)
    print("Near-Duplicate Detection")
    print("=" * 60)

    try:
        print("\n[1] Loading records...")
        records = load_all_records(data_dir)
        print(f"  Loaded {len(records):,} records")

        print("\n[2] Running near-duplicate detection...")
        genuine, cross_split = run_near_duplicate_check(
            records, n_pairs, args.seed, args.threshold
        )

        print(f"\n  Genuine near-dups found: {len(genuine)}")
        print(f"  Cross-split (heldout boundary) near-dups: {len(cross_split)}")

        # Hard fail: any cross-split near-dup
        passed = len(cross_split) == 0

        print("\n[3] Writing report...")
        write_report(
            output_dir,
            len(records),
            n_pairs,
            args.seed,
            args.threshold,
            genuine,
            cross_split,
            passed,
        )

        if not passed:
            print(f"\n❌ HARD FAIL: {len(cross_split)} genuine near-duplicate(s) cross the heldout_systems boundary!")
            return 1

        print(f"\n✅ PASSED — 0 genuine near-dups crossing heldout boundary (out of {len(genuine)} total near-dups, including intentional)")
        return 0

    except Exception as e:
        print(f"\nScript error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
