"""Tests for deterministic shuffle ordering mode in EvalRunner.

Verifies:
- shuffle_seed=42 produces a different item order than shuffle_seed=None
- Two runs with shuffle_seed=42 produce the same item order (determinism)
- Manifest records order_mode and shuffle_seed correctly
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


def _make_items(n: int = 20) -> list[dict]:
    """Create synthetic items for testing."""
    return [
        {
            "id": f"item_{i:03d}",
            "question": f"Is system {i} chaotic?",
            "ground_truth": "TRUE" if i % 2 == 0 else "FALSE",
            "type": "atomic",
        }
        for i in range(n)
    ]


def _run_with_config(items, shuffle_seed, output_dir):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from chaosbench.eval.run import EvalRunner, RunConfig
    from chaosbench.eval.providers import MockProvider

    provider = MockProvider(default="TRUE")
    cfg = RunConfig(
        provider=provider,
        output_dir=output_dir,
        max_items=None,
        seed=42,
        workers=1,
        retries=0,
        strict_parsing=False,
        shuffle_seed=shuffle_seed,
    )
    runner = EvalRunner(cfg)
    return runner.run(items=list(items), dataset="canonical")


def test_shuffle_seed_changes_order():
    """shuffle_seed=42 must produce a different item order than shuffle_seed=None."""
    items = _make_items(20)

    def _load_ids(result):
        preds = [json.loads(l) for l in Path(result["predictions_path"]).read_text().splitlines() if l.strip()]
        return [p["id"] for p in preds]

    # Use separate temp dirs to avoid run_id timestamp collision
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        result_canonical = _run_with_config(items, shuffle_seed=None, output_dir=tmp1)
        result_shuffled = _run_with_config(items, shuffle_seed=42, output_dir=tmp2)

        ids_canonical = _load_ids(result_canonical)
        ids_shuffled = _load_ids(result_shuffled)

    # Both runs evaluated all items
    assert set(ids_canonical) == set(ids_shuffled)
    # But in different order
    assert ids_canonical != ids_shuffled, (
        "shuffle_seed=42 should produce a different order than canonical"
    )


def test_shuffle_seed_is_deterministic():
    """Two runs with the same shuffle_seed must produce identical item order."""
    items = _make_items(20)

    def _load_ids(result):
        preds = [json.loads(l) for l in Path(result["predictions_path"]).read_text().splitlines() if l.strip()]
        return [p["id"] for p in preds]

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        result_a = _run_with_config(items, shuffle_seed=42, output_dir=tmp1)
        result_b = _run_with_config(items, shuffle_seed=42, output_dir=tmp2)

        ids_a = _load_ids(result_a)
        ids_b = _load_ids(result_b)

    assert ids_a == ids_b, "Same shuffle_seed must yield identical item order"


def test_manifest_records_order_mode_canonical():
    """Manifest must record order_mode='canonical' when shuffle_seed is None."""
    items = _make_items(10)

    with tempfile.TemporaryDirectory() as tmp:
        result = _run_with_config(items, shuffle_seed=None, output_dir=tmp)
        manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert manifest.get("order_mode") == "canonical"
    assert manifest.get("shuffle_seed") is None


def test_manifest_records_order_mode_shuffled():
    """Manifest must record order_mode='shuffled' and shuffle_seed when shuffled."""
    items = _make_items(10)

    with tempfile.TemporaryDirectory() as tmp:
        result = _run_with_config(items, shuffle_seed=99, output_dir=tmp)
        manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert manifest.get("order_mode") == "shuffled"
    assert manifest.get("shuffle_seed") == 99
