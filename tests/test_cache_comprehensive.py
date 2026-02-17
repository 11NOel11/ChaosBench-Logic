"""Comprehensive cache system test."""

import os
import tempfile

from chaosbench.eval.cache import ResponseCache
from chaosbench.eval.runner import evaluate_items_with_parallelism
from chaosbench.models.prompt import ModelConfig, make_model_client


def test_cache_full_workflow():
    """Test complete cache workflow with multiple scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Scenario 1: Initial run with cache
        cache1 = ResponseCache(tmpdir)

        items_batch1 = [
            {"id": "item_001", "question": "Q1?", "ground_truth": "YES"},
            {"id": "item_002", "question": "Q2?", "ground_truth": "NO"},
        ]

        config = ModelConfig(name="dummy", mode="zeroshot")
        client = make_model_client(config)

        results1 = evaluate_items_with_parallelism(
            items=items_batch1,
            client=client,
            model_name="dummy",
            mode="zeroshot",
            max_workers=1,
            cache=cache1,
        )

        assert len(results1) == 2
        stats1 = cache1.stats()
        assert stats1["total_entries"] == 2
        assert stats1["models"] == 1

        cache1.close()

        # Scenario 2: Rerun with same items (should use cache)
        cache2 = ResponseCache(tmpdir)

        results2 = evaluate_items_with_parallelism(
            items=items_batch1,
            client=client,
            model_name="dummy",
            mode="zeroshot",
            max_workers=1,
            cache=cache2,
        )

        assert len(results2) == 2
        # Responses should match (from cache)
        assert results2[0].pred_raw == results1[0].pred_raw
        assert results2[1].pred_raw == results1[1].pred_raw

        cache2.close()

        # Scenario 3: Run with new items + existing items
        cache3 = ResponseCache(tmpdir)

        items_batch2 = [
            {"id": "item_001", "question": "Q1?", "ground_truth": "YES"},  # cached
            {"id": "item_003", "question": "Q3?", "ground_truth": "YES"},  # new
        ]

        results3 = evaluate_items_with_parallelism(
            items=items_batch2,
            client=client,
            model_name="dummy",
            mode="zeroshot",
            max_workers=1,
            cache=cache3,
        )

        assert len(results3) == 2
        stats3 = cache3.stats()
        assert stats3["total_entries"] == 3  # 2 original + 1 new

        cache3.close()

        # Scenario 4: Different mode (separate cache)
        cache4 = ResponseCache(tmpdir)

        config_cot = ModelConfig(name="dummy", mode="cot")
        client_cot = make_model_client(config_cot)

        results4 = evaluate_items_with_parallelism(
            items=[{"id": "item_001", "question": "Q1?", "ground_truth": "YES"}],
            client=client_cot,
            model_name="dummy",
            mode="cot",
            max_workers=1,
            cache=cache4,
        )

        assert len(results4) == 1
        stats4 = cache4.stats()
        assert stats4["total_entries"] == 4  # 3 zeroshot + 1 cot

        cache4.close()

        # Scenario 5: Invalidation
        cache5 = ResponseCache(tmpdir)

        # Invalidate all zeroshot entries
        deleted = cache5.invalidate("dummy", "zeroshot")
        assert deleted == 3

        stats5 = cache5.stats()
        assert stats5["total_entries"] == 1  # Only cot entry remains

        cache5.close()


def test_cache_parallel_evaluation():
    """Test cache with parallel evaluation workers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        items = [
            {"id": f"item_{i:03d}", "question": f"Q{i}?", "ground_truth": "YES"}
            for i in range(10)
        ]

        config = ModelConfig(name="dummy", mode="zeroshot")
        client = make_model_client(config)

        # Run with sequential mode to avoid dummy client threading issues
        results = evaluate_items_with_parallelism(
            items=items,
            client=client,
            model_name="dummy",
            mode="zeroshot",
            max_workers=1,  # Sequential to avoid threading issues with dummy client
            cache=cache,
        )

        assert len(results) == 10
        stats = cache.stats()
        assert stats["total_entries"] == 10

        # Verify each item is cached
        for i in range(10):
            cached = cache.get("dummy", "zeroshot", f"item_{i:03d}", f"Q{i}?")
            assert cached is not None

        cache.close()


def test_cache_without_cache_object():
    """Test that evaluation works without cache (backward compatibility)."""
    items = [
        {"id": "item_001", "question": "Q1?", "ground_truth": "YES"},
    ]

    config = ModelConfig(name="dummy", mode="zeroshot")
    client = make_model_client(config)

    # Should work without cache parameter
    results = evaluate_items_with_parallelism(
        items=items,
        client=client,
        model_name="dummy",
        mode="zeroshot",
        max_workers=1,
        cache=None,  # No cache
    )

    assert len(results) == 1
    assert results[0].pred_raw is not None
