"""End-to-end test for cache with evaluation runner."""

import os
import tempfile

from chaosbench.eval.cache import ResponseCache
from chaosbench.eval.runner import evaluate_items_with_parallelism
from chaosbench.models.prompt import ModelConfig, make_model_client


def test_cache_populates_during_evaluation():
    """Test that cache is populated during evaluation run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        # Create test items
        items = [
            {
                "id": "test_001",
                "question": "Is the system stable?",
                "ground_truth": "YES",
                "system_id": "lorenz",
                "task_family": "stability",
            },
            {
                "id": "test_002",
                "question": "Does the system oscillate?",
                "ground_truth": "NO",
                "system_id": "lorenz",
                "task_family": "dynamics",
            },
        ]

        config = ModelConfig(name="dummy", mode="zeroshot")
        client = make_model_client(config)

        # Run evaluation with cache
        results = evaluate_items_with_parallelism(
            items=items,
            client=client,
            numeric_fact_map={},
            model_name="dummy",
            mode="zeroshot",
            max_workers=1,
            cache=cache,
        )

        assert len(results) == 2

        # Verify cache was populated
        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["models"] == 1

        # Verify we can retrieve from cache
        cached1 = cache.get("dummy", "zeroshot", "test_001", "Is the system stable?")
        assert cached1 is not None

        cached2 = cache.get("dummy", "zeroshot", "test_002", "Does the system oscillate?")
        assert cached2 is not None

        cache.close()


def test_cache_reuse_on_second_run():
    """Test that cache is reused on subsequent evaluation runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run - populate cache
        cache1 = ResponseCache(tmpdir)

        items = [
            {
                "id": "test_001",
                "question": "Is the system stable?",
                "ground_truth": "YES",
            },
        ]

        config = ModelConfig(name="dummy", mode="zeroshot")
        client = make_model_client(config)

        results1 = evaluate_items_with_parallelism(
            items=items,
            client=client,
            model_name="dummy",
            mode="zeroshot",
            max_workers=1,
            cache=cache1,
        )

        response1 = results1[0].pred_raw
        cache1.close()

        # Second run - should use cache
        cache2 = ResponseCache(tmpdir)
        results2 = evaluate_items_with_parallelism(
            items=items,
            client=client,
            model_name="dummy",
            mode="zeroshot",
            max_workers=1,
            cache=cache2,
        )

        response2 = results2[0].pred_raw
        cache2.close()

        # Should get same response from cache
        assert response1 == response2

        # Verify cache stats
        cache3 = ResponseCache(tmpdir)
        stats = cache3.stats()
        assert stats["total_entries"] == 1
        cache3.close()
