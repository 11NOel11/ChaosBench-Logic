"""Integration test for cache with evaluation runner."""

import json
import os
import tempfile

from chaosbench.eval.cache import ResponseCache
from chaosbench.eval.runner import evaluate_single_item_robust
from chaosbench.models.prompt import ModelConfig, make_model_client


def test_cache_integration_with_evaluate_single_item():
    """Test that cache integration works with evaluate_single_item_robust."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        # Create a test item
        item = {
            "id": "test_001",
            "question": "Is the system stable?",
            "ground_truth": "YES",
            "system_id": "lorenz",
            "task_family": "stability",
        }

        # Use dummy client
        config = ModelConfig(name="dummy", mode="zeroshot")
        client = make_model_client(config)

        # First call - should call model and cache
        result1 = evaluate_single_item_robust(
            item=item,
            client=client,
            numeric_fact_map={},
            delay=0.0,
            cache=cache,
            model_name="dummy",
            mode="zeroshot",
        )

        assert result1.pred_raw is not None
        initial_response = result1.pred_raw

        # Check cache was populated
        cached = cache.get("dummy", "zeroshot", "test_001", "Is the system stable?")
        assert cached == initial_response

        # Second call - should use cache (same response)
        result2 = evaluate_single_item_robust(
            item=item,
            client=client,
            numeric_fact_map={},
            delay=0.0,
            cache=cache,
            model_name="dummy",
            mode="zeroshot",
        )

        assert result2.pred_raw == initial_response

        # Verify stats
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["models"] == 1

        cache.close()


def test_cache_respects_different_modes():
    """Test that cache treats different modes as separate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        item = {
            "id": "test_001",
            "question": "Is the system stable?",
            "ground_truth": "YES",
        }

        config_zeroshot = ModelConfig(name="dummy", mode="zeroshot")
        client_zeroshot = make_model_client(config_zeroshot)

        config_cot = ModelConfig(name="dummy", mode="cot")
        client_cot = make_model_client(config_cot)

        # Evaluate with zeroshot
        result1 = evaluate_single_item_robust(
            item=item,
            client=client_zeroshot,
            numeric_fact_map={},
            cache=cache,
            model_name="dummy",
            mode="zeroshot",
        )

        # Evaluate with cot (should be separate cache entry)
        result2 = evaluate_single_item_robust(
            item=item,
            client=client_cot,
            numeric_fact_map={},
            cache=cache,
            model_name="dummy",
            mode="cot",
        )

        # Should have 2 separate cache entries
        stats = cache.stats()
        assert stats["total_entries"] == 2

        cache.close()


def test_cache_miss_calls_model():
    """Test that cache miss results in model call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        item = {
            "id": "test_001",
            "question": "Is the system stable?",
            "ground_truth": "YES",
        }

        config = ModelConfig(name="dummy", mode="zeroshot")
        client = make_model_client(config)

        # First call - cache miss
        result = evaluate_single_item_robust(
            item=item,
            client=client,
            numeric_fact_map={},
            cache=cache,
            model_name="dummy",
            mode="zeroshot",
        )

        assert result.pred_raw is not None
        assert result.pred_norm is not None

        # Verify cache was populated
        cached = cache.get("dummy", "zeroshot", "test_001", "Is the system stable?")
        assert cached == result.pred_raw

        cache.close()
