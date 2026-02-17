"""Tests for evaluation response cache."""

import os
import tempfile
from pathlib import Path

import pytest

from chaosbench.eval.cache import ResponseCache


def test_cache_creation():
    """Test that cache creates DB file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)
        db_path = os.path.join(tmpdir, "response_cache.db")
        assert os.path.exists(db_path), "DB file should be created"
        cache.close()


def test_put_then_get_hit():
    """Test cache hit after put."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        model = "gpt4"
        mode = "zeroshot"
        item_id = "item_001"
        question = "Is the system stable?"
        response = "YES"

        cache.put(model, mode, item_id, question, response)
        cached = cache.get(model, mode, item_id, question)

        assert cached == response, "Should retrieve cached response"
        cache.close()


def test_get_without_put_miss():
    """Test cache miss when no entry exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        cached = cache.get("gpt4", "zeroshot", "item_999", "Does this exist?")

        assert cached is None, "Should return None for cache miss"
        cache.close()


def test_invalidate_by_model_mode():
    """Test invalidating all entries for a model+mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        # Add multiple entries
        cache.put("gpt4", "zeroshot", "item_001", "Q1", "YES")
        cache.put("gpt4", "zeroshot", "item_002", "Q2", "NO")
        cache.put("gpt4", "cot", "item_003", "Q3", "YES")
        cache.put("claude3", "zeroshot", "item_004", "Q4", "NO")

        # Invalidate gpt4 zeroshot
        deleted = cache.invalidate("gpt4", "zeroshot")

        assert deleted == 2, "Should delete 2 entries"
        assert cache.get("gpt4", "zeroshot", "item_001", "Q1") is None
        assert cache.get("gpt4", "zeroshot", "item_002", "Q2") is None
        assert cache.get("gpt4", "cot", "item_003", "Q3") == "YES"
        assert cache.get("claude3", "zeroshot", "item_004", "Q4") == "NO"

        cache.close()


def test_invalidate_by_model_mode_item():
    """Test invalidating specific item for model+mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        # Add multiple entries
        cache.put("gpt4", "zeroshot", "item_001", "Q1", "YES")
        cache.put("gpt4", "zeroshot", "item_002", "Q2", "NO")

        # Invalidate only item_001
        deleted = cache.invalidate("gpt4", "zeroshot", "item_001")

        assert deleted == 1, "Should delete 1 entry"
        assert cache.get("gpt4", "zeroshot", "item_001", "Q1") is None
        assert cache.get("gpt4", "zeroshot", "item_002", "Q2") == "NO"

        cache.close()


def test_stats():
    """Test cache statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        # Empty cache
        stats = cache.stats()
        assert stats["total_entries"] == 0
        assert stats["models"] == 0

        # Add entries
        cache.put("gpt4", "zeroshot", "item_001", "Q1", "YES")
        cache.put("gpt4", "cot", "item_002", "Q2", "NO")
        cache.put("claude3", "zeroshot", "item_003", "Q3", "YES")

        stats = cache.stats()
        assert stats["total_entries"] == 3
        assert stats["models"] == 2

        cache.close()


def test_context_manager():
    """Test context manager usage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "response_cache.db")

        with ResponseCache(tmpdir) as cache:
            cache.put("gpt4", "zeroshot", "item_001", "Q1", "YES")
            assert cache.get("gpt4", "zeroshot", "item_001", "Q1") == "YES"

        # DB file should exist after context exit
        assert os.path.exists(db_path)


def test_question_hash_collision_avoidance():
    """Test that different questions with same item_id are cached separately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        model = "gpt4"
        mode = "zeroshot"
        item_id = "item_001"

        q1 = "Is the system stable?"
        r1 = "YES"

        q2 = "Is the system unstable?"
        r2 = "NO"

        cache.put(model, mode, item_id, q1, r1)
        cache.put(model, mode, item_id, q2, r2)

        assert cache.get(model, mode, item_id, q1) == r1
        assert cache.get(model, mode, item_id, q2) == r2

        cache.close()


def test_replace_existing_entry():
    """Test that put replaces existing entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        model = "gpt4"
        mode = "zeroshot"
        item_id = "item_001"
        question = "Is the system stable?"

        cache.put(model, mode, item_id, question, "YES")
        cache.put(model, mode, item_id, question, "NO")

        cached = cache.get(model, mode, item_id, question)
        assert cached == "NO", "Should have replaced previous entry"

        stats = cache.stats()
        assert stats["total_entries"] == 1, "Should have only 1 entry after replace"

        cache.close()


def test_cache_persistence():
    """Test that cache persists across connections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache and add entry
        cache1 = ResponseCache(tmpdir)
        cache1.put("gpt4", "zeroshot", "item_001", "Q1", "YES")
        cache1.close()

        # Open new connection
        cache2 = ResponseCache(tmpdir)
        cached = cache2.get("gpt4", "zeroshot", "item_001", "Q1")
        assert cached == "YES", "Cache should persist across connections"
        cache2.close()


def test_unicode_questions_and_responses():
    """Test that cache handles unicode properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        question = "系统是否稳定？"  # Chinese
        response = "是的"  # Chinese "yes"

        cache.put("gpt4", "zeroshot", "item_001", question, response)
        cached = cache.get("gpt4", "zeroshot", "item_001", question)

        assert cached == response
        cache.close()


def test_special_characters_in_keys():
    """Test that cache handles special characters in keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(tmpdir)

        model = "gpt-4-turbo"
        mode = "zero-shot-cot"
        item_id = "batch_8_item_001"
        question = "Test question with 'quotes' and \"double quotes\""
        response = "Response with \n newlines and \t tabs"

        cache.put(model, mode, item_id, question, response)
        cached = cache.get(model, mode, item_id, question)

        assert cached == response
        cache.close()
