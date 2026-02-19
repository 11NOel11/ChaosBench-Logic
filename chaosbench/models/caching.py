"""Content-addressable completion cache with JSONL storage."""

import hashlib
import json
import os
from typing import Any, Dict, Optional


class CompletionCache:
    """Content-addressable cache for model completions.

    Keys are computed from model name, prompt text, and generation parameters.
    Storage is JSONL-based in a configurable cache directory.

    Attributes:
        cache_dir: Directory for cache files.
        index: In-memory lookup from cache key to response.
    """

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.index: Dict[str, str] = {}
        self._load_existing()

    def _load_existing(self):
        """Load existing cache entries from disk."""
        if not os.path.exists(self.cache_dir):
            return

        cache_file = os.path.join(self.cache_dir, "completions.jsonl")
        if not os.path.exists(cache_file):
            return

        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.index[entry["key"]] = entry["response"]

    def _compute_key(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Compute cache key from request parameters.

        Args:
            model: Model identifier.
            prompt: Prompt text.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.

        Returns:
            SHA-256 hex digest cache key.
        """
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Look up a cached completion.

        Args:
            model: Model identifier.
            prompt: Prompt text.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.

        Returns:
            Cached response string, or None if not found.
        """
        key = self._compute_key(model, prompt, temperature, max_tokens)
        return self.index.get(key)

    def put(
        self,
        model: str,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Store a completion in the cache.

        Args:
            model: Model identifier.
            prompt: Prompt text.
            response: Model response to cache.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.

        Returns:
            The cache key.
        """
        key = self._compute_key(model, prompt, temperature, max_tokens)
        self.index[key] = response

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "completions.jsonl")
        entry = {"key": key, "model": model, "response": response}
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return key

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, key: str) -> bool:
        return key in self.index
