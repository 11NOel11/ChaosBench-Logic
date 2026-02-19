"""Bounded concurrency and exponential backoff retry for model calls."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple


PROVIDER_LIMITS = {
    "openai": {"max_concurrent": 5, "base_delay": 0.1},
    "anthropic": {"max_concurrent": 4, "base_delay": 0.15},
    "google": {"max_concurrent": 8, "base_delay": 0.05},
    "huggingface": {"max_concurrent": 3, "base_delay": 0.15},
    "default": {"max_concurrent": 2, "base_delay": 0.2},
}


def get_provider_limits(provider: str) -> Dict[str, Any]:
    """Get concurrency limits for a provider.

    Args:
        provider: Provider name (e.g. "openai", "anthropic").

    Returns:
        Dict with max_concurrent and base_delay.
    """
    return PROVIDER_LIMITS.get(provider, PROVIDER_LIMITS["default"])


def infer_provider(model_name: str) -> str:
    """Infer provider from model name.

    Args:
        model_name: Model identifier string.

    Returns:
        Provider name string.
    """
    m = model_name.lower()
    if "gpt" in m or "openai" in m:
        return "openai"
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gemini" in m or "google" in m:
        return "google"
    if any(kw in m for kw in ["llama", "mixtral", "openhermes", "hf"]):
        return "huggingface"
    return "default"


def retry_with_backoff(
    func: Callable,
    max_retries: int = 4,
    initial_delay: float = 1.0,
) -> Tuple[Any, Optional[str], Optional[str]]:
    """Retry a callable with exponential backoff.

    Args:
        func: Callable to retry.
        max_retries: Maximum retry attempts.
        initial_delay: Initial backoff delay in seconds.

    Returns:
        Tuple of (result, error_type, error_message).
        On success: (result, None, None).
        On failure: (None, error_type_str, error_message_str).
    """
    last_error = None
    last_error_type = "OtherError"

    for attempt in range(max_retries + 1):
        try:
            result = func()
            return result, None, None
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if "429" in error_str or "rate limit" in error_str:
                error_type = "RateLimitError"
                is_retryable = True
            elif "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                error_type = "AuthError"
                is_retryable = False
            elif "api key" in error_str or "api_key" in error_str:
                error_type = "InvalidAPIKeyError"
                is_retryable = False
            elif any(code in error_str for code in ["500", "502", "503", "504"]):
                error_type = "ServerError"
                is_retryable = True
            elif "timeout" in error_str:
                error_type = "TimeoutError"
                is_retryable = True
            else:
                error_type = "OtherError"
                is_retryable = False

            last_error_type = error_type

            if not is_retryable:
                return None, error_type, f"{error_type}: {str(e)[:200]}"

            if attempt < max_retries:
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                return (
                    None,
                    error_type,
                    f"All {max_retries} retries failed - {error_type}: {str(e)[:200]}",
                )

    return None, last_error_type, f"Unexpected exit: {str(last_error)[:200]}"


def run_parallel(
    tasks: List[Callable],
    max_workers: int = 4,
    delay: float = 0.0,
) -> List[Tuple[int, Any, Optional[str]]]:
    """Run callables in parallel with bounded concurrency.

    Args:
        tasks: List of callables to execute.
        max_workers: Maximum concurrent workers.
        delay: Delay between task submissions in seconds.

    Returns:
        List of (index, result, error_string) tuples, ordered by completion.
    """
    results: List[Tuple[int, Any, Optional[str]]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, task in enumerate(tasks):
            if delay > 0 and i > 0:
                time.sleep(delay)
            future_to_idx[executor.submit(task)] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result, None))
            except Exception as e:
                results.append((idx, None, str(e)))

    return results
