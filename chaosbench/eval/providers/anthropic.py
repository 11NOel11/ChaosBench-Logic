"""Anthropic provider (api.anthropic.com) — uses urllib.request only, no SDK."""

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Optional

from chaosbench.eval.providers.base import Provider
from chaosbench.eval.providers.types import ProviderResponse

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_STRICT_SUFFIX = "\n\nReturn exactly one token: TRUE or FALSE. No explanation."

# Build an SSL context that trusts certifi's CA bundle when available.
try:
    import certifi as _certifi
    _SSL_CTX: Optional[ssl.SSLContext] = ssl.create_default_context(
        cafile=_certifi.where()
    )
except ImportError:
    _SSL_CTX = None


class AnthropicProvider(Provider):
    """Provider that calls the Anthropic Messages API.

    Reads the API key from the ANTHROPIC_API_KEY environment variable.
    Uses urllib.request only — no anthropic SDK dependency.

    Args:
        model: Anthropic model ID, e.g. "claude-sonnet-4-6" or "claude-haiku-4-5".
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum tokens to generate (default 16 suppresses CoT).
        timeout: HTTP timeout in seconds.
        retries: Retry count on 5xx / connection errors.
        strict_suffix: If True, append strict TRUE/FALSE instruction to every prompt.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 16,
        timeout: int = 60,
        retries: int = 2,
        strict_suffix: bool = True,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._retries = retries
        self._strict_suffix = strict_suffix

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def _get_api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        return key

    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        if self._strict_suffix:
            prompt = prompt + _STRICT_SUFFIX

        try:
            api_key = self._get_api_key()
        except RuntimeError as e:
            return ProviderResponse(text="", latency_s=0.0, error=str(e))

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        # Anthropic API does not accept temperature=0.0 without explicit support;
        # include it to allow deterministic sampling where the API honours it.
        if temperature is not None:
            payload["temperature"] = temperature

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _ENDPOINT,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": _API_VERSION,
            },
            method="POST",
        )

        last_error: Optional[str] = None
        start = time.monotonic()
        for attempt in range(self._retries + 1):
            start = time.monotonic()
            try:
                kw = {"context": _SSL_CTX} if _SSL_CTX is not None else {}
                with urllib.request.urlopen(req, timeout=self._timeout, **kw) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency = time.monotonic() - start
                text = (body["content"][0]["text"] or "").strip()
                raw = {
                    "prompt_tokens": body.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": body.get("usage", {}).get("output_tokens", 0),
                }
                return ProviderResponse(text=text[:200], raw=raw, latency_s=latency)
            except urllib.error.HTTPError as e:
                status = e.code
                last_error = f"HTTPError {status}: {e.reason}"
                if status == 429:
                    # Rate limited — back off and retry
                    if attempt < self._retries:
                        time.sleep(2.0 * (attempt + 1))
                elif status < 500:
                    break  # other 4xx: non-retryable
                elif attempt < self._retries:
                    time.sleep(1.0 * (attempt + 1))
            except urllib.error.URLError as e:
                last_error = f"URLError: {e.reason}"
                if attempt < self._retries:
                    time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                break

        latency = time.monotonic() - start
        return ProviderResponse(text="", latency_s=latency, error=last_error)
