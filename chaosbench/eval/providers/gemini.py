"""Google Gemini provider (generativelanguage.googleapis.com) — uses urllib.request only, no SDK."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from chaosbench.eval.providers.base import Provider
from chaosbench.eval.providers.types import ProviderResponse

_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)
_STRICT_SUFFIX = "\n\nReturn exactly one token: TRUE or FALSE. No explanation."


class GeminiProvider(Provider):
    """Provider that calls the Google Gemini generateContent API.

    Reads the API key from the GEMINI_API_KEY environment variable.
    The key is passed as a URL query parameter — no Authorization header.
    Uses urllib.request only — no google-generativeai SDK dependency.

    Args:
        model: Gemini model ID, e.g. "gemini-2.0-flash" or "gemini-1.5-pro".
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum output tokens (default 16 suppresses CoT).
        timeout: HTTP timeout in seconds.
        retries: Retry count on 5xx / connection errors.
        strict_suffix: If True, append strict TRUE/FALSE instruction to every prompt.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
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
        return f"gemini/{self._model}"

    def _get_api_key(self) -> str:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export it before running: export GEMINI_API_KEY=AIza..."
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

        url = _ENDPOINT_TEMPLATE.format(model=self._model, key=api_key)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Optional[str] = None
        start = time.monotonic()
        for attempt in range(self._retries + 1):
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency = time.monotonic() - start
                text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
                usage = body.get("usageMetadata", {})
                raw = {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                }
                return ProviderResponse(text=text[:200], raw=raw, latency_s=latency)
            except urllib.error.HTTPError as e:
                status = e.code
                last_error = f"HTTPError {status}: {e.reason}"
                if status < 500:
                    break  # client-side error, don't retry
                if attempt < self._retries:
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
