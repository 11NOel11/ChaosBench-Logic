"""Groq provider (api.groq.com) — OpenAI-compatible, uses urllib.request only."""

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Optional

from chaosbench.eval.providers.base import Provider
from chaosbench.eval.providers.types import ProviderResponse

try:
    import certifi as _certifi
    _SSL_CTX: Optional[ssl.SSLContext] = ssl.create_default_context(
        cafile=_certifi.where()
    )
except ImportError:
    _SSL_CTX = None

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_STRICT_SUFFIX = "\n\nReturn exactly one token: TRUE or FALSE. No explanation."


class GroqProvider(Provider):
    """Provider for Groq inference (api.groq.com).

    Supports fast inference for open models (e.g. llama-3.3-70b-versatile,
    qwen-qwq-32b, mixtral-8x7b-32768).
    Reads API key from GROQ_API_KEY environment variable.

    Args:
        model: Groq model ID, e.g. "llama-3.3-70b-versatile".
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum output tokens (16 suppresses CoT).
        timeout: HTTP timeout in seconds.
        retries: Retry count on 5xx / 429 errors.
        strict_suffix: If True, append strict TRUE/FALSE instruction to every prompt.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
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
        return f"groq/{self._model}"

    def _get_api_key(self) -> str:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Export it before running: export GROQ_API_KEY=gsk_..."
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
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _ENDPOINT,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
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
                text = (body["choices"][0]["message"]["content"] or "").strip()
                usage = body.get("usage", {})
                raw = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                return ProviderResponse(text=text[:200], raw=raw, latency_s=latency)
            except urllib.error.HTTPError as e:
                status = e.code
                last_error = f"HTTPError {status}: {e.reason}"
                if status == 429:
                    if attempt < self._retries:
                        time.sleep(2.0 * (attempt + 1))
                elif status < 500:
                    break
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
