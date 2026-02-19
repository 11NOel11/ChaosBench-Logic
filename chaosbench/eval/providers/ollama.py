"""Ollama local provider (http://localhost:11434)."""

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from chaosbench.eval.providers.base import Provider
from chaosbench.eval.providers.types import ProviderResponse

_DEFAULT_HOST = "http://localhost:11434"
_GENERATE_ENDPOINT = "/api/generate"


class OllamaProvider(Provider):
    """Provider that calls a local Ollama server.

    Uses the HTTP /api/generate endpoint (no streaming) which avoids
    subprocess overhead and works well with concurrent calls.

    Args:
        model: Ollama model tag, e.g. "qwen2.5:7b".
        host: Base URL for the Ollama server.
        temperature: Sampling temperature (0.0 = greedy).
        top_p: Nucleus sampling probability.
        max_tokens: Maximum tokens to generate.
        timeout: HTTP request timeout in seconds.
        retries: Number of retries on transient errors (5xx / connection errors).
    """

    def __init__(
        self,
        model: str,
        host: str = _DEFAULT_HOST,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_tokens: int = 16,
        timeout: int = 60,
        retries: int = 2,
    ):
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._retries = retries

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self._top_p is not None:
            payload["options"]["top_p"] = self._top_p

        url = self._host + _GENERATE_ENDPOINT
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Optional[str] = None
        for attempt in range(self._retries + 1):
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency = time.monotonic() - start
                text = body.get("response", "").strip()
                return ProviderResponse(text=text, raw=body, latency_s=latency)
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
