"""Mock provider for testing (no network calls)."""

import time
from typing import Callable, Dict, Iterator, Optional, Union

from chaosbench.eval.providers.base import Provider
from chaosbench.eval.providers.types import ProviderResponse


class MockProvider(Provider):
    """Deterministic mock provider for unit tests and smoke tests.

    Responses can be specified as:
    - A single default string returned for every prompt.
    - A dict mapping item_id (extracted from prompt) to response string.
    - A callable (prompt: str) -> str.
    - An iterable of strings consumed in order (cycles on exhaustion).
    """

    def __init__(
        self,
        responses: Union[str, Dict[str, str], Callable[[str], str], None] = None,
        default: str = "TRUE",
        latency_s: float = 0.0,
    ):
        self._default = default
        self._latency_s = latency_s
        self._call_index = 0

        if responses is None:
            self._mode = "default"
        elif isinstance(responses, str):
            self._default = responses
            self._mode = "default"
        elif isinstance(responses, dict):
            self._responses_dict = responses
            self._mode = "dict"
        elif callable(responses):
            self._responses_fn = responses
            self._mode = "callable"
        elif hasattr(responses, "__iter__"):
            self._responses_list = list(responses)
            self._mode = "list"
        else:
            self._mode = "default"

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        start = time.monotonic()
        if self._latency_s:
            time.sleep(self._latency_s)

        if self._mode == "default":
            text = self._default
        elif self._mode == "dict":
            # Try to find an id in the prompt; fall back to default
            text = self._default
            for key, val in self._responses_dict.items():
                if key in prompt:
                    text = val
                    break
        elif self._mode == "callable":
            text = self._responses_fn(prompt)
        elif self._mode == "list":
            idx = self._call_index % len(self._responses_list)
            text = self._responses_list[idx]
        else:
            text = self._default

        self._call_index += 1
        latency = time.monotonic() - start
        return ProviderResponse(text=text, latency_s=latency)
