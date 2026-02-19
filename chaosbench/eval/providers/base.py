"""Abstract provider interface."""

from abc import ABC, abstractmethod

from chaosbench.eval.providers.types import ProviderResponse


class Provider(ABC):
    """Abstract base class for all model providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider/model identifier."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.).

        Returns:
            ProviderResponse with text, latency, and optional error.
        """
        ...
