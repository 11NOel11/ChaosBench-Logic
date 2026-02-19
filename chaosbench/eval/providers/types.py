"""Response types for evaluation providers."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProviderResponse:
    """Response from a model provider."""

    text: str
    raw: Dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None
