"""Backward-compatibility shim for clients.

All functionality has moved to chaosbench.models.adapters.
This file re-exports everything so existing imports continue to work.
"""

from chaosbench.models.prompt import ModelClient, ModelConfig
from chaosbench.models.adapters import build_client

try:
    from chaosbench.models.adapters.openai_adapter import OpenAIClient, OPENAI_AVAILABLE
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from chaosbench.models.adapters.anthropic_adapter import ClaudeClient, ANTHROPIC_AVAILABLE
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from chaosbench.models.adapters.gemini_adapter import GeminiClient
except ImportError:
    pass

try:
    from chaosbench.models.adapters.hf_local import (
        HFaceClient,
        MixtralClient,
        OpenHermesClient,
        HUGGINGFACE_AVAILABLE,
    )
except ImportError:
    HUGGINGFACE_AVAILABLE = False
