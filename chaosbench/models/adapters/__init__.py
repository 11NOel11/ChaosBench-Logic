"""Provider-specific model adapters."""

from chaosbench.models.prompt import ModelConfig, ModelClient


def build_client(config: ModelConfig) -> ModelClient:
    """Factory to create provider-specific model clients.

    Args:
        config: Model configuration with name identifying the provider.

    Returns:
        A ModelClient instance for the specified provider.

    Raises:
        ImportError: If the required provider package is not installed.
        ValueError: If the model name is not recognized.
    """
    m = config.name.lower()

    if m in ("gpt4", "gpt-4", "openai"):
        from chaosbench.models.adapters.openai_adapter import OpenAIClient
        return OpenAIClient(config)

    if m in ("claude", "claude3", "opus", "sonnet"):
        from chaosbench.models.adapters.anthropic_adapter import ClaudeClient
        return ClaudeClient(config)

    if m in ("gemini", "google"):
        from chaosbench.models.adapters.gemini_adapter import GeminiClient
        return GeminiClient(config)

    if m in ("llama", "llama3", "hf", "huggingface"):
        from chaosbench.models.adapters.hf_local import HFaceClient
        return HFaceClient(config)

    if m in ("mixtral", "mixtral8x7b", "mixtral-8x7b"):
        from chaosbench.models.adapters.hf_local import MixtralClient
        return MixtralClient(config)

    if m in ("openhermes", "openhermes2.5", "openhermes-2.5"):
        from chaosbench.models.adapters.hf_local import OpenHermesClient
        return OpenHermesClient(config)

    raise ValueError(f"Unknown model {config.name}")
