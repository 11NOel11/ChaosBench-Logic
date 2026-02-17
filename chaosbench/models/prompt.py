"""Prompt building and model client base classes."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a model client.

    Attributes:
        name: Model identifier, e.g. "gpt4", "claude3", "gemini".
        mode: Prompting mode: "zeroshot", "cot", or "tool".
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
    """

    name: str
    mode: str
    temperature: float = 0.0
    max_tokens: int = 512


class ModelClient:
    """Base interface for model backends. Subclasses implement call()."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def build_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a prompt string based on mode and optional context.

        Args:
            question: The question text.
            context: Optional dict with keys like "numeric_facts".

        Returns:
            Formatted prompt string.
        """
        context = context or {}
        mode = self.config.mode

        if mode == "zeroshot":
            return (
                "Answer the following question with either YES or NO.\n"
                "Provide your answer in this exact format:\n"
                "FINAL_ANSWER: YES\n"
                "or\n"
                "FINAL_ANSWER: NO\n\n"
                f"Question: {question}\n\n"
                "FINAL_ANSWER:"
            )

        if mode == "cot":
            return (
                "You are a careful mathematical assistant.\n"
                "Think step by step, then provide your final answer.\n\n"
                f"Question: {question}\n\n"
                "Instructions:\n"
                "1. Reason through the problem step by step\n"
                "2. On the last line, write your final answer in this exact format:\n"
                "   FINAL_ANSWER: YES\n"
                "   or\n"
                "   FINAL_ANSWER: NO\n\n"
                "Your response:"
            )

        if mode == "tool":
            numeric = context.get("numeric_facts", "")
            return (
                "You are a reasoning assistant with access to numeric facts "
                "about a dynamical system (e.g., Lyapunov exponents, attractor type).\n"
                "Use these facts to reason logically.\n\n"
                f"Facts:\n{numeric}\n\n"
                f"Question: {question}\n\n"
                "Instructions:\n"
                "1. Use the provided facts to reason step by step\n"
                "2. On the last line, write your final answer in this exact format:\n"
                "   FINAL_ANSWER: YES\n"
                "   or\n"
                "   FINAL_ANSWER: NO\n\n"
                "Your response:"
            )

        return question

    def call(self, prompt: str) -> str:
        """Override this for each model backend.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Model response text.
        """
        raise NotImplementedError("Implement call() in a subclass or factory.")


class DummyEchoModel(ModelClient):
    """Fallback model that always answers YES. For pipeline testing only."""

    def call(self, prompt: str) -> str:
        """Returns a fixed YES answer.

        Args:
            prompt: Ignored.

        Returns:
            "Final answer: YES".
        """
        return "Final answer: YES"


def make_model_client(config: ModelConfig) -> ModelClient:
    """Factory to create model clients.

    Args:
        config: Model configuration.

    Returns:
        A ModelClient instance for the specified model.
    """
    if config.name.lower() in ("dummy", "test"):
        return DummyEchoModel(config)

    try:
        from chaosbench.models.adapters import build_client
        return build_client(config)
    except (ImportError, ValueError) as e:
        print(f"Warning: Could not load model '{config.name}': {e}")
        print("Falling back to DummyEchoModel for testing")
        return DummyEchoModel(config)
