"""Anthropic Claude model adapter."""

import os

from chaosbench.models.prompt import ModelClient


try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeClient(ModelClient):
    """Anthropic Claude client."""

    def __init__(self, config):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def call(self, prompt: str) -> str:
        """Call Anthropic API.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Model response text.
        """
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        for content_block in response.content:
            if hasattr(content_block, "text"):
                return content_block.text
        return str(response.content[0])
