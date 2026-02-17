"""OpenAI GPT-4 / GPT-4o model adapter."""

import os

from chaosbench.models.prompt import ModelClient


try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIClient(ModelClient):
    """OpenAI GPT-4 / GPT-4o client."""

    def __init__(self, config):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def call(self, prompt: str) -> str:
        """Call OpenAI API.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Model response text.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""
