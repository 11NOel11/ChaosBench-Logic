"""Google Gemini model adapter."""

import os

from chaosbench.models.prompt import ModelClient


try:
    import google.generativeai as genai
except ImportError:
    genai = None


class GeminiClient(ModelClient):
    """Google Gemini client."""

    def __init__(self, config):
        super().__init__(config)
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")

    def call(self, prompt: str) -> str:
        """Call Gemini API.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Model response text.
        """
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )
        return response.text
