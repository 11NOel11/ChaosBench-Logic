"""HuggingFace Inference API adapters for LLaMA-3, Mixtral, OpenHermes."""

import os

from chaosbench.models.prompt import ModelClient


try:
    from huggingface_hub import InferenceClient as HFInferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


def _check_hf():
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "huggingface-hub package not installed. Run: pip install huggingface-hub"
        )


class HFaceClient(ModelClient):
    """HuggingFace client for LLaMA-3 70B Instruct."""

    def __init__(self, config):
        super().__init__(config)
        _check_hf()
        self.client = HFInferenceClient(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            token=os.getenv("HF_API_KEY"),
        )

    def call(self, prompt: str) -> str:
        """Call HuggingFace Inference API for LLaMA-3.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Model response text.
        """
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            content = str(response)

        if content is None:
            raise RuntimeError("LLaMA-3 API returned None content")
        if content == "":
            raise RuntimeError(
                f"LLaMA-3 API returned empty string (check max_tokens={self.config.max_tokens})"
            )

        return content


class MixtralClient(ModelClient):
    """HuggingFace client for Mixtral 8x7B Instruct."""

    def __init__(self, config):
        super().__init__(config)
        _check_hf()
        self.client = HFInferenceClient(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            token=os.getenv("HF_API_KEY"),
        )

    def call(self, prompt: str) -> str:
        """Call HuggingFace Inference API for Mixtral.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Model response text.
        """
        formatted_prompt = f"[INST] {prompt} [/INST]"

        response = self.client.text_generation(
            formatted_prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            return_full_text=False,
        )

        if response is None:
            raise RuntimeError("Mixtral API returned None content")
        if response == "":
            raise RuntimeError(
                f"Mixtral API returned empty string (check max_tokens={self.config.max_tokens})"
            )

        return response


class OpenHermesClient(ModelClient):
    """HuggingFace client for OpenHermes-2.5-Mistral-7B."""

    def __init__(self, config):
        super().__init__(config)
        _check_hf()
        self.client = HFInferenceClient(
            "teknium/OpenHermes-2.5-Mistral-7B",
            token=os.getenv("HF_API_KEY"),
        )

    def call(self, prompt: str) -> str:
        """Call HuggingFace Inference API for OpenHermes.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Model response text.
        """
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )

        response = self.client.text_generation(
            formatted_prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            return_full_text=False,
            stop_sequences=["<|im_end|>"],
        )

        if response is None:
            raise RuntimeError("OpenHermes API returned None content")
        if response == "":
            raise RuntimeError(
                f"OpenHermes API returned empty string (check max_tokens={self.config.max_tokens})"
            )

        return response
