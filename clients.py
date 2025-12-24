import os
from typing import Optional, Dict, Any

# Import ModelClient directly - no circular import since we import late
import sys
import importlib

# Get ModelClient at runtime
eval_module = importlib.import_module('eval_chaosbench')
ModelClient = eval_module.ModelClient

############################################
# 1. OpenAI GPT-4 / GPT-4.1 / o3-mini
############################################

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if OPENAI_AVAILABLE:
    class OpenAIClient(ModelClient):
        def __init__(self, config):
            super().__init__(config)
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def call(self, prompt: str) -> str:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or gpt-4o-mini, gpt-4-turbo
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""


############################################
# 2. Anthropic Claude Opus 4
############################################

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

if ANTHROPIC_AVAILABLE:
    class ClaudeClient(ModelClient):
        def __init__(self, config):
            super().__init__(config)
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        def call(self, prompt: str) -> str:
            # Use Claude 3.5 Sonnet (latest production model)
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            # Handle different content block types - extract text from TextBlock
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    return content_block.text  # type: ignore
            return str(response.content[0])


############################################
# 3. Google Gemini 1.5 Pro / Ultra
############################################

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

class GeminiClient(ModelClient):
    def __init__(self, config):
        super().__init__(config)
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # type: ignore
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")  # type: ignore

    def call(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.config.temperature, "max_output_tokens": self.config.max_tokens}
        )
        return response.text


############################################
# 4. HuggingFace Inference API (Llama-3)
############################################

try:
    from huggingface_hub import InferenceClient as HFInferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

if HUGGINGFACE_AVAILABLE:
    class HFaceClient(ModelClient):
        """HuggingFace client for LLaMA-3 70B Instruct"""
        def __init__(self, config):
            super().__init__(config)
            self.client = HFInferenceClient(
                "meta-llama/Meta-Llama-3-70B-Instruct",
                token=os.getenv("HF_API_KEY")
            )

        def call(self, prompt: str) -> str:
            # Use chat_completion (conversational) API - required for this model
            messages = [{"role": "user", "content": prompt}]
            
            try:
                # Use chat_completion since text_generation is not supported for this provider
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                
                # Extract response content
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content
                elif isinstance(response, dict) and 'choices' in response:
                    content = response['choices'][0]['message']['content']
                else:
                    # Fallback: try to extract from response object
                    try:
                        content = str(response)
                    except:
                        content = None
                
                if content is None:
                    raise RuntimeError(f"LLaMA-3 API returned None content")
                if content == "":
                    raise RuntimeError(f"LLaMA-3 API returned empty string (check max_tokens={self.config.max_tokens})")
                    
                return content
                
            except Exception as e:
                error_msg = str(e)
                if "None" in error_msg or "empty" in error_msg.lower():
                    print(f"  [LLaMA-3] Empty/None response: {error_msg}")
                raise RuntimeError(f"LLaMA-3: {error_msg}")

    class MixtralClient(ModelClient):
        """HuggingFace client for Mixtral 8x7B Instruct"""
        def __init__(self, config):
            super().__init__(config)
            self.client = HFInferenceClient(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                token=os.getenv("HF_API_KEY")
            )

        def call(self, prompt: str) -> str:
            # Mixtral uses instruction format: wrap in [INST] tags
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            try:
                # Use text_generation instead of chat_completion
                response = self.client.text_generation(
                    formatted_prompt,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    return_full_text=False,
                )
                
                if response is None:
                    raise RuntimeError(f"Mixtral API returned None content")
                if response == "":
                    raise RuntimeError(f"Mixtral API returned empty string (check max_tokens={self.config.max_tokens})")
                    
                return response
                
            except Exception as e:
                error_msg = str(e)
                if "None" in error_msg or "empty" in error_msg.lower():
                    print(f"  [Mixtral] Empty/None response: {error_msg}")
                raise RuntimeError(f"Mixtral: {error_msg}")

    class OpenHermesClient(ModelClient):
        """HuggingFace client for OpenHermes-2.5-Mistral-7B"""
        def __init__(self, config):
            super().__init__(config)
            self.client = HFInferenceClient(
                "teknium/OpenHermes-2.5-Mistral-7B",
                token=os.getenv("HF_API_KEY")
            )

        def call(self, prompt: str) -> str:
            # OpenHermes uses ChatML format
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            try:
                # Use text_generation instead of chat_completion
                response = self.client.text_generation(
                    formatted_prompt,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    return_full_text=False,
                    stop_sequences=["<|im_end|>"],
                )
                
                if response is None:
                    raise RuntimeError(f"OpenHermes API returned None content")
                if response == "":
                    raise RuntimeError(f"OpenHermes API returned empty string (check max_tokens={self.config.max_tokens})")
                    
                return response
                
            except Exception as e:
                error_msg = str(e)
                if "None" in error_msg or "empty" in error_msg.lower():
                    print(f"  [OpenHermes] Empty/None response: {error_msg}")
                raise RuntimeError(f"OpenHermes: {error_msg}")
        

############################################
# 5. Factory
############################################

def build_client(config):
    m = config.name.lower()

    # OpenAI GPT-4
    if m in ("gpt4", "gpt-4", "openai"):
        if OPENAI_AVAILABLE:
            return OpenAIClient(config)
        raise ImportError("openai package not installed. Run: pip install openai")

    # Anthropic Claude
    if m in ("claude", "claude3", "opus", "sonnet"):
        if ANTHROPIC_AVAILABLE:
            return ClaudeClient(config)
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    # Google Gemini
    if m in ("gemini", "google"):
        return GeminiClient(config)

    # Meta LLaMA-3
    if m in ("llama", "llama3", "hf", "huggingface"):
        if HUGGINGFACE_AVAILABLE:
            return HFaceClient(config)
        raise ImportError("huggingface-hub package not installed. Run: pip install huggingface-hub")

    # Mixtral 8x7B
    if m in ("mixtral", "mixtral8x7b", "mixtral-8x7b"):
        if HUGGINGFACE_AVAILABLE:
            return MixtralClient(config)
        raise ImportError("huggingface-hub package not installed. Run: pip install huggingface-hub")

    # OpenHermes 2.5
    if m in ("openhermes", "openhermes2.5", "openhermes-2.5"):
        if HUGGINGFACE_AVAILABLE:
            return OpenHermesClient(config)
        raise ImportError("huggingface-hub package not installed. Run: pip install huggingface-hub")

    raise ValueError(f"Unknown model {config.name}")
