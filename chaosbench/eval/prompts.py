"""Versioned prompt templates for evaluation.

Each template is registered with a version string and a SHA256 hash of its
content is stored in the run manifest for reproducibility.
"""

import hashlib
from typing import Optional

# ---------------------------------------------------------------------------
# Template v1 (default)
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTION_V1 = (
    "You are an expert in dynamical systems and chaos theory. "
    "Answer the following question about a dynamical system with EXACTLY one word: "
    "TRUE or FALSE. Do not add any explanation."
)

_USER_TEMPLATE_V1 = "{question}"

_FORCED_FORMAT_REPROMPT_V1 = (
    "Your previous answer was not clear. "
    "Reply with EXACTLY one word: TRUE or FALSE."
)

_PROMPT_VERSION = "v1"


def build_prompt(question: str, system_id: Optional[str] = None) -> str:
    """Build the full evaluation prompt for a question.

    Args:
        question: The question text from the dataset.
        system_id: Optional system identifier for context (not used in v1).

    Returns:
        Full prompt string to send to the model.
    """
    return f"{_SYSTEM_INSTRUCTION_V1}\n\nQuestion: {question}"


def build_reprompt(original_prompt: str, original_response: str) -> str:
    """Build a retry prompt when the first response was INVALID.

    Args:
        original_prompt: The original prompt that produced an invalid response.
        original_response: The invalid response text.

    Returns:
        A shorter forced-format reprompt.
    """
    return (
        f"{original_prompt}\n\n"
        f"Your response: {original_response[:200]}\n\n"
        f"{_FORCED_FORMAT_REPROMPT_V1}"
    )


def get_prompt_version() -> str:
    return _PROMPT_VERSION


def get_prompt_hash() -> str:
    """SHA256 of system instruction + user template (for manifest logging)."""
    content = _SYSTEM_INSTRUCTION_V1 + "\n" + _USER_TEMPLATE_V1
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
