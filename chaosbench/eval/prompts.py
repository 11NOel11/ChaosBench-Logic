"""Versioned prompt templates for evaluation.

Each template is registered with a version string and a SHA256 hash of its
content is stored in the run manifest for reproducibility.
"""

import hashlib
import os
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTION_V1 = (
    "You are an expert in dynamical systems and chaos theory. "
    "Answer the following question about a dynamical system with EXACTLY one word: "
    "TRUE or FALSE. Do not add any explanation."
)

_USER_TEMPLATE_V1 = "{question}"

_FORCED_FORMAT_REPROMPT_V1 = (
    "Your previous answer was not clear. Reply with EXACTLY one word: TRUE or FALSE."
)

_SYSTEM_INSTRUCTION_V1_COMPACT = (
    "You are evaluating a statement about a dynamical system. "
    "Return EXACTLY one token: TRUE or FALSE."
)

_SYSTEM_INSTRUCTION_V1_LOGICCHECK = (
    "You are an expert in dynamical systems and logic. "
    "First decide whether the proposition is true under standard definitions, "
    "then output EXACTLY one word: TRUE or FALSE."
)

_PROMPT_VARIANTS: Dict[str, Tuple[str, str, str]] = {
    "v1": (
        _SYSTEM_INSTRUCTION_V1,
        _USER_TEMPLATE_V1,
        _FORCED_FORMAT_REPROMPT_V1,
    ),
    "v1_compact": (
        _SYSTEM_INSTRUCTION_V1_COMPACT,
        _USER_TEMPLATE_V1,
        _FORCED_FORMAT_REPROMPT_V1,
    ),
    "v1_logiccheck": (
        _SYSTEM_INSTRUCTION_V1_LOGICCHECK,
        _USER_TEMPLATE_V1,
        _FORCED_FORMAT_REPROMPT_V1,
    ),
}

_PROMPT_VARIANT_ENV = "CHAOSBENCH_PROMPT_VARIANT"
_DEFAULT_PROMPT_VARIANT = "v1"


def _active_prompt_variant() -> str:
    candidate = os.getenv(_PROMPT_VARIANT_ENV, _DEFAULT_PROMPT_VARIANT).strip()
    if candidate in _PROMPT_VARIANTS:
        return candidate
    return _DEFAULT_PROMPT_VARIANT


def _active_prompt_tuple() -> Tuple[str, str, str]:
    return _PROMPT_VARIANTS[_active_prompt_variant()]


def build_prompt(question: str, system_id: Optional[str] = None) -> str:
    """Build the full evaluation prompt for a question.

    Args:
        question: The question text from the dataset.
        system_id: Optional system identifier for context (not used in v1).

    Returns:
        Full prompt string to send to the model.
    """
    system_instruction, user_template, _ = _active_prompt_tuple()
    rendered_question = user_template.format(
        question=question, system_id=system_id or ""
    )
    return f"{system_instruction}\n\nQuestion: {rendered_question}"


def build_reprompt(original_prompt: str, original_response: str) -> str:
    """Build a retry prompt when the first response was INVALID.

    Args:
        original_prompt: The original prompt that produced an invalid response.
        original_response: The invalid response text.

    Returns:
        A shorter forced-format reprompt.
    """
    _, _, forced_format_reprompt = _active_prompt_tuple()
    return (
        f"{original_prompt}\n\n"
        f"Your response: {original_response[:200]}\n\n"
        f"{forced_format_reprompt}"
    )


def get_prompt_version() -> str:
    return _active_prompt_variant()


def get_prompt_hash() -> str:
    """SHA256 of system instruction + user template (for manifest logging)."""
    system_instruction, user_template, _ = _active_prompt_tuple()
    content = system_instruction + "\n" + user_template
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
