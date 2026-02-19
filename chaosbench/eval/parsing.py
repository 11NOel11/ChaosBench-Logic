"""Robust output parsing with 3-way outcomes: VALID_TRUE / VALID_FALSE / INVALID.

Design decisions:
- VALID_TRUE / VALID_FALSE: parsed successfully into a binary label.
- INVALID: cannot extract an unambiguous TRUE/FALSE from the output.
- YES/NO are normalised to TRUE/FALSE.
- strict_mode (default): final token / final non-empty line must be the label.
- lenient_mode: scan full text for "Answer: TRUE" style patterns first.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ParseOutcome(str, Enum):
    """Three-way parsing outcome."""

    VALID_TRUE = "VALID_TRUE"
    VALID_FALSE = "VALID_FALSE"
    INVALID = "INVALID"


# Mapping from raw tokens to canonical label
_TRUE_TOKENS = {"yes", "true", "y", "t"}
_FALSE_TOKENS = {"no", "false", "n", "f"}

_CLEAN_RE = re.compile(r"[*_`#\[\]()\"']")


@dataclass
class ParsedLabel:
    """Result of parsing a model response."""

    outcome: ParseOutcome
    label: Optional[str]  # "TRUE", "FALSE", or None
    confidence: float  # 1.0 if unambiguous, 0.5 if heuristic, 0.0 if invalid
    reason: str  # human-readable explanation


def _clean(text: str) -> str:
    return _CLEAN_RE.sub("", text)


def _token_to_label(token: str) -> Optional[str]:
    t = token.strip().lower().rstrip(".,;:!?")
    if t in _TRUE_TOKENS:
        return "TRUE"
    if t in _FALSE_TOKENS:
        return "FALSE"
    return None


# Ordered list of (pattern, group_index) for answer extraction
_ANSWER_PATTERNS = [
    # Explicit markers: FINAL_ANSWER: TRUE
    (re.compile(
        r"FINAL[_\s\-]*ANSWER\s*[:=]\s*([^\n.,;]+)",
        re.IGNORECASE,
    ), 1),
    # "answer is TRUE", "the answer is: FALSE"
    (re.compile(
        r"(?:the\s+)?answer\s+is\s*[:=]?\s*([^\n.,;]+)",
        re.IGNORECASE,
    ), 1),
    # "Answer: TRUE"
    (re.compile(
        r"\banswer\s*[:=]\s*([^\n.,;]+)",
        re.IGNORECASE,
    ), 1),
    # "Therefore: TRUE" / "therefore, TRUE"
    (re.compile(
        r"\btherefore\s*[,:]?\s*([^\n.,;]+)",
        re.IGNORECASE,
    ), 1),
    # "Conclusion: FALSE"
    (re.compile(
        r"\bconclusion\s*[:=]\s*([^\n.,;]+)",
        re.IGNORECASE,
    ), 1),
    # "So the answer is TRUE"
    (re.compile(
        r"\bso\s+(?:the\s+answer\s+is\s+)?([^\n.,;]+)",
        re.IGNORECASE,
    ), 1),
    # "My final answer: TRUE"
    (re.compile(
        r"(?:my\s+)?(?:final|ultimate)\s+answer\s*[:=]?\s*([^\n.,;]+)",
        re.IGNORECASE,
    ), 1),
]


def _extract_label_from_text(text: str) -> tuple[Optional[str], str]:
    """Try pattern-based extraction; return (label_or_None, reason)."""
    cleaned = _clean(text)
    for pattern, grp in _ANSWER_PATTERNS:
        m = pattern.search(cleaned)
        if m:
            candidate = m.group(grp).strip()
            # Take the last token in the candidate
            tokens = candidate.split()
            for tok in reversed(tokens):
                lbl = _token_to_label(tok)
                if lbl:
                    return lbl, f"matched pattern: {pattern.pattern[:40]}"
    return None, "no pattern matched"


def _extract_label_from_final_tokens(text: str) -> tuple[Optional[str], str]:
    """Check the last non-empty line for a binary label."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None, "empty text"
    last_line = lines[-1]
    tokens = _clean(last_line).split()
    if not tokens:
        return None, "empty last line"
    # Check if any token in the last line (prefer last) is a label
    for tok in reversed(tokens):
        lbl = _token_to_label(tok)
        if lbl:
            return lbl, "final line token"
    return None, f"last line tokens unrecognized: {tokens[:4]}"


def _check_ambiguous(text: str) -> bool:
    """Return True if the text contains signals of deliberate ambiguity."""
    lower = text.lower()
    ambiguous_phrases = [
        "it depends",
        "cannot determine",
        "cannot be determined",
        "not enough information",
        "ambiguous",
        "both true and false",
        "both false and true",
        "neither true nor false",
    ]
    return any(phrase in lower for phrase in ambiguous_phrases)


def parse_label(text: Optional[str], strict: bool = True) -> ParsedLabel:
    """Parse a model response into a ParsedLabel.

    Args:
        text: Raw model output.
        strict: If True, prefer the final-line token heuristic.
                If False (lenient), try pattern matching first.

    Returns:
        ParsedLabel with outcome (VALID_TRUE/VALID_FALSE/INVALID), label, confidence, reason.
    """
    if not text or not text.strip():
        return ParsedLabel(
            outcome=ParseOutcome.INVALID,
            label=None,
            confidence=0.0,
            reason="empty response",
        )

    if _check_ambiguous(text):
        return ParsedLabel(
            outcome=ParseOutcome.INVALID,
            label=None,
            confidence=0.0,
            reason="response signals ambiguity",
        )

    if strict:
        lbl, reason = _extract_label_from_final_tokens(text)
        if lbl is None:
            lbl, reason = _extract_label_from_text(text)
    else:
        lbl, reason = _extract_label_from_text(text)
        if lbl is None:
            lbl, reason = _extract_label_from_final_tokens(text)

    if lbl is None:
        return ParsedLabel(
            outcome=ParseOutcome.INVALID,
            label=None,
            confidence=0.0,
            reason=reason,
        )

    outcome = ParseOutcome.VALID_TRUE if lbl == "TRUE" else ParseOutcome.VALID_FALSE
    confidence = 1.0 if "FINAL_ANSWER" in text.upper() else 0.9
    return ParsedLabel(outcome=outcome, label=lbl, confidence=confidence, reason=reason)


def outcome_to_label(outcome: ParseOutcome) -> Optional[str]:
    """Convert outcome to label string or None."""
    if outcome == ParseOutcome.VALID_TRUE:
        return "TRUE"
    if outcome == ParseOutcome.VALID_FALSE:
        return "FALSE"
    return None
