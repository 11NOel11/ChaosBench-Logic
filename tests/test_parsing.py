"""Tests for chaosbench.eval.parsing: 3-way outcome parser."""

import pytest

from chaosbench.eval.parsing import (
    ParseOutcome,
    ParsedLabel,
    parse_label,
    outcome_to_label,
)


class TestParseLabel:
    # --- Exact TRUE/FALSE ---
    def test_exact_true(self):
        r = parse_label("TRUE")
        assert r.outcome == ParseOutcome.VALID_TRUE
        assert r.label == "TRUE"

    def test_exact_false(self):
        r = parse_label("FALSE")
        assert r.outcome == ParseOutcome.VALID_FALSE
        assert r.label == "FALSE"

    # --- YES/NO normalization ---
    def test_yes_normalized_to_true(self):
        r = parse_label("Yes")
        assert r.outcome == ParseOutcome.VALID_TRUE
        assert r.label == "TRUE"

    def test_no_normalized_to_false(self):
        r = parse_label("No")
        assert r.outcome == ParseOutcome.VALID_FALSE
        assert r.label == "FALSE"

    def test_yes_upper(self):
        r = parse_label("YES")
        assert r.outcome == ParseOutcome.VALID_TRUE

    def test_no_upper(self):
        r = parse_label("NO")
        assert r.outcome == ParseOutcome.VALID_FALSE

    # --- Variants with trailing punctuation ---
    def test_true_with_period(self):
        r = parse_label("True.")
        assert r.outcome == ParseOutcome.VALID_TRUE

    def test_false_with_newline(self):
        r = parse_label("FALSE\n")
        assert r.outcome == ParseOutcome.VALID_FALSE

    # --- "Answer: TRUE" style ---
    def test_answer_colon_true(self):
        r = parse_label("Answer: TRUE")
        assert r.outcome == ParseOutcome.VALID_TRUE

    def test_answer_is_false(self):
        r = parse_label("The answer is FALSE.")
        assert r.outcome == ParseOutcome.VALID_FALSE

    def test_final_answer_marker(self):
        r = parse_label("Let me think... FINAL_ANSWER: TRUE")
        assert r.outcome == ParseOutcome.VALID_TRUE

    def test_therefore_true(self):
        r = parse_label("Therefore: TRUE")
        assert r.outcome == ParseOutcome.VALID_TRUE

    def test_conclusion_false(self):
        r = parse_label("Conclusion: FALSE")
        assert r.outcome == ParseOutcome.VALID_FALSE

    # --- CoT with final answer on last line ---
    def test_cot_last_line_true(self):
        text = (
            "The Lorenz system exhibits sensitive dependence on initial conditions.\n"
            "Therefore the statement is TRUE."
        )
        r = parse_label(text)
        assert r.outcome == ParseOutcome.VALID_TRUE

    def test_cot_last_line_false(self):
        text = (
            "This system has a stable fixed point.\n"
            "FALSE"
        )
        r = parse_label(text)
        assert r.outcome == ParseOutcome.VALID_FALSE

    # --- INVALID cases ---
    def test_empty_string_is_invalid(self):
        r = parse_label("")
        assert r.outcome == ParseOutcome.INVALID
        assert r.label is None

    def test_none_is_invalid(self):
        r = parse_label(None)
        assert r.outcome == ParseOutcome.INVALID

    def test_ambiguous_it_depends(self):
        r = parse_label("It depends on the context.")
        assert r.outcome == ParseOutcome.INVALID

    def test_ambiguous_cannot_determine(self):
        r = parse_label("I cannot determine the answer from the given information.")
        assert r.outcome == ParseOutcome.INVALID

    def test_long_explanation_no_label(self):
        r = parse_label(
            "The system dynamics depend on multiple factors including the initial "
            "conditions, parameter values, and the specific trajectory. Without "
            "further specification it is not possible to give a definitive answer."
        )
        assert r.outcome == ParseOutcome.INVALID

    def test_random_text_is_invalid(self):
        r = parse_label("The quick brown fox jumps over the lazy dog")
        assert r.outcome == ParseOutcome.INVALID

    # --- Lenient mode ---
    def test_lenient_finds_embedded_true(self):
        text = "Based on analysis: Answer: TRUE and I am confident."
        r = parse_label(text, strict=False)
        assert r.outcome == ParseOutcome.VALID_TRUE

    # --- confidence and reason ---
    def test_valid_has_nonzero_confidence(self):
        r = parse_label("TRUE")
        assert r.confidence > 0

    def test_invalid_has_zero_confidence(self):
        r = parse_label("")
        assert r.confidence == 0.0

    def test_reason_is_string(self):
        r = parse_label("TRUE")
        assert isinstance(r.reason, str)


class TestOutcomeToLabel:
    def test_valid_true(self):
        assert outcome_to_label(ParseOutcome.VALID_TRUE) == "TRUE"

    def test_valid_false(self):
        assert outcome_to_label(ParseOutcome.VALID_FALSE) == "FALSE"

    def test_invalid_returns_none(self):
        assert outcome_to_label(ParseOutcome.INVALID) is None


class TestParseOutcomeEnum:
    def test_values(self):
        assert ParseOutcome.VALID_TRUE == "VALID_TRUE"
        assert ParseOutcome.VALID_FALSE == "VALID_FALSE"
        assert ParseOutcome.INVALID == "INVALID"
