"""Tests for duplicate detection and grouping logic."""

import pytest
from chaosbench.data.grouping import (
    compute_group_id,
    is_accidental_duplicate,
    _extract_predicate,
    _normalize_text,
)
from chaosbench.quality.gates import _text_hash


def test_normalize_text_deterministic():
    """Text normalization should be deterministic."""
    text1 = "Is Lorenz-63 chaotic?"
    text2 = "Is Lorenz-63 chaotic?"
    assert _normalize_text(text1) == _normalize_text(text2)

    # Punctuation is stripped, whitespace collapsed
    text3 = "  Is   Lorenz63   chaotic?  "
    text4 = "is lorenz63 chaotic"
    assert _normalize_text(text3) == _normalize_text(text4)


def test_text_hash_deterministic():
    """Text hashing should be deterministic."""
    text1 = "Is Lorenz-63 chaotic?"
    text2 = "Is Lorenz-63 chaotic?"
    assert _text_hash(text1) == _text_hash(text2)

    text3 = "Is Lorenz-63 chaotic?"
    text4 = "Is Lorenz-84 chaotic?"
    assert _text_hash(text3) != _text_hash(text4)


def test_extract_predicate():
    """Predicate extraction from question text."""
    q1 = "Is Lorenz-63 chaotic?"
    assert _extract_predicate(q1) == "chaotic"

    q2 = "Is Double pendulum deterministic?"
    assert _extract_predicate(q2) == "deterministic"

    q3 = "Is Van der Pol oscillator poslyap?"
    assert _extract_predicate(q3) == "poslyap"

    q4 = "Is Unknown System xyzabc?"
    assert _extract_predicate(q4) == "unknown"


def test_compute_group_id_perturbation():
    """Perturbation items should get stable group_id."""
    item1 = {
        "id": "perturb_paraphrase_0063",
        "question": "Is Lorenz-63 chaotic?",
        "type": "perturbation",
        "system_id": "lorenz63",
        "ground_truth": "TRUE",
    }

    item2 = {
        "id": "perturb_distractor_0063",
        "question": "The system has a strange attractor. Is Lorenz-63 chaotic?",
        "type": "perturbation",
        "system_id": "lorenz63",
        "ground_truth": "TRUE",
    }

    # Same base_id (0063) should produce same group_id
    gid1 = compute_group_id(item1)
    gid2 = compute_group_id(item2)
    assert gid1 is not None
    assert gid1 == gid2


def test_compute_group_id_atomic():
    """Atomic items should not get group_id (singletons)."""
    item = {
        "id": "atomic_q0001",
        "question": "Is Lorenz-63 chaotic?",
        "type": "atomic",
        "system_id": "lorenz63",
        "ground_truth": "TRUE",
    }

    assert compute_group_id(item) is None


def test_is_accidental_duplicate():
    """Detect accidental duplicates (same question + system + answer)."""
    item1 = {
        "id": "perturb_entity_swap_1817",
        "question": "Is Ornstein–Uhlenbeck process deterministic?",
        "type": "perturbation",
        "system_id": "stochastic_ou",
        "ground_truth": "FALSE",
    }

    item2 = {
        "id": "perturb_entity_swap_1887",
        "question": "Is Ornstein–Uhlenbeck process deterministic?",
        "type": "perturbation",
        "system_id": "stochastic_ou",
        "ground_truth": "FALSE",
    }

    item3 = {
        "id": "perturb_paraphrase_0999",
        "question": "Is Ornstein–Uhlenbeck process deterministic?",
        "type": "perturbation",
        "system_id": "stochastic_ou",
        "ground_truth": "TRUE",  # Different ground truth
    }

    # Same question + system + ground_truth → accidental duplicate
    assert is_accidental_duplicate(item1, item2) is True

    # Different ground_truth → not duplicate
    assert is_accidental_duplicate(item1, item3) is False

    # Same ID → not duplicate
    assert is_accidental_duplicate(item1, item1) is False


def test_group_id_determinism():
    """Group IDs should be stable across calls."""
    item = {
        "id": "perturb_paraphrase_0123",
        "question": "Is Van der Pol oscillator periodic?",
        "type": "perturbation",
        "system_id": "vdp",
        "ground_truth": "FALSE",
    }

    gid1 = compute_group_id(item)
    gid2 = compute_group_id(item)
    assert gid1 == gid2
