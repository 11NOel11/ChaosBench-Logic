"""Tests for provider-threshold routing in M5 runner."""

from __future__ import annotations

import json

from scripts.run_m5_instance_guardrail import (
    load_provider_reference_dists,
    load_provider_thresholds,
    parse_provider_step_multipliers,
    resolve_provider_reference_path,
    resolve_provider_thresholds_path,
    resolve_threshold,
)


def test_resolve_threshold_prefers_exact_then_prefix_then_default() -> None:
    mapping = {
        "openai": 0.12,
        "openai/gpt-4o": 0.2,
        "deepseek": 0.0,
    }
    assert resolve_threshold("openai/gpt-4o", 0.05, mapping) == 0.2
    assert resolve_threshold("openai/gpt-5.2", 0.05, mapping) == 0.12
    assert resolve_threshold("anthropic/claude-sonnet-4-6", 0.05, mapping) == 0.05


def test_load_provider_thresholds_normalizes_lowercase(tmp_path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "OpenAI": 0.11,
                "DEEPSEEK/DeepSeek-Chat": 0.0,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_provider_thresholds(path)
    assert loaded["openai"] == 0.11
    assert loaded["deepseek/deepseek-chat"] == 0.0


def test_load_provider_thresholds_accepts_prior_payload(tmp_path) -> None:
    path = tmp_path / "threshold_priors.json"
    path.write_text(
        json.dumps(
            {
                "openai": {"threshold": 0.13, "threshold_std": 0.02},
                "gemini": {"threshold": 0.54},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_provider_thresholds(path)
    assert loaded["openai"] == 0.13
    assert loaded["gemini"] == 0.54


def test_resolve_provider_thresholds_path_prefers_explicit_then_default() -> None:
    explicit_path, explicit_source = resolve_provider_thresholds_path(
        provider_thresholds_json="bar.json",
        auto_provider_thresholds_json=False,
    )
    assert explicit_path == "bar.json"
    assert explicit_source == "explicit"

    none_path, none_source = resolve_provider_thresholds_path(
        provider_thresholds_json=None,
        auto_provider_thresholds_json=False,
    )
    assert none_path is None
    assert none_source == "none"

    default_path, default_source = resolve_provider_thresholds_path(
        provider_thresholds_json=None,
        auto_provider_thresholds_json=True,
    )
    assert default_path is not None
    assert default_path.endswith("m5_provider_thresholds_crossfit_v1.json")
    assert default_source == "auto_default_for_transfer"


def test_load_provider_reference_dists_normalizes_keys(tmp_path) -> None:
    path = tmp_path / "provider_refs.json"
    path.write_text(
        json.dumps(
            {
                "OpenAI/GPT-4o": {"multi_hop|m1|s2": 2.0, "x": 0.0},
                "": {"unused": 1.0},
                "deepseek": "invalid",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_provider_reference_dists(path)
    assert "openai/gpt-4o" in loaded
    assert loaded["openai/gpt-4o"]["multi_hop|m1|s2"] == 2.0
    assert "deepseek" not in loaded


def test_resolve_provider_reference_path_prefers_explicit_then_sibling(
    tmp_path,
) -> None:
    explicit_path, explicit_source = resolve_provider_reference_path(
        provider_reference_json="refs.json",
        auto_provider_reference_json=False,
        provider_thresholds_path=None,
    )
    assert explicit_path == "refs.json"
    assert explicit_source == "explicit"

    threshold_path = tmp_path / "provider_thresholds_crossfit_v1.json"
    threshold_path.write_text("{}", encoding="utf-8")
    sibling = tmp_path / "provider_reference_dists_crossfit_v1.json"
    sibling.write_text("{}", encoding="utf-8")

    sibling_path, sibling_source = resolve_provider_reference_path(
        provider_reference_json=None,
        auto_provider_reference_json=True,
        provider_thresholds_path=str(threshold_path),
    )
    assert sibling_path is not None
    assert sibling_path.endswith("provider_reference_dists_crossfit_v1.json")
    assert sibling_source == "auto_sibling_of_threshold_map"


def test_parse_provider_step_multipliers_accepts_prefix_entries() -> None:
    parsed = parse_provider_step_multipliers("openai=0.25, openrouter/meta-llama=0.5")
    assert parsed["openai"] == 0.25
    assert parsed["openrouter/meta-llama"] == 0.5


def test_parse_provider_step_multipliers_rejects_negative() -> None:
    try:
        parse_provider_step_multipliers("openai=-0.1")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative multiplier")
