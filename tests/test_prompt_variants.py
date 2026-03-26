"""Tests for prompt variant selection and hashing."""

from __future__ import annotations

from chaosbench.eval import prompts


def test_default_prompt_variant_is_v1(monkeypatch):
    monkeypatch.delenv("CHAOSBENCH_PROMPT_VARIANT", raising=False)
    assert prompts.get_prompt_version() == "v1"
    assert prompts.get_prompt_hash() == "5881f664c444e3d3"


def test_prompt_variant_override_changes_version_and_hash(monkeypatch):
    monkeypatch.setenv("CHAOSBENCH_PROMPT_VARIANT", "v1_compact")
    compact_version = prompts.get_prompt_version()
    compact_hash = prompts.get_prompt_hash()

    monkeypatch.setenv("CHAOSBENCH_PROMPT_VARIANT", "v1_logiccheck")
    logic_version = prompts.get_prompt_version()
    logic_hash = prompts.get_prompt_hash()

    assert compact_version == "v1_compact"
    assert logic_version == "v1_logiccheck"
    assert compact_hash != logic_hash
    assert compact_hash != "5881f664c444e3d3"
    assert logic_hash != "5881f664c444e3d3"


def test_unknown_prompt_variant_falls_back_to_v1(monkeypatch):
    monkeypatch.setenv("CHAOSBENCH_PROMPT_VARIANT", "unknown_variant")
    assert prompts.get_prompt_version() == "v1"
    assert prompts.get_prompt_hash() == "5881f664c444e3d3"
