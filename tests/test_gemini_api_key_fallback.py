"""Tests for Gemini API key environment fallback behavior."""

from __future__ import annotations

from chaosbench.eval.providers.gemini import GeminiProvider


def test_gemini_provider_uses_google_api_key_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    provider = GeminiProvider()
    assert provider._get_api_key() == "google-key"


def test_gemini_provider_prefers_gemini_api_key_when_both_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    provider = GeminiProvider()
    assert provider._get_api_key() == "gemini-key"
