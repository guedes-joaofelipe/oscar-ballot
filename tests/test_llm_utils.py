"""Tests for LLM utility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils import llm
from utils.llm import _extract_json_payload


def test_extract_json_payload_from_plain_json() -> None:
    """Plain JSON should parse successfully."""
    payload = _extract_json_payload('{"votes":[{"category_id":"c1","nominee_id":"n1"}]}')
    assert payload["votes"][0]["category_id"] == "c1"


def test_extract_json_payload_from_markdown_block() -> None:
    """JSON fenced in markdown should parse successfully."""
    raw_content = """```json
{"votes":[{"category_id":"c1","nominee_id":"n1"}]}
```"""
    payload = _extract_json_payload(raw_content)
    assert payload["votes"][0]["nominee_id"] == "n1"


def test_call_model_json_returns_parsed_payload(monkeypatch) -> None:
    """LLM call should return parsed JSON content from first attempt."""
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"winner":"nom-1"}'))]
    )

    class FakeClient:
        def __init__(self, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: fake_response))

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    result = llm.call_model_json(
        model_config={"api_key_id": "dummy", "model": "gpt-x"},
        api_keys={"dummy": {"API_KEY": "secret", "API_ENDPOINT": "https://example.com/"}},
        system_prompt="system",
        user_prompt="user",
    )

    assert result == {"winner": "nom-1"}


def test_call_model_json_raises_after_max_attempts(monkeypatch) -> None:
    """LLM call should raise ValueError after repeated invalid payloads."""
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not valid json"))]
    )

    class FakeClient:
        def __init__(self, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: fake_response))

    monkeypatch.setattr(llm, "OpenAI", FakeClient)

    with pytest.raises(ValueError, match="Model response was not valid JSON payload."):
        llm.call_model_json(
            model_config={"api_key_id": "dummy", "model": "gpt-x"},
            api_keys={"dummy": {"API_KEY": "secret", "API_ENDPOINT": "https://example.com/"}},
            system_prompt="system",
            user_prompt="user",
            max_attempts=2,
        )
