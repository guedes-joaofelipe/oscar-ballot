"""Tests for LLM utility helpers."""

from __future__ import annotations

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
