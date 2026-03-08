"""Utilities to call vLLM chat-completion endpoints and parse JSON responses."""

from __future__ import annotations

import json
from typing import Any

import requests

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _extract_json_payload(raw_content: str) -> Any:
    """Extract JSON payload from plain text or markdown fenced blocks.

    Parameters
    ----------
    raw_content : str
        Raw model response content.

    Returns
    -------
    Any
        Parsed JSON payload.
    """
    content = raw_content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        for part in parts:
            stripped_part = part.strip()
            if not stripped_part:
                continue
            if stripped_part.startswith("json"):
                stripped_part = stripped_part[4:].strip()
            try:
                return json.loads(stripped_part)
            except json.JSONDecodeError:
                continue
    return json.loads(content)


def _build_chat_completions_url(base_url: str) -> str:
    """Build a vLLM chat completions URL from base URL.

    Parameters
    ----------
    base_url : str
        Base API URL configured for the model provider.

    Returns
    -------
    str
        Chat completions endpoint URL.
    """
    normalized_url = base_url.rstrip("/")
    if normalized_url.endswith("/chat/completions"):
        return normalized_url
    return f"{normalized_url}/chat/completions"


def call_model_json(
    model_config: dict,
    api_keys: dict,
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = 2,
) -> Any:
    """Call a configured vLLM-compatible model and parse JSON output.

    Parameters
    ----------
    model_config : dict
        Voter or judge model configuration.
    api_keys : dict
        API keys and endpoints map.
    system_prompt : str
        System instruction for the model.
    user_prompt : str
        User payload for the model.
    max_attempts : int, default=2
        Maximum parsing attempts before raising an error.

    Returns
    -------
    Any
        Parsed JSON response.
    """
    api_key_id = model_config["api_key_id"]
    api_key_config = api_keys[api_key_id]
    chat_completions_url = _build_chat_completions_url(api_key_config["API_ENDPOINT"])
    headers = {
        "Authorization": f"Bearer {api_key_config['API_KEY']}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        payload = {
            "model": model_config["model"],
            "temperature": model_config.get("temperature", 0.7),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            chat_completions_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        response_json = response.json()
        try:
            content = response_json["choices"][0]["message"]["content"] or ""
            parsed_content = _extract_json_payload(content)
            return parsed_content
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            last_error = error
            LOGGER.warning(
                "Failed to parse model response as JSON for model '%s' (attempt %s/%s).",
                model_config["model"],
                attempt + 1,
                max_attempts,
            )

    raise ValueError("Model response was not valid JSON payload.") from last_error
