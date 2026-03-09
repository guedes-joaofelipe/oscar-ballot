"""Tests for judge prompt generation."""

from __future__ import annotations

from prompts.judge import get_judge_voter_prompt


def test_get_judge_voter_prompt_supports_repetitions() -> None:
    """Prompt should repeat sections when repetition arguments are provided."""
    categories = [{"id": "best_picture", "name": "Best Picture", "description": "Top film"}]
    votes = [{"category_id": "best_picture", "nominee_id": "nom-1", "explanation": "Strong momentum"}]

    prompt = get_judge_voter_prompt(
        categories=categories,
        votes=votes,
        system_prompt_repetitions=2,
        user_prompt_repetitions=2,
    )

    assert prompt["system_prompt"].count("You are a movie expert") == 2
    assert prompt["user_prompt"].count("Help me decide which is the best nominee") == 2
    assert "<CATEGORY_ID>best_picture</CATEGORY_ID>" in prompt["user_prompt"]
