"""Tests for predictions orchestrator."""

from __future__ import annotations

from orchestrators import predictions


def test_extract_vote_list_accepts_list() -> None:
    """List payload should pass through unchanged."""
    payload = [{"category_id": "best_picture", "nominee_id": "nom-1"}]
    assert predictions._extract_vote_list(payload) == payload


def test_extract_vote_list_accepts_votes_key() -> None:
    """Dictionary payload should support votes key."""
    payload = {"votes": [{"category_id": "best_picture", "nominee_id": "nom-1"}]}
    assert predictions._extract_vote_list(payload) == payload["votes"]


def test_run_builds_predictions_with_mocked_llm(monkeypatch) -> None:
    """Run should combine voter votes and decisions from multiple judges."""

    voters_payload = {
        "votes": [
            {
                "category_id": "best_picture",
                "nominee_id": "nom-1",
                "explanation": "Strong directing and performances.",
            }
        ]
    }
    judge_payload = {
        "votes": [
            {
                "category_id": "best_picture",
                "nominee_id": "nom-1",
                "decision": "Consensus pick from experts.",
            }
        ]
    }

    state = {"calls": 0}

    def fake_call_model_json(
        model_config: dict,
        api_keys: dict,
        system_prompt: str,
        user_prompt: str,
        max_attempts: int = 2,
    ) -> dict:
        del model_config, api_keys, system_prompt, user_prompt, max_attempts
        state["calls"] += 1
        if state["calls"] == 1:
            return voters_payload
        return judge_payload

    monkeypatch.setattr(predictions, "call_model_json", fake_call_model_json)

    voters_config = {
        "voter_a": {
            "api_key_id": "dummy",
            "model": "dummy-model",
            "temperature": 0.0,
        }
    }
    judge_config = {
        "judge_a": {
            "api_key_id": "dummy",
            "model": "dummy-model",
            "temperature": 0.0,
        },
        "judge_b": {
            "api_key_id": "dummy",
            "model": "dummy-model",
            "temperature": 0.0,
            "system_prompt_repetitions": 2,
        },
    }
    categories_config = {
        "best_picture": {
            "name": "Best Picture",
            "description": "",
            "nominees": [{"id": "nom-1", "name": "Movie 1"}],
        }
    }
    api_keys = {"dummy": {"API_ENDPOINT": "http://example.com", "API_KEY": "redacted"}}

    results = predictions.run(
        voters_config=voters_config,
        categories_config=categories_config,
        judge_config=judge_config,
        api_keys=api_keys,
    )

    assert len(results) == 3
    assert results[0]["category_id"] == "best_picture"
    assert results[0]["predicted_winner_id"] == "nom-1"
    assert results[0]["voter_id"] == "voter_a"
    assert results[0]["is_judge"] is False
    assert results[1]["category_id"] == "best_picture"
    assert results[1]["predicted_winner_id"] == "nom-1"
    assert results[1]["voter_id"] == "judge_a"
    assert results[1]["is_judge"] is True
    assert results[2]["category_id"] == "best_picture"
    assert results[2]["predicted_winner_id"] == "nom-1"
    assert results[2]["voter_id"] == "judge_b"
    assert results[2]["is_judge"] is True
