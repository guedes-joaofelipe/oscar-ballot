import json
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from prompts.judge import JudgeResponse, get_judge_voter_prompt
from prompts.voters import VoterResponse, get_voter_prompt
from utils.llm import call_model_json
from utils.logger import get_logger

LOGGER = get_logger(__name__)
TIMEZONE = ZoneInfo("America/Sao_Paulo")


class Predictions(BaseModel):
    """Prediction row model."""

    timestamp: datetime = Field(description="The timestamp of the prediction")
    category_id: str = Field(description="The oscar award category id")
    voter_id: str = Field(description="The voter id")
    predicted_winner_id: str = Field(description="The predicted winner id")
    explanation: str = Field(description="The explanation for the prediction")
    is_judge: bool = Field(description="Whether the prediction is from a judge")


class PredictionsResults(BaseModel):
    """Collection of prediction rows."""

    predictions: list[Predictions]


def _normalize_categories(categories_config: dict) -> list[dict]:
    """Normalize categories config into a list with explicit ids"""
    categories: list[dict] = [
        {
            "id": category_id,
            "name": category_data.get("name", ""),
            "description": category_data.get("description", ""),
            "nominees": category_data.get("nominees", []),
        }
        for category_id, category_data in categories_config.items()
    ]
    return categories


def _extract_vote_list(payload: object) -> list[dict]:
    """Extract a list of votes from model payload."""
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "votes" in payload and isinstance(payload["votes"], list):
            return payload["votes"]
        if "predictions" in payload and isinstance(payload["predictions"], list):
            return payload["predictions"]
        if {"category_id", "nominee_id"}.issubset(payload.keys()):
            return [payload]

    raise ValueError("Model payload must include a list of votes.")


def _build_category_nominee_map(categories: list[dict]) -> dict[str, set[str]]:
    """Build a category to nominee id map."""
    category_nominee_map: dict[str, set[str]] = {
        category["id"]: {nominee["id"] for nominee in category.get("nominees", []) if "id" in nominee} for category in categories
    }
    return category_nominee_map


def _validate_vote_identifiers(
    category_id: str,
    nominee_id: str,
    category_nominee_map: dict[str, set[str]],
) -> None:
    """Validate category and nominee identifiers.

    Parameters
    ----------
    category_id : str
        Category identifier.
    nominee_id : str
        Nominee identifier.
    category_nominee_map : dict[str, set[str]]
        Allowed nominee ids by category id.
    """
    if category_id not in category_nominee_map:
        raise ValueError(f"Unknown category_id: {category_id}")
    if nominee_id not in category_nominee_map[category_id]:
        raise ValueError(f"Unknown nominee_id '{nominee_id}' for category_id '{category_id}'.")


def run(
    voters_config: dict,
    categories_config: dict,
    judge_config: dict,
    api_keys: dict,
) -> list[dict]:
    """
    Run the predictions using the voters and categories config.

    The predictions are run by the following steps:
    1. For each voter config:
        1.1. Get the voter prompt
        1.2. Get the voter response using the LLM API call
        1.3. Validate the voter response
    2. For each judge:
        2.1. Get the judge prompt
        2.2. Get the judge response using the LLM API call
        2.3. Validate the judge response
    3. Return the predictions

    Args:
        voters_config: The voters config.
        categories_config: The categories config.
        judge_config: The judges config.
        api_keys: The keys for the LLMs.

    Returns:
        The predictions.
    """
    categories = _normalize_categories(categories_config)
    category_nominee_map = _build_category_nominee_map(categories)
    timestamp = datetime.now(TIMEZONE)

    expert_votes: list[dict] = []
    predictions: list[dict] = []
    for voter_id, voter_model_config in voters_config.items():
        LOGGER.info(f"Running voter model '{voter_id}'.")
        voter_prompt = get_voter_prompt(categories)
        voter_payload = call_model_json(
            model_config=voter_model_config,
            api_keys=api_keys,
            system_prompt=voter_prompt["system_prompt"],
            user_prompt=voter_prompt["user_prompt"],
        )
        for vote in _extract_vote_list(voter_payload):
            parsed_vote = VoterResponse(**vote)
            _validate_vote_identifiers(
                category_id=parsed_vote.category_id,
                nominee_id=parsed_vote.nominee_id,
                category_nominee_map=category_nominee_map,
            )
            expert_votes.append(
                {
                    "voter_id": voter_id,
                    "category_id": parsed_vote.category_id,
                    "nominee_id": parsed_vote.nominee_id,
                    "explanation": parsed_vote.explanation,
                }
            )
            predictions.append(
                {
                    "timestamp": timestamp,
                    "category_id": parsed_vote.category_id,
                    "voter_id": voter_id,
                    "predicted_winner_id": parsed_vote.nominee_id,
                    "explanation": parsed_vote.explanation,
                    "is_judge": False
                }
            )

    if not expert_votes:
        raise ValueError("No expert votes were generated by voters.")

    if not judge_config:
        raise ValueError("No judges were configured.")

    judge_prompt = get_judge_voter_prompt(categories=categories, votes=expert_votes)
    for judge_id, judge_model_config in judge_config.items():
        LOGGER.info(f"Running judge model '{judge_id}'.")
        judge_payload = call_model_json(
            model_config=judge_model_config,
            api_keys=api_keys,
            system_prompt=judge_prompt["system_prompt"],
            user_prompt=judge_prompt["user_prompt"],
        )

        for vote in _extract_vote_list(judge_payload):
            parsed_vote = JudgeResponse(**vote)
            _validate_vote_identifiers(
                category_id=parsed_vote.category_id,
                nominee_id=parsed_vote.nominee_id,
                category_nominee_map=category_nominee_map,
            )
            predictions.append(
                {
                    "timestamp": timestamp,
                    "category_id": parsed_vote.category_id,
                    "voter_id": judge_id,
                    "predicted_winner_id": parsed_vote.nominee_id,
                    "explanation": parsed_vote.decision,
                    "is_judge": True
                }
            )
    return predictions
