from concurrent.futures import ThreadPoolExecutor
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


def _run_voter_model(
    voter_id: str,
    voter_model_config: dict,
    categories: list[dict],
    api_keys: dict,
    category_nominee_map: dict[str, set[str]],
    timestamp: datetime,
    imdb_metadata: dict = None
) -> tuple[list[dict], list[dict]]:
    """Run one voter model and build expert votes and predictions rows."""
    LOGGER.info(f"Running voter model '{voter_id}'.")
    voter_prompt = get_voter_prompt(
        categories=categories,
        imdb_metadata=imdb_metadata,
        system_prompt_repetitions=voter_model_config.get("system_prompt_repetitions", 1),
        user_prompt_repetitions=voter_model_config.get("user_prompt_repetitions", 1),
    )
    voter_payload = call_model_json(
        model_config=voter_model_config,
        api_keys=api_keys,
        system_prompt=voter_prompt["system_prompt"],
        user_prompt=voter_prompt["user_prompt"],
        timeout=voter_model_config.get("timeout", 120),
    )
    expert_votes: list[dict] = []
    predictions: list[dict] = []
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
                "is_judge": False,
            }
        )
    return expert_votes, predictions


def _run_judge_model(
    judge_id: str,
    judge_model_config: dict,
    categories: list[dict],
    expert_votes: list[dict],
    api_keys: dict,
    category_nominee_map: dict[str, set[str]],
    timestamp: datetime,
) -> list[dict]:
    """Run one judge model and build predictions rows."""
    LOGGER.info(f"Running judge model '{judge_id}'.")
    judge_prompt = get_judge_voter_prompt(
        categories=categories,
        votes=expert_votes,
        system_prompt_repetitions=judge_model_config.get("system_prompt_repetitions", 1),
        user_prompt_repetitions=judge_model_config.get("user_prompt_repetitions", 1),
    )
    judge_payload = call_model_json(
        model_config=judge_model_config,
        api_keys=api_keys,
        system_prompt=judge_prompt["system_prompt"],
        user_prompt=judge_prompt["user_prompt"],
    )
    predictions: list[dict] = []
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
                "is_judge": True,
            }
        )
    return predictions


def run(
    voters_config: dict,
    categories_config: dict,
    judge_config: dict,
    api_keys: dict,
    max_judge_workers: int = 1,
    max_voter_workers: int = 1,
    imdb_metadata: dict = None,
) -> list[dict]:
    """ Run the predictions using the voters and categories config.

    Args:
        voters_config: The voters config.
        categories_config: The categories config.
        judge_config: The judges config.
        api_keys: The keys for the LLMs.
        max_judge_workers: The maximum number of judge workers.
        max_voter_workers: The maximum number of voter workers.
        imdb_metadata: The IMDb metadata (optional).

    Returns:
        The predictions.
    """
    categories = _normalize_categories(categories_config)
    category_nominee_map = _build_category_nominee_map(categories)
    timestamp = datetime.now(TIMEZONE)

    expert_votes: list[dict] = []
    predictions: list[dict] = []
    voter_items = list(voters_config.items())
    max_voter_workers = min(len(voter_items), max_voter_workers)
    with ThreadPoolExecutor(max_workers=max_voter_workers) as executor:
        futures = [
            executor.submit(
                _run_voter_model,
                voter_id=voter_id,
                imdb_metadata=imdb_metadata,
                voter_model_config=voter_model_config,
                categories=categories,
                api_keys=api_keys,
                category_nominee_map=category_nominee_map,
                timestamp=timestamp,
            )
            for voter_id, voter_model_config in voter_items
        ]
        for future in futures:
            voter_expert_votes, voter_predictions = future.result()
            expert_votes.extend(voter_expert_votes)
            predictions.extend(voter_predictions)

    if not expert_votes:
        raise ValueError("No expert votes were generated by voters.")

    if not judge_config:
        raise ValueError("No judges were configured.")

    judge_items = list(judge_config.items())
    max_judge_workers = min(len(judge_items), max_judge_workers)
    with ThreadPoolExecutor(max_workers=max_judge_workers) as executor:
        futures = [
            executor.submit(
                _run_judge_model,
                judge_id=judge_id,
                judge_model_config=judge_model_config,
                categories=categories,
                expert_votes=expert_votes,
                api_keys=api_keys,
                category_nominee_map=category_nominee_map,
                timestamp=timestamp,
            )
            for judge_id, judge_model_config in judge_items
        ]
        for future in futures:
            predictions.extend(future.result())
    return predictions
