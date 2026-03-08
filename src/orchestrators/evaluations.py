from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _parse_bool(value: object) -> bool:
    """Convert string-like booleans to bool.

    Parameters
    ----------
    value : object
        Raw value from CSV input.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def run(predictions: list[dict], winners: dict[str, str]) -> list[dict]:
    """
    Run the evaluations using the predictions and winners.

    The evaluations are run by the following steps:
    1. For each prediction:
        1.1. Compare the prediction with the winner
        1.2. Calculate the score
    2. Return the evaluations

    Args:
        predictions: The predictions.
        winners: The winners.

    Returns:
        The evaluations.
    """
    evaluation_rows: list[dict] = []
    for prediction in predictions:
        category_id = prediction["category_id"]
        winner_nominee_id = winners.get(category_id)
        if winner_nominee_id is None:
            LOGGER.warning("Skipping unknown category in winners: %s", category_id)
            continue

        predicted_winner_id = prediction["predicted_winner_id"]
        is_correct = predicted_winner_id == winner_nominee_id

        evaluation_rows.append(
            {
                "timestamp": prediction["timestamp"],
                "category_id": category_id,
                "voter_id": prediction["voter_id"],
                "predicted_winner_id": predicted_winner_id,
                "winner_nominee_id": winner_nominee_id,
                "is_judge": _parse_bool(prediction.get("is_judge", False)),
                "is_correct": is_correct,
            }
        )
    return evaluation_rows


def calculate_scores(evaluations: list[dict]) -> list[dict]:
    """Aggregate per-voter accuracy scores from evaluation rows.

    Parameters
    ----------
    evaluations : list[dict]
        Evaluation rows from `run`.

    Returns
    -------
    list[dict]
        Score rows by voter.
    """
    by_voter: dict[str, dict] = {}
    for row in evaluations:
        voter_id = row["voter_id"]
        if voter_id not in by_voter:
            by_voter[voter_id] = {
                "voter_id": voter_id,
                "is_judge": _parse_bool(row.get("is_judge", False)),
                "total_predictions": 0,
                "correct_predictions": 0,
            }
        by_voter[voter_id]["total_predictions"] += 1
        if _parse_bool(row["is_correct"]):
            by_voter[voter_id]["correct_predictions"] += 1

    score_rows: list[dict] = []
    for score in by_voter.values():
        total_predictions = score["total_predictions"]
        correct_predictions = score["correct_predictions"]
        accuracy = correct_predictions / total_predictions if total_predictions else 0.0
        score_rows.append(
            {
                **score,
                "accuracy": round(accuracy, 4),
            }
        )

    score_rows.sort(key=lambda row: (row["accuracy"], row["correct_predictions"]), reverse=True)
    return score_rows
