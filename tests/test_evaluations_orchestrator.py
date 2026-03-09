"""Tests for evaluations orchestrator."""

from __future__ import annotations

from orchestrators import evaluations


def test_run_generates_evaluation_rows_and_skips_unknown_category() -> None:
    """Run should compare winners and skip categories not present in winners map."""
    predictions = [
        {
            "timestamp": "2026-03-08T12:00:00-03:00",
            "category_id": "best_picture",
            "voter_id": "voter_a",
            "predicted_winner_id": "nom-1",
            "is_judge": False,
        },
        {
            "timestamp": "2026-03-08T12:00:00-03:00",
            "category_id": "missing_category",
            "voter_id": "voter_a",
            "predicted_winner_id": "nom-1",
            "is_judge": False,
        },
    ]
    winners = {"best_picture": "nom-1"}

    results = evaluations.run(predictions=predictions, winners=winners)

    assert len(results) == 1
    assert results[0]["category_id"] == "best_picture"
    assert results[0]["is_correct"] is True


def test_calculate_scores_includes_accuracy_precision_and_recall() -> None:
    """Score rows should include accuracy, precision, and recall values."""
    evaluations_rows = [
        {"voter_id": "voter_a", "is_judge": False, "is_correct": True},
        {"voter_id": "voter_a", "is_judge": False, "is_correct": False},
        {"voter_id": "judge_a", "is_judge": True, "is_correct": True},
    ]

    scores = evaluations.calculate_scores(evaluations_rows)

    assert len(scores) == 2
    voter_a = next(score for score in scores if score["voter_id"] == "voter_a")
    assert voter_a["total_predictions"] == 2
    assert voter_a["correct_predictions"] == 1
    assert voter_a["accuracy"] == 0.5
    assert voter_a["precision"] == 0.5
    assert voter_a["recall"] == 0.5
