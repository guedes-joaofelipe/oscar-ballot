from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orchestrators import evaluations
from utils import files
from utils.logger import get_logger

LOGGER = get_logger(__name__)
FILES_CONFIG_PATH = "configs/files.yaml"


def _resolve_winners_path(files_config: dict, run_example: bool) -> str:
    """Resolve winners source path based on execution mode. """
    if run_example:
        return files_config["WINNERS_EXAMPLE_DATA_PATH"]
    return files_config["WINNERS_DATA_PATH"]


def _resolve_predictions_file(predictions_pattern: str) -> str:
    """Resolve the latest predictions CSV file from a glob pattern."""
    predictions_files = glob.glob(predictions_pattern)
    if not predictions_files:
        raise FileNotFoundError(f"No predictions files found at {predictions_pattern}")
    return max(predictions_files, key=os.path.getmtime)


def _resolve_output_paths(files_config: dict, run_example: bool) -> tuple[str, str]:
    """Resolve evaluations and scores output paths."""
    if run_example:
        return files_config["EVALUATIONS_EXAMPLE_DATA_PATH"], files_config["SCORES_EXAMPLE_DATA_PATH"]
    return files_config["EVALUATIONS_DATA_PATH"], files_config["SCORES_DATA_PATH"]


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path when missing."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def main(run_example: bool = False):
    """
    Main function to run the evaluations.
    """
    files_config = files.load_yaml(FILES_CONFIG_PATH)
    predictions_pattern = files_config["PREDICTIONS_DATA_PATH"]
    predictions_file = _resolve_predictions_file(predictions_pattern)
    LOGGER.info(f"Loading predictions from {predictions_file}")
    predictions = files.load_csv(predictions_file)
    if not predictions:
        raise ValueError(f"No predictions rows found in {predictions_file}")

    winners_path = _resolve_winners_path(files_config, run_example)
    LOGGER.info(f"Loading winners from {winners_path}")
    winners = files.load_yaml(winners_path)
    if not winners:
        raise FileNotFoundError(f"No winners found at {winners_path}")

    LOGGER.info("Running evaluations")
    evaluations_results = evaluations.run(predictions=predictions, winners=winners)
    if not evaluations_results:
        raise ValueError("No evaluation rows generated.")

    scores_results = evaluations.calculate_scores(evaluations_results)
    if not scores_results:
        raise ValueError("No score rows generated.")

    evaluations_output_path, scores_output_path = _resolve_output_paths(files_config, run_example)
    _ensure_parent_dir(evaluations_output_path)
    _ensure_parent_dir(scores_output_path)

    LOGGER.info(f"Saving evaluations results to {evaluations_output_path}")
    files.save_csv(evaluations_output_path, evaluations_results)

    LOGGER.info(f"Saving scores results to {scores_output_path}")
    files.save_csv(scores_output_path, scores_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Oscar winner evaluations using configured predictions and winners."
    )
    parser.add_argument("--example", action="store_true", help="Run example evaluations")
    args = parser.parse_args()
    main(run_example=args.example)