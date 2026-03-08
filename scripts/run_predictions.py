"""Run Oscar winner predictions using configured LLM voters and judge."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orchestrators import predictions
from orchestrators.predictions import Predictions
from utils import files
from utils.logger import get_logger

LOGGER = get_logger(__name__)
FILES_CONFIG_PATH = "configs/files.yaml"


def validate_and_convert_predictions_results(
    predictions_results_list: list[dict],
) -> list[dict]:
    """Validate prediction rows and return normalized dictionaries."""
    predictions_results: list[dict] = []
    for prediction_result in predictions_results_list:
        parsed_prediction = Predictions(**prediction_result)
        predictions_results.append(parsed_prediction.model_dump())

    return predictions_results


def main(run_example: bool = False):
    """
    Main function to run the predictions.
    """
    files_config = files.load_yaml(FILES_CONFIG_PATH)
    LOGGER.info(f"Loading API keys from {files_config['API_KEYS_PATH']}")
    api_keys = files.load_yaml(files_config['API_KEYS_PATH'])

    LOGGER.info(f"Loading voters config from {files_config['VOTERS_CONFIG_PATH']}")
    voters_config = files.load_yaml(files_config['VOTERS_CONFIG_PATH'])

    LOGGER.info(f"Loading categories config from {files_config['CATEGORIES_CONFIG_PATH']}")
    categories_config = files.load_yaml(files_config['CATEGORIES_CONFIG_PATH'])

    LOGGER.info(f"Loading judge config from {files_config['JUDGE_CONFIG_PATH']}")
    judge_config = files.load_yaml(files_config['JUDGE_CONFIG_PATH'])

    LOGGER.info(f"Running predictions")
    predictions_results_list = predictions.run(
        voters_config=voters_config,
        categories_config=categories_config,
        judge_config=judge_config,
        api_keys=api_keys,
    )

    LOGGER.info(f"Validating and converting predictions results")
    predictions_results = validate_and_convert_predictions_results(predictions_results_list)

    os.makedirs(files_config['VOTERS_DATA_PATH'], exist_ok=True)
    file_id = "example" if run_example else datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    output_file_path = os.path.join(files_config['VOTERS_DATA_PATH'], f"votes-{file_id}.csv")
    LOGGER.info(f"Saving predictions results to {output_file_path}")
    if not predictions_results:
        raise ValueError("No predictions generated to persist.")

    files.save_csv(output_file_path, predictions_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Oscar winner predictions using configured LLM voters and judge.")
    parser.add_argument("--example", action="store_true", help="Run example predictions")
    args = parser.parse_args()
    main(run_example=args.example)
