import csv

import yaml


def load_yaml(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def load_csv(path: str) -> list[dict]:
    """Load a CSV file and return rows as dictionaries.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    list[dict]
        Parsed CSV rows.
    """
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def save_csv(path: str, data: list[dict]) -> None:
    """Save a list of dictionaries to a CSV file.

    Parameters
    ----------
    path : str
        Output CSV path.
    data : list[dict]
        Rows to be saved.
    """
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
