"""Tests for file utility helpers."""

from __future__ import annotations

from pathlib import Path

from utils import files


def test_save_and_load_csv_round_trip(tmp_path: Path) -> None:
    """CSV rows should persist and load with string values."""
    csv_path = tmp_path / "sample.csv"
    rows = [{"name": "Alice", "score": 10}, {"name": "Bob", "score": 7}]

    files.save_csv(str(csv_path), rows)
    loaded_rows = files.load_csv(str(csv_path))

    assert loaded_rows == [{"name": "Alice", "score": "10"}, {"name": "Bob", "score": "7"}]


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    """YAML mapping should be loaded as a Python dictionary."""
    yaml_path = tmp_path / "sample.yaml"
    yaml_path.write_text("winner: nom-1\ncategory: best_picture\n", encoding="utf-8")

    loaded_yaml = files.load_yaml(str(yaml_path))

    assert loaded_yaml == {"winner": "nom-1", "category": "best_picture"}
