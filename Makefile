setup:
	# Install dependencies using uv
	uv sync

lint:
	# Lint the code using ruff
	ruff check src

format:
	# Format the code using ruff
	ruff format src

test:
	# Test the code using pytest
	if command -v uv >/dev/null 2>&1; then uv run python -m pytest tests; else python3 -m pytest tests; fi

coverage:
	# Generate coverage report using pytest
	if command -v uv >/dev/null 2>&1; then uv run python -m pytest --cov=src tests; else python3 -m pytest --cov=src tests; fi

run-predictions:
	python3 scripts/run_predictions.py

run-evaluations:
	python3 scripts/run_evaluations.py

clean:
	# Clean the code using ruff
	ruff clean src