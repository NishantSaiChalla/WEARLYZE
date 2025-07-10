.PHONY: help install install-dev test lint format clean run-api train

help:
	@echo "Available commands:"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting"
	@echo "  make format       Format code"
	@echo "  make clean        Clean up generated files"
	@echo "  make run-api      Run the API server"
	@echo "  make train        Train the model"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=fashion_detection --cov-report=html --cov-report=term

lint:
	flake8 fashion_detection/ tests/
	mypy fashion_detection/ tests/
	black --check fashion_detection/ tests/
	isort --check-only fashion_detection/ tests/

format:
	black fashion_detection/ tests/
	isort fashion_detection/ tests/

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

run-api:
	uvicorn inference.api:app --reload --host 0.0.0.0 --port 8000

train:
	python -m training.train --config config/default_config.yaml

docker-build:
	docker build -t fashion-detection:latest .

docker-run:
	docker run -p 8000:8000 --gpus all fashion-detection:latest

setup-hooks:
	pre-commit install
	pre-commit run --all-files