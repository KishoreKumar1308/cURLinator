.PHONY: help setup install test lint format clean run

help: ## Show this help message
	@echo "cURLinator - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Run initial setup
	@bash scripts/setup.sh

install: ## Install dependencies
	uv sync --extra dev

test: ## Run tests with coverage
	@bash scripts/test.sh

lint: ## Run linting and type checking
	@bash scripts/lint.sh

format: ## Format code with Black
	uv run black src/ tests/

clean: ## Clean up generated files
	rm -rf .venv
	rm -rf htmlcov
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run: ## Run the API server (development)
	uv run uvicorn curlinator.api.main:app --reload

shell: ## Start IPython shell with project context
	uv run ipython

.DEFAULT_GOAL := help

