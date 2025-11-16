#!/bin/bash

# Run linting and formatting checks

set -e

echo "🔍 Running code quality checks..."

echo ""
echo "📝 Formatting with Black..."
uv run black src/ tests/

echo ""
echo "🔎 Linting with Ruff..."
uv run ruff check src/ tests/ --fix

echo ""
echo "🔬 Type checking with MyPy..."
uv run mypy src/

echo ""
echo "✅ All checks passed!"

