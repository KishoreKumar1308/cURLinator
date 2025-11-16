#!/bin/bash

# Run tests with coverage

set -e

echo "🧪 Running tests..."

uv run pytest \
    --cov=curlinator \
    --cov-report=html \
    --cov-report=term \
    -v \
    "$@"

echo ""
echo "✅ Tests complete! Coverage report generated in htmlcov/"

