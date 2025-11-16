#!/bin/bash

# cURLinator Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up cURLinator development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ Found uv $(uv --version)"

# Sync dependencies
echo "📦 Installing dependencies..."
uv sync --extra dev

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
source .venv/bin/activate
playwright install chromium

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys"
fi

echo ""
echo "✨ Setup complete! Next steps:"
echo ""
echo "1. Edit .env file with your API keys:"
echo "   - OPENAI_API_KEY or ANTHROPIC_API_KEY"
echo "   - PINECONE_API_KEY and PINECONE_ENVIRONMENT"
echo ""
echo "2. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "3. Run tests to verify setup:"
echo "   uv run pytest"
echo ""
echo "4. Start building the agents!"
echo ""

