"""Embedding model configuration and provider enum."""

from enum import Enum


class EmbeddingProvider(str, Enum):
    """Embedding model provider options."""

    LOCAL = "local"  # HuggingFace BAAI/bge-small-en-v1.5 (free, slower, ~90MB download)
    OPENAI = "openai"  # OpenAI text-embedding-3-small (fast, costs money)
    GEMINI = "gemini"  # Google Gemini gemini-embedding-001 (fast, costs money)
    AUTO = "auto"  # Automatically select based on available API keys
