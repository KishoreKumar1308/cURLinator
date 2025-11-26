"""
Shared utilities for embedding model management.

This module provides functions to create embedding models from different providers.
API keys are passed as parameters rather than read from settings, making it suitable
for both standalone library usage and server integration.
"""

import logging
from typing import Any

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from curlinator.models.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


def get_embedding_model(
    provider: EmbeddingProvider | str,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
    openai_api_base: str | None = None,
    openai_model: str = "text-embedding-3-small",
    gemini_model: str = "gemini-embedding-001",
    local_model: str = "BAAI/bge-small-en-v1.5",
    local_cache_folder: str = "./data/models",
) -> tuple[Any, str, str]:
    """
    Get embedding model instance based on provider.

    Args:
        provider: EmbeddingProvider enum or string ("local", "openai", "gemini", "auto")
        openai_api_key: OpenAI API key (required for OpenAI provider)
        gemini_api_key: Gemini API key (required for Gemini provider)
        openai_api_base: Custom OpenAI API base URL (optional)
        openai_model: OpenAI embedding model name (default: text-embedding-3-small)
        gemini_model: Gemini embedding model name (default: gemini-embedding-001)
        local_model: HuggingFace model name for local embeddings (default: BAAI/bge-small-en-v1.5)
        local_cache_folder: Cache folder for local models (default: ./data/models)

    Returns:
        Tuple of (embed_model_instance, provider_name, model_name)

    Raises:
        ValueError: If provider is invalid or required API key is missing
    """
    # Convert string to enum if needed
    if isinstance(provider, str):
        provider = provider.lower()
        if provider == "local":
            provider = EmbeddingProvider.LOCAL
        elif provider == "openai":
            provider = EmbeddingProvider.OPENAI
        elif provider == "gemini":
            provider = EmbeddingProvider.GEMINI
        elif provider == "auto":
            provider = EmbeddingProvider.AUTO
        else:
            raise ValueError(f"Invalid embedding provider: {provider}")

    # Handle AUTO provider - choose based on available API keys
    if provider == EmbeddingProvider.AUTO:
        if openai_api_key:
            provider = EmbeddingProvider.OPENAI
            logger.info("AUTO mode: Selected OpenAI embeddings (API key provided)")
        elif gemini_api_key:
            provider = EmbeddingProvider.GEMINI
            logger.info("AUTO mode: Selected Gemini embeddings (API key provided)")
        else:
            provider = EmbeddingProvider.LOCAL
            logger.info("AUTO mode: Selected local embeddings (no API keys provided)")

    # Instantiate the selected provider
    if provider == EmbeddingProvider.LOCAL:
        embed_model = HuggingFaceEmbedding(model_name=local_model, cache_folder=local_cache_folder)
        logger.info(f"Using local embedding model: {local_model}")
        return embed_model, "local", local_model

    elif provider == EmbeddingProvider.OPENAI:
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key is required for OpenAI embeddings. "
                "Please provide openai_api_key parameter."
            )

        # Build OpenAI embedding kwargs
        embed_kwargs = {
            "model": openai_model,
            "api_key": openai_api_key,
        }

        # Add custom base URL if configured
        if openai_api_base:
            embed_kwargs["api_base"] = openai_api_base

        embed_model = OpenAIEmbedding(**embed_kwargs)
        logger.info(f"Using OpenAI embedding model: {openai_model}")
        return embed_model, "openai", openai_model

    elif provider == EmbeddingProvider.GEMINI:
        if not gemini_api_key:
            raise ValueError(
                "Gemini API key is required for Gemini embeddings. "
                "Please provide gemini_api_key parameter."
            )

        embed_model = GeminiEmbedding(
            model_name=gemini_model,
            api_key=gemini_api_key,
        )
        logger.info(f"Using Gemini embedding model: {gemini_model}")
        return embed_model, "gemini", gemini_model

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
