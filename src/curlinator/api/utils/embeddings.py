"""
Shared utilities for embedding model management.
"""

import logging
from typing import Tuple, Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

from curlinator.api.models.crawl import EmbeddingProvider
from curlinator.config import get_settings

logger = logging.getLogger(__name__)


def get_embedding_model(provider: EmbeddingProvider | str) -> Tuple[Any, str, str]:
    """
    Get embedding model instance based on provider.
    
    Args:
        provider: EmbeddingProvider enum or string ("local", "openai", "gemini", "auto")
        
    Returns:
        Tuple of (embed_model_instance, provider_name, model_name)
        
    Raises:
        ValueError: If provider is invalid or API key is missing
    """
    settings = get_settings()
    
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
        if settings.openai_api_key:
            provider = EmbeddingProvider.OPENAI
            logger.info("AUTO mode: Selected OpenAI embeddings (API key found)")
        elif settings.gemini_api_key:
            provider = EmbeddingProvider.GEMINI
            logger.info("AUTO mode: Selected Gemini embeddings (API key found)")
        else:
            provider = EmbeddingProvider.LOCAL
            logger.info("AUTO mode: Selected local embeddings (no API keys found)")
    
    # Instantiate the selected provider
    if provider == EmbeddingProvider.LOCAL:
        model_name = "BAAI/bge-small-en-v1.5"
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            cache_folder="./data/models"
        )
        logger.info(f"Using local embedding model: {model_name}")
        return embed_model, "local", model_name
    
    elif provider == EmbeddingProvider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        model_name = settings.default_embedding_model_openai
        
        # Build OpenAI embedding kwargs
        embed_kwargs = {
            "model": model_name,
            "api_key": settings.openai_api_key,
        }
        
        # Add custom base URL if configured
        if settings.openai_api_base:
            embed_kwargs["api_base"] = settings.openai_api_base
        
        embed_model = OpenAIEmbedding(**embed_kwargs)
        logger.info(f"Using OpenAI embedding model: {model_name}")
        return embed_model, "openai", model_name
    
    elif provider == EmbeddingProvider.GEMINI:
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        
        model_name = settings.default_embedding_model_gemini
        embed_model = GeminiEmbedding(
            model_name=model_name,
            api_key=settings.gemini_api_key,
        )
        logger.info(f"Using Gemini embedding model: {model_name}")
        return embed_model, "gemini", model_name
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

