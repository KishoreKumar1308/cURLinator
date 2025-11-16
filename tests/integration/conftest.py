"""Shared fixtures and configuration for integration tests"""

import pytest
import os

# Set TESTING environment variable BEFORE any imports
# This must be done before importing the app to disable rate limiting
os.environ["TESTING"] = "true"

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini

from curlinator.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for all integration tests"""
    # Configure local embedding model to avoid API dependencies
    # This runs once for the entire test session
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="./data/models"
    )

    # Configure LLM if API key is available
    # This is needed for QueryFusionRetriever and other LlamaIndex components
    settings = get_settings()
    if settings.default_llm_provider == "openai" and settings.openai_api_key:
        Settings.llm = OpenAI(
            model=settings.default_model_openai,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base
        )
    elif settings.default_llm_provider == "anthropic" and settings.anthropic_api_key:
        Settings.llm = Anthropic(
            model=settings.default_model_anthropic,
            api_key=settings.anthropic_api_key
        )
    elif settings.default_llm_provider == "gemini" and settings.gemini_api_key:
        Settings.llm = Gemini(
            model=settings.default_model_gemini,
            api_key=settings.gemini_api_key
        )

    yield

    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture(scope="session")
def check_api_key():
    """Check if API key is available for LLM tests"""
    from curlinator.config import get_settings
    
    settings = get_settings()
    has_key = bool(
        settings.openai_api_key or 
        settings.anthropic_api_key or 
        settings.gemini_api_key
    )
    
    return has_key

