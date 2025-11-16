"""Shared fixtures and configuration for integration tests"""

import pytest
import os
import logging

# Set TESTING environment variable BEFORE any imports
# This must be done before importing the app to disable rate limiting
os.environ["TESTING"] = "true"

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini

from curlinator.config import get_settings
from curlinator.api.utils.llm_validation import is_valid_api_key

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for all integration tests"""
    # Configure local embedding model to avoid API dependencies
    # This runs once for the entire test session
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="./data/models"
    )

    # Configure LLM if a VALID API key is available
    # This is needed for QueryFusionRetriever and other LlamaIndex components
    # Skip LLM initialization if only test/placeholder keys are present
    settings = get_settings()
    llm_configured = False

    if settings.default_llm_provider == "openai" and is_valid_api_key(settings.openai_api_key, "openai"):
        try:
            Settings.llm = OpenAI(
                model=settings.default_model_openai,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base
            )
            llm_configured = True
            logger.info(f"✅ Configured OpenAI LLM: {settings.default_model_openai}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI LLM: {e}")
    elif settings.default_llm_provider == "anthropic" and is_valid_api_key(settings.anthropic_api_key, "anthropic"):
        try:
            Settings.llm = Anthropic(
                model=settings.default_model_anthropic,
                api_key=settings.anthropic_api_key
            )
            llm_configured = True
            logger.info(f"✅ Configured Anthropic LLM: {settings.default_model_anthropic}")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic LLM: {e}")
    elif settings.default_llm_provider == "gemini" and is_valid_api_key(settings.gemini_api_key, "gemini"):
        try:
            Settings.llm = Gemini(
                model=settings.default_model_gemini,
                api_key=settings.gemini_api_key
            )
            llm_configured = True
            logger.info(f"✅ Configured Gemini LLM: {settings.default_model_gemini}")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini LLM: {e}")

    if not llm_configured:
        logger.warning(
            "⚠️  No valid LLM API key found - LLM will not be initialized. "
            "Tests that require LLM will be skipped or use mocks."
        )

    yield

    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture(scope="session")
def check_api_key():
    """Check if a VALID API key is available for LLM tests"""
    from curlinator.config import get_settings

    settings = get_settings()
    has_key = bool(
        is_valid_api_key(settings.openai_api_key, "openai") or
        is_valid_api_key(settings.anthropic_api_key, "anthropic") or
        is_valid_api_key(settings.gemini_api_key, "gemini")
    )

    return has_key

