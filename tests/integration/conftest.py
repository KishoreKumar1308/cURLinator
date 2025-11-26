"""Shared fixtures and configuration for core library integration tests"""

import logging
import os

import pytest
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI

from curlinator.config import get_settings
from curlinator.utils.llm_validation import is_valid_api_key

logger = logging.getLogger(__name__)


def _clear_settings_cache():
    """Clear the settings cache to force reload after environment changes"""
    get_settings.cache_clear()


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
    # Try all providers in order: OpenAI, Anthropic, Gemini
    settings = get_settings()
    llm_configured = False

    # Try OpenAI first (most common)
    if is_valid_api_key(settings.openai_api_key, "openai"):
        try:
            Settings.llm = OpenAI(
                model=settings.default_model_openai,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base
            )
            llm_configured = True
            # Override default provider for tests to use OpenAI
            os.environ["DEFAULT_LLM_PROVIDER"] = "openai"
            _clear_settings_cache()  # Clear cache to reload settings with new provider
            logger.info(f"✅ Configured OpenAI LLM for tests: {settings.default_model_openai}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI LLM: {e}")

    # Try Anthropic if OpenAI not available
    if not llm_configured and is_valid_api_key(settings.anthropic_api_key, "anthropic"):
        try:
            Settings.llm = Anthropic(
                model=settings.default_model_anthropic,
                api_key=settings.anthropic_api_key
            )
            llm_configured = True
            # Override default provider for tests to use Anthropic
            os.environ["DEFAULT_LLM_PROVIDER"] = "anthropic"
            _clear_settings_cache()  # Clear cache to reload settings with new provider
            logger.info(f"✅ Configured Anthropic LLM for tests: {settings.default_model_anthropic}")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic LLM: {e}")

    # Try Gemini if neither OpenAI nor Anthropic available
    if not llm_configured and is_valid_api_key(settings.gemini_api_key, "gemini"):
        try:
            Settings.llm = Gemini(
                model=settings.default_model_gemini,
                api_key=settings.gemini_api_key
            )
            llm_configured = True
            # Override default provider for tests to use Gemini
            os.environ["DEFAULT_LLM_PROVIDER"] = "gemini"
            _clear_settings_cache()  # Clear cache to reload settings with new provider
            logger.info(f"✅ Configured Gemini LLM for tests: {settings.default_model_gemini}")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini LLM: {e}")

    if not llm_configured:
        logger.warning(
            "⚠️  No valid LLM API key found - LLM will not be initialized. "
            "Tests that require LLM will be skipped or use mocks."
        )

    yield


def _has_valid_llm_api_key():
    """
    Check if a VALID API key is available for LLM tests.

    Returns True if at least one valid (non-test/placeholder) API key is configured.
    Used by pytest.mark.skipif to skip tests that require real LLM API access.
    """
    from curlinator.config import get_settings

    settings = get_settings()
    has_key = bool(
        is_valid_api_key(settings.openai_api_key, "openai") or
        is_valid_api_key(settings.anthropic_api_key, "anthropic") or
        is_valid_api_key(settings.gemini_api_key, "gemini")
    )

    return has_key


@pytest.fixture(scope="session")
def check_api_key():
    """Check if a VALID API key is available for LLM tests"""
    return _has_valid_llm_api_key()


# Pytest marker for tests that require a real LLM API key
requires_llm = pytest.mark.skipif(
    not _has_valid_llm_api_key(),
    reason="Requires valid LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY). "
           "Skipping in CI to avoid API costs and rate limits. Run locally with valid API key to test."
)

