"""
Utilities for validating LLM API keys and preventing initialization with test/placeholder keys.

This module provides centralized validation logic to ensure that LLM initialization
only happens with valid API keys, not test placeholders like 'test-key-not-real'.
"""

import logging

logger = logging.getLogger(__name__)


def is_valid_api_key(api_key: str | None, provider: str) -> bool:
    """
    Check if an API key is valid (not a test/placeholder key).

    This function validates that an API key:
    1. Exists (is not None or empty)
    2. Doesn't contain common test/placeholder patterns
    3. Matches the expected format for the provider

    Args:
        api_key: The API key to validate (can be None)
        provider: The provider name ("openai", "anthropic", "gemini")

    Returns:
        True if the key appears to be valid, False otherwise

    Examples:
        >>> is_valid_api_key("sk-test-key-not-real", "openai")
        False
        >>> is_valid_api_key("test-key-not-real", "gemini")
        False
        >>> is_valid_api_key("sk-1234567890123456789012345678901234567890", "openai")
        True
        >>> is_valid_api_key(None, "openai")
        False
        >>> is_valid_api_key("", "openai")
        False
    """
    if not api_key:
        return False

    # Common test/placeholder patterns
    test_patterns = [
        "test-key",
        "not-real",
        "placeholder",
        "dummy",
        "fake",
        "mock",
        "example",
        "invalid",
    ]

    # Check if key contains any test patterns (case-insensitive)
    api_key_lower = api_key.lower()
    if any(pattern in api_key_lower for pattern in test_patterns):
        logger.debug(f"Detected test/placeholder API key for {provider}: {api_key[:20]}...")
        return False

    # Provider-specific format validation
    if provider == "openai":
        # OpenAI keys start with 'sk-' and are at least 20 chars
        # Real keys are typically 48+ characters
        is_valid = api_key.startswith("sk-") and len(api_key) > 20
        if not is_valid:
            logger.debug(f"OpenAI API key failed format validation: {api_key[:20]}...")
        return is_valid

    elif provider == "anthropic":
        # Anthropic keys start with 'sk-ant-' and are at least 20 chars
        # Real keys are typically 100+ characters
        is_valid = api_key.startswith("sk-ant-") and len(api_key) > 20
        if not is_valid:
            logger.debug(f"Anthropic API key failed format validation: {api_key[:20]}...")
        return is_valid

    elif provider == "gemini":
        # Gemini keys are typically 39 characters long (AIzaSy...)
        # Minimum length check to avoid test keys
        is_valid = len(api_key) >= 30
        if not is_valid:
            logger.debug(f"Gemini API key failed format validation: {api_key[:20]}...")
        return is_valid

    # Unknown provider - be conservative and reject
    logger.warning(f"Unknown provider '{provider}' - rejecting API key")
    return False


def validate_llm_config(provider: str, api_key: str | None) -> bool:
    """
    Validate LLM configuration before attempting initialization.

    This is a convenience wrapper around is_valid_api_key that also logs
    appropriate messages for debugging.

    Args:
        provider: The LLM provider name ("openai", "anthropic", "gemini")
        api_key: The API key to validate

    Returns:
        True if the configuration is valid and LLM can be initialized, False otherwise

    Examples:
        >>> validate_llm_config("openai", "sk-test-key-not-real")
        False
        >>> validate_llm_config("gemini", None)
        False
    """
    if not api_key:
        logger.info(f"No {provider.upper()} API key configured - skipping LLM initialization")
        return False

    if not is_valid_api_key(api_key, provider):
        logger.info(
            f"Invalid or test {provider.upper()} API key detected - skipping LLM initialization. "
            f"Key preview: {api_key[:20]}..."
        )
        return False

    return True
