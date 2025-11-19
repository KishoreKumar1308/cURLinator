"""
LLM factory for creating user-specific LLM instances.
"""

import logging
from typing import Optional
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini

from curlinator.api.db.models import UserSettings
from curlinator.api.utils.encryption import decrypt_api_key

logger = logging.getLogger(__name__)


def create_llm_from_user_settings(user_settings: UserSettings) -> Optional[object]:
    """
    Create an LLM instance based on user settings.
    
    Args:
        user_settings: UserSettings model instance
        
    Returns:
        LLM instance or None if user has no API key configured
    """
    provider = user_settings.preferred_llm_provider
    
    if not provider:
        return None
    
    # Get the encrypted API key for the selected provider
    encrypted_key = None
    if provider == "openai":
        encrypted_key = user_settings.user_openai_api_key_encrypted
    elif provider == "anthropic":
        encrypted_key = user_settings.user_anthropic_api_key_encrypted
    elif provider == "gemini":
        encrypted_key = user_settings.user_gemini_api_key_encrypted
    
    if not encrypted_key:
        logger.warning(f"User selected {provider} but no API key configured")
        return None
    
    # Decrypt the API key
    api_key = decrypt_api_key(encrypted_key)
    if not api_key:
        logger.error(f"Failed to decrypt {provider} API key for user")
        return None
    
    # Create LLM instance
    try:
        if provider == "openai":
            model = user_settings.preferred_llm_model or "gpt-4o-mini"
            return OpenAI(model=model, api_key=api_key)
        elif provider == "anthropic":
            model = user_settings.preferred_llm_model or "claude-3-5-sonnet-20241022"
            return Anthropic(model=model, api_key=api_key)
        elif provider == "gemini":
            model = user_settings.preferred_llm_model or "gemini-2.5-flash"
            return Gemini(model=model, api_key=api_key)
        else:
            logger.error(f"Unknown LLM provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"Failed to create {provider} LLM: {e}")
        return None


def validate_api_key(provider: str, api_key: str) -> tuple[bool, Optional[str]]:
    """
    Validate an API key by making a test API call.
    
    Args:
        provider: LLM provider ("openai", "anthropic", "gemini")
        api_key: API key to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if provider == "openai":
            llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
            # Make a minimal test call
            llm.complete("test")
            return True, None
        elif provider == "anthropic":
            llm = Anthropic(model="claude-3-5-sonnet-20241022", api_key=api_key)
            llm.complete("test")
            return True, None
        elif provider == "gemini":
            llm = Gemini(model="gemini-2.5-flash", api_key=api_key)
            llm.complete("test")
            return True, None
        else:
            return False, f"Unknown provider: {provider}"
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"API key validation failed for {provider}: {error_msg}")
        return False, error_msg

