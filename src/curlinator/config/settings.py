"""Core library settings (simplified - no database, JWT, or encryption settings)"""

import logging
import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Core library settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Provider Configuration
    openai_api_key: str = ""
    openai_api_base: str | None = None  # For custom OpenAI-compatible APIs
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Vector Database Configuration (Chroma - Local Persistent Storage)
    vector_db_path: str = "./data/vector_db"
    vector_collection_name: str = "api_documentation"

    # Application Configuration
    environment: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"

    # Agent Configuration
    default_llm_provider: Literal["openai", "anthropic", "gemini"] = "gemini"
    default_model_gemini: str = "gemini-2.5-flash"
    default_model_openai: str = "gpt-4-turbo-preview"
    default_model_anthropic: str = "claude-3-5-sonnet-20241022"

    default_embedding_model_gemini: str = "gemini-embedding-001"
    default_embedding_model_openai: str = "text-embedding-3-small"

    # Crawler Configuration
    crawler_timeout: int = 30000  # milliseconds
    crawler_max_pages: int = 50
    crawler_rate_limit: float = 2.0  # requests per second


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def validate_environment() -> None:
    """
    Validate required environment variables on startup.

    For core library, this is optional - users can provide API keys programmatically.

    Raises:
        ValueError: If required environment variables are missing or invalid
    """
    # Skip validation in test mode
    if os.getenv("TESTING") == "true":
        logger.info("Skipping environment validation in test mode")
        return

    settings = get_settings()

    # Check if at least one LLM API key is configured
    has_llm_key = bool(
        settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key
    )

    if not has_llm_key:
        logger.warning(
            "No LLM API keys configured. You can provide them programmatically "
            "when creating agents, or set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY."
        )
