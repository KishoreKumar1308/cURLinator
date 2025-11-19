"""Application settings and configuration"""

import os
import logging
from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Provider Configuration
    openai_api_key: str = ""
    openai_api_base: Optional[str] = None  # For custom OpenAI-compatible APIs
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Vector Database Configuration (Chroma - Local Persistent Storage)
    vector_db_path: str = "./data/vector_db"
    vector_collection_name: str = "api_documentation"

    # Application Configuration
    environment: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"

    # API Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

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

    # Error Tracking (Sentry)
    sentry_dsn: str = ""
    sentry_environment: Optional[str] = None  # Defaults to environment if not set
    sentry_traces_sample_rate: float = 0.1  # 10% of requests

    # User API Key Encryption (Required for BYOK)
    api_key_encryption_key: str = ""  # Fernet encryption key for user API keys


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def validate_environment() -> None:
    """
    Validate required environment variables on startup.

    Raises:
        ValueError: If required environment variables are missing or invalid
    """
    # Skip validation in test mode
    if os.getenv("TESTING") == "true":
        logger.info("Skipping environment validation in test mode")
        return

    errors = []
    warnings = []

    # Get environment
    environment = os.getenv("ENVIRONMENT", "development")
    is_production = environment == "production"

    # Check DATABASE_URL (always required)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        errors.append(
            "DATABASE_URL is required. "
            "Set it to your PostgreSQL connection string: "
            "postgresql://user:password@host:port/database"
        )
    elif not database_url.startswith("postgresql://"):
        warnings.append(
            f"DATABASE_URL should start with 'postgresql://', got: {database_url[:20]}..."
        )

    # Check JWT_SECRET (required in production, warning in development)
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        if is_production:
            errors.append(
                "JWT_SECRET is required in production. "
                "Generate a secure secret with: openssl rand -hex 32"
            )
        else:
            warnings.append(
                "JWT_SECRET not set. Using default (insecure for production). "
                "Generate a secure secret with: openssl rand -hex 32"
            )
    elif jwt_secret in ["your-secret-key-change-this-in-production", "your-super-secret-key-change-in-production"]:
        if is_production:
            errors.append(
                "JWT_SECRET is using the default value. "
                "This is insecure! Generate a secure secret with: openssl rand -hex 32"
            )
        else:
            warnings.append(
                "JWT_SECRET is using the default value. "
                "Change this before deploying to production!"
            )

    # Check LLM API keys (at least one required)
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not any([openai_key, gemini_key, anthropic_key]):
        errors.append(
            "At least one LLM API key is required. "
            "Set one of: OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY"
        )
    else:
        # Log which providers are configured
        configured_providers = []
        if openai_key:
            configured_providers.append("OpenAI")
        if gemini_key:
            configured_providers.append("Gemini")
        if anthropic_key:
            configured_providers.append("Anthropic")

        logger.info(f"Configured LLM providers: {', '.join(configured_providers)}")

    # Check CHROMA_PERSIST_DIR (warning if not set)
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    if not os.path.exists(chroma_dir):
        logger.info(f"Chroma directory will be created at: {chroma_dir}")

    # Log warnings
    for warning in warnings:
        logger.warning(f"⚠️  {warning}")

    # Raise errors if any
    if errors:
        error_message = "\n\n❌ Environment validation failed:\n\n"
        for i, error in enumerate(errors, 1):
            error_message += f"{i}. {error}\n"
        error_message += "\nPlease check your .env file and set the required variables."
        error_message += "\nSee .env.example for reference."
        raise ValueError(error_message)

    logger.info("✅ Environment validation passed")

