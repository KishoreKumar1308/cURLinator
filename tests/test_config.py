"""Test configuration and settings"""

import pytest
from curlinator.config import Settings, get_settings


def test_settings_initialization() -> None:
    """Test that settings can be initialized"""
    settings = Settings()
    assert settings is not None
    assert settings.environment in ["development", "staging", "production", "test"]


def test_get_settings_returns_same_instance() -> None:
    """Test that get_settings returns cached instance"""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_default_values() -> None:
    """Test default configuration values"""
    settings = Settings()
    # Environment can be "development" by default, or "test" if ENVIRONMENT=test is set
    assert settings.environment in ["development", "test"]
    assert settings.log_level == "INFO"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.default_llm_provider in ["openai", "anthropic", "gemini"]
    assert settings.crawler_timeout == 30000
    assert settings.crawler_max_pages == 50

