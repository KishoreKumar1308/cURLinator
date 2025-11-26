"""
Unit tests for embedding model utility functions.
"""

import pytest

from curlinator.models.embeddings import EmbeddingProvider
from curlinator.utils.embeddings import get_embedding_model


class TestGetEmbeddingModel:
    """Test the get_embedding_model function."""

    def test_local_provider_enum(self):
        """Test LOCAL provider with enum input."""
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.LOCAL)

        assert provider_name == "local"
        assert model_name == "BAAI/bge-small-en-v1.5"
        assert embed_model is not None

    def test_local_provider_string(self):
        """Test LOCAL provider with string input."""
        embed_model, provider_name, model_name = get_embedding_model("local")

        assert provider_name == "local"
        assert model_name == "BAAI/bge-small-en-v1.5"
        assert embed_model is not None

    def test_local_provider_string_uppercase(self):
        """Test LOCAL provider with uppercase string input."""
        embed_model, provider_name, model_name = get_embedding_model("LOCAL")

        assert provider_name == "local"
        assert model_name == "BAAI/bge-small-en-v1.5"
        assert embed_model is not None

    def test_openai_provider_with_api_key(self):
        """Test OPENAI provider when API key is available."""
        embed_model, provider_name, model_name = get_embedding_model(
            EmbeddingProvider.OPENAI,
            openai_api_key="sk-test1234567890123456789012345678901234567890",
        )

        assert provider_name == "openai"
        assert model_name == "text-embedding-3-small"
        assert embed_model is not None

    def test_openai_provider_without_api_key(self):
        """Test OPENAI provider when API key is missing."""

        with pytest.raises(Exception) as exc_info:
            get_embedding_model(EmbeddingProvider.OPENAI)

        assert "OpenAI API key is required" in str(exc_info.value)

    def test_gemini_provider_with_api_key(self):
        """Test GEMINI provider when API key is available."""
        embed_model, provider_name, model_name = get_embedding_model(
            EmbeddingProvider.GEMINI, gemini_api_key="AIzaSyTest1234567890123456789012345"
        )

        assert provider_name == "gemini"
        assert model_name == "gemini-embedding-001"
        assert embed_model is not None

    def test_gemini_provider_without_api_key(self):
        """Test GEMINI provider when API key is missing."""
        with pytest.raises(Exception) as exc_info:
            get_embedding_model(EmbeddingProvider.GEMINI)

        assert "Gemini API key is required" in str(exc_info.value)

    def test_auto_mode_selects_openai(self):
        """Test AUTO mode selects OpenAI when API key is available."""
        embed_model, provider_name, model_name = get_embedding_model(
            EmbeddingProvider.AUTO, openai_api_key="sk-test1234567890123456789012345678901234567890"
        )

        assert provider_name == "openai"
        assert model_name == "text-embedding-3-small"

    def test_auto_mode_selects_gemini(self):
        """Test AUTO mode selects Gemini when only Gemini API key is available."""
        embed_model, provider_name, model_name = get_embedding_model(
            EmbeddingProvider.AUTO, gemini_api_key="AIzaSyTest1234567890123456789012345"
        )

        assert provider_name == "gemini"
        assert model_name == "gemini-embedding-001"

    def test_auto_mode_selects_local(self):
        """Test AUTO mode selects LOCAL when no API keys are available."""
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.AUTO)

        assert provider_name == "local"
        assert model_name == "BAAI/bge-small-en-v1.5"

    def test_auto_mode_prefers_openai_over_gemini(self):
        """Test AUTO mode prefers OpenAI when both API keys are available."""
        embed_model, provider_name, model_name = get_embedding_model(
            EmbeddingProvider.AUTO,
            openai_api_key="sk-test1234567890123456789012345678901234567890",
            gemini_api_key="AIzaSyTest1234567890123456789012345",
        )

        assert provider_name == "openai"
        assert model_name == "text-embedding-3-small"

    def test_invalid_provider_string(self):
        """Test that invalid provider string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_embedding_model("invalid_provider")

        assert "Invalid embedding provider" in str(exc_info.value)

    def test_string_input_case_insensitive(self):
        """Test that string input is case-insensitive."""
        # Test various case combinations
        for provider_str in ["local", "LOCAL", "Local", "LoCaL"]:
            embed_model, provider_name, model_name = get_embedding_model(provider_str)
            assert provider_name == "local"

    def test_auto_string_input(self):
        """Test AUTO mode with string input."""
        embed_model, provider_name, model_name = get_embedding_model("auto")

        assert provider_name == "local"
        assert model_name == "BAAI/bge-small-en-v1.5"
