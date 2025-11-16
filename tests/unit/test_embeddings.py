"""
Unit tests for embedding model utility functions.
"""

import pytest
from unittest.mock import patch, MagicMock
from curlinator.api.utils.embeddings import get_embedding_model
from curlinator.api.models.crawl import EmbeddingProvider


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
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_openai_provider_with_api_key(self, mock_get_settings):
        """Test OPENAI provider when API key is available."""
        # Mock settings with OpenAI API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.default_embedding_model_openai = "text-embedding-3-small"
        mock_settings.openai_api_base = None
        mock_get_settings.return_value = mock_settings
        
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.OPENAI)
        
        assert provider_name == "openai"
        assert model_name == "text-embedding-3-small"
        assert embed_model is not None
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_openai_provider_without_api_key(self, mock_get_settings):
        """Test OPENAI provider when API key is missing."""
        # Mock settings without OpenAI API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None
        mock_get_settings.return_value = mock_settings
        
        with pytest.raises(Exception) as exc_info:
            get_embedding_model(EmbeddingProvider.OPENAI)
        
        assert "OPENAI_API_KEY" in str(exc_info.value)
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_gemini_provider_with_api_key(self, mock_get_settings):
        """Test GEMINI provider when API key is available."""
        # Mock settings with Gemini API key
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = "test-gemini-key"
        mock_settings.default_embedding_model_gemini = "models/embedding-001"
        mock_get_settings.return_value = mock_settings
        
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.GEMINI)
        
        assert provider_name == "gemini"
        assert model_name == "models/embedding-001"
        assert embed_model is not None
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_gemini_provider_without_api_key(self, mock_get_settings):
        """Test GEMINI provider when API key is missing."""
        # Mock settings without Gemini API key
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        mock_get_settings.return_value = mock_settings
        
        with pytest.raises(Exception) as exc_info:
            get_embedding_model(EmbeddingProvider.GEMINI)
        
        assert "GEMINI_API_KEY" in str(exc_info.value)
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_auto_mode_selects_openai(self, mock_get_settings):
        """Test AUTO mode selects OpenAI when API key is available."""
        # Mock settings with OpenAI API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.gemini_api_key = None
        mock_settings.default_embedding_model_openai = "text-embedding-3-small"
        mock_settings.openai_api_base = None
        mock_get_settings.return_value = mock_settings
        
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.AUTO)
        
        assert provider_name == "openai"
        assert model_name == "text-embedding-3-small"
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_auto_mode_selects_gemini(self, mock_get_settings):
        """Test AUTO mode selects Gemini when only Gemini API key is available."""
        # Mock settings with only Gemini API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None
        mock_settings.gemini_api_key = "test-gemini-key"
        mock_settings.default_embedding_model_gemini = "models/embedding-001"
        mock_get_settings.return_value = mock_settings
        
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.AUTO)
        
        assert provider_name == "gemini"
        assert model_name == "models/embedding-001"
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_auto_mode_selects_local(self, mock_get_settings):
        """Test AUTO mode selects LOCAL when no API keys are available."""
        # Mock settings without any API keys
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None
        mock_settings.gemini_api_key = None
        mock_get_settings.return_value = mock_settings
        
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.AUTO)
        
        assert provider_name == "local"
        assert model_name == "BAAI/bge-small-en-v1.5"
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_auto_mode_prefers_openai_over_gemini(self, mock_get_settings):
        """Test AUTO mode prefers OpenAI when both API keys are available."""
        # Mock settings with both API keys
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.gemini_api_key = "test-gemini-key"
        mock_settings.default_embedding_model_openai = "text-embedding-3-small"
        mock_settings.openai_api_base = None
        mock_get_settings.return_value = mock_settings
        
        embed_model, provider_name, model_name = get_embedding_model(EmbeddingProvider.AUTO)
        
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
    
    @patch('curlinator.api.utils.embeddings.get_settings')
    def test_auto_string_input(self, mock_get_settings):
        """Test AUTO mode with string input."""
        # Mock settings without any API keys
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None
        mock_settings.gemini_api_key = None
        mock_get_settings.return_value = mock_settings
        
        embed_model, provider_name, model_name = get_embedding_model("auto")
        
        assert provider_name == "local"
        assert model_name == "BAAI/bge-small-en-v1.5"

