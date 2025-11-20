"""
Integration tests for freemium user flow.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import patch, MagicMock, AsyncMock

from curlinator.api.db.models import UserSettings, DocumentationCollection


class TestFreemiumChatFlow:
    """Test freemium flow for chat endpoint."""

    def test_free_tier_user_can_send_messages_within_limit(
        self, client: TestClient, auth_headers: dict, test_user_id: str,
        db_session: Session, test_collection: DocumentationCollection
    ):
        """Test that free tier users can send messages within their daily limit."""
        # Ensure user has no API key
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        if not settings:
            settings = UserSettings(user_id=test_user_id)
            db_session.add(settings)

        settings.user_openai_api_key_encrypted = None
        settings.user_anthropic_api_key_encrypted = None
        settings.user_gemini_api_key_encrypted = None
        settings.free_messages_used = 0
        db_session.commit()

        # Mock the ChatAgent to avoid actual LLM calls
        with patch('curlinator.api.routes.chat.ChatAgent') as mock_agent:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value={
                "response": "Test response",
                "curl_command": "curl -X GET https://api.example.com",
                "sources": []
            })
            mock_agent.return_value = mock_instance

            # Send a message (should succeed)
            response = client.post(
                "/api/v1/chat",
                headers=auth_headers,
                json={
                    "collection_name": test_collection.name,
                    "message": "Test message"
                }
            )
            assert response.status_code == 200

            # Verify counter was incremented
            db_session.refresh(settings)
            assert settings.free_messages_used == 1

    def test_free_tier_user_blocked_after_limit(
        self, client: TestClient, auth_headers: dict, test_user_id: str,
        db_session: Session, test_collection: DocumentationCollection
    ):
        """Test that free tier users are blocked after exceeding daily limit."""
        # Set user to limit
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        if not settings:
            settings = UserSettings(user_id=test_user_id)
            db_session.add(settings)
        
        settings.user_openai_api_key_encrypted = None
        settings.user_anthropic_api_key_encrypted = None
        settings.user_gemini_api_key_encrypted = None
        settings.free_messages_used = 10
        settings.free_messages_limit = 10
        db_session.commit()

        # Try to send a message (should fail with 402)
        response = client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={
                "collection_name": test_collection.name,
                "message": "Test message"
            }
        )
        assert response.status_code == 402

        data = response.json()
        assert data["error"] == "Free message limit exceeded"
        assert data["free_messages_used"] == 10
        assert data["free_messages_limit"] == 10
        assert data["upgrade_options"]["byok"] is True
        assert "API key" in data["suggestion"]

    def test_byok_user_unlimited_messages(
        self, client: TestClient, auth_headers: dict, test_user_id: str,
        db_session: Session, test_collection: DocumentationCollection
    ):
        """Test that BYOK users can send unlimited messages."""
        # Give user an API key
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        if not settings:
            settings = UserSettings(user_id=test_user_id)
            db_session.add(settings)
        
        from curlinator.api.utils.encryption import encrypt_api_key
        settings.user_openai_api_key_encrypted = encrypt_api_key("sk-test-key")
        settings.preferred_llm_provider = "openai"
        settings.free_messages_used = 100  # Already exceeded limit
        db_session.commit()

        # Mock the ChatAgent
        with patch('curlinator.api.routes.chat.ChatAgent') as mock_agent:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value={
                "response": "Test response",
                "curl_command": "curl -X GET https://api.example.com",
                "sources": []
            })
            mock_agent.return_value = mock_instance

            # Should succeed even though free limit is exceeded
            response = client.post(
                "/api/v1/chat",
                headers=auth_headers,
                json={
                    "collection_name": test_collection.name,
                    "message": "Test message"
                }
            )
            assert response.status_code == 200

            # Counter should NOT be incremented for BYOK users
            db_session.refresh(settings)
            assert settings.free_messages_used == 100  # Unchanged

    def test_daily_reset_of_free_messages(
        self, client: TestClient, auth_headers: dict, test_user_id: str,
        db_session: Session, test_collection: DocumentationCollection
    ):
        """Test that free message counter resets daily."""
        from datetime import datetime, timezone, timedelta

        # Set user to limit with yesterday's date
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        if not settings:
            settings = UserSettings(user_id=test_user_id)
            db_session.add(settings)
        
        settings.user_openai_api_key_encrypted = None
        settings.free_messages_used = 10
        settings.free_messages_limit = 10
        settings.last_message_reset_date = datetime.now(timezone.utc) - timedelta(days=1)
        db_session.commit()

        # Mock the ChatAgent
        with patch('curlinator.api.routes.chat.ChatAgent') as mock_agent:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value={
                "response": "Test response",
                "curl_command": "curl -X GET https://api.example.com",
                "sources": []
            })
            mock_agent.return_value = mock_instance

            # Should succeed because counter should reset
            response = client.post(
                "/api/v1/chat",
                headers=auth_headers,
                json={
                    "collection_name": test_collection.name,
                    "message": "Test message"
                }
            )
            assert response.status_code == 200

            # Verify counter was reset and incremented
            db_session.refresh(settings)
            assert settings.free_messages_used == 1


class TestFreemiumCrawlFlow:
    """Test freemium flow for crawl endpoint."""

    def test_free_tier_user_forced_local_embeddings(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test that free tier users are forced to use local embeddings."""
        # Ensure user has no API key
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        if settings:
            settings.user_openai_api_key_encrypted = None
            settings.user_anthropic_api_key_encrypted = None
            settings.user_gemini_api_key_encrypted = None
            db_session.commit()

        # Try to crawl with OpenAI embeddings (should fail with 402)
        response = client.post(
            "/api/v1/crawl",
            headers=auth_headers,
            json={
                "url": "https://docs.example.com",
                "max_pages": 10,
                "max_depth": 2,
                "embedding_provider": "openai"
            }
        )
        assert response.status_code == 402

        data = response.json()
        assert data["error"] == "API-based embeddings require API key"
        assert data["allowed_provider"] == "local"
        assert data["upgrade_options"]["byok"] is True

    def test_free_tier_user_can_use_local_embeddings(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test that free tier users can use local embeddings."""
        # Ensure user has no API key
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        if settings:
            settings.user_openai_api_key_encrypted = None
            db_session.commit()

        # Mock the crawl agents (patch where they're actually imported and used)
        with patch('curlinator.api.services.incremental_crawler.DocumentationAgent') as mock_doc_agent, \
             patch('curlinator.api.services.incremental_crawler.ChatAgent') as mock_chat_agent:
            
            mock_doc_instance = MagicMock()
            mock_doc_instance.execute.return_value = [MagicMock()]  # Return mock documents
            mock_doc_agent.return_value = mock_doc_instance

            mock_chat_instance = MagicMock()
            mock_chat_agent.return_value = mock_chat_instance

            # Should succeed with local embeddings
            response = client.post(
                "/api/v1/crawl",
                headers=auth_headers,
                json={
                    "url": "https://docs.example.com",
                    "max_pages": 10,
                    "max_depth": 2,
                    "embedding_provider": "LOCAL"
                }
            )
            # May fail for other reasons (rate limit, etc.) but not 402
            assert response.status_code != 402

    def test_byok_user_can_use_any_embeddings(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test that BYOK users can use any embedding provider."""
        # Give user an API key
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        if not settings:
            settings = UserSettings(user_id=test_user_id)
            db_session.add(settings)
        
        from curlinator.api.utils.encryption import encrypt_api_key
        settings.user_openai_api_key_encrypted = encrypt_api_key("sk-test-key")
        db_session.commit()

        # Mock the crawl agents (patch where they're actually imported and used)
        with patch('curlinator.api.services.incremental_crawler.DocumentationAgent') as mock_doc_agent, \
             patch('curlinator.api.services.incremental_crawler.ChatAgent') as mock_chat_agent:
            
            mock_doc_instance = MagicMock()
            mock_doc_instance.execute.return_value = [MagicMock()]
            mock_doc_agent.return_value = mock_doc_instance

            mock_chat_instance = MagicMock()
            mock_chat_agent.return_value = mock_chat_instance

            # Should succeed with OpenAI embeddings
            response = client.post(
                "/api/v1/crawl",
                headers=auth_headers,
                json={
                    "url": "https://docs.example.com",
                    "max_pages": 10,
                    "max_depth": 2,
                    "embedding_provider": "OPENAI"
                }
            )
            # May fail for other reasons but not 402
            assert response.status_code != 402

