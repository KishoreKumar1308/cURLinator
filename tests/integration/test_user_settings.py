"""
Integration tests for user settings endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from curlinator.api.db.models import UserSettings
from curlinator.api.utils.encryption import encrypt_api_key, decrypt_api_key


class TestUserSettingsEndpoints:
    """Test user settings CRUD operations."""

    def test_get_settings_creates_default_if_not_exists(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test GET /api/v1/settings creates default settings if they don't exist."""
        # Ensure no settings exist
        db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).delete()
        db_session.commit()

        response = client.get("/api/v1/settings", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["preferred_llm_provider"] is None
        assert data["preferred_llm_model"] is None
        assert data["has_openai_key"] is False
        assert data["has_anthropic_key"] is False
        assert data["has_gemini_key"] is False
        assert data["preferred_embedding_provider"] == "local"
        assert data["default_max_pages"] == 50
        assert data["default_max_depth"] == 3
        assert data["free_messages_used"] == 0
        assert data["free_messages_limit"] == 10
        assert data["free_messages_remaining"] == 10

    def test_update_llm_preferences(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test updating LLM preferences."""
        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={
                "preferred_llm_provider": "openai",
                "preferred_llm_model": "gpt-4o-mini",
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["preferred_llm_provider"] == "openai"
        assert data["preferred_llm_model"] == "gpt-4o-mini"

    def test_update_api_key_encrypts_before_storage(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test that API keys are encrypted before storage."""
        test_api_key = "sk-test-key-1234567890"

        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={"openai_api_key": test_api_key}
        )
        assert response.status_code == 200
        assert response.json()["has_openai_key"] is True

        # Verify key is encrypted in database
        settings = db_session.query(UserSettings).filter(UserSettings.user_id == test_user_id).first()
        assert settings.user_openai_api_key_encrypted is not None
        assert settings.user_openai_api_key_encrypted != test_api_key  # Should be encrypted

        # Verify we can decrypt it
        decrypted = decrypt_api_key(settings.user_openai_api_key_encrypted)
        assert decrypted == test_api_key

    def test_update_multiple_api_keys(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test updating multiple API keys at once."""
        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={
                "openai_api_key": "sk-openai-test-key",
                "anthropic_api_key": "sk-ant-test-key",
                "gemini_api_key": "gemini-test-key",
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["has_openai_key"] is True
        assert data["has_anthropic_key"] is True
        assert data["has_gemini_key"] is True

    def test_remove_api_key_with_empty_string(
        self, client: TestClient, auth_headers: dict, test_user_id: str, db_session: Session
    ):
        """Test removing an API key by setting it to empty string."""
        # First add a key
        client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={"openai_api_key": "sk-test-key"}
        )

        # Then remove it
        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={"openai_api_key": ""}
        )
        assert response.status_code == 200
        assert response.json()["has_openai_key"] is False

    def test_update_embedding_preferences(
        self, client: TestClient, auth_headers: dict
    ):
        """Test updating embedding preferences."""
        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={"preferred_embedding_provider": "openai"}
        )
        assert response.status_code == 200
        assert response.json()["preferred_embedding_provider"] == "openai"

    def test_update_crawl_defaults(
        self, client: TestClient, auth_headers: dict
    ):
        """Test updating crawl default settings."""
        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={
                "default_max_pages": 100,
                "default_max_depth": 5,
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["default_max_pages"] == 100
        assert data["default_max_depth"] == 5

    def test_validate_crawl_defaults_range(
        self, client: TestClient, auth_headers: dict
    ):
        """Test that crawl defaults are validated."""
        # Test max_pages too high
        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={"default_max_pages": 2000}
        )
        assert response.status_code == 422

        # Test max_depth too high
        response = client.patch(
            "/api/v1/settings",
            headers=auth_headers,
            json={"default_max_depth": 20}
        )
        assert response.status_code == 422

    def test_settings_require_authentication(self, client: TestClient):
        """Test that settings endpoints require authentication."""
        response = client.get("/api/v1/settings")
        assert response.status_code in [401, 403]  # Either unauthorized or forbidden

        response = client.patch("/api/v1/settings", json={})
        assert response.status_code in [401, 403]  # Either unauthorized or forbidden


class TestAPIKeyValidation:
    """Test API key validation endpoint."""

    def test_validate_api_key_endpoint_exists(
        self, client: TestClient, auth_headers: dict
    ):
        """Test that validate-api-key endpoint is accessible."""
        response = client.post(
            "/api/v1/settings/validate-api-key",
            headers=auth_headers,
            json={
                "provider": "openai",
                "api_key": "sk-test-invalid-key"
            }
        )
        # Should return 200 even if key is invalid (validation result in response)
        assert response.status_code == 200

    def test_validate_api_key_requires_authentication(self, client: TestClient):
        """Test that validation endpoint requires authentication."""
        response = client.post(
            "/api/v1/settings/validate-api-key",
            json={
                "provider": "openai",
                "api_key": "sk-test-key"
            }
        )
        assert response.status_code in [401, 403]  # Either unauthorized or forbidden

    @pytest.mark.skip(reason="Rate limiting is disabled in test mode (TESTING=true)")
    def test_validate_api_key_rate_limited(
        self, client: TestClient, auth_headers: dict
    ):
        """Test that validation endpoint is rate limited."""
        # Make 6 requests (limit is 5/minute)
        for i in range(6):
            response = client.post(
                "/api/v1/settings/validate-api-key",
                headers=auth_headers,
                json={
                    "provider": "openai",
                    "api_key": f"sk-test-key-{i}"
                }
            )
            if i < 5:
                assert response.status_code == 200
            else:
                assert response.status_code == 429  # Rate limit exceeded


class TestEncryptionHelpers:
    """Test encryption helper functions."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption work correctly."""
        original = "sk-test-api-key-1234567890"
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted)

        assert encrypted != original
        assert decrypted == original

    def test_encrypt_empty_string(self):
        """Test encrypting empty string."""
        encrypted = encrypt_api_key("")
        assert encrypted == ""

    def test_decrypt_empty_string(self):
        """Test decrypting empty string."""
        decrypted = decrypt_api_key("")
        assert decrypted is None

    def test_decrypt_invalid_data(self):
        """Test decrypting invalid data returns None."""
        decrypted = decrypt_api_key("invalid-encrypted-data")
        assert decrypted is None

