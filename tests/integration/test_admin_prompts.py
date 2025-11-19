"""
Integration tests for admin prompt customization endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from curlinator.api.db.models import User, UserSettings, SystemConfig
from curlinator.api.auth import get_password_hash, create_access_token


@pytest.fixture
def admin_user(db_session):
    """Create an admin user directly in the database."""
    user = User(
        email="admin@example.com",
        hashed_password=get_password_hash("AdminPassword123"),
        is_active=True,
        role="admin",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user):
    """Create an authentication token for the admin user."""
    return create_access_token(data={"sub": admin_user.id})


@pytest.fixture
def auth_token(test_user):
    """Create an authentication token for a regular (non-admin) user."""
    return create_access_token(data={"sub": test_user.id})


class TestAdminPromptEndpoints:
    """Test admin prompt customization endpoints."""

    def test_update_system_prompt_success(
        self, client: TestClient, admin_token: str, db_session: Session
    ):
        """Test updating system-wide prompt."""
        response = client.patch(
            "/api/v1/admin/system-prompt",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "prompt": "You are a helpful API assistant specialized in REST APIs.",
                "description": "Custom system prompt for testing"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "System-wide prompt updated successfully"
        assert "You are a helpful API assistant" in data["prompt_preview"]
        assert "updated_at" in data
        assert "updated_by" in data

        # Verify in database
        system_prompt = db_session.query(SystemConfig).filter(
            SystemConfig.config_key == "system_prompt"
        ).first()
        assert system_prompt is not None
        assert system_prompt.config_value == "You are a helpful API assistant specialized in REST APIs."
        assert system_prompt.description == "Custom system prompt for testing"

    def test_update_system_prompt_validation_error(
        self, client: TestClient, admin_token: str
    ):
        """Test prompt validation (empty prompt)."""
        response = client.patch(
            "/api/v1/admin/system-prompt",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"prompt": "   "}  # Only whitespace
        )

        assert response.status_code == 422
        assert "Prompt cannot be empty" in str(response.json())

    def test_update_system_prompt_too_long(
        self, client: TestClient, admin_token: str
    ):
        """Test prompt validation (exceeds max length)."""
        long_prompt = "A" * 10001  # Exceeds 10,000 char limit
        response = client.patch(
            "/api/v1/admin/system-prompt",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"prompt": long_prompt}
        )

        assert response.status_code == 422
        # Pydantic's built-in validation message
        assert "at most 10000 characters" in str(response.json())

    def test_update_system_prompt_requires_admin(
        self, client: TestClient, auth_token: str
    ):
        """Test that non-admin users cannot update system prompt."""
        response = client.patch(
            "/api/v1/admin/system-prompt",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"prompt": "Test prompt"}
        )

        assert response.status_code == 403
        response_data = response.json()
        assert "Admin access required" in response_data.get("detail", str(response_data))

    def test_update_user_prompt_success(
        self, client: TestClient, admin_token: str, test_user: User, db_session: Session
    ):
        """Test setting per-user custom prompt."""
        response = client.patch(
            f"/api/v1/admin/users/{test_user.id}/prompt",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "prompt": "You are a concise API assistant. Keep responses brief.",
                "variant_name": "concise_variant"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "Custom prompt set for user" in data["message"]
        assert "You are a concise API assistant" in data["prompt_preview"]

        # Verify in database
        user_settings = db_session.query(UserSettings).filter(
            UserSettings.user_id == test_user.id
        ).first()
        assert user_settings is not None
        assert user_settings.custom_system_prompt == "You are a concise API assistant. Keep responses brief."
        assert user_settings.prompt_variant_name == "concise_variant"
        assert user_settings.prompt_updated_at is not None

    def test_update_user_prompt_user_not_found(
        self, client: TestClient, admin_token: str
    ):
        """Test setting prompt for non-existent user."""
        response = client.patch(
            "/api/v1/admin/users/nonexistent-user-id/prompt",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"prompt": "Test prompt"}
        )

        assert response.status_code == 404
        response_data = response.json()
        assert "not found" in response_data.get("detail", str(response_data))

    def test_update_user_prompt_requires_admin(
        self, client: TestClient, auth_token: str, test_user: User
    ):
        """Test that non-admin users cannot update user prompts."""
        response = client.patch(
            f"/api/v1/admin/users/{test_user.id}/prompt",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"prompt": "Test prompt"}
        )

        assert response.status_code == 403
        response_data = response.json()
        assert "Admin access required" in response_data.get("detail", str(response_data))

    def test_get_prompts_overview_with_system_prompt(
        self, client: TestClient, admin_token: str, db_session: Session
    ):
        """Test getting prompts overview with system prompt configured."""
        # Set system prompt
        system_prompt = SystemConfig(
            config_key="system_prompt",
            config_value="Test system prompt",
            description="Test description"
        )
        db_session.add(system_prompt)
        db_session.commit()

        response = client.get(
            "/api/v1/admin/prompts",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["system_prompt"]["prompt"] == "Test system prompt"
        assert data["system_prompt"]["description"] == "Test description"
        assert data["system_prompt"]["is_default"] is False
        assert "users_with_custom_prompts" in data
        assert "total_users_with_custom_prompts" in data

    def test_get_prompts_overview_default_prompt(
        self, client: TestClient, admin_token: str
    ):
        """Test getting prompts overview with no system prompt (uses default)."""
        response = client.get(
            "/api/v1/admin/prompts",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["system_prompt"]["is_default"] is True
        assert "cURLinator" in data["system_prompt"]["prompt"]
        assert data["system_prompt"]["updated_at"] is None

    def test_get_prompts_overview_with_custom_user_prompts(
        self, client: TestClient, admin_token: str, test_user: User, db_session: Session
    ):
        """Test prompts overview includes users with custom prompts."""
        # Create user settings with custom prompt
        user_settings = UserSettings(
            user_id=test_user.id,
            custom_system_prompt="Custom prompt for testing",
            prompt_variant_name="test_variant"
        )
        db_session.add(user_settings)
        db_session.commit()

        response = client.get(
            "/api/v1/admin/prompts",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_users_with_custom_prompts"] == 1
        assert len(data["users_with_custom_prompts"]) == 1
        
        user_info = data["users_with_custom_prompts"][0]
        assert user_info["user_id"] == test_user.id
        assert user_info["user_email"] == test_user.email
        assert user_info["variant_name"] == "test_variant"
        assert "Custom prompt for testing" in user_info["prompt_preview"]

    def test_get_prompts_overview_requires_admin(
        self, client: TestClient, auth_token: str
    ):
        """Test that non-admin users cannot view prompts overview."""
        response = client.get(
            "/api/v1/admin/prompts",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 403
        response_data = response.json()
        assert "Admin access required" in response_data.get("detail", str(response_data))

    def test_delete_user_prompt_success(
        self, client: TestClient, admin_token: str, test_user: User, db_session: Session
    ):
        """Test deleting user custom prompt."""
        # Create user settings with custom prompt
        user_settings = UserSettings(
            user_id=test_user.id,
            custom_system_prompt="Custom prompt to delete",
            prompt_variant_name="delete_variant"
        )
        db_session.add(user_settings)
        db_session.commit()

        response = client.delete(
            f"/api/v1/admin/users/{test_user.id}/prompt",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "Custom prompt removed" in data["message"]
        assert data["user_id"] == test_user.id
        assert data["user_email"] == test_user.email

        # Verify in database
        db_session.refresh(user_settings)
        assert user_settings.custom_system_prompt is None
        assert user_settings.prompt_variant_name is None
        assert user_settings.prompt_updated_at is None

    def test_delete_user_prompt_user_not_found(
        self, client: TestClient, admin_token: str
    ):
        """Test deleting prompt for non-existent user."""
        response = client.delete(
            "/api/v1/admin/users/nonexistent-user-id/prompt",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 404
        response_data = response.json()
        assert "not found" in response_data.get("detail", str(response_data))

    def test_delete_user_prompt_no_custom_prompt(
        self, client: TestClient, admin_token: str, test_user: User, db_session: Session
    ):
        """Test deleting prompt when user has no custom prompt."""
        # Ensure user has no custom prompt
        user_settings = db_session.query(UserSettings).filter(
            UserSettings.user_id == test_user.id
        ).first()
        if user_settings:
            user_settings.custom_system_prompt = None
            db_session.commit()

        response = client.delete(
            f"/api/v1/admin/users/{test_user.id}/prompt",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 404
        response_data = response.json()
        assert "does not have a custom prompt" in response_data.get("detail", str(response_data))

    def test_delete_user_prompt_requires_admin(
        self, client: TestClient, auth_token: str, test_user: User
    ):
        """Test that non-admin users cannot delete user prompts."""
        response = client.delete(
            f"/api/v1/admin/users/{test_user.id}/prompt",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 403
        response_data = response.json()
        assert "Admin access required" in response_data.get("detail", str(response_data))

