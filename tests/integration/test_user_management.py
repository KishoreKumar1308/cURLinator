"""
Integration tests for user management endpoints.

Tests for:
- Password change (PATCH /auth/password)
- User self-deletion (DELETE /auth/me)
- Admin user management endpoints
- Metrics endpoint security
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import tempfile
import os

from curlinator.api.main import app
from curlinator.api.database import Base, get_db
from curlinator.api.db.models import User, DocumentationCollection
from curlinator.api.auth import get_password_hash, create_access_token, verify_password
from curlinator.config import get_settings


# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("TestPassword123"),
        is_active=True,
        role="user",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def admin_user(db_session):
    """Create an admin user."""
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


@pytest.fixture(scope="function")
def auth_token(test_user):
    """Create an authentication token for the test user."""
    return create_access_token(data={"sub": test_user.id})


@pytest.fixture(scope="function")
def admin_token(admin_user):
    """Create an authentication token for the admin user."""
    return create_access_token(data={"sub": admin_user.id})


# ============================================================================
# Password Change Tests
# ============================================================================

class TestPasswordChange:
    """Test password change endpoint."""
    
    def test_change_password_success(self, client, test_user, auth_token, db_session):
        """Test successful password change."""
        response = client.patch(
            "/api/v1/auth/password",
            json={
                "current_password": "TestPassword123",
                "new_password": "NewPassword456"
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Password changed successfully"
        assert data["user_id"] == test_user.id
        
        # Verify password was actually changed
        db_session.refresh(test_user)
        assert verify_password("NewPassword456", test_user.hashed_password)
    
    def test_change_password_wrong_current_password(self, client, auth_token):
        """Test password change with incorrect current password."""
        response = client.patch(
            "/api/v1/auth/password",
            json={
                "current_password": "WrongPassword",
                "new_password": "NewPassword456"
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 401
        assert "Current password is incorrect" in response.json()["message"]

    def test_change_password_same_as_current(self, client, auth_token):
        """Test password change with new password same as current."""
        response = client.patch(
            "/api/v1/auth/password",
            json={
                "current_password": "TestPassword123",
                "new_password": "TestPassword123"
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 400
        assert "must be different from current password" in response.json()["message"]
    
    def test_change_password_weak_password(self, client, auth_token):
        """Test password change with weak new password."""
        response = client.patch(
            "/api/v1/auth/password",
            json={
                "current_password": "TestPassword123",
                "new_password": "weak"
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_change_password_unauthenticated(self, client):
        """Test password change without authentication."""
        response = client.patch(
            "/api/v1/auth/password",
            json={
                "current_password": "TestPassword123",
                "new_password": "NewPassword456"
            }
        )
        
        assert response.status_code == 403  # No credentials provided


# ============================================================================
# User Self-Deletion Tests
# ============================================================================

class TestUserSelfDeletion:
    """Test user self-deletion endpoint."""
    
    def test_delete_account_success(self, client, test_user, auth_token, db_session):
        """Test successful account deletion."""
        response = client.request(
            "DELETE",
            "/api/v1/auth/me",
            json={"password": "TestPassword123"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 204

        # Verify user was deleted
        deleted_user = db_session.query(User).filter(User.id == test_user.id).first()
        assert deleted_user is None

    def test_delete_account_wrong_password(self, client, auth_token):
        """Test account deletion with incorrect password."""
        response = client.request(
            "DELETE",
            "/api/v1/auth/me",
            json={"password": "WrongPassword"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 401
        assert "Password is incorrect" in response.json()["message"]

    def test_delete_account_unauthenticated(self, client):
        """Test account deletion without authentication."""
        response = client.request(
            "DELETE",
            "/api/v1/auth/me",
            json={"password": "TestPassword123"}
        )

        assert response.status_code == 403  # No credentials provided


# ============================================================================
# Metrics Endpoint Security Tests
# ============================================================================

class TestMetricsEndpointSecurity:
    """Test metrics endpoint security."""
    
    def test_metrics_with_admin_token(self, client, admin_token):
        """Test metrics endpoint with admin token."""
        response = client.get(
            "/metrics",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_with_user_token(self, client, auth_token):
        """Test metrics endpoint with regular user token (should fail)."""
        response = client.get(
            "/metrics",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 403
        assert "Admin access required" in response.json()["message"]
    
    def test_metrics_without_auth(self, client):
        """Test metrics endpoint without authentication."""
        response = client.get("/metrics")

        assert response.status_code == 401


# ============================================================================
# Admin Endpoints Tests
# ============================================================================

class TestAdminListUsers:
    """Test admin list users endpoint."""

    def test_list_users_as_admin(self, client, admin_token, test_user, admin_user):
        """Test listing users as admin."""
        response = client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        users = response.json()
        assert len(users) >= 2  # At least test_user and admin_user

        # Check user structure
        user_emails = [u["email"] for u in users]
        assert "test@example.com" in user_emails
        assert "admin@example.com" in user_emails

    def test_list_users_as_regular_user(self, client, auth_token):
        """Test listing users as regular user (should fail)."""
        response = client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 403
        assert "Admin access required" in response.json()["message"]

    def test_list_users_without_auth(self, client):
        """Test listing users without authentication."""
        response = client.get("/api/v1/admin/users")

        assert response.status_code == 403


class TestAdminGetUser:
    """Test admin get user endpoint."""

    def test_get_user_as_admin(self, client, admin_token, test_user):
        """Test getting user details as admin."""
        response = client.get(
            f"/api/v1/admin/users/{test_user.id}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        user = response.json()
        assert user["email"] == "test@example.com"
        assert user["role"] == "user"
        assert user["is_active"] is True

    def test_get_user_not_found(self, client, admin_token):
        """Test getting non-existent user."""
        response = client.get(
            "/api/v1/admin/users/nonexistent-id",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 404


class TestAdminUpdateUserStatus:
    """Test admin update user status endpoint."""

    def test_deactivate_user_as_admin(self, client, admin_token, test_user, db_session):
        """Test deactivating user as admin."""
        response = client.patch(
            f"/api/v1/admin/users/{test_user.id}/status",
            json={"is_active": False},
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False

        # Verify in database
        db_session.refresh(test_user)
        assert test_user.is_active is False

    def test_admin_cannot_deactivate_self(self, client, admin_token, admin_user):
        """Test that admin cannot deactivate themselves."""
        response = client.patch(
            f"/api/v1/admin/users/{admin_user.id}/status",
            json={"is_active": False},
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 400
        assert "cannot deactivate your own account" in response.json()["message"]


class TestAdminResetPassword:
    """Test admin reset password endpoint."""

    def test_reset_user_password_as_admin(self, client, admin_token, test_user, db_session):
        """Test resetting user password as admin."""
        response = client.patch(
            f"/api/v1/admin/users/{test_user.id}/password",
            json={"new_password": "AdminResetPass123"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Password reset successfully"

        # Verify password was changed
        db_session.refresh(test_user)
        assert verify_password("AdminResetPass123", test_user.hashed_password)


class TestAdminDeleteUser:
    """Test admin delete user endpoint."""

    def test_delete_user_as_admin(self, client, admin_token, test_user, db_session):
        """Test deleting user as admin."""
        user_id = test_user.id

        response = client.delete(
            f"/api/v1/admin/users/{user_id}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 204

        # Verify user was deleted
        deleted_user = db_session.query(User).filter(User.id == user_id).first()
        assert deleted_user is None

    def test_admin_cannot_delete_self(self, client, admin_token, admin_user):
        """Test that admin cannot delete themselves."""
        response = client.delete(
            f"/api/v1/admin/users/{admin_user.id}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 400
        assert "cannot delete your own account" in response.json()["message"]

    def test_delete_user_not_found(self, client, admin_token):
        """Test deleting non-existent user."""
        response = client.delete(
            "/api/v1/admin/users/nonexistent-id",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 404

