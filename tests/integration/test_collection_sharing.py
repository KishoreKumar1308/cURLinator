"""
Integration tests for collection sharing functionality.

Tests cover:
- Sharing collections with other users
- Permission levels (VIEW vs CHAT)
- Visibility settings (PRIVATE vs PUBLIC)
- Access control and authorization
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from curlinator.api.main import app
from curlinator.api.database import get_db, Base
from curlinator.api.db.models import (
    DocumentationCollection,
    CollectionShare,
    CollectionVisibility,
    SharePermission,
)

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sharing.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Create and drop test database for each test."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def owner_user(client):
    """Create owner user and return auth token."""
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "OwnerPass123"}
    )
    assert response.status_code == 201
    data = response.json()
    return {"token": data["access_token"], "user_id": data["user"]["id"], "email": data["user"]["email"]}


@pytest.fixture
def viewer_user(client):
    """Create viewer user and return auth token."""
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "viewer@example.com", "password": "ViewerPass123"}
    )
    assert response.status_code == 201
    data = response.json()
    return {"token": data["access_token"], "user_id": data["user"]["id"], "email": data["user"]["email"]}


@pytest.fixture
def test_collection(owner_user):
    """Create a test collection."""
    db = TestingSessionLocal()
    try:
        collection = DocumentationCollection(
            name="test-api-docs",
            url="https://api.example.com/docs",
            domain="api.example.com",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            pages_crawled=10,
            owner_id=owner_user["user_id"],
            visibility=CollectionVisibility.PRIVATE,
        )
        db.add(collection)
        db.commit()
        db.refresh(collection)
        return {"id": collection.id, "name": collection.name, "owner_id": collection.owner_id}
    finally:
        db.close()


class TestCollectionSharing:
    """Test collection sharing functionality."""
    
    def test_share_collection_with_view_permission(self, client, owner_user, viewer_user, test_collection):
        """Test sharing a collection with VIEW permission."""
        response = client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": viewer_user["email"], "permission": "view"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["user_email"] == viewer_user["email"]
        assert data["permission"] == "view"

    def test_share_collection_with_chat_permission(self, client, owner_user, viewer_user, test_collection):
        """Test sharing a collection with CHAT permission."""
        response = client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": viewer_user["email"], "permission": "chat"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["permission"] == "chat"

    def test_non_owner_cannot_share_collection(self, client, viewer_user, test_collection):
        """Test that non-owners cannot share a collection."""
        response = client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": "other@example.com", "permission": "view"},
            headers={"Authorization": f"Bearer {viewer_user['token']}"}
        )

        assert response.status_code == 404

    def test_list_collection_shares(self, client, owner_user, viewer_user, test_collection):
        """Test listing all shares for a collection."""
        # First share the collection
        client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": viewer_user["email"], "permission": "view"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        # List shares
        response = client.get(
            f"/api/v1/collections/{test_collection['name']}/shares",
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["user_email"] == viewer_user["email"]

    def test_update_share_permission(self, client, owner_user, viewer_user, test_collection):
        """Test updating share permission."""
        # First share with VIEW
        client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": viewer_user["email"], "permission": "view"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        # Update to CHAT
        response = client.patch(
            f"/api/v1/collections/{test_collection['name']}/shares/{viewer_user['email']}",
            json={"permission": "chat"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["permission"] == "chat"

    def test_revoke_share(self, client, owner_user, viewer_user, test_collection):
        """Test revoking a share."""
        # First share the collection
        client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": viewer_user["email"], "permission": "view"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        # Revoke share
        response = client.delete(
            f"/api/v1/collections/{test_collection['name']}/shares/{viewer_user['email']}",
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        assert response.status_code == 204

        # Verify share is gone
        response = client.get(
            f"/api/v1/collections/{test_collection['name']}/shares",
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )
        assert len(response.json()) == 0


class TestCollectionVisibility:
    """Test collection visibility settings."""

    def test_update_collection_to_public(self, client, owner_user, test_collection):
        """Test updating collection visibility to PUBLIC."""
        response = client.patch(
            f"/api/v1/collections/{test_collection['name']}/visibility",
            json={"visibility": "public"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        assert response.status_code == 200

    def test_public_collection_accessible_to_all(self, client, owner_user, viewer_user, test_collection):
        """Test that public collections are accessible to all users."""
        # Set to public
        client.patch(
            f"/api/v1/collections/{test_collection['name']}/visibility",
            json={"visibility": "public"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        # Viewer should be able to access it
        response = client.get(
            f"/api/v1/collections/{test_collection['name']}",
            headers={"Authorization": f"Bearer {viewer_user['token']}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == test_collection["name"]

    def test_private_collection_not_accessible_to_others(self, client, viewer_user, test_collection):
        """Test that private collections are not accessible to non-owners."""
        response = client.get(
            f"/api/v1/collections/{test_collection['name']}",
            headers={"Authorization": f"Bearer {viewer_user['token']}"}
        )

        assert response.status_code == 404


class TestSharedCollectionAccess:
    """Test access control for shared collections."""

    def test_shared_collection_in_list(self, client, owner_user, viewer_user, test_collection):
        """Test that shared collections appear in user's list."""
        # Share with viewer
        client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": viewer_user["email"], "permission": "view"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        # Viewer should see it in their list
        response = client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {viewer_user['token']}"}
        )

        assert response.status_code == 200
        data = response.json()
        collection_names = [c["name"] for c in data]
        assert test_collection["name"] in collection_names

    def test_view_permission_allows_viewing(self, client, owner_user, viewer_user, test_collection):
        """Test that VIEW permission allows viewing collection."""
        # Share with VIEW permission
        client.post(
            f"/api/v1/collections/{test_collection['name']}/share",
            json={"user_email": viewer_user["email"], "permission": "view"},
            headers={"Authorization": f"Bearer {owner_user['token']}"}
        )

        # Viewer should be able to view
        response = client.get(
            f"/api/v1/collections/{test_collection['name']}",
            headers={"Authorization": f"Bearer {viewer_user['token']}"}
        )

        assert response.status_code == 200

