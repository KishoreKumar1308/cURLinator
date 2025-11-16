"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import chromadb
import tempfile
import os

from curlinator.api.main import app
from curlinator.api.database import Base, get_db
from curlinator.api.db.models import User, DocumentationCollection
from curlinator.api.auth import get_password_hash, create_access_token
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
        hashed_password=get_password_hash("testpassword"),
        is_active=True,
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
def test_collection(db_session, test_user):
    """Create a test collection."""
    collection = DocumentationCollection(
        name="test_collection",
        url="https://example.com",
        domain="example.com",
        pages_crawled=5,
        owner_id=test_user.id,
        embedding_provider="local",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    db_session.add(collection)
    db_session.commit()
    db_session.refresh(collection)
    return collection


class TestChatEndpointAuthentication:
    """Test chat endpoint authentication."""
    
    def test_chat_without_authentication(self, client):
        """Test that chat endpoint requires authentication."""
        response = client.post(
            "/api/v1/chat",
            json={
                "collection_name": "test_collection",
                "message": "What is this API about?"
            }
        )

        assert response.status_code == 403
        data = response.json()
        # Check for error in either "error" or "message" field (structured error response)
        error_text = data.get("error", "") + data.get("message", "")
        assert "Not authenticated" in error_text
    
    def test_chat_with_invalid_token(self, client):
        """Test chat endpoint with invalid token."""
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": "Bearer invalid_token"},
            json={
                "collection_name": "test_collection",
                "message": "What is this API about?"
            }
        )
        
        assert response.status_code == 401
    
    def test_chat_with_valid_token_nonexistent_collection(self, client, auth_token):
        """Test chat endpoint with valid token but nonexistent collection."""
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "collection_name": "nonexistent_collection",
                "message": "What is this API about?"
            }
        )

        assert response.status_code == 404
        response_data = response.json()
        # Check the structured error format
        assert response_data.get("error") == "RESOURCE_NOT_FOUND"
        assert "Collection" in response_data.get("message", "")
        assert "not found" in response_data.get("message", "")


class TestCollectionListEndpoint:
    """Test collection listing endpoint."""
    
    def test_list_collections_without_authentication(self, client):
        """Test that list collections requires authentication."""
        response = client.get("/api/v1/collections")

        assert response.status_code == 403
    
    def test_list_collections_empty(self, client, auth_token):
        """Test listing collections when user has none."""
        response = client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        assert response.json() == []
    
    def test_list_collections_with_data(self, client, auth_token, test_collection):
        """Test listing collections when user has collections."""
        response = client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        collections = response.json()
        assert len(collections) == 1
        assert collections[0]["name"] == "test_collection"
        assert collections[0]["url"] == "https://example.com"
        assert collections[0]["embedding_provider"] == "local"
        assert collections[0]["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert collections[0]["pages_crawled"] == 5
    
    def test_list_collections_pagination(self, client, auth_token, db_session, test_user):
        """Test collection listing with pagination."""
        # Create multiple collections
        for i in range(5):
            collection = DocumentationCollection(
                name=f"test_collection_{i}",
                url=f"https://example{i}.com",
                domain=f"example{i}.com",
                pages_crawled=i,
                owner_id=test_user.id,
                embedding_provider="local",
                embedding_model="BAAI/bge-small-en-v1.5",
            )
            db_session.add(collection)
        db_session.commit()
        
        # Test with limit
        response = client.get(
            "/api/v1/collections?limit=3",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 3
        
        # Test with skip
        response = client.get(
            "/api/v1/collections?skip=2&limit=3",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 3


class TestCollectionDetailEndpoint:
    """Test collection detail endpoint."""
    
    def test_get_collection_without_authentication(self, client):
        """Test that get collection requires authentication."""
        response = client.get("/api/v1/collections/test_collection")

        assert response.status_code == 403
    
    def test_get_nonexistent_collection(self, client, auth_token):
        """Test getting a collection that doesn't exist."""
        response = client.get(
            "/api/v1/collections/nonexistent",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 404
        response_data = response.json()
        # Check the structured error format
        assert response_data.get("error") == "RESOURCE_NOT_FOUND"
        assert "Collection" in response_data.get("message", "")
        assert "not found" in response_data.get("message", "")
    
    def test_get_collection_success(self, client, auth_token, test_collection):
        """Test successfully getting collection details."""
        response = client.get(
            f"/api/v1/collections/{test_collection.name}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == test_collection.name
        assert data["url"] == test_collection.url
        assert data["domain"] == test_collection.domain
        assert data["embedding_provider"] == "local"
        assert data["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert data["pages_crawled"] == 5
        assert "document_count" in data


class TestCollectionDeleteEndpoint:
    """Test collection deletion endpoint."""
    
    def test_delete_collection_without_authentication(self, client):
        """Test that delete collection requires authentication."""
        response = client.delete("/api/v1/collections/test_collection")

        assert response.status_code == 403
    
    def test_delete_nonexistent_collection(self, client, auth_token):
        """Test deleting a collection that doesn't exist."""
        response = client.delete(
            "/api/v1/collections/nonexistent",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 404
    
    def test_delete_collection_success(self, client, auth_token, test_collection, db_session):
        """Test successfully deleting a collection."""
        collection_name = test_collection.name
        
        response = client.delete(
            f"/api/v1/collections/{collection_name}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 204

        # Verify collection was deleted from database
        deleted_collection = db_session.query(DocumentationCollection).filter(
            DocumentationCollection.name == collection_name
        ).first()
        assert deleted_collection is None


class TestCrawlValidationErrors:
    """Test crawl endpoint validation error handling."""

    def test_crawl_with_invalid_url_scheme(self, client, auth_token):
        """Test crawl with FTP URL returns 422."""
        response = client.post(
            "/api/v1/crawl",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"url": "ftp://example.com/docs"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "Invalid URL" in data["error"] or "url" in str(data).lower()

    def test_crawl_with_localhost_url(self, client, auth_token):
        """Test crawl with localhost URL returns 422."""
        response = client.post(
            "/api/v1/crawl",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"url": "http://localhost:8000/docs"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        # Check for either structured error or validation error
        if "message" in data:
            assert "localhost" in data["message"].lower()
        else:
            assert "localhost" in str(data).lower()

    def test_crawl_with_private_ip(self, client, auth_token):
        """Test crawl with private IP returns 422."""
        response = client.post(
            "/api/v1/crawl",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"url": "http://192.168.1.1/docs"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        # Check for either structured error or validation error
        if "message" in data:
            assert "private" in data["message"].lower() or "192.168" in data["message"]
        else:
            assert "private" in str(data).lower() or "192.168" in str(data)

    def test_crawl_with_cloud_metadata_endpoint(self, client, auth_token):
        """Test crawl with cloud metadata endpoint returns 422."""
        response = client.post(
            "/api/v1/crawl",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"url": "http://169.254.169.254/latest/meta-data"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        # Check for either structured error or validation error
        if "message" in data:
            assert "169.254.169.254" in data["message"] or "metadata" in data["message"].lower()
        else:
            assert "169.254.169.254" in str(data) or "metadata" in str(data).lower()

    def test_crawl_with_invalid_max_pages(self, client, auth_token):
        """Test crawl with max_pages > 1000 returns 422."""
        response = client.post(
            "/api/v1/crawl",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "url": "https://example.com/docs",
                "max_pages": 2000
            }
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data

    def test_crawl_without_authentication(self, client):
        """Test crawl without auth token returns 403."""
        response = client.post(
            "/api/v1/crawl",
            json={"url": "https://example.com/docs"}
        )
        assert response.status_code == 403


class TestAuthErrorHandling:
    """Test authentication endpoint error handling."""

    def test_register_with_weak_password(self, client):
        """Test registration with weak password returns 422."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "weak"  # Too short, no uppercase, no digit
            }
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data

    def test_register_with_no_uppercase(self, client):
        """Test registration with password missing uppercase returns 422."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "lowercase123"  # No uppercase
            }
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        # Check for password validation error
        if "details" in data:
            error_msg = str(data["details"])
            assert "uppercase" in error_msg.lower()

    def test_register_with_no_digit(self, client):
        """Test registration with password missing digit returns 422."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "NoDigitHere"  # No digit
            }
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        # Check for password validation error
        if "details" in data:
            error_msg = str(data["details"])
            assert "digit" in error_msg.lower()

    def test_register_with_duplicate_email(self, client, test_user):
        """Test registration with existing email returns 400."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "password": "ValidPass123"
            }
        )
        assert response.status_code in [400, 409]  # Either 400 or 409 is acceptable
        data = response.json()
        assert "error" in data
        # Check for duplicate email error
        if "message" in data:
            assert "already" in data["message"].lower() or "registered" in data["message"].lower()
        else:
            assert "already" in data["error"].lower() or "registered" in data["error"].lower()

    def test_login_with_invalid_credentials(self, client):
        """Test login with wrong password returns 401."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "WrongPass123"
            }
        )
        assert response.status_code == 401
        data = response.json()
        assert "error" in data

    def test_login_with_invalid_email_format(self, client):
        """Test login with invalid email format returns 422."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "not-an-email",
                "password": "ValidPass123"
            }
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data


class TestRateLimitHeaders:
    """Test rate limit headers are present in responses."""

    def test_auth_endpoint_has_rate_limit_headers(self, client):
        """Test that auth endpoints include rate limit headers."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123"
            }
        )

        # Check that rate limit headers are present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # Verify header values are reasonable
        assert int(response.headers["X-RateLimit-Limit"]) == 10  # 10/minute for auth
        assert int(response.headers["X-RateLimit-Remaining"]) >= 0
        assert int(response.headers["X-RateLimit-Reset"]) > 0

    def test_register_endpoint_has_rate_limit_headers(self, client):
        """Test that register endpoint includes rate limit headers."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "ValidPass123"
            }
        )

        # Check that rate limit headers are present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # Verify header values
        assert int(response.headers["X-RateLimit-Limit"]) == 10  # 10/minute for auth

    def test_non_rate_limited_endpoint_no_headers(self, client, test_user, auth_token):
        """Test that non-rate-limited endpoints don't have rate limit headers."""
        response = client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        # Collections endpoint is not rate-limited, so no headers
        assert "X-RateLimit-Limit" not in response.headers

