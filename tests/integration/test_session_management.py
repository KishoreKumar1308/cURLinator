"""
Integration tests for conversation history persistence and session management.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from curlinator.api.main import app
from curlinator.api.database import Base, get_db
from curlinator.api.db.models import User, DocumentationCollection, ChatSession, ChatMessage


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sessions.db"
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
def test_user(client):
    """Create a test user and return auth token."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "TestPass123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    return {
        "token": data["access_token"],
        "user_id": data["user"]["id"],
        "email": data["user"]["email"]
    }


@pytest.fixture
def test_collection(test_user):
    """Create a test collection in the database."""
    db = TestingSessionLocal()
    try:
        collection = DocumentationCollection(
            name="test_collection",
            url="https://example.com/docs",
            domain="example.com",
            pages_crawled=10,
            owner_id=test_user["user_id"],
            embedding_provider="local",
            embedding_model="BAAI/bge-small-en-v1.5"
        )
        db.add(collection)
        db.commit()
        db.refresh(collection)
        return {
            "id": collection.id,
            "name": collection.name
        }
    finally:
        db.close()


class TestChatWithSessionPersistence:
    """Test chat endpoint with session persistence."""

    def test_chat_creates_new_session_when_no_session_id(self, client, test_user, test_collection):
        """Test that chat creates a new session when session_id is not provided."""
        # Note: This test will fail because we don't have a real Chroma collection
        # We're testing the session creation logic, not the actual chat functionality
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            json={
                "collection_name": test_collection["name"],
                "message": "Hello, how do I get started?"
            }
        )

        # We expect this to fail at the ChatAgent loading stage (404)
        # because we don't have a real Chroma collection
        # But we can verify the session was created in the database
        assert response.status_code == 404
        error_data = response.json()
        assert "Collection not found" in error_data["message"] or "vector store" in error_data["message"].lower()

        # Verify session was created in database
        db = TestingSessionLocal()
        try:
            sessions = db.query(ChatSession).filter(
                ChatSession.user_id == test_user["user_id"],
                ChatSession.collection_id == test_collection["id"]
            ).all()
            # Session should be created before the ChatAgent error
            assert len(sessions) == 1
        finally:
            db.close()

    def test_chat_uses_existing_session_when_session_id_provided(self, client, test_user, test_collection):
        """Test that chat uses existing session when session_id is provided."""
        # Create a session manually
        db = TestingSessionLocal()
        try:
            session = ChatSession(
                user_id=test_user["user_id"],
                collection_id=test_collection["id"]
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.id
        finally:
            db.close()

        # Try to chat with the session
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            json={
                "collection_name": test_collection["name"],
                "message": "Hello again!",
                "session_id": session_id
            }
        )

        # Should fail at ChatAgent loading (no real Chroma collection)
        assert response.status_code == 404

        # Verify no new session was created
        db = TestingSessionLocal()
        try:
            sessions = db.query(ChatSession).filter(
                ChatSession.user_id == test_user["user_id"]
            ).all()
            assert len(sessions) == 1
            assert sessions[0].id == session_id
        finally:
            db.close()

    def test_chat_rejects_invalid_session_id(self, client, test_user, test_collection):
        """Test that chat rejects invalid session_id."""
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            json={
                "collection_name": test_collection["name"],
                "message": "Hello!",
                "session_id": "invalid-session-id"
            }
        )

        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["message"].lower() or "don't have access" in error_data["message"].lower()

    def test_chat_rejects_session_from_different_user(self, client, test_user, test_collection):
        """Test that chat rejects session_id from a different user."""
        # Create another user
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "other@example.com",
                "password": "OtherPass123"
            }
        )
        assert response.status_code == 201
        other_user = response.json()

        # Create a session for the other user
        db = TestingSessionLocal()
        try:
            session = ChatSession(
                user_id=other_user["user"]["id"],
                collection_id=test_collection["id"]
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.id
        finally:
            db.close()

        # Try to use the session with the first user's token
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            json={
                "collection_name": test_collection["name"],
                "message": "Hello!",
                "session_id": session_id
            }
        )

        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["message"].lower() or "don't have access" in error_data["message"].lower()


class TestSessionManagementEndpoints:
    """Test session management endpoints."""

    def test_list_sessions_empty(self, client, test_user):
        """Test listing sessions when user has no sessions."""
        response = client.get(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 200
        assert response.json() == []

    def test_list_sessions_with_data(self, client, test_user, test_collection):
        """Test listing sessions with data."""
        # Create sessions manually
        db = TestingSessionLocal()
        try:
            session1 = ChatSession(
                user_id=test_user["user_id"],
                collection_id=test_collection["id"]
            )
            session2 = ChatSession(
                user_id=test_user["user_id"],
                collection_id=test_collection["id"]
            )
            db.add(session1)
            db.add(session2)
            db.commit()
            db.refresh(session1)
            db.refresh(session2)

            # Store IDs before closing session
            session1_id = session1.id
            session2_id = session2.id

            # Add messages to session1
            msg1 = ChatMessage(
                session_id=session1.id,
                role="user",
                content="Hello"
            )
            msg2 = ChatMessage(
                session_id=session1.id,
                role="assistant",
                content="Hi there!"
            )
            db.add(msg1)
            db.add(msg2)
            db.commit()
        finally:
            db.close()

        response = client.get(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 2

        # Find session1 in the response
        session1_data = next(s for s in sessions if s["id"] == session1_id)
        assert session1_data["collection_name"] == test_collection["name"]
        assert session1_data["message_count"] == 2

        # Find session2 in the response
        session2_data = next(s for s in sessions if s["id"] == session2_id)
        assert session2_data["message_count"] == 0


    def test_get_session_detail(self, client, test_user, test_collection):
        """Test getting session details with messages."""
        # Create session with messages
        db = TestingSessionLocal()
        try:
            session = ChatSession(
                user_id=test_user["user_id"],
                collection_id=test_collection["id"]
            )
            db.add(session)
            db.commit()
            db.refresh(session)

            # Add messages
            msg1 = ChatMessage(
                session_id=session.id,
                role="user",
                content="How do I authenticate?"
            )
            msg2 = ChatMessage(
                session_id=session.id,
                role="assistant",
                content="Use the /auth/login endpoint",
                curl_command="curl -X POST https://api.example.com/auth/login"
            )
            db.add(msg1)
            db.add(msg2)
            db.commit()
            db.refresh(msg1)
            db.refresh(msg2)
            session_id = session.id
        finally:
            db.close()

        response = client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["collection_name"] == test_collection["name"]
        assert len(data["messages"]) == 2

        # Check first message
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "How do I authenticate?"
        assert data["messages"][0]["curl_command"] is None

        # Check second message
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Use the /auth/login endpoint"
        assert data["messages"][1]["curl_command"] == "curl -X POST https://api.example.com/auth/login"

    def test_get_session_not_found(self, client, test_user):
        """Test getting non-existent session."""
        response = client.get(
            "/api/v1/sessions/invalid-session-id",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["message"].lower() or "don't have access" in error_data["message"].lower()

    def test_get_session_from_different_user(self, client, test_user, test_collection):
        """Test that user cannot access another user's session."""
        # Create another user
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "other@example.com",
                "password": "OtherPass123"
            }
        )
        assert response.status_code == 201
        other_user = response.json()

        # Create a session for the other user
        db = TestingSessionLocal()
        try:
            session = ChatSession(
                user_id=other_user["user"]["id"],
                collection_id=test_collection["id"]
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.id
        finally:
            db.close()

        # Try to access with first user's token
        response = client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["message"].lower() or "don't have access" in error_data["message"].lower()

    def test_delete_session(self, client, test_user, test_collection):
        """Test deleting a session."""
        # Create session with messages
        db = TestingSessionLocal()
        try:
            session = ChatSession(
                user_id=test_user["user_id"],
                collection_id=test_collection["id"]
            )
            db.add(session)
            db.commit()
            db.refresh(session)

            # Add messages
            msg = ChatMessage(
                session_id=session.id,
                role="user",
                content="Test message"
            )
            db.add(msg)
            db.commit()
            session_id = session.id
        finally:
            db.close()

        # Delete the session
        response = client.delete(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Session deleted successfully"
        assert response.json()["session_id"] == session_id

        # Verify session is deleted
        db = TestingSessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            assert session is None

            # Verify messages are also deleted (cascade)
            messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
            assert len(messages) == 0
        finally:
            db.close()

    def test_delete_session_not_found(self, client, test_user):
        """Test deleting non-existent session."""
        response = client.delete(
            "/api/v1/sessions/invalid-session-id",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["message"].lower() or "don't have access" in error_data["message"].lower()


    def test_reset_session(self, client, test_user, test_collection):
        """Test resetting a session (clearing messages)."""
        # Create session with messages
        db = TestingSessionLocal()
        try:
            session = ChatSession(
                user_id=test_user["user_id"],
                collection_id=test_collection["id"]
            )
            db.add(session)
            db.commit()
            db.refresh(session)

            # Add messages
            for i in range(5):
                msg = ChatMessage(
                    session_id=session.id,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}"
                )
                db.add(msg)
            db.commit()
            session_id = session.id
        finally:
            db.close()

        # Reset the session
        response = client.post(
            f"/api/v1/sessions/{session_id}/reset",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Session reset successfully"
        assert response.json()["session_id"] == session_id
        assert response.json()["messages_deleted"] == 5

        # Verify session still exists but messages are deleted
        db = TestingSessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            assert session is not None

            messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
            assert len(messages) == 0
        finally:
            db.close()

    def test_reset_session_not_found(self, client, test_user):
        """Test resetting non-existent session."""
        response = client.post(
            "/api/v1/sessions/invalid-session-id/reset",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["message"].lower() or "don't have access" in error_data["message"].lower()

    def test_session_requires_authentication(self, client):
        """Test that session endpoints require authentication."""
        # List sessions
        response = client.get("/api/v1/sessions")
        assert response.status_code == 403

        # Get session
        response = client.get("/api/v1/sessions/some-id")
        assert response.status_code == 403

        # Delete session
        response = client.delete("/api/v1/sessions/some-id")
        assert response.status_code == 403

        # Reset session
        response = client.post("/api/v1/sessions/some-id/reset")
        assert response.status_code == 403

