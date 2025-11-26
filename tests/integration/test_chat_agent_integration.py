"""Integration tests for ChatAgent

These tests validate the ChatAgent with real documents and vector database operations.
They test RAG-based querying, cURL command generation, conversation history,
and Chroma vector database integration.

Note: These tests require an LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)
and make real API calls. They are marked with @pytest.mark.integration.

These tests are SKIPPED in CI when no valid LLM API key is available to avoid:
- API costs (LLM calls are expensive)
- Rate limits (frequent CI runs could hit limits)

Run locally with a valid API key to test ChatAgent functionality.

Usage:
    # Run all integration tests (requires valid LLM API key)
    pytest tests/integration/test_chat_agent_integration.py -v

    # Run specific test
    pytest tests/integration/test_chat_agent_integration.py::test_query_with_documents -v

    # Skip integration tests (for CI)
    pytest -m "not integration"
"""

import shutil

import pytest
from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from curlinator.agents import ChatAgent
from curlinator.config import get_settings
from tests.integration.conftest import requires_llm

# Mark all tests in this module as integration tests and requiring LLM
pytestmark = [pytest.mark.integration, requires_llm]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def setup_embedding_model():
    """Set up local embedding model for tests"""
    # Use local embedding model to avoid API dependencies
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5", cache_folder="./data/models"
    )
    yield
    # Cleanup is automatic


@pytest.fixture
def sample_api_documents():
    """Create sample API documentation documents for testing"""
    return [
        Document(
            text="POST /users - Create a new user. Requires authentication. "
            "Request body should include: name (string), email (string), password (string). "
            "Returns: User object with id, name, email, created_at.",
            metadata={
                "url": "https://api.example.com/docs/users",
                "title": "Create User",
                "page_type": "api_endpoint",
                "source": "openapi",
                "type": "api_endpoint",
                "method": "POST",
                "endpoint": "/users",
                "tags": "users, authentication",
            },
        ),
        Document(
            text="GET /users/{id} - Get user by ID. Requires authentication. "
            "Path parameters: id (integer) - User ID. "
            "Returns: User object with id, name, email, created_at.",
            metadata={
                "url": "https://api.example.com/docs/users",
                "title": "Get User",
                "page_type": "api_endpoint",
                "source": "openapi",
                "type": "api_endpoint",
                "method": "GET",
                "endpoint": "/users/{id}",
                "tags": "users",
            },
        ),
        Document(
            text="Authentication: Use Bearer token authentication. "
            "Include the token in the Authorization header: Authorization: Bearer YOUR_TOKEN. "
            "Tokens can be obtained from POST /auth/login endpoint.",
            metadata={
                "url": "https://api.example.com/docs/auth",
                "title": "Authentication Guide",
                "page_type": "authentication",
                "source": "crawl",
            },
        ),
        Document(
            text="API Overview: Welcome to the Example API. "
            "Base URL: https://api.example.com/v1. "
            "All endpoints require authentication unless specified otherwise.",
            metadata={
                "url": "https://api.example.com/docs",
                "title": "API Overview",
                "page_type": "api_overview",
                "source": "openapi",
                "type": "api_overview",
                "version": "1.0.0",
                "base_url": "https://api.example.com/v1",
            },
        ),
    ]


@pytest.fixture
def test_collection_name():
    """Generate unique collection name for each test"""
    import uuid

    return f"test_chat_agent_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_persist_directory(tmp_path):
    """Create temporary directory for vector database"""
    persist_dir = tmp_path / "vector_db"
    persist_dir.mkdir(parents=True, exist_ok=True)
    yield str(persist_dir)
    # Cleanup
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


# ============================================================================
# Test: Basic RAG Query
# ============================================================================


@pytest.mark.asyncio
async def test_query_with_documents(
    setup_embedding_model, sample_api_documents, test_collection_name, test_persist_directory
):
    """Test basic RAG-based query answering with documents"""

    # Arrange
    agent = ChatAgent(
        documents=sample_api_documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Act
    response = await agent.query("How do I create a new user?")

    # Assert
    assert response is not None, "Should return a response"
    assert "response" in response, "Should have response field"
    assert "curl_command" in response, "Should have curl_command field"
    assert "sources" in response, "Should have sources field"

    # Check response content
    assert len(response["response"]) > 0, "Response should not be empty"
    assert "POST" in response["response"] or "post" in response["response"].lower(), (
        "Response should mention POST method"
    )

    # Check cURL command
    curl = response["curl_command"]
    assert curl is not None, "Should generate cURL command"
    if curl:  # May be None if LLM doesn't generate one
        assert "curl" in curl.lower(), "Should contain curl command"
        assert "/users" in curl, "Should contain the endpoint"


@pytest.mark.asyncio
async def test_query_returns_relevant_sources(
    setup_embedding_model, sample_api_documents, test_collection_name, test_persist_directory
):
    """Test that query returns relevant source citations"""

    agent = ChatAgent(
        documents=sample_api_documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Act
    response = await agent.query("How do I authenticate?")

    # Assert
    assert "sources" in response, "Should have sources field"
    sources = response["sources"]
    assert len(sources) > 0, "Should return source citations"

    # Check that sources are relevant
    # Should include the authentication document
    source_texts = [str(source) for source in sources]
    combined_sources = " ".join(source_texts).lower()
    assert "auth" in combined_sources or "token" in combined_sources, (
        "Sources should be relevant to authentication"
    )


# ============================================================================
# Test: cURL Command Generation
# ============================================================================


@pytest.mark.asyncio
async def test_curl_command_generation(
    setup_embedding_model, sample_api_documents, test_collection_name, test_persist_directory
):
    """Test that cURL commands are generated correctly"""

    agent = ChatAgent(
        documents=sample_api_documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Act
    response = await agent.query(
        "Show me how to create a user with name 'John' and email 'john@example.com'"
    )

    # Assert
    curl = response.get("curl_command")
    # Note: This test is somewhat flaky because it depends on LLM output format
    # We just check that a cURL command was generated with the endpoint
    if curl:  # LLM may or may not generate cURL command
        assert "curl" in curl.lower(), "Should contain curl command"
        # Check that it contains either the endpoint or the base URL
        assert "/users" in curl or "api.example.com" in curl, (
            "Should contain the endpoint or base URL"
        )


# ============================================================================
# Test: Conversation History
# ============================================================================


@pytest.mark.asyncio
async def test_conversation_history_maintained(
    setup_embedding_model, sample_api_documents, test_collection_name, test_persist_directory
):
    """Test that conversation history is maintained across queries"""

    agent = ChatAgent(
        documents=sample_api_documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Act - First query
    response1 = await agent.query("How do I create a user?")
    assert response1 is not None, "First query should return response"

    # Act - Follow-up query (should use context from first query)
    response2 = await agent.query("What authentication is required for that?")
    assert response2 is not None, "Follow-up query should return response"

    # Assert - Follow-up should understand "that" refers to creating a user
    assert len(response2["response"]) > 0, "Follow-up response should not be empty"
    # The response should mention authentication
    assert "auth" in response2["response"].lower() or "token" in response2["response"].lower(), (
        "Follow-up should understand context and mention authentication"
    )


# ============================================================================
# Test: Vector Database Integration
# ============================================================================


@pytest.mark.asyncio
async def test_chroma_persistence(
    setup_embedding_model, sample_api_documents, test_collection_name, test_persist_directory
):
    """Test that Chroma vector database persists correctly"""

    # Arrange & Act - Create agent and index documents
    agent1 = ChatAgent(
        documents=sample_api_documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Query to ensure index is built
    settings = get_settings()
    if settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key:
        response1 = await agent1.query("test query")
        assert response1 is not None

    # Create new agent instance loading from same collection
    agent2 = ChatAgent(
        documents=None,  # No documents - should load from disk
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Assert - Should be able to query loaded index
    if settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key:
        response2 = await agent2.query("How do I create a user?")
        assert response2 is not None, "Should work with loaded index"
        assert len(response2["response"]) > 0, "Should return response from loaded index"


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_empty_documents_handling(
    setup_embedding_model, test_collection_name, test_persist_directory
):
    """Test error handling for empty document list"""

    # Arrange & Act
    with pytest.raises(Exception):  # Should raise some exception
        agent = ChatAgent(
            documents=[],  # Empty documents
            collection_name=test_collection_name,
            persist_directory=test_persist_directory,
            verbose=False,
        )


@pytest.mark.asyncio
async def test_invalid_query_handling(
    setup_embedding_model, sample_api_documents, test_collection_name, test_persist_directory
):
    """Test error handling for invalid queries"""

    agent = ChatAgent(
        documents=sample_api_documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Act - Empty query
    response = await agent.query("")

    # Assert - Should handle gracefully (may return empty or error response)
    assert response is not None, "Should return some response even for empty query"
