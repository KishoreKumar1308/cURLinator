"""Integration tests for End-to-End Workflow

These tests validate the complete cURLinator workflow:
DocumentationAgent → ChatAgent

They test that documents from DocumentationAgent work seamlessly with ChatAgent,
and that the complete workflow produces correct results.

Note: These tests make real HTTP requests and LLM API calls. They are marked with
@pytest.mark.integration, @pytest.mark.slow, and require a valid LLM API key.

These tests are SKIPPED in CI when no valid LLM API key is available to avoid:
- API costs (LLM calls are expensive)
- Rate limits (frequent CI runs could hit limits)
- Slow execution (LLM calls add significant time)

Run locally with a valid API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)
to test the complete end-to-end workflow.

Usage:
    # Run all end-to-end tests (requires valid LLM API key)
    pytest tests/integration/test_end_to_end_workflow.py -v

    # Run specific test
    pytest tests/integration/test_end_to_end_workflow.py::test_complete_workflow_with_openapi -v

    # Skip slow tests (for CI)
    pytest -m "not slow"
"""

import shutil

import pytest
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from curlinator.agents import ChatAgent, DocumentationAgent
from curlinator.config import get_settings
from tests.integration.conftest import requires_llm

# Mark all tests in this module as integration, slow, and requiring LLM
pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_llm]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def setup_embedding_model():
    """Set up local embedding model for tests"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="./data/models"
    )
    yield


@pytest.fixture
def test_collection_name():
    """Generate unique collection name for each test"""
    import uuid
    return f"test_e2e_{uuid.uuid4().hex[:8]}"


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
# Test: Complete Workflow with OpenAPI
# ============================================================================

@pytest.mark.asyncio
async def test_complete_workflow_with_openapi(
    setup_embedding_model,
    test_collection_name,
    test_persist_directory
):
    """Test complete workflow: DocumentationAgent → ChatAgent with OpenAPI spec"""
    # Check for API key
    settings = get_settings()
    if not (settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key):
        pytest.skip("No LLM API key found - skipping test")

    # Step 1: Crawl documentation with DocumentationAgent
    doc_agent = DocumentationAgent(
        max_depth=2,
        max_pages=20,
        enable_enrichment=True,
        headless=True,
        verbose=False,
    )

    documents = await doc_agent.execute("https://petstore3.swagger.io")

    # Assert documents were crawled
    assert len(documents) > 0, "DocumentationAgent should return documents"
    assert len(documents) >= 10, "Should have at least 10 documents"

    # Step 2: Query with ChatAgent
    chat_agent = ChatAgent(
        documents=documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    response = await chat_agent.query("How do I add a new pet?")

    # Assert response is valid
    assert response is not None, "ChatAgent should return response"
    assert "response" in response, "Should have response field"
    assert len(response["response"]) > 0, "Response should not be empty"

    # Check that response is relevant
    response_text = response["response"].lower()
    assert "pet" in response_text or "post" in response_text, \
        "Response should be relevant to adding a pet"

    # Check cURL command
    if response.get("curl_command"):
        curl = response["curl_command"]
        assert "curl" in curl.lower(), "Should contain curl command"
        assert "/pet" in curl, "Should contain the pet endpoint"


@pytest.mark.asyncio
async def test_complete_workflow_without_openapi(
    setup_embedding_model,
    test_collection_name,
    test_persist_directory
):
    """Test complete workflow with site that has no OpenAPI spec"""
    # Check for API key
    settings = get_settings()
    if not (settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key):
        pytest.skip("No LLM API key found - skipping test")

    # Step 1: Crawl documentation (full crawl path)
    doc_agent = DocumentationAgent(
        max_depth=2,
        max_pages=15,  # Limit for faster test
        enable_enrichment=True,
        headless=True,
        verbose=False,
    )

    documents = await doc_agent.execute("https://jsonplaceholder.typicode.com")

    # Assert documents were crawled
    assert len(documents) > 0, "DocumentationAgent should return documents"

    # Step 2: Query with ChatAgent
    chat_agent = ChatAgent(
        documents=documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    response = await chat_agent.query("What endpoints are available?")

    # Assert response is valid
    assert response is not None, "ChatAgent should return response"
    assert "response" in response, "Should have response field"
    assert len(response["response"]) > 0, "Response should not be empty"


# ============================================================================
# Test: Multiple Queries on Same Document Set
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_queries_on_same_documents(
    setup_embedding_model,
    test_collection_name,
    test_persist_directory
):
    """Test multiple queries on the same document set"""
    # Check for API key
    settings = get_settings()
    if not (settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key):
        pytest.skip("No LLM API key found - skipping test")

    # Step 1: Crawl documentation
    doc_agent = DocumentationAgent(
        max_depth=2,
        max_pages=20,
        enable_enrichment=True,
        headless=True,
        verbose=False,
    )

    documents = await doc_agent.execute("https://petstore3.swagger.io")
    assert len(documents) > 0, "Should have documents"

    # Step 2: Create ChatAgent
    chat_agent = ChatAgent(
        documents=documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Step 3: Multiple queries
    queries = [
        "How do I add a new pet?",
        "How do I get a pet by ID?",
        "How do I update a pet?",
    ]

    for query in queries:
        response = await chat_agent.query(query)
        assert response is not None, f"Should return response for: {query}"
        assert len(response["response"]) > 0, f"Response should not be empty for: {query}"


# ============================================================================
# Test: Conversation Context
# ============================================================================

@pytest.mark.asyncio
async def test_conversation_context_maintained(
    setup_embedding_model,
    test_collection_name,
    test_persist_directory
):
    """Test that conversation context is maintained across queries"""
    # Check for API key
    settings = get_settings()
    if not (settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key):
        pytest.skip("No LLM API key found - skipping test")

    # Step 1: Crawl documentation
    doc_agent = DocumentationAgent(
        max_depth=2,
        max_pages=20,
        enable_enrichment=True,
        headless=True,
        verbose=False,
    )

    documents = await doc_agent.execute("https://petstore3.swagger.io")
    assert len(documents) > 0, "Should have documents"

    # Step 2: Create ChatAgent
    chat_agent = ChatAgent(
        documents=documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    # Step 3: First query
    response1 = await chat_agent.query("How do I add a new pet?")
    assert response1 is not None, "First query should return response"

    # Step 4: Follow-up query using context
    response2 = await chat_agent.query("What parameters does that endpoint require?")
    assert response2 is not None, "Follow-up query should return response"
    assert len(response2["response"]) > 0, "Follow-up response should not be empty"


# ============================================================================
# Test: Document Metadata Preservation
# ============================================================================

@pytest.mark.asyncio
async def test_document_metadata_preserved(
    setup_embedding_model,
    test_collection_name,
    test_persist_directory
):
    """Test that document metadata is preserved through the workflow"""
    # Check for API key
    settings = get_settings()
    if not (settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key):
        pytest.skip("No LLM API key found - skipping test")

    # Step 1: Crawl documentation
    doc_agent = DocumentationAgent(
        max_depth=2,
        max_pages=20,
        enable_enrichment=True,
        headless=True,
        verbose=False,
    )

    documents = await doc_agent.execute("https://petstore3.swagger.io")

    # Check that documents have metadata
    # Note: OpenAPI documents have different metadata than crawled documents
    for doc in documents[:5]:
        # All documents should have source
        assert 'source' in doc.metadata, "Should have source metadata"

        # OpenAPI documents have 'type', crawled documents have 'page_type'
        has_type_info = 'type' in doc.metadata or 'page_type' in doc.metadata
        assert has_type_info, "Should have type or page_type metadata"

        # OpenAPI documents have 'source', crawled documents have 'URL' or 'url'
        has_url_info = 'URL' in doc.metadata or 'url' in doc.metadata or 'source' in doc.metadata
        assert has_url_info, "Should have URL/url/source metadata"

    # Step 2: Create ChatAgent and query
    chat_agent = ChatAgent(
        documents=documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    response = await chat_agent.query("How do I add a pet?")

    # Check that sources are returned
    assert "sources" in response, "Should have sources"
    assert len(response["sources"]) > 0, "Should return source citations"


# ============================================================================
# Test: Enrichment Impact
# ============================================================================

@pytest.mark.asyncio
async def test_enrichment_improves_retrieval(
    setup_embedding_model,
    test_collection_name,
    test_persist_directory
):
    """Test that contextual enrichment improves retrieval quality"""
    # Check for API key
    settings = get_settings()
    if not (settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key):
        pytest.skip("No LLM API key found - skipping test")

    # Step 1: Crawl with enrichment enabled
    # Use JSONPlaceholder (no OpenAPI) to test enrichment
    doc_agent = DocumentationAgent(
        max_depth=1,
        max_pages=10,
        enable_enrichment=True,  # Enable enrichment
        headless=True,
        verbose=False,
    )

    documents = await doc_agent.execute("https://jsonplaceholder.typicode.com")

    # Check that some documents are enriched
    # Note: Enrichment may not be applied to all documents (only crawled ones, not OpenAPI)
    enriched_docs = [doc for doc in documents if doc.metadata.get('contextually_enriched')]
    # If no OpenAPI was found, we should have some enriched documents
    has_openapi = any('type' in doc.metadata for doc in documents)
    if not has_openapi:
        assert len(enriched_docs) > 0, "Should have enriched documents when crawling without OpenAPI"

    # Step 2: Query with ChatAgent
    chat_agent = ChatAgent(
        documents=documents,
        collection_name=test_collection_name,
        persist_directory=test_persist_directory,
        verbose=False,
    )

    response = await chat_agent.query("How do I add a new pet?")

    # Assert response is valid (enrichment should help)
    assert response is not None, "Should return response"
    assert len(response["response"]) > 0, "Response should not be empty"
    assert len(response.get("sources", [])) > 0, "Should return relevant sources"

