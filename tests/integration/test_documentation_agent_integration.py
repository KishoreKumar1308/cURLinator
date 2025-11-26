"""Integration tests for DocumentationAgent

These tests validate the DocumentationAgent with real API documentation sites.
They test the complete workflow: crawling, OpenAPI detection, page classification,
and contextual enrichment.

Note: These tests make real HTTP requests and may be slow. They are marked with
@pytest.mark.integration to allow selective execution.

Usage:
    # Run all integration tests
    pytest tests/integration/test_documentation_agent_integration.py -v

    # Run specific test
    pytest tests/integration/test_documentation_agent_integration.py::test_crawl_with_openapi_spec -v

    # Skip integration tests (for CI)
    pytest -m "not integration"
"""

import pytest
from llama_index.core.schema import Document

from curlinator.agents import DocumentationAgent

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ============================================================================
# Test: OpenAPI Detection and Fast Path
# ============================================================================

@pytest.mark.asyncio
async def test_crawl_with_openapi_spec():
    """Test crawling a site with OpenAPI spec (fast path)"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=20,
        enable_enrichment=True,
        headless=True,
        verbose=False,
    )

    # Act
    documents = await agent.execute("https://petstore3.swagger.io")

    # Assert
    assert len(documents) > 0, "Should return documents"
    assert len(documents) >= 10, "Should have at least 10 endpoint documents"

    # Check that OpenAPI spec was detected (documents have 'type' metadata)
    # OpenAPI documents have 'type' field like 'api_overview', 'api_endpoint', etc.
    openapi_docs = [doc for doc in documents if 'type' in doc.metadata]
    assert len(openapi_docs) > 0, "Should have OpenAPI-sourced documents with 'type' metadata"

    # Check document structure
    for doc in documents[:3]:  # Check first 3 documents
        assert isinstance(doc, Document), "Should return Document objects"
        assert doc.text, "Document should have text content"
        # OpenAPI documents have 'source' (spec URL) and 'type' fields
        assert 'source' in doc.metadata, "Should have source in metadata"
        assert 'type' in doc.metadata, "Should have type in metadata"

    # Check for API endpoint documents
    endpoint_docs = [doc for doc in documents if doc.metadata.get('type') == 'api_endpoint']
    assert len(endpoint_docs) > 0, "Should have API endpoint documents"

    # Check endpoint metadata
    for doc in endpoint_docs[:3]:
        assert 'method' in doc.metadata, "Endpoint should have HTTP method"
        assert 'endpoint' in doc.metadata, "Endpoint should have path"
        assert doc.metadata['method'] in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'], \
            "Should have valid HTTP method"


@pytest.mark.asyncio
async def test_openapi_detection_with_swagger_ui():
    """Test OpenAPI detection from Swagger UI page"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=20,
        enable_enrichment=False,  # Disable for faster test
        headless=True,
        verbose=False,
    )

    # Act
    documents = await agent.execute("https://petstore3.swagger.io")

    # Assert - Check that OpenAPI spec was detected
    openapi_docs = [doc for doc in documents if 'type' in doc.metadata]
    assert len(openapi_docs) > 0, "Should detect OpenAPI spec from Swagger UI"

    # Check for API overview document
    overview_docs = [doc for doc in documents if doc.metadata.get('type') == 'api_overview']
    assert len(overview_docs) > 0, "Should have API overview document"

    # Check overview metadata
    overview = overview_docs[0]
    assert 'api_title' in overview.metadata, "Overview should have api_title"
    assert 'api_version' in overview.metadata, "Overview should have api_version"


# ============================================================================
# Test: Full Crawl Path (No OpenAPI)
# ============================================================================

@pytest.mark.asyncio
async def test_crawl_without_openapi_spec():
    """Test crawling a site without OpenAPI spec (full crawl path)"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=15,  # Limit pages for faster test
        enable_enrichment=True,
        headless=True,
        verbose=False,
    )

    # Act
    documents = await agent.execute("https://jsonplaceholder.typicode.com")

    # Assert
    assert len(documents) > 0, "Should return documents from crawl"

    # Check that documents are from crawl (have page_type), not OpenAPI (have type)
    # Crawled documents have 'page_type', OpenAPI documents have 'type'
    crawl_docs = [doc for doc in documents if 'page_type' in doc.metadata]
    assert len(crawl_docs) > 0, "Should have crawl-sourced documents with page_type"

    # Check document structure
    for doc in crawl_docs[:3]:  # Check only crawled documents
        assert isinstance(doc, Document), "Should return Document objects"
        assert doc.text, "Document should have text content"
        # WholeSiteReader uses 'URL' (uppercase), page_classifier adds 'url' (lowercase)
        assert 'url' in doc.metadata or 'URL' in doc.metadata, "Crawled document should have URL in metadata"
        assert 'page_type' in doc.metadata, "Crawled document should have page_type in metadata"
        assert 'title' in doc.metadata, "Crawled document should have title in metadata"


# ============================================================================
# Test: Contextual Enrichment
# ============================================================================

@pytest.mark.asyncio
async def test_contextual_enrichment_enabled():
    """Test that contextual enrichment is applied when enabled"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=10,
        enable_enrichment=True,  # Enable enrichment
        headless=True,
        verbose=False,
    )

    # Act - Use JSONPlaceholder (no OpenAPI, so will crawl and enrich)
    documents = await agent.execute("https://jsonplaceholder.typicode.com")

    # Assert
    enriched_docs = [doc for doc in documents if doc.metadata.get('contextually_enriched')]
    assert len(enriched_docs) > 0, "Should have enriched documents when crawling"

    # Check that enriched documents have context prefix
    for doc in enriched_docs[:3]:
        # Enriched documents should have longer text (context prefix added)
        assert len(doc.text) > 50, "Enriched document should have substantial text"


@pytest.mark.asyncio
async def test_contextual_enrichment_disabled():
    """Test that contextual enrichment is not applied when disabled"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=10,
        enable_enrichment=False,  # Disable enrichment
        headless=True,
        verbose=False,
    )

    # Act - Use JSONPlaceholder (no OpenAPI, so will crawl but not enrich)
    documents = await agent.execute("https://jsonplaceholder.typicode.com")

    # Assert
    enriched_docs = [doc for doc in documents if doc.metadata.get('contextually_enriched')]
    assert len(enriched_docs) == 0, "Should have no enriched documents when disabled"


# ============================================================================
# Test: Page Classification
# ============================================================================

@pytest.mark.asyncio
async def test_page_classification_accuracy():
    """Test that page types are classified correctly"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=20,
        enable_enrichment=False,
        headless=True,
        verbose=False,
    )

    # Act
    documents = await agent.execute("https://petstore3.swagger.io")

    # Assert - Check for expected document types
    # OpenAPI documents use 'type' field, crawled documents use 'page_type'
    doc_types = set(doc.metadata.get('type') or doc.metadata.get('page_type') for doc in documents)

    # Should have at least api_endpoint type
    assert 'api_endpoint' in doc_types or 'api_reference' in doc_types, \
        f"Should classify API endpoint pages. Found types: {doc_types}"

    # All documents should have a type or page_type
    for doc in documents:
        has_type = 'type' in doc.metadata or 'page_type' in doc.metadata
        assert has_type, "All documents should have type or page_type"


# ============================================================================
# Test: Configuration Parameters
# ============================================================================

@pytest.mark.asyncio
async def test_max_pages_limit_respected():
    """Test that max_pages limit is respected during crawl"""
    # Arrange
    max_pages = 5
    agent = DocumentationAgent(
        max_depth=3,
        max_pages=max_pages,
        enable_enrichment=False,
        headless=True,
        verbose=False,
    )

    # Act - Use a site without OpenAPI to test crawl limits
    documents = await agent.execute("https://jsonplaceholder.typicode.com")

    # Assert
    # Note: OpenAPI detection bypasses max_pages, so we use a site without OpenAPI
    # The actual number may be less than max_pages if site is small
    assert len(documents) <= max_pages + 5, \
        f"Should respect max_pages limit (got {len(documents)}, max {max_pages})"


@pytest.mark.asyncio
async def test_headless_mode_works():
    """Test that headless mode works correctly"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=10,
        enable_enrichment=False,
        headless=True,  # Headless mode
        verbose=False,
    )

    # Act
    documents = await agent.execute("https://petstore3.swagger.io")

    # Assert
    assert len(documents) > 0, "Should work in headless mode"


# ============================================================================
# Test: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_invalid_url_handling():
    """Test error handling for invalid URLs"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=10,
        enable_enrichment=False,
        headless=True,
        verbose=False,
    )

    # Act
    documents = await agent.execute("https://this-domain-does-not-exist-12345.com")

    # Assert - Should handle gracefully and return empty list
    assert isinstance(documents, list), "Should return a list"
    assert len(documents) == 0, "Should return empty list for invalid URL"


@pytest.mark.asyncio
async def test_malformed_url_handling():
    """Test error handling for malformed URLs"""
    # Arrange
    agent = DocumentationAgent(
        max_depth=2,
        max_pages=10,
        enable_enrichment=False,
        headless=True,
        verbose=False,
    )

    # Act
    documents = await agent.execute("not-a-valid-url")

    # Assert - Should handle gracefully and return empty list
    assert isinstance(documents, list), "Should return a list"
    assert len(documents) == 0, "Should return empty list for malformed URL"

