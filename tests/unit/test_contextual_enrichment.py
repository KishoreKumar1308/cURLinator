"""Tests for contextual enrichment utilities.

Tests the contextual_enrichment module that:
- Adds contextual prefixes to documents before embedding
- Generates context based on document metadata
- Preserves original metadata
- Improves retrieval accuracy (based on Anthropic's research)
"""

import pytest
from llama_index.core.schema import Document

from curlinator.utils.contextual_enrichment import (
    _get_endpoint_context,
    _get_page_type_context,
    _get_type_context,
    _get_url_context,
    enrich_document_with_context,
    enrich_documents_batch,
    generate_contextual_prefix,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def api_endpoint_document():
    """Sample API endpoint document"""
    return Document(
        text="POST /v1/customers - Creates a new customer in the system",
        metadata={
            "url": "https://stripe.com/docs/api/customers/create",
            "type": "api_endpoint",
            "endpoint": "/v1/customers",
            "method": "POST",
            "title": "Create Customer",
        },
    )


@pytest.fixture
def api_overview_document():
    """Sample API overview document"""
    return Document(
        text="The Stripe API allows you to process payments and manage customers.",
        metadata={
            "url": "https://stripe.com/docs/api",
            "type": "api_overview",
            "api_title": "Stripe API",
            "title": "API Overview",
        },
    )


@pytest.fixture
def authentication_document():
    """Sample authentication document"""
    return Document(
        text="Use your API key in the Authorization header with Bearer token.",
        metadata={
            "url": "https://stripe.com/docs/api/authentication",
            "type": "authentication",
            "page_type": "authentication",
            "title": "Authentication",
        },
    )


@pytest.fixture
def guide_document():
    """Sample guide document"""
    return Document(
        text="This guide explains how to implement pagination in your API calls.",
        metadata={
            "url": "https://stripe.com/docs/guides/pagination",
            "type": "guide",
            "page_type": "guide",
            "title": "Pagination Guide",
        },
    )


@pytest.fixture
def tutorial_document():
    """Sample tutorial document"""
    return Document(
        text="Step 1: Install the SDK. Step 2: Configure your API key.",
        metadata={
            "url": "https://stripe.com/docs/tutorials/quickstart",
            "page_type": "tutorial",
            "title": "Quickstart Tutorial",
        },
    )


@pytest.fixture
def minimal_document():
    """Document with minimal metadata"""
    return Document(
        text="Some content without much metadata.", metadata={"url": "https://example.com/page"}
    )


# ============================================================================
# Test enrich_document_with_context
# ============================================================================


class TestEnrichDocumentWithContext:
    """Tests for document enrichment with contextual prefixes"""

    def test_adds_contextual_prefix_to_text(self, api_endpoint_document):
        """Test adds contextual prefix to document text"""
        enriched = enrich_document_with_context(api_endpoint_document, "Stripe API documentation")

        # Should have prefix + original text
        assert len(enriched.text) > len(api_endpoint_document.text)
        assert api_endpoint_document.text in enriched.text
        assert "Stripe API documentation" in enriched.text

    def test_preserves_original_metadata(self, api_endpoint_document):
        """Test preserves all original metadata"""
        enriched = enrich_document_with_context(api_endpoint_document, "Stripe API documentation")

        # All original metadata should be preserved
        assert enriched.metadata["url"] == api_endpoint_document.metadata["url"]
        assert enriched.metadata["type"] == api_endpoint_document.metadata["type"]
        assert enriched.metadata["endpoint"] == api_endpoint_document.metadata["endpoint"]
        assert enriched.metadata["method"] == api_endpoint_document.metadata["method"]

    def test_adds_enrichment_metadata(self, api_endpoint_document):
        """Test adds enrichment metadata flags"""
        enriched = enrich_document_with_context(api_endpoint_document, "Stripe API documentation")

        assert enriched.metadata["contextually_enriched"] is True
        assert "original_text_length" in enriched.metadata
        assert enriched.metadata["original_text_length"] == len(api_endpoint_document.text)

    def test_preserves_document_id(self, api_endpoint_document):
        """Test preserves document ID"""
        api_endpoint_document.id_ = "test-doc-123"

        enriched = enrich_document_with_context(api_endpoint_document, "Stripe API documentation")

        assert enriched.id_ == "test-doc-123"

    def test_preserves_embedding(self, api_endpoint_document):
        """Test preserves existing embedding"""
        api_endpoint_document.embedding = [0.1, 0.2, 0.3]

        enriched = enrich_document_with_context(api_endpoint_document, "Stripe API documentation")

        assert enriched.embedding == [0.1, 0.2, 0.3]

    def test_preserves_excluded_metadata_keys(self, api_endpoint_document):
        """Test preserves excluded metadata keys"""
        api_endpoint_document.excluded_embed_metadata_keys = ["url"]
        api_endpoint_document.excluded_llm_metadata_keys = ["type"]

        enriched = enrich_document_with_context(api_endpoint_document, "Stripe API documentation")

        assert enriched.excluded_embed_metadata_keys == ["url"]
        assert enriched.excluded_llm_metadata_keys == ["type"]


# ============================================================================
# Test generate_contextual_prefix
# ============================================================================


class TestGenerateContextualPrefix:
    """Tests for contextual prefix generation"""

    def test_generates_prefix_for_api_endpoint(self, api_endpoint_document):
        """Test generates appropriate prefix for API endpoint"""
        prefix = generate_contextual_prefix(api_endpoint_document, "Stripe API documentation")

        assert "Stripe API documentation" in prefix
        assert "api endpoint" in prefix.lower() or "endpoint" in prefix.lower()
        assert "POST" in prefix
        assert "/v1/customers" in prefix

    def test_generates_prefix_for_api_overview(self, api_overview_document):
        """Test generates appropriate prefix for API overview"""
        prefix = generate_contextual_prefix(api_overview_document, "Stripe API documentation")

        assert "Stripe API documentation" in prefix
        assert "overview" in prefix.lower()
        assert "Stripe API" in prefix

    def test_generates_prefix_for_authentication(self, authentication_document):
        """Test generates appropriate prefix for authentication"""
        prefix = generate_contextual_prefix(authentication_document, "Stripe API documentation")

        assert "Stripe API documentation" in prefix
        assert "authentication" in prefix.lower()

    def test_generates_prefix_for_guide(self, guide_document):
        """Test generates appropriate prefix for guide"""
        prefix = generate_contextual_prefix(guide_document, "Stripe API documentation")

        assert "Stripe API documentation" in prefix
        assert "guide" in prefix.lower()

    def test_generates_prefix_for_tutorial(self, tutorial_document):
        """Test generates appropriate prefix for tutorial"""
        prefix = generate_contextual_prefix(tutorial_document, "Stripe API documentation")

        assert "Stripe API documentation" in prefix
        assert "tutorial" in prefix.lower()

    def test_includes_title_when_available(self, api_endpoint_document):
        """Test includes title in prefix when available"""
        prefix = generate_contextual_prefix(api_endpoint_document, "Stripe API documentation")

        assert "Create Customer" in prefix

    def test_excludes_untitled_from_prefix(self):
        """Test excludes 'Untitled' from prefix"""
        doc = Document(text="Content", metadata={"url": "https://example.com", "title": "Untitled"})

        prefix = generate_contextual_prefix(doc, "API documentation")

        assert "Untitled" not in prefix

    def test_includes_url_context(self, api_endpoint_document):
        """Test includes URL path context"""
        prefix = generate_contextual_prefix(api_endpoint_document, "Stripe API documentation")

        # Should include meaningful URL segments
        assert "customers" in prefix.lower() or "create" in prefix.lower()

    def test_handles_minimal_metadata(self, minimal_document):
        """Test handles document with minimal metadata"""
        prefix = generate_contextual_prefix(minimal_document, "API documentation")

        # Should at least have base context
        assert "API documentation" in prefix
        assert prefix.endswith(".")

    def test_creates_coherent_sentence(self, api_endpoint_document):
        """Test creates grammatically coherent sentence"""
        prefix = generate_contextual_prefix(api_endpoint_document, "Stripe API documentation")

        # Should start with capital letter and end with period
        assert prefix[0].isupper()
        assert prefix.endswith(".")

        # Should not have double periods or commas
        assert ".." not in prefix
        assert ",," not in prefix


# ============================================================================
# Test enrich_documents_batch
# ============================================================================


class TestEnrichDocumentsBatch:
    """Tests for batch document enrichment"""

    def test_enriches_multiple_documents(
        self, api_endpoint_document, guide_document, authentication_document
    ):
        """Test enriches multiple documents in batch"""
        documents = [api_endpoint_document, guide_document, authentication_document]

        enriched = enrich_documents_batch(documents, "Stripe API documentation", verbose=False)

        assert len(enriched) == 3
        assert all(doc.metadata.get("contextually_enriched") for doc in enriched)

    def test_preserves_order(self, api_endpoint_document, guide_document, authentication_document):
        """Test preserves document order"""
        documents = [api_endpoint_document, guide_document, authentication_document]

        enriched = enrich_documents_batch(documents, "Stripe API documentation", verbose=False)

        # Check order is preserved by checking metadata
        assert enriched[0].metadata["type"] == "api_endpoint"
        assert enriched[1].metadata["type"] == "guide"
        assert enriched[2].metadata["type"] == "authentication"

    def test_handles_empty_list(self):
        """Test handles empty document list"""
        enriched = enrich_documents_batch([], "API documentation", verbose=False)

        assert enriched == []

    def test_handles_single_document(self, api_endpoint_document):
        """Test handles single document"""
        enriched = enrich_documents_batch(
            [api_endpoint_document], "Stripe API documentation", verbose=False
        )

        assert len(enriched) == 1
        assert enriched[0].metadata["contextually_enriched"] is True

    def test_continues_on_error(self, api_endpoint_document, guide_document):
        """Test continues processing when one document fails"""
        # Create a document with metadata that will cause an error during enrichment
        # (e.g., missing required fields that generate_contextual_prefix expects)
        bad_document = Document(text="Bad doc", metadata={})

        # Patch enrich_document_with_context to raise an error for the bad document
        from unittest.mock import patch

        from curlinator.utils.contextual_enrichment import (
            enrich_document_with_context as real_enrich,
        )

        def mock_enrich(doc, site_context):
            if doc.text == "Bad doc":
                raise ValueError("Simulated error")
            # Call the real function for other documents
            return real_enrich(doc, site_context)

        documents = [api_endpoint_document, bad_document, guide_document]

        with patch(
            "curlinator.utils.contextual_enrichment.enrich_document_with_context",
            side_effect=mock_enrich,
        ):
            enriched = enrich_documents_batch(documents, "API documentation", verbose=False)

        # Should have 3 documents (bad one kept as original)
        assert len(enriched) == 3

        # First and third should be enriched
        assert enriched[0].metadata.get("contextually_enriched") is True
        assert enriched[2].metadata.get("contextually_enriched") is True

        # Second (bad) document should not be enriched
        assert enriched[1].metadata.get("contextually_enriched") is not True

    def test_verbose_mode_logs_progress(self, api_endpoint_document, guide_document, caplog):
        """Test verbose mode logs progress"""
        import logging

        # Create 10+ documents to trigger progress logging
        documents = [api_endpoint_document] * 15

        with caplog.at_level(logging.INFO):
            enrich_documents_batch(documents, "API documentation", verbose=True)

        # Should log progress at 10 documents
        assert any("10/" in record.message for record in caplog.records)
        # Should log completion
        assert any(
            "Enriched" in record.message and "15" in record.message for record in caplog.records
        )


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions"""

    def test_get_type_context_for_known_types(self):
        """Test _get_type_context returns descriptions for known types"""
        assert "endpoint" in _get_type_context("api_endpoint").lower()
        assert "overview" in _get_type_context("api_overview").lower()
        assert "authentication" in _get_type_context("authentication").lower()
        assert "guide" in _get_type_context("guide").lower()
        assert "tutorial" in _get_type_context("tutorial").lower()
        assert "quickstart" in _get_type_context("quickstart").lower()
        assert "sdk" in _get_type_context("sdk").lower()
        assert "webhook" in _get_type_context("webhook").lower()
        assert "error" in _get_type_context("error").lower()
        assert "changelog" in _get_type_context("changelog").lower()

    def test_get_type_context_for_unknown_type(self):
        """Test _get_type_context handles unknown types"""
        result = _get_type_context("custom_type")
        assert "custom_type" in result

    def test_get_endpoint_context_with_method_and_endpoint(self):
        """Test _get_endpoint_context with both method and endpoint"""
        metadata = {"method": "POST", "endpoint": "/v1/customers"}

        result = _get_endpoint_context(metadata)

        assert result is not None
        assert "POST" in result
        assert "/v1/customers" in result

    def test_get_endpoint_context_with_endpoint_only(self):
        """Test _get_endpoint_context with endpoint only"""
        metadata = {"endpoint": "/v1/customers"}

        result = _get_endpoint_context(metadata)

        assert result is not None
        assert "/v1/customers" in result

    def test_get_endpoint_context_returns_none_when_missing(self):
        """Test _get_endpoint_context returns None when endpoint missing"""
        metadata = {"method": "POST"}

        result = _get_endpoint_context(metadata)

        assert result is None

    def test_get_page_type_context_for_known_types(self):
        """Test _get_page_type_context returns context for known types"""
        assert _get_page_type_context("api_reference") is not None
        assert _get_page_type_context("guide") is not None
        assert _get_page_type_context("tutorial") is not None
        assert _get_page_type_context("overview") is not None
        assert _get_page_type_context("authentication") is not None
        assert _get_page_type_context("quickstart") is not None
        assert _get_page_type_context("sdk") is not None
        assert _get_page_type_context("webhook") is not None
        assert _get_page_type_context("error") is not None
        assert _get_page_type_context("changelog") is not None

    def test_get_page_type_context_returns_none_for_unknown(self):
        """Test _get_page_type_context returns None for unknown types"""
        assert _get_page_type_context("unknown_type") is None

    def test_get_url_context_extracts_meaningful_segments(self):
        """Test _get_url_context extracts meaningful URL segments"""
        url = "https://stripe.com/docs/api/customers/create"

        result = _get_url_context(url)

        assert result is not None
        assert "customers" in result
        assert "create" in result

    def test_get_url_context_handles_single_segment(self):
        """Test _get_url_context handles single segment"""
        url = "https://example.com/authentication"

        result = _get_url_context(url)

        assert result is not None
        assert "authentication" in result

    def test_get_url_context_filters_numeric_segments(self):
        """Test _get_url_context filters out numeric segments"""
        url = "https://example.com/docs/123/api/456/endpoint"

        result = _get_url_context(url)

        # Should not include "123" or "456"
        assert "123" not in result
        assert "456" not in result
        # Should include meaningful segments
        assert "api" in result or "endpoint" in result

    def test_get_url_context_returns_none_for_empty_path(self):
        """Test _get_url_context returns None for empty path"""
        url = "https://example.com"

        result = _get_url_context(url)

        assert result is None

    def test_get_url_context_returns_none_for_root_path(self):
        """Test _get_url_context returns None for root path"""
        url = "https://example.com/"

        result = _get_url_context(url)

        assert result is None

    def test_get_url_context_handles_malformed_url(self):
        """Test _get_url_context handles malformed URLs gracefully"""
        url = "not-a-valid-url"

        result = _get_url_context(url)

        # Should not crash, should return None or handle gracefully
        assert result is None or isinstance(result, str)


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Tests for real-world integration scenarios"""

    def test_enrichment_improves_context_for_retrieval(self, api_endpoint_document):
        """Test enrichment adds context that would improve retrieval"""
        enriched = enrich_document_with_context(api_endpoint_document, "Stripe API documentation")

        # Original text is about creating customers
        # Enriched text should add context about Stripe, API, endpoint type
        enriched_lower = enriched.text.lower()

        # Should add "stripe" context
        assert "stripe" in enriched_lower

        # Should add "api" context
        assert "api" in enriched_lower

        # Should still contain original content
        assert "customer" in enriched_lower

    def test_different_document_types_get_different_context(
        self, api_endpoint_document, guide_document, authentication_document
    ):
        """Test different document types get appropriately different context"""
        endpoint_enriched = enrich_document_with_context(api_endpoint_document, "Stripe API")
        guide_enriched = enrich_document_with_context(guide_document, "Stripe API")
        auth_enriched = enrich_document_with_context(authentication_document, "Stripe API")

        # Each should have different context
        endpoint_prefix = endpoint_enriched.text.split("\n\n")[0]
        guide_prefix = guide_enriched.text.split("\n\n")[0]
        auth_prefix = auth_enriched.text.split("\n\n")[0]

        # Prefixes should be different
        assert endpoint_prefix != guide_prefix
        assert guide_prefix != auth_prefix
        assert endpoint_prefix != auth_prefix

        # But all should mention "Stripe API"
        assert "Stripe API" in endpoint_prefix
        assert "Stripe API" in guide_prefix
        assert "Stripe API" in auth_prefix

    def test_enrichment_is_idempotent(self, api_endpoint_document):
        """Test enriching an already enriched document doesn't break"""
        first_enrichment = enrich_document_with_context(api_endpoint_document, "Stripe API")

        # Enrich again
        second_enrichment = enrich_document_with_context(first_enrichment, "Stripe API")

        # Should still be valid document
        assert isinstance(second_enrichment, Document)
        assert len(second_enrichment.text) > 0
        assert second_enrichment.metadata["contextually_enriched"] is True
