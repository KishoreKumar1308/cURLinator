"""Tests for DocumentationAgent (Refactored Version)

Tests the new LlamaIndex-based DocumentationAgent that:
- Uses WholeSiteReader for crawling
- Integrates openapi_detector, page_classifier, contextual_enrichment utilities
- Outputs List[Document] compatible with ChatAgent
"""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.schema import Document

from curlinator.agents import DocumentationAgent

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    return MockLLM()


@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing"""
    return [
        Document(
            text="This is a guide about authentication",
            metadata={
                "url": "https://docs.example.com/auth",
                "title": "Authentication Guide",
            },
        ),
        Document(
            text="API reference for creating users",
            metadata={
                "url": "https://docs.example.com/api/users",
                "title": "Create User",
            },
        ),
        Document(
            text="Tutorial on getting started",
            metadata={
                "url": "https://docs.example.com/tutorial",
                "title": "Getting Started",
            },
        ),
    ]


@pytest.fixture
def sample_openapi_documents():
    """Create sample OpenAPI-parsed documents"""
    return [
        Document(
            text="POST /users - Create a new user",
            metadata={
                "url": "https://api.example.com/openapi.json",
                "endpoint": "/users",
                "method": "POST",
                "source": "openapi",
            },
        ),
        Document(
            text="GET /users/{id} - Get user by ID",
            metadata={
                "url": "https://api.example.com/openapi.json",
                "endpoint": "/users/{id}",
                "method": "GET",
                "source": "openapi",
            },
        ),
    ]


# ============================================================================
# Test Initialization
# ============================================================================


class TestDocumentationAgentInitialization:
    """Tests for DocumentationAgent initialization"""

    def test_creates_agent_with_defaults(self, mock_llm) -> None:
        """Test agent creation with default settings"""
        agent = DocumentationAgent(llm=mock_llm)

        assert agent.max_depth == 3
        assert agent.max_pages == 50
        assert agent.enable_enrichment is True
        assert agent.use_llm_classification is False
        assert agent.llm is not None
        assert agent.verbose is False

    def test_creates_agent_with_custom_settings(self, mock_llm) -> None:
        """Test agent creation with custom settings"""
        agent = DocumentationAgent(
            llm=mock_llm,
            max_depth=5,
            max_pages=100,
            enable_enrichment=False,
            use_llm_classification=True,
            verbose=True,
        )

        assert agent.max_depth == 5
        assert agent.max_pages == 100
        assert agent.enable_enrichment is False
        assert agent.use_llm_classification is True
        assert agent.verbose is True

    def test_agent_inherits_from_base_agent(self, mock_llm) -> None:
        """Test that agent inherits from BaseAgent correctly"""
        from curlinator.agents.base import BaseAgent

        agent = DocumentationAgent(llm=mock_llm)

        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, "llm")
        assert hasattr(agent, "settings")
        assert hasattr(agent, "_log")
        assert hasattr(agent, "execute")

    def test_agent_has_required_methods(self, mock_llm) -> None:
        """Test that agent has all required methods"""
        agent = DocumentationAgent(llm=mock_llm)

        assert hasattr(agent, "execute")
        assert hasattr(agent, "_detect_openapi")
        assert hasattr(agent, "_crawl_with_reader")
        assert hasattr(agent, "_classify_pages")
        assert hasattr(agent, "_enrich_documents")


# ============================================================================
# Test _detect_openapi Method
# ============================================================================


class TestDetectOpenAPI:
    """Tests for _detect_openapi method"""

    @pytest.mark.asyncio
    async def test_returns_documents_when_spec_found(
        self, mock_llm, sample_openapi_documents
    ) -> None:
        """Test that OpenAPI spec detection returns documents when found"""
        agent = DocumentationAgent(llm=mock_llm, verbose=True)

        with (
            patch("curlinator.agents.documentation_agent.detect_openapi_spec") as mock_detect,
            patch("curlinator.agents.documentation_agent.parse_openapi_to_documents") as mock_parse,
        ):
            # Mock successful detection
            mock_detect.return_value = "https://api.example.com/openapi.json"
            mock_parse.return_value = sample_openapi_documents

            result = await agent._detect_openapi("https://api.example.com")

            assert result is not None
            assert len(result) == 2
            assert all(isinstance(doc, Document) for doc in result)
            mock_detect.assert_called_once_with("https://api.example.com")
            mock_parse.assert_called_once_with("https://api.example.com/openapi.json")

    @pytest.mark.asyncio
    async def test_returns_none_when_spec_not_found(self, mock_llm) -> None:
        """Test that OpenAPI spec detection returns None when not found"""
        agent = DocumentationAgent(llm=mock_llm)

        with patch("curlinator.agents.documentation_agent.detect_openapi_spec") as mock_detect:
            # Mock no spec found
            mock_detect.return_value = None

            result = await agent._detect_openapi("https://api.example.com")

            assert result is None
            mock_detect.assert_called_once_with("https://api.example.com")

    @pytest.mark.asyncio
    async def test_handles_detection_error_gracefully(self, mock_llm) -> None:
        """Test that detection errors are handled gracefully"""
        agent = DocumentationAgent(llm=mock_llm)

        with patch("curlinator.agents.documentation_agent.detect_openapi_spec") as mock_detect:
            # Mock error during detection
            mock_detect.side_effect = Exception("Network error")

            result = await agent._detect_openapi("https://api.example.com")

            assert result is None  # Should return None on error

    @pytest.mark.asyncio
    async def test_handles_parsing_error_gracefully(self, mock_llm) -> None:
        """Test that parsing errors are handled gracefully"""
        agent = DocumentationAgent(llm=mock_llm)

        with (
            patch("curlinator.agents.documentation_agent.detect_openapi_spec") as mock_detect,
            patch("curlinator.agents.documentation_agent.parse_openapi_to_documents") as mock_parse,
        ):
            mock_detect.return_value = "https://api.example.com/openapi.json"
            mock_parse.side_effect = Exception("Parse error")

            result = await agent._detect_openapi("https://api.example.com")

            assert result is None  # Should return None on error


# ============================================================================
# Test _crawl_with_reader Method
# ============================================================================


class TestCrawlWithReader:
    """Tests for _crawl_with_reader method"""

    @pytest.mark.asyncio
    async def test_successful_crawl_returns_documents(self, mock_llm, sample_documents) -> None:
        """Test successful crawl returns list of documents"""
        agent = DocumentationAgent(llm=mock_llm, max_depth=3, max_pages=50)

        with (
            patch("curlinator.agents.documentation_agent.WholeSiteReader") as mock_reader_class,
            patch.object(agent, "_create_webdriver") as mock_create_driver,
        ):
            # Mock WebDriver
            mock_driver = MagicMock()
            mock_create_driver.return_value = mock_driver

            # Mock WholeSiteReader
            mock_reader = MagicMock()
            mock_reader.load_data.return_value = sample_documents
            mock_reader_class.return_value = mock_reader

            result = await agent._crawl_with_reader("https://docs.example.com")

            assert result is not None
            assert len(result) == 3
            assert all(isinstance(doc, Document) for doc in result)

            # Verify WholeSiteReader was initialized correctly with driver
            mock_reader_class.assert_called_once_with(
                prefix="https://docs.example.com",
                max_depth=3,
                driver=mock_driver,
            )
            # Note: We don't check mock_reader.load_data.assert_called_once_with() anymore
            # because DocumentationAgent overrides load_data with a custom implementation
            # that respects max_pages limit

            # Verify driver was created and cleaned up
            mock_create_driver.assert_called_once()
            mock_driver.quit.assert_called_once()

    @pytest.mark.asyncio
    async def test_limits_to_max_pages(self, mock_llm) -> None:
        """Test that results are limited to max_pages"""
        agent = DocumentationAgent(llm=mock_llm, max_pages=2)

        # Create more documents than max_pages
        many_documents = [
            Document(text=f"Page {i}", metadata={"url": f"https://example.com/page{i}"})
            for i in range(10)
        ]

        with patch("curlinator.agents.documentation_agent.WholeSiteReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.load_data.return_value = many_documents
            mock_reader_class.return_value = mock_reader

            result = await agent._crawl_with_reader("https://docs.example.com")

            assert len(result) == 2  # Should be limited to max_pages

    @pytest.mark.asyncio
    async def test_handles_crawl_error_gracefully(self, mock_llm) -> None:
        """Test that crawl errors are handled gracefully"""
        agent = DocumentationAgent(llm=mock_llm)

        with patch("curlinator.agents.documentation_agent.WholeSiteReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.load_data.side_effect = Exception("Crawl failed")
            mock_reader_class.return_value = mock_reader

            result = await agent._crawl_with_reader("https://docs.example.com")

            assert result == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_documents_found(self, mock_llm) -> None:
        """Test that empty list is returned when no documents found"""
        agent = DocumentationAgent(llm=mock_llm)

        with patch("curlinator.agents.documentation_agent.WholeSiteReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.load_data.return_value = []
            mock_reader_class.return_value = mock_reader

            result = await agent._crawl_with_reader("https://docs.example.com")

            assert result == []


# ============================================================================
# Test _classify_pages Method
# ============================================================================


class TestClassifyPages:
    """Tests for _classify_pages method"""

    @pytest.mark.asyncio
    async def test_successful_classification_updates_metadata(
        self, mock_llm, sample_documents
    ) -> None:
        """Test successful classification updates document metadata"""
        agent = DocumentationAgent(llm=mock_llm, use_llm_classification=False)

        with (
            patch("curlinator.agents.documentation_agent.classify_page_type") as mock_classify,
            patch("curlinator.agents.documentation_agent.extract_page_metadata") as mock_extract,
        ):
            # Mock classification results
            mock_classify.side_effect = ["guide", "api_reference", "tutorial"]
            mock_extract.side_effect = [
                {
                    "title": "Auth Guide",
                    "description": "Authentication guide",
                    "headings": ["Overview", "Setup"],
                },
                {
                    "title": "Create User API",
                    "description": "API for creating users",
                    "headings": ["Request", "Response"],
                },
                {
                    "title": "Getting Started",
                    "description": "Tutorial for beginners",
                    "headings": ["Step 1", "Step 2"],
                },
            ]

            result = await agent._classify_pages(sample_documents)

            assert len(result) == 3
            assert result[0].metadata["page_type"] == "guide"
            assert result[0].metadata["title"] == "Auth Guide"
            assert result[1].metadata["page_type"] == "api_reference"
            assert result[2].metadata["page_type"] == "tutorial"

            # Verify all documents have classified_at timestamp
            for doc in result:
                assert "classified_at" in doc.metadata

    @pytest.mark.asyncio
    async def test_handles_documents_without_urls(self, mock_llm) -> None:
        """Test handling of documents without URLs in metadata"""
        agent = DocumentationAgent(llm=mock_llm)

        docs_without_urls = [Document(text="Content without URL", metadata={})]

        with (
            patch("curlinator.agents.documentation_agent.classify_page_type") as mock_classify,
            patch("curlinator.agents.documentation_agent.extract_page_metadata") as mock_extract,
        ):
            mock_classify.return_value = "unknown"
            mock_extract.return_value = {"title": "Untitled", "description": "", "headings": []}

            result = await agent._classify_pages(docs_without_urls)

            assert len(result) == 1
            assert result[0].metadata["page_type"] == "unknown"

    @pytest.mark.asyncio
    async def test_keeps_document_on_classification_error(self, mock_llm, sample_documents) -> None:
        """Test that documents are kept even if classification fails"""
        agent = DocumentationAgent(llm=mock_llm)

        with patch("curlinator.agents.documentation_agent.classify_page_type") as mock_classify:
            # Mock classification error
            mock_classify.side_effect = Exception("Classification failed")

            result = await agent._classify_pages(sample_documents)

            # Should still return all documents
            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_uses_llm_when_enabled(self, mock_llm, sample_documents) -> None:
        """Test that LLM is used for classification when enabled"""
        agent = DocumentationAgent(llm=mock_llm, use_llm_classification=True)

        with (
            patch("curlinator.agents.documentation_agent.classify_page_type") as mock_classify,
            patch("curlinator.agents.documentation_agent.extract_page_metadata") as mock_extract,
        ):
            mock_classify.return_value = "api_reference"
            mock_extract.return_value = {"title": "Test", "description": "", "headings": []}

            await agent._classify_pages(sample_documents)

            # Verify LLM was passed to classification functions
            for call in mock_classify.call_args_list:
                assert call[1]["llm"] == mock_llm
                assert call[1]["use_llm_fallback"] is True


# ============================================================================
# Test _enrich_documents Method
# ============================================================================


class TestEnrichDocuments:
    """Tests for _enrich_documents method"""

    @pytest.mark.asyncio
    async def test_successful_enrichment_adds_context(self, mock_llm, sample_documents) -> None:
        """Test successful enrichment adds contextual prefixes"""
        agent = DocumentationAgent(llm=mock_llm, enable_enrichment=True)

        with patch(
            "curlinator.agents.documentation_agent.enrich_document_with_context"
        ) as mock_enrich:
            # Mock enrichment - return modified documents
            def enrich_side_effect(doc, context):
                enriched = Document(
                    text=f"[Context: {context}] {doc.text}", metadata=doc.metadata.copy()
                )
                return enriched

            mock_enrich.side_effect = enrich_side_effect

            result = await agent._enrich_documents(sample_documents, "https://docs.example.com")

            assert len(result) == 3
            # Verify enrichment was called for each document
            assert mock_enrich.call_count == 3

            # Verify context was extracted from base_url
            for call in mock_enrich.call_args_list:
                context = call[0][1]
                assert "docs.example.com" in context
                assert "API documentation" in context

    @pytest.mark.asyncio
    async def test_extracts_site_context_from_base_url(self, mock_llm, sample_documents) -> None:
        """Test that site context is correctly extracted from base_url"""
        agent = DocumentationAgent(llm=mock_llm)

        with patch(
            "curlinator.agents.documentation_agent.enrich_document_with_context"
        ) as mock_enrich:
            mock_enrich.side_effect = lambda doc, ctx: doc

            # Test with www prefix
            await agent._enrich_documents(sample_documents, "https://www.stripe.com/docs")
            context = mock_enrich.call_args_list[0][0][1]
            assert "stripe.com" in context  # www should be removed

            mock_enrich.reset_mock()

            # Test without www
            await agent._enrich_documents(sample_documents, "https://api.github.com/docs")
            context = mock_enrich.call_args_list[0][0][1]
            assert "api.github.com" in context

    @pytest.mark.asyncio
    async def test_keeps_original_document_on_enrichment_error(
        self, mock_llm, sample_documents
    ) -> None:
        """Test that original documents are kept if enrichment fails"""
        agent = DocumentationAgent(llm=mock_llm)

        with patch(
            "curlinator.agents.documentation_agent.enrich_document_with_context"
        ) as mock_enrich:
            # Mock enrichment error
            mock_enrich.side_effect = Exception("Enrichment failed")

            result = await agent._enrich_documents(sample_documents, "https://docs.example.com")

            # Should still return all original documents
            assert len(result) == 3
            assert result[0].text == sample_documents[0].text


# ============================================================================
# Test Main execute() Method
# ============================================================================


class TestExecute:
    """Tests for main execute method"""

    @pytest.mark.asyncio
    async def test_fast_path_returns_early_when_openapi_found(
        self, mock_llm, sample_openapi_documents
    ) -> None:
        """Test fast path: returns early when OpenAPI spec is found"""
        agent = DocumentationAgent(llm=mock_llm, verbose=True)

        with (
            patch.object(agent, "_detect_openapi") as mock_detect,
            patch.object(agent, "_crawl_with_reader") as mock_crawl,
        ):
            # Mock OpenAPI detection success
            mock_detect.return_value = sample_openapi_documents

            result = await agent.execute("https://api.example.com")

            # Should return OpenAPI documents
            assert result == sample_openapi_documents
            assert len(result) == 2

            # Should NOT call crawl (fast path)
            mock_detect.assert_called_once_with("https://api.example.com")
            mock_crawl.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_workflow_when_no_openapi_spec(self, mock_llm, sample_documents) -> None:
        """Test full workflow: crawl → classify → enrich when no OpenAPI spec"""
        agent = DocumentationAgent(llm=mock_llm, enable_enrichment=True)

        # Create enriched documents for final result
        enriched_docs = [
            Document(
                text=f"[Enriched] {doc.text}",
                metadata={**doc.metadata, "page_type": "guide", "enriched": True},
            )
            for doc in sample_documents
        ]

        with (
            patch.object(agent, "_detect_openapi") as mock_detect,
            patch.object(agent, "_crawl_with_reader") as mock_crawl,
            patch.object(agent, "_classify_pages") as mock_classify,
            patch.object(agent, "_enrich_documents") as mock_enrich,
        ):
            # Mock workflow
            mock_detect.return_value = None  # No OpenAPI spec
            mock_crawl.return_value = sample_documents
            mock_classify.return_value = sample_documents
            mock_enrich.return_value = enriched_docs

            result = await agent.execute("https://docs.example.com")

            # Verify full workflow was executed
            mock_detect.assert_called_once_with("https://docs.example.com")
            mock_crawl.assert_called_once_with("https://docs.example.com")
            mock_classify.assert_called_once_with(sample_documents)
            mock_enrich.assert_called_once_with(sample_documents, "https://docs.example.com")

            # Should return enriched documents
            assert result == enriched_docs
            assert len(result) == 3
            assert all(doc.metadata.get("enriched") for doc in result)

    @pytest.mark.asyncio
    async def test_skips_enrichment_when_disabled(self, mock_llm, sample_documents) -> None:
        """Test that enrichment is skipped when enable_enrichment=False"""
        agent = DocumentationAgent(llm=mock_llm, enable_enrichment=False)

        with (
            patch.object(agent, "_detect_openapi") as mock_detect,
            patch.object(agent, "_crawl_with_reader") as mock_crawl,
            patch.object(agent, "_classify_pages") as mock_classify,
            patch.object(agent, "_enrich_documents") as mock_enrich,
        ):
            mock_detect.return_value = None
            mock_crawl.return_value = sample_documents
            mock_classify.return_value = sample_documents

            result = await agent.execute("https://docs.example.com")

            # Should NOT call enrichment
            mock_enrich.assert_not_called()

            # Should return classified documents directly
            assert result == sample_documents

    @pytest.mark.asyncio
    async def test_handles_empty_crawl_results(self, mock_llm) -> None:
        """Test handling of empty crawl results"""
        agent = DocumentationAgent(llm=mock_llm)

        with (
            patch.object(agent, "_detect_openapi") as mock_detect,
            patch.object(agent, "_crawl_with_reader") as mock_crawl,
        ):
            mock_detect.return_value = None
            mock_crawl.return_value = []  # No documents found

            result = await agent.execute("https://docs.example.com")

            # Should return empty list
            assert result == []

    @pytest.mark.asyncio
    async def test_workflow_with_all_steps(self, mock_llm) -> None:
        """Test complete workflow with all steps executed"""
        agent = DocumentationAgent(
            llm=mock_llm,
            max_depth=3,
            max_pages=10,
            enable_enrichment=True,
            use_llm_classification=True,
            verbose=True,
        )

        # Create test documents for each stage
        raw_docs = [Document(text="Raw content", metadata={"url": "https://example.com/page1"})]
        classified_docs = [
            Document(
                text="Raw content",
                metadata={"url": "https://example.com/page1", "page_type": "guide"},
            )
        ]
        enriched_docs = [
            Document(
                text="[Context] Raw content",
                metadata={"url": "https://example.com/page1", "page_type": "guide"},
            )
        ]

        with (
            patch.object(agent, "_detect_openapi") as mock_detect,
            patch.object(agent, "_crawl_with_reader") as mock_crawl,
            patch.object(agent, "_classify_pages") as mock_classify,
            patch.object(agent, "_enrich_documents") as mock_enrich,
        ):
            mock_detect.return_value = None
            mock_crawl.return_value = raw_docs
            mock_classify.return_value = classified_docs
            mock_enrich.return_value = enriched_docs

            result = await agent.execute("https://docs.example.com")

            # Verify all steps were called in order
            assert mock_detect.called
            assert mock_crawl.called
            assert mock_classify.called
            assert mock_enrich.called

            # Verify final result
            assert result == enriched_docs
            assert len(result) == 1
            assert result[0].metadata["page_type"] == "guide"


# ============================================================================
# Integration-Style Tests
# ============================================================================


class TestDocumentationAgentIntegration:
    """Integration-style tests with minimal mocking"""

    @pytest.mark.asyncio
    async def test_agent_returns_list_of_documents(self, mock_llm) -> None:
        """Test that agent always returns List[Document]"""
        agent = DocumentationAgent(llm=mock_llm)

        # Mock only the external dependencies
        with (
            patch("curlinator.agents.documentation_agent.detect_openapi_spec") as mock_detect,
            patch("curlinator.agents.documentation_agent.WholeSiteReader") as mock_reader_class,
        ):
            mock_detect.return_value = None

            mock_reader = MagicMock()
            mock_reader.load_data.return_value = []
            mock_reader_class.return_value = mock_reader

            result = await agent.execute("https://docs.example.com")

            # Should always return a list
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_agent_output_compatible_with_chat_agent(
        self, mock_llm, sample_documents
    ) -> None:
        """Test that agent output is compatible with ChatAgent"""
        agent = DocumentationAgent(llm=mock_llm)

        with (
            patch("curlinator.agents.documentation_agent.detect_openapi_spec") as mock_detect,
            patch("curlinator.agents.documentation_agent.WholeSiteReader") as mock_reader_class,
            patch("curlinator.agents.documentation_agent.classify_page_type") as mock_classify,
            patch("curlinator.agents.documentation_agent.extract_page_metadata") as mock_extract,
            patch(
                "curlinator.agents.documentation_agent.enrich_document_with_context"
            ) as mock_enrich,
        ):
            mock_detect.return_value = None

            mock_reader = MagicMock()
            mock_reader.load_data.return_value = sample_documents
            mock_reader_class.return_value = mock_reader

            mock_classify.return_value = "guide"
            mock_extract.return_value = {"title": "Test", "description": "", "headings": []}
            mock_enrich.side_effect = lambda doc, ctx: doc

            result = await agent.execute("https://docs.example.com")

            # Verify output format is compatible with ChatAgent
            assert isinstance(result, list)
            assert all(isinstance(doc, Document) for doc in result)
            assert all(hasattr(doc, "text") for doc in result)
            assert all(hasattr(doc, "metadata") for doc in result)
            assert all("page_type" in doc.metadata for doc in result)

    def test_agent_configuration_options(self, mock_llm) -> None:
        """Test various configuration options"""
        # Test minimal configuration
        agent1 = DocumentationAgent(llm=mock_llm)
        assert agent1.max_depth == 3
        assert agent1.max_pages == 50

        # Test custom configuration
        agent2 = DocumentationAgent(
            llm=mock_llm,
            max_depth=5,
            max_pages=100,
            enable_enrichment=False,
            use_llm_classification=True,
        )
        assert agent2.max_depth == 5
        assert agent2.max_pages == 100
        assert agent2.enable_enrichment is False
        assert agent2.use_llm_classification is True
