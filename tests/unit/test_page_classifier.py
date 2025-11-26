"""Tests for documentation page classification and metadata extraction utilities.

Tests the page_classifier module that:
- Classifies page types using rule-based pattern matching
- Falls back to LLM for ambiguous cases
- Extracts structured metadata (title, description, headings)
- Returns headings as newline-separated string for Chroma compatibility
"""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import MockLLM

from curlinator.utils.page_classifier import (
    VALID_PAGE_TYPES,
    _classify_with_llm,
    _classify_with_rules,
    _contains_keywords,
    _extract_breadcrumbs,
    _extract_endpoints,
    _matches_patterns,
    classify_page_type,
    extract_page_metadata,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    return MockLLM()


@pytest.fixture
def authentication_page_html():
    """Sample authentication page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authentication Guide</title>
        <meta name="description" content="Learn how to authenticate with our API">
    </head>
    <body>
        <h1>Authentication</h1>
        <p>This guide explains how to authenticate with our API using API keys and OAuth.</p>
        <h2>API Key Authentication</h2>
        <p>Use your API key in the Authorization header with Bearer token.</p>
        <h3>Example</h3>
        <pre><code>Authorization: Bearer YOUR_API_KEY</code></pre>
    </body>
    </html>
    """


@pytest.fixture
def api_reference_page_html():
    """Sample API reference page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Create User - API Reference</title></head>
    <body>
        <h1>Create User</h1>
        <p>POST /api/users - Creates a new user in the system</p>
        <h2>Parameters</h2>
        <p>name (string, required): User's full name</p>
        <h2>Response</h2>
        <p>Returns the created user object</p>
        <pre><code>POST /api/users</code></pre>
    </body>
    </html>
    """


@pytest.fixture
def quickstart_page_html():
    """Sample quickstart page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Getting Started</title></head>
    <body>
        <h1>Quick Start Guide</h1>
        <p>Getting started with our API is easy. Follow these steps to make your first request.</p>
        <h2>Step 1: Get your API key</h2>
        <h2>Step 2: Make your first request</h2>
    </body>
    </html>
    """


@pytest.fixture
def tutorial_page_html():
    """Sample tutorial page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Tutorial: Building a Chat App</title></head>
    <body>
        <h1>Tutorial: Building a Chat App</h1>
        <p>This step-by-step tutorial shows you how to build a chat application.</p>
        <h2>Step 1: Set up your project</h2>
        <h2>Step 2: Create the UI</h2>
        <h2>Step 3: Connect to the API</h2>
    </body>
    </html>
    """


@pytest.fixture
def sdk_page_html():
    """Sample SDK documentation page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Python SDK</title></head>
    <body>
        <h1>Python SDK</h1>
        <p>Install the SDK using pip:</p>
        <pre><code>pip install our-sdk</code></pre>
        <p>Then import it in your code:</p>
        <pre><code>import our_sdk</code></pre>
    </body>
    </html>
    """


@pytest.fixture
def webhook_page_html():
    """Sample webhook documentation page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Webhooks</title></head>
    <body>
        <h1>Webhook Events</h1>
        <p>Our API sends webhook notifications for various events and callbacks.</p>
        <h2>Event Types</h2>
        <p>user.created - Triggered when a new user is created</p>
        <p>user.updated - Triggered when a user is updated</p>
    </body>
    </html>
    """


@pytest.fixture
def error_page_html():
    """Sample error documentation page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Error Codes</title></head>
    <body>
        <h1>Error Handling</h1>
        <p>This page documents error codes and troubleshooting steps.</p>
        <h2>Common Error Codes</h2>
        <p>400 - Bad Request: Invalid input</p>
        <p>401 - Unauthorized: Invalid API key</p>
    </body>
    </html>
    """


@pytest.fixture
def changelog_page_html():
    """Sample changelog page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>API Changelog</title></head>
    <body>
        <h1>Changelog</h1>
        <h2>Version 2.0.0 - 2024-01-15</h2>
        <p>Major release with breaking changes</p>
        <h2>Version 1.5.0 - 2023-12-01</h2>
        <p>Added new endpoints</p>
    </body>
    </html>
    """


@pytest.fixture
def guide_page_html():
    """Sample guide page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>How to Use Pagination</title></head>
    <body>
        <h1>Pagination Guide</h1>
        <p>Learn how to paginate through large result sets.</p>
        <h2>How to use pagination</h2>
        <p>Use the limit and offset parameters to control pagination.</p>
    </body>
    </html>
    """


@pytest.fixture
def overview_page_html():
    """Sample overview page HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>API Overview</title></head>
    <body>
        <h1>Introduction to Our API</h1>
        <p>This overview explains what is our API and how it works.</p>
        <h2>What is the API?</h2>
        <p>Our API provides access to user data and analytics.</p>
    </body>
    </html>
    """


@pytest.fixture
def ambiguous_page_html():
    """Sample ambiguous page that's hard to classify"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Some Page</title></head>
    <body>
        <h1>Random Content</h1>
        <p>This page has no clear indicators of its type.</p>
    </body>
    </html>
    """


@pytest.fixture
def page_with_breadcrumbs():
    """Sample page with breadcrumb navigation"""
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <nav class="breadcrumbs">
            <a href="/">Home</a>
            <a href="/docs">Documentation</a>
            <a href="/docs/api">API Reference</a>
        </nav>
        <h1>API Endpoint</h1>
    </body>
    </html>
    """


@pytest.fixture
def page_with_endpoints():
    """Sample page with API endpoints"""
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>User Management</h1>
        <pre><code>GET /api/users</code></pre>
        <pre><code>POST /api/users</code></pre>
        <pre><code>DELETE /api/users/{id}</code></pre>
    </body>
    </html>
    """


# ============================================================================
# Test classify_page_type (Rule-based)
# ============================================================================


class TestClassifyPageTypeRuleBased:
    """Tests for rule-based page type classification"""

    def test_classifies_authentication_page_by_url(self, authentication_page_html):
        """Test classifies authentication page by URL pattern"""
        result = classify_page_type(authentication_page_html, "https://docs.example.com/auth")
        assert result == "authentication"

    def test_classifies_authentication_page_by_content(self, authentication_page_html):
        """Test classifies authentication page by content keywords"""
        result = classify_page_type(authentication_page_html, "https://docs.example.com/some-page")
        assert result == "authentication"

    def test_classifies_quickstart_page(self, quickstart_page_html):
        """Test classifies quickstart/getting started page"""
        result = classify_page_type(
            quickstart_page_html, "https://docs.example.com/getting-started"
        )
        assert result == "quickstart"

    def test_classifies_api_reference_page(self, api_reference_page_html):
        """Test classifies API reference page with HTTP methods"""
        result = classify_page_type(
            api_reference_page_html, "https://docs.example.com/api/users/create"
        )
        assert result == "api_reference"

    def test_classifies_sdk_page(self, sdk_page_html):
        """Test classifies SDK documentation page"""
        result = classify_page_type(sdk_page_html, "https://docs.example.com/sdk/python")
        assert result == "sdk"

    def test_classifies_webhook_page(self, webhook_page_html):
        """Test classifies webhook documentation page"""
        result = classify_page_type(webhook_page_html, "https://docs.example.com/webhooks")
        assert result == "webhook"

    def test_classifies_error_page(self, error_page_html):
        """Test classifies error documentation page"""
        result = classify_page_type(error_page_html, "https://docs.example.com/errors")
        assert result == "error"

    def test_classifies_changelog_page(self, changelog_page_html):
        """Test classifies changelog page"""
        result = classify_page_type(changelog_page_html, "https://docs.example.com/changelog")
        assert result == "changelog"

    def test_classifies_tutorial_page(self, tutorial_page_html):
        """Test classifies tutorial page"""
        result = classify_page_type(
            tutorial_page_html, "https://docs.example.com/tutorials/chat-app"
        )
        assert result == "tutorial"

    def test_classifies_guide_page(self, guide_page_html):
        """Test classifies guide page"""
        result = classify_page_type(guide_page_html, "https://docs.example.com/guides/pagination")
        assert result == "guide"

    def test_classifies_overview_page(self, overview_page_html):
        """Test classifies overview page"""
        result = classify_page_type(overview_page_html, "https://docs.example.com/overview")
        assert result == "overview"

    def test_returns_unknown_for_ambiguous_page(self, ambiguous_page_html):
        """Test returns 'unknown' for ambiguous pages"""
        result = classify_page_type(ambiguous_page_html, "https://docs.example.com/random")
        assert result == "unknown"

    def test_prioritizes_specific_classifications(self):
        """Test prioritizes more specific classifications over generic ones"""
        # Page with both auth and API reference indicators
        html = """
        <html>
        <body>
            <h1>Authentication API</h1>
            <p>Use API key for authentication</p>
            <p>GET /auth/token</p>
        </body>
        </html>
        """
        result = classify_page_type(html, "https://docs.example.com/auth")
        # Should classify as authentication (more specific) not api_reference
        assert result == "authentication"


# ============================================================================
# Test classify_page_type (LLM Fallback)
# ============================================================================


class TestClassifyPageTypeLLMFallback:
    """Tests for LLM fallback classification"""

    def test_uses_llm_fallback_when_enabled(self, ambiguous_page_html):
        """Test uses LLM fallback when rule-based returns 'unknown'"""
        # Mock the _classify_with_llm function directly
        with patch(
            "curlinator.utils.page_classifier._classify_with_llm", return_value="guide"
        ) as mock_classify:
            result = classify_page_type(
                ambiguous_page_html,
                "https://docs.example.com/random",
                llm=MockLLM(),
                use_llm_fallback=True,
            )

            assert result == "guide"
            mock_classify.assert_called_once()

    def test_does_not_use_llm_when_rule_based_succeeds(self, authentication_page_html):
        """Test does not use LLM when rule-based classification succeeds"""
        with patch("curlinator.utils.page_classifier._classify_with_llm") as mock_classify:
            result = classify_page_type(
                authentication_page_html,
                "https://docs.example.com/auth",
                llm=MockLLM(),
                use_llm_fallback=True,
            )

            assert result == "authentication"
            # LLM should not be called
            mock_classify.assert_not_called()

    def test_does_not_use_llm_when_disabled(self, ambiguous_page_html):
        """Test does not use LLM when use_llm_fallback=False"""
        with patch("curlinator.utils.page_classifier._classify_with_llm") as mock_classify:
            result = classify_page_type(
                ambiguous_page_html,
                "https://docs.example.com/random",
                llm=MockLLM(),
                use_llm_fallback=False,
            )

            assert result == "unknown"
            mock_classify.assert_not_called()

    def test_returns_unknown_when_llm_returns_invalid_type(self, ambiguous_page_html):
        """Test returns 'unknown' when LLM returns invalid page type"""
        # Test _classify_with_llm directly to test validation logic
        mock_llm = MockLLM()
        mock_response = MagicMock()
        mock_response.text = "invalid_type"

        with patch.object(type(mock_llm), "complete", return_value=mock_response):
            result = _classify_with_llm(
                ambiguous_page_html, "https://docs.example.com/random", mock_llm
            )

            assert result == "unknown"

    def test_handles_llm_error_gracefully(self, ambiguous_page_html):
        """Test handles LLM errors gracefully"""
        # Test _classify_with_llm directly to test error handling
        mock_llm = MockLLM()

        with patch.object(type(mock_llm), "complete", side_effect=Exception("LLM error")):
            result = _classify_with_llm(
                ambiguous_page_html, "https://docs.example.com/random", mock_llm
            )

            assert result == "unknown"


# ============================================================================
# Test extract_page_metadata
# ============================================================================


class TestExtractPageMetadata:
    """Tests for page metadata extraction"""

    def test_extracts_title_from_title_tag(self, authentication_page_html):
        """Test extracts title from <title> tag"""
        metadata = extract_page_metadata(authentication_page_html, "https://docs.example.com/auth")

        assert metadata["title"] == "Authentication Guide"

    def test_extracts_title_from_h1_when_no_title_tag(self):
        """Test falls back to h1 when no title tag"""
        html = """
        <html>
        <body>
            <h1>My Page Title</h1>
            <p>Content</p>
        </body>
        </html>
        """
        metadata = extract_page_metadata(html, "https://docs.example.com/page")

        assert metadata["title"] == "My Page Title"

    def test_extracts_description_from_meta_tag(self, authentication_page_html):
        """Test extracts description from meta description tag"""
        metadata = extract_page_metadata(authentication_page_html, "https://docs.example.com/auth")

        assert metadata["description"] == "Learn how to authenticate with our API"

    def test_extracts_description_from_first_paragraph(self):
        """Test falls back to first paragraph when no meta description"""
        html = """
        <html>
        <body>
            <h1>Title</h1>
            <p>This is the first paragraph that should be used as description.</p>
            <p>This is the second paragraph.</p>
        </body>
        </html>
        """
        metadata = extract_page_metadata(html, "https://docs.example.com/page")

        assert "first paragraph" in metadata["description"]

    def test_truncates_long_description(self):
        """Test truncates description longer than 200 characters"""
        long_text = "A" * 250
        html = f"""
        <html>
        <body>
            <h1>Title</h1>
            <p>{long_text}</p>
        </body>
        </html>
        """
        metadata = extract_page_metadata(html, "https://docs.example.com/page")

        assert len(metadata["description"]) <= 203  # 200 + "..."
        assert metadata["description"].endswith("...")

    def test_extracts_headings_as_string(self, authentication_page_html):
        """Test extracts headings as newline-separated string for Chroma compatibility"""
        metadata = extract_page_metadata(authentication_page_html, "https://docs.example.com/auth")

        # Should be a string, not a list
        assert isinstance(metadata["headings"], str)

        # Should contain all headings
        assert "h1: Authentication" in metadata["headings"]
        assert "h2: API Key Authentication" in metadata["headings"]
        assert "h3: Example" in metadata["headings"]

        # Should be newline-separated
        assert "\n" in metadata["headings"]

    def test_returns_empty_string_when_no_headings(self):
        """Test returns empty string when no headings found"""
        html = """
        <html>
        <body>
            <p>No headings here</p>
        </body>
        </html>
        """
        metadata = extract_page_metadata(html, "https://docs.example.com/page")

        assert metadata["headings"] == ""

    def test_extracts_page_type(self, authentication_page_html):
        """Test extracts page type classification"""
        metadata = extract_page_metadata(authentication_page_html, "https://docs.example.com/auth")

        assert metadata["page_type"] == "authentication"

    def test_counts_code_blocks(self, api_reference_page_html):
        """Test counts code blocks in page"""
        metadata = extract_page_metadata(
            api_reference_page_html, "https://docs.example.com/api/users"
        )

        assert metadata["code_block_count"] >= 1

    def test_extracts_http_methods(self, api_reference_page_html):
        """Test extracts HTTP methods mentioned in page"""
        metadata = extract_page_metadata(
            api_reference_page_html, "https://docs.example.com/api/users"
        )

        assert "POST" in metadata["http_methods"]

    def test_extracts_breadcrumbs(self, page_with_breadcrumbs):
        """Test extracts breadcrumb navigation"""
        metadata = extract_page_metadata(
            page_with_breadcrumbs, "https://docs.example.com/docs/api/endpoint"
        )

        assert "breadcrumbs" in metadata
        assert "Home" in metadata["breadcrumbs"]
        assert "Documentation" in metadata["breadcrumbs"]
        assert "API Reference" in metadata["breadcrumbs"]

    def test_extracts_endpoints(self, page_with_endpoints):
        """Test extracts API endpoints from page"""
        metadata = extract_page_metadata(page_with_endpoints, "https://docs.example.com/api/users")

        assert "endpoints" in metadata
        assert "GET /api/users" in metadata["endpoints"]
        assert "POST /api/users" in metadata["endpoints"]
        assert "DELETE /api/users/{id}" in metadata["endpoints"]

    def test_includes_url_in_metadata(self, authentication_page_html):
        """Test includes URL in metadata"""
        url = "https://docs.example.com/auth"
        metadata = extract_page_metadata(authentication_page_html, url)

        assert metadata["url"] == url

    def test_handles_empty_page(self):
        """Test handles empty page gracefully"""
        html = "<html><body></body></html>"
        metadata = extract_page_metadata(html, "https://docs.example.com/empty")

        assert metadata["title"] == "Untitled"
        assert metadata["description"] == ""
        assert metadata["headings"] == ""
        assert metadata["page_type"] == "unknown"

    def test_handles_malformed_html(self):
        """Test handles malformed HTML gracefully"""
        html = "<html><body><h1>Broken HTML"
        metadata = extract_page_metadata(html, "https://docs.example.com/broken")

        # Should not crash, should extract what it can
        assert "title" in metadata
        assert "page_type" in metadata


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions"""

    def test_matches_patterns_returns_true_when_match(self):
        """Test _matches_patterns returns True when pattern matches"""
        assert _matches_patterns("auth/login", ["auth", "login"]) is True
        assert _matches_patterns("api/reference", ["reference"]) is True

    def test_matches_patterns_returns_false_when_no_match(self):
        """Test _matches_patterns returns False when no pattern matches"""
        assert _matches_patterns("random/path", ["auth", "api"]) is False

    def test_contains_keywords_with_threshold(self):
        """Test _contains_keywords respects threshold"""
        text = "authentication api key bearer token"

        # Should pass with threshold=2 (has 3 keywords)
        assert (
            _contains_keywords(text, ["authentication", "api key", "bearer"], threshold=2) is True
        )

        # Should fail with threshold=5 (only has 3 keywords)
        assert (
            _contains_keywords(
                text, ["authentication", "api key", "bearer", "oauth", "jwt"], threshold=5
            )
            is False
        )

    def test_extract_breadcrumbs_from_various_patterns(self):
        """Test _extract_breadcrumbs handles various HTML patterns"""
        from bs4 import BeautifulSoup

        # Pattern 1: class="breadcrumb"
        html1 = '<nav class="breadcrumb"><a>Home</a><a>Docs</a></nav>'
        soup1 = BeautifulSoup(html1, "lxml")
        result1 = _extract_breadcrumbs(soup1)
        assert result1 == ["Home", "Docs"]

        # Pattern 2: role="navigation"
        html2 = '<nav role="navigation"><li>Home</li><li>Docs</li></nav>'
        soup2 = BeautifulSoup(html2, "lxml")
        result2 = _extract_breadcrumbs(soup2)
        assert result2 == ["Home", "Docs"]

    def test_extract_breadcrumbs_returns_none_when_not_found(self):
        """Test _extract_breadcrumbs returns None when no breadcrumbs found"""
        from bs4 import BeautifulSoup

        html = "<html><body><p>No breadcrumbs</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = _extract_breadcrumbs(soup)

        assert result is None

    def test_extract_endpoints_from_code_blocks(self):
        """Test _extract_endpoints extracts endpoints from code blocks"""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <pre><code>GET /api/users</code></pre>
            <pre><code>POST /v1/customers</code></pre>
            <code>DELETE /api/users/{id}</code>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        result = _extract_endpoints(soup)

        assert "GET /api/users" in result
        assert "POST /v1/customers" in result
        assert "DELETE /api/users/{id}" in result

    def test_extract_endpoints_returns_none_when_not_found(self):
        """Test _extract_endpoints returns None when no endpoints found"""
        from bs4 import BeautifulSoup

        html = "<html><body><p>No endpoints here</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = _extract_endpoints(soup)

        assert result is None

    def test_extract_endpoints_deduplicates(self):
        """Test _extract_endpoints removes duplicates"""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <pre><code>GET /api/users</code></pre>
            <pre><code>GET /api/users</code></pre>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        result = _extract_endpoints(soup)

        # Should only have one instance
        assert result.count("GET /api/users") == 1


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_valid_page_types_constant(self):
        """Test VALID_PAGE_TYPES contains expected types"""
        expected_types = [
            "api_reference",
            "guide",
            "tutorial",
            "overview",
            "authentication",
            "quickstart",
            "sdk",
            "webhook",
            "error",
            "changelog",
            "unknown",
        ]

        for expected in expected_types:
            assert expected in VALID_PAGE_TYPES

    def test_classify_with_rules_is_case_insensitive(self):
        """Test rule-based classification is case-insensitive"""
        html_upper = "<html><body><h1>AUTHENTICATION</h1><p>API KEY</p></body></html>"
        html_lower = "<html><body><h1>authentication</h1><p>api key</p></body></html>"

        result_upper = _classify_with_rules(html_upper, "https://example.com")
        result_lower = _classify_with_rules(html_lower, "https://example.com")

        assert result_upper == result_lower == "authentication"

    def test_handles_unicode_content(self):
        """Test handles Unicode content in HTML"""
        html = """
        <html>
        <head><title>认证指南</title></head>
        <body>
            <h1>Authentication 認証</h1>
            <p>API キー authentication</p>
        </body>
        </html>
        """
        metadata = extract_page_metadata(html, "https://docs.example.com/auth")

        assert "认证指南" in metadata["title"]
        assert "Authentication 認証" in metadata["headings"]
