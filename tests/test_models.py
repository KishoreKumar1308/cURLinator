"""Test data models"""

from curlinator.models import (
    APIEndpoint,
    APIParameter,
    APISectionSummary,
    APISpecification,
    CodeExample,
    CrawlStatistics,
    DocumentationPage,
    DocumentationSource,
    OpenAPIInfo,
    PageSummary,
)
from curlinator.models.api_spec import HTTPMethod, ParameterLocation


def test_api_parameter_creation() -> None:
    """Test creating an API parameter"""
    param = APIParameter(
        name="user_id",
        location=ParameterLocation.PATH,
        type="string",
        description="User identifier",
        required=True,
    )
    assert param.name == "user_id"
    assert param.location == ParameterLocation.PATH
    assert param.required is True


def test_api_endpoint_creation() -> None:
    """Test creating an API endpoint"""
    endpoint = APIEndpoint(
        path="/users/{user_id}",
        method=HTTPMethod.GET,
        summary="Get user by ID",
        description="Retrieve a single user by their ID",
        auth_required=True,
    )
    assert endpoint.path == "/users/{user_id}"
    assert endpoint.method == HTTPMethod.GET
    assert endpoint.auth_required is True
    assert len(endpoint.parameters) == 0


def test_api_specification_creation() -> None:
    """Test creating an API specification"""
    spec = APISpecification(
        title="Test API",
        version="1.0.0",
        base_url="https://api.example.com",
        description="A test API specification",
    )
    assert spec.title == "Test API"
    assert spec.base_url == "https://api.example.com"
    assert len(spec.endpoints) == 0


def test_documentation_page_creation() -> None:
    """Test creating a documentation page"""
    page = DocumentationPage(
        url="https://api.example.com/docs/getting-started",
        title="Getting Started",
        content="<html><body>Documentation content</body></html>",
    )
    assert page.url == "https://api.example.com/docs/getting-started"
    assert page.title == "Getting Started"
    assert page.content_type == "html"
    assert page.summary is None


def test_documentation_source_creation() -> None:
    """Test creating a documentation source"""
    source = DocumentationSource(
        base_url="https://api.example.com/docs",
        has_openapi_spec=True,
    )
    assert str(source.base_url) == "https://api.example.com/docs"
    assert source.has_openapi_spec is True
    assert len(source.pages) == 0
    assert source.completeness_score == 0.0


def test_code_example_creation() -> None:
    """Test creating a code example"""
    example = CodeExample(
        language="python",
        code='print("Hello, World!")',
        description="Simple hello world example",
        context="Getting started guide",
    )
    assert example.language == "python"
    assert 'print("Hello, World!")' in example.code
    assert example.description == "Simple hello world example"


def test_api_section_summary_creation() -> None:
    """Test creating API section summary"""
    summary = APISectionSummary(
        endpoints_mentioned=["GET /users", "POST /users"],
        authentication_methods=["bearer", "api_key"],
        rate_limits="100 requests per minute",
    )
    assert len(summary.endpoints_mentioned) == 2
    assert "bearer" in summary.authentication_methods
    assert summary.rate_limits == "100 requests per minute"


def test_page_summary_creation() -> None:
    """Test creating page summary"""
    api_content = APISectionSummary(endpoints_mentioned=["GET /users"])
    code_example = CodeExample(language="curl", code="curl https://api.example.com")

    summary = PageSummary(
        url="https://api.example.com/docs",
        title="API Documentation",
        page_type="api_doc",
        summary="Comprehensive API documentation",
        key_topics=["authentication", "endpoints"],
        api_content=api_content,
        code_examples=[code_example],
        relevance_score=0.9,
    )
    assert summary.url == "https://api.example.com/docs"
    assert summary.page_type == "api_doc"
    assert len(summary.key_topics) == 2
    assert len(summary.code_examples) == 1
    assert summary.relevance_score == 0.9


def test_openapi_info_creation() -> None:
    """Test creating OpenAPI info"""
    info = OpenAPIInfo(
        spec_url="https://api.example.com/openapi.json",
        version="3.0",
        title="Example API",
        api_version="1.0.0",
        endpoint_count=50,
        has_authentication=True,
        base_url="https://api.example.com",
    )
    assert info.version == "3.0"
    assert info.endpoint_count == 50
    assert info.has_authentication is True


def test_crawl_statistics_creation() -> None:
    """Test creating crawl statistics"""
    stats = CrawlStatistics(
        total_pages_visited=10,
        pages_analyzed=8,
        code_examples_found=15,
        endpoints_discovered=25,
        crawl_duration_seconds=30.5,
        stopped_reason="sufficient_coverage",
    )
    assert stats.total_pages_visited == 10
    assert stats.pages_analyzed == 8
    assert stats.stopped_reason == "sufficient_coverage"


def test_documentation_source_add_page() -> None:
    """Test adding pages to documentation source"""
    source = DocumentationSource(base_url="https://api.example.com")

    # Create page with summary
    page_summary = PageSummary(
        url="https://api.example.com/docs",
        summary="Test page",
        key_topics=["auth"],
        code_examples=[CodeExample(language="curl", code="curl test")],
    )
    page = DocumentationPage(
        url="https://api.example.com/docs",
        content="<html>Test</html>",
        summary=page_summary,
    )

    source.add_page(page)

    assert len(source.pages) == 1
    assert source.crawl_statistics.total_pages_visited == 1
    assert source.crawl_statistics.pages_analyzed == 1
    assert source.crawl_statistics.code_examples_found == 1
    assert "auth" in source.key_topics


def test_documentation_source_estimate_completeness() -> None:
    """Test completeness estimation"""
    source = DocumentationSource(base_url="https://api.example.com")

    # Empty source should have low completeness
    score = source.estimate_completeness()
    assert score == 0.0

    # Add OpenAPI info
    source.has_openapi_spec = True
    source.openapi_info = OpenAPIInfo(
        spec_url="test",
        version="3.0",
        title="Test",
        api_version="1.0",
        endpoint_count=50,
        has_authentication=True,
    )

    # Add some code examples
    source.crawl_statistics.code_examples_found = 5
    source.crawl_statistics.pages_analyzed = 3
    source.key_topics = ["auth", "webhooks"]

    score = source.estimate_completeness()
    assert score > 0.5  # Should have good completeness
    assert source.completeness_score == score


def test_documentation_source_get_summary_text() -> None:
    """Test getting human-readable summary"""
    source = DocumentationSource(
        base_url="https://api.example.com",
        has_openapi_spec=True,
    )
    source.openapi_info = OpenAPIInfo(
        spec_url="test",
        version="3.0",
        title="Test API",
        api_version="1.0",
        endpoint_count=100,
        has_authentication=True,
    )
    source.crawl_statistics.total_pages_visited = 5
    source.crawl_statistics.code_examples_found = 10
    source.completeness_score = 0.85

    summary = source.get_summary_text()

    assert "api.example.com" in summary
    assert "100 endpoints" in summary
    assert "5" in summary  # pages
    assert "10" in summary  # code examples
    assert "85%" in summary  # completeness


def test_documentation_page_with_summary() -> None:
    """Test documentation page with full summary"""
    code_example = CodeExample(
        language="python",
        code="import requests",
        description="Making a request",
    )

    api_content = APISectionSummary(
        endpoints_mentioned=["GET /users", "POST /users"],
        authentication_methods=["bearer"],
    )

    page_summary = PageSummary(
        url="https://api.example.com/docs/users",
        title="Users API",
        page_type="reference",
        summary="Documentation for user management endpoints",
        key_topics=["users", "crud"],
        api_content=api_content,
        code_examples=[code_example],
        headings=["Overview", "Authentication", "Endpoints"],
        relevance_score=0.95,
    )

    page = DocumentationPage(
        url="https://api.example.com/docs/users",
        title="Users API",
        content="<html>...</html>",
        summary=page_summary,
    )

    assert page.summary is not None
    assert len(page.summary.code_examples) == 1
    assert len(page.summary.api_content.endpoints_mentioned) == 2
    assert page.summary.relevance_score == 0.95

