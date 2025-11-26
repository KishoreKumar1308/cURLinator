"""Documentation source data models"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class CodeExample(BaseModel):
    """A code example extracted from documentation"""

    language: str  # python, javascript, curl, bash, etc.
    code: str
    description: str | None = None
    context: str | None = None  # What page/section it came from


class APISectionSummary(BaseModel):
    """Summary of API-related content from a page"""

    endpoints_mentioned: list[str] = Field(default_factory=list)
    authentication_methods: list[str] = Field(default_factory=list)
    rate_limits: str | None = None
    webhooks_info: str | None = None
    error_codes: list[str] = Field(default_factory=list)


class PageSummary(BaseModel):
    """Analyzed summary of a documentation page"""

    url: str
    title: str | None = None
    page_type: Literal["tutorial", "reference", "guide", "api_doc", "other"] = "other"

    # LLM-generated summary
    summary: str
    key_topics: list[str] = Field(default_factory=list)

    # Extracted structured content
    api_content: APISectionSummary = Field(default_factory=APISectionSummary)
    code_examples: list[CodeExample] = Field(default_factory=list)

    # Metadata
    headings: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0  # 0-1, how relevant/useful is this page
    crawled_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentationPage(BaseModel):
    """Single documentation page with raw content and analysis"""

    url: str
    title: str | None = None
    content: str  # Raw HTML/text content
    content_type: str = "html"
    summary: PageSummary | None = None  # Analyzed summary (optional)
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)


class OpenAPIInfo(BaseModel):
    """Information about discovered OpenAPI specification"""

    spec_url: str
    version: str  # "2.0", "3.0", "3.1"
    title: str
    api_version: str
    endpoint_count: int
    has_authentication: bool
    base_url: str | None = None
    spec_content: dict[str, Any] | None = None  # The actual OpenAPI spec


class CrawlStatistics(BaseModel):
    """Statistics about the crawling process"""

    total_pages_visited: int = 0
    pages_analyzed: int = 0
    code_examples_found: int = 0
    endpoints_discovered: int = 0
    crawl_duration_seconds: float = 0.0
    stopped_reason: str = "completed"  # completed, max_pages, sufficient_coverage, error


class DocumentationSource(BaseModel):
    """Complete documentation source with structured data"""

    # Basic information
    base_url: HttpUrl
    title: str | None = None
    description: str | None = None

    # OpenAPI specification (if found)
    has_openapi_spec: bool = False
    openapi_info: OpenAPIInfo | None = None

    # Crawled pages with raw content
    pages: list[DocumentationPage] = Field(default_factory=list)

    # Aggregated data (derived from pages)
    all_code_examples: list[CodeExample] = Field(default_factory=list)
    all_endpoints_mentioned: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)

    # Crawl metadata
    crawl_statistics: CrawlStatistics = Field(default_factory=CrawlStatistics)
    crawl_strategy: str = "intelligent_react"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Overall assessment
    completeness_score: float = 0.0  # 0-1, estimated completeness
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_summary_text(self) -> str:
        """Generate human-readable summary of the documentation source"""
        spec_info = (
            f"OpenAPI spec found ({self.openapi_info.endpoint_count} endpoints)"
            if self.has_openapi_spec and self.openapi_info
            else "No OpenAPI spec"
        )
        return f"""Documentation Source: {self.base_url}
- {spec_info}
- Pages crawled: {self.crawl_statistics.total_pages_visited}
- Code examples: {self.crawl_statistics.code_examples_found}
- Completeness: {self.completeness_score:.0%}"""

    def add_page(self, page: DocumentationPage) -> None:
        """Add a page and update aggregated statistics"""
        self.pages.append(page)
        self.crawl_statistics.total_pages_visited += 1

        # Update aggregated data if page has summary
        if page.summary:
            self.crawl_statistics.pages_analyzed += 1

            # Add code examples (with deduplication)
            for code_ex in page.summary.code_examples:
                # Check if this exact code already exists
                code_hash = hash(code_ex.code)
                existing_hashes = {hash(ex.code) for ex in self.all_code_examples}

                if code_hash not in existing_hashes:
                    self.all_code_examples.append(code_ex)
                    self.crawl_statistics.code_examples_found += 1

            # Add endpoints (with deduplication)
            for endpoint in page.summary.api_content.endpoints_mentioned:
                if endpoint not in self.all_endpoints_mentioned:
                    self.all_endpoints_mentioned.append(endpoint)

            # Merge key topics (unique)
            for topic in page.summary.key_topics:
                if topic not in self.key_topics:
                    self.key_topics.append(topic)

    def estimate_completeness(self) -> float:
        """
        Estimate documentation completeness based on what was found.

        Returns value between 0 and 1.
        """
        score = 0.0

        # OpenAPI spec found (40% of completeness)
        if self.has_openapi_spec and self.openapi_info:
            if self.openapi_info.endpoint_count > 0:
                score += 0.4

        # Code examples found (30% of completeness)
        if self.crawl_statistics.code_examples_found > 0:
            # Diminishing returns: 1 example = 0.1, 5+ examples = 0.3
            example_score = min(self.crawl_statistics.code_examples_found * 0.06, 0.3)
            score += example_score

        # Multiple pages analyzed (20% of completeness)
        if self.crawl_statistics.pages_analyzed > 0:
            # 1 page = 0.05, 4+ pages = 0.2
            page_score = min(self.crawl_statistics.pages_analyzed * 0.05, 0.2)
            score += page_score

        # Key topics covered (10% of completeness)
        if len(self.key_topics) > 0:
            topic_score = min(len(self.key_topics) * 0.02, 0.1)
            score += topic_score

        self.completeness_score = min(score, 1.0)
        return self.completeness_score
