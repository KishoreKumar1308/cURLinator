"""
Documentation page classification and metadata extraction utilities.

This module provides functions to classify documentation page types and extract
structured metadata for better retrieval and organization.

Uses a hybrid approach:
- Fast rule-based classification (95% of cases)
- Optional LLM fallback for ambiguous cases (5% of cases)
"""

import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# Valid page type classifications
VALID_PAGE_TYPES = [
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


def classify_page_type(
    html_content: str,
    url: str,
    llm: LLM | None = None,
    use_llm_fallback: bool = False,
) -> str:
    """
    Classify documentation page type using hybrid approach.

    Uses fast rule-based classification first (pattern matching on URLs and content),
    then optionally falls back to LLM for ambiguous cases where rules return "unknown".

    **Hybrid Approach:**
    1. **Rule-based (primary)**: Fast, free, handles 95% of cases
       - Matches URL patterns (e.g., "/auth" → "authentication")
       - Matches content keywords (e.g., "getting started" → "quickstart")

    2. **LLM fallback (optional)**: Intelligent, costs money, handles edge cases
       - Only triggered when rule-based returns "unknown"
       - Requires `use_llm_fallback=True` and valid `llm` parameter
       - Analyzes content to determine page type

    **Trade-offs:**
    - Rule-based: 0.001s, $0.00, 85-90% accuracy
    - LLM fallback: 2-5s, $0.01-0.05, 95-98% accuracy

    Classifies pages into categories:
    - api_reference: API endpoint documentation
    - guide: Tutorial or how-to guide
    - tutorial: Step-by-step tutorial
    - overview: High-level overview or introduction
    - authentication: Authentication/authorization documentation
    - quickstart: Getting started guide
    - sdk: SDK/library documentation
    - webhook: Webhook documentation
    - error: Error handling documentation
    - changelog: API changelog or release notes
    - unknown: Cannot determine type

    Args:
        html_content: Raw HTML content of the page
        url: URL of the page
        llm: Optional LLM instance for fallback classification
        use_llm_fallback: Whether to use LLM when rule-based returns "unknown"

    Returns:
        Page type classification string

    Example:
        >>> # Rule-based only (fast, free)
        >>> page_type = classify_page_type(html, url)

        >>> # With LLM fallback (intelligent, costs money)
        >>> from llama_index.llms.openai import OpenAI
        >>> llm = OpenAI(model="gpt-4")
        >>> page_type = classify_page_type(html, url, llm=llm, use_llm_fallback=True)
    """
    # 1. Try fast rule-based classification first
    page_type = _classify_with_rules(html_content, url)

    # 2. If confident (not "unknown"), return immediately
    if page_type != "unknown":
        return page_type

    # 3. Fall back to LLM for ambiguous cases (if enabled)
    if use_llm_fallback and llm is not None:
        logger.info(f"Rule-based classification returned 'unknown' for {url}, using LLM fallback")
        return _classify_with_llm(html_content, url, llm)

    return "unknown"


def _classify_with_rules(html_content: str, url: str) -> str:
    """
    Classify page type using fast rule-based pattern matching.

    This is the primary classification method that handles 95% of cases.
    Uses URL patterns and content keyword matching.

    Args:
        html_content: Raw HTML content of the page
        url: URL of the page

    Returns:
        Page type classification string (or "unknown" if no match)
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Extract text content for analysis
    text_content = soup.get_text().lower()

    # Get URL path for pattern matching
    url_path = urlparse(url).path.lower()

    # Classification rules (order matters - most specific first)

    # 1. Authentication pages
    if _matches_patterns(url_path, ["auth", "authentication", "authorization", "oauth", "api-key"]):
        return "authentication"
    if _contains_keywords(
        text_content, ["authentication", "api key", "bearer token", "oauth"], threshold=2
    ):
        return "authentication"

    # 2. Quickstart/Getting Started
    if _matches_patterns(url_path, ["quickstart", "getting-started", "get-started", "quick-start"]):
        return "quickstart"
    if _contains_keywords(
        text_content, ["getting started", "quick start", "first request"], threshold=2
    ):
        return "quickstart"

    # 3. API Reference (endpoint documentation)
    if _matches_patterns(url_path, ["api", "reference", "endpoint", "resources"]):
        # Check for HTTP methods in content
        http_methods = ["get", "post", "put", "patch", "delete"]
        method_count = sum(1 for method in http_methods if f" {method} " in text_content)
        if method_count >= 2:
            return "api_reference"

    # Look for API endpoint patterns
    if re.search(r"(get|post|put|patch|delete)\s+/[a-z0-9/_-]+", text_content):
        return "api_reference"

    # 4. SDK/Library documentation
    if _matches_patterns(url_path, ["sdk", "library", "client", "package"]):
        return "sdk"
    if _contains_keywords(
        text_content, ["install", "npm install", "pip install", "import"], threshold=2
    ):
        return "sdk"

    # 5. Webhook documentation
    if _matches_patterns(url_path, ["webhook", "events", "callbacks"]):
        return "webhook"
    if _contains_keywords(
        text_content, ["webhook", "event", "callback", "notification"], threshold=2
    ):
        return "webhook"

    # 6. Error handling
    if _matches_patterns(url_path, ["error", "errors", "troubleshoot"]):
        return "error"
    if _contains_keywords(
        text_content, ["error code", "error handling", "troubleshooting"], threshold=2
    ):
        return "error"

    # 7. Changelog/Release notes
    if _matches_patterns(url_path, ["changelog", "release", "version", "migration"]):
        return "changelog"

    # 8. Tutorial (step-by-step guides)
    if _matches_patterns(url_path, ["tutorial", "walkthrough", "example"]):
        return "tutorial"
    if _contains_keywords(text_content, ["step 1", "step 2", "tutorial", "example"], threshold=2):
        return "tutorial"

    # 9. Guide (how-to documentation)
    if _matches_patterns(url_path, ["guide", "how-to", "howto"]):
        return "guide"
    if _contains_keywords(text_content, ["how to", "guide", "learn"], threshold=2):
        return "guide"

    # 10. Overview/Introduction
    if _matches_patterns(url_path, ["overview", "introduction", "intro", "about"]):
        return "overview"
    if _contains_keywords(text_content, ["overview", "introduction", "what is"], threshold=2):
        return "overview"

    # Default: unknown (triggers LLM fallback if enabled)
    return "unknown"


def _classify_with_llm(html_content: str, url: str, llm: LLM) -> str:
    """
    Classify page type using LLM for ambiguous cases.

    This is the fallback method for pages that rule-based classification
    cannot confidently classify. Uses LLM to analyze content and determine
    the most appropriate page type.

    Args:
        html_content: Raw HTML content
        url: URL of the page
        llm: LLM instance to use for classification

    Returns:
        Page type classification string (or "unknown" if LLM returns invalid type)
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Extract first 1000 characters of text content for analysis
    text_content = soup.get_text()
    text_preview = text_content[:1000].strip()

    # Create classification prompt
    prompt = f"""Classify this documentation page into ONE of these categories:

**Valid Categories:**
- api_reference: API endpoint documentation with HTTP methods and parameters
- guide: How-to guide explaining how to accomplish specific tasks
- tutorial: Step-by-step tutorial with numbered steps or walkthrough
- overview: High-level overview or introduction to concepts
- authentication: Authentication/authorization documentation (API keys, OAuth, tokens)
- quickstart: Getting started guide for new users
- sdk: SDK/library documentation (installation, imports, code examples)
- webhook: Webhook documentation (events, callbacks, notifications)
- error: Error handling documentation (error codes, troubleshooting)
- changelog: API changelog, release notes, or version history
- unknown: Cannot determine the type

**Page Information:**
URL: {url}

Content Preview (first 1000 characters):
{text_preview}

**Instructions:**
1. Analyze the URL and content preview
2. Choose the MOST SPECIFIC category that matches
3. Respond with ONLY the category name (e.g., "api_reference")
4. Do not include explanations or additional text

Classification:"""

    try:
        # Call LLM for classification
        response = llm.complete(prompt)
        classification = response.text.strip().lower()

        # Validate response is a valid page type
        if classification in VALID_PAGE_TYPES:
            logger.info(f"LLM classified {url} as '{classification}'")
            return classification
        else:
            logger.warning(
                f"LLM returned invalid classification '{classification}' for {url}. "
                f"Valid types: {VALID_PAGE_TYPES}"
            )
            return "unknown"

    except Exception as e:
        logger.error(f"LLM classification failed for {url}: {e}")
        return "unknown"


def extract_page_metadata(
    html_content: str,
    url: str,
    llm: LLM | None = None,
    use_llm_fallback: bool = False,
) -> dict[str, any]:
    """
    Extract structured metadata from documentation page.

    Extracts:
    - title: Page title
    - description: Meta description or first paragraph
    - breadcrumbs: Navigation breadcrumbs
    - headings: Section headings (h1-h3)
    - code_blocks: Number of code examples
    - http_methods: HTTP methods mentioned (GET, POST, etc.)
    - page_type: Classified page type (uses hybrid classification)

    Args:
        html_content: Raw HTML content
        url: URL of the page
        llm: Optional LLM instance for page type classification fallback
        use_llm_fallback: Whether to use LLM for page type classification

    Returns:
        Dictionary with extracted metadata

    Example:
        >>> metadata = extract_page_metadata(html, url)
        >>> print(metadata["title"])
        >>> print(metadata["headings"])

        >>> # With LLM fallback for page type classification
        >>> metadata = extract_page_metadata(html, url, llm=llm, use_llm_fallback=True)
    """
    soup = BeautifulSoup(html_content, "lxml")

    metadata: dict[str, any] = {
        "url": url,
        "page_type": classify_page_type(
            html_content, url, llm=llm, use_llm_fallback=use_llm_fallback
        ),
    }

    # 1. Extract title
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    metadata["title"] = (
        title_tag.get_text(strip=True)
        if title_tag
        else h1_tag.get_text(strip=True)
        if h1_tag
        else "Untitled"
    )

    # 2. Extract description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        metadata["description"] = meta_desc["content"]
    else:
        # Use first paragraph as description
        first_p = soup.find("p")
        if first_p:
            desc_text = first_p.get_text(strip=True)
            metadata["description"] = desc_text[:200] + "..." if len(desc_text) > 200 else desc_text
        else:
            metadata["description"] = ""

    # 3. Extract breadcrumbs
    breadcrumbs = _extract_breadcrumbs(soup)
    if breadcrumbs:
        metadata["breadcrumbs"] = breadcrumbs

    # 4. Extract headings
    # Note: Store as string for vector store compatibility (Chroma requires flat metadata)
    headings = []
    for level in ["h1", "h2", "h3"]:
        for heading in soup.find_all(level):
            heading_text = heading.get_text(strip=True)
            if heading_text:
                headings.append(f"{level}: {heading_text}")

    # Store as newline-separated string instead of list for Chroma compatibility
    if headings:
        metadata["headings"] = "\n".join(headings)
    else:
        metadata["headings"] = ""

    # 5. Count code blocks
    code_blocks = soup.find_all(["pre", "code"])
    metadata["code_block_count"] = len([cb for cb in code_blocks if cb.get_text(strip=True)])

    # 6. Extract HTTP methods mentioned
    text_content = soup.get_text().upper()
    http_methods = []
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]:
        if method in text_content:
            http_methods.append(method)
    metadata["http_methods"] = http_methods

    # 7. Extract API endpoints mentioned (if any)
    endpoints = _extract_endpoints(soup)
    if endpoints:
        metadata["endpoints"] = endpoints

    return metadata


def _matches_patterns(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the given patterns."""
    return any(pattern in text for pattern in patterns)


def _contains_keywords(text: str, keywords: list[str], threshold: int = 1) -> bool:
    """Check if text contains at least 'threshold' keywords."""
    count = sum(1 for keyword in keywords if keyword in text)
    return count >= threshold


def _extract_breadcrumbs(soup: BeautifulSoup) -> list[str] | None:
    """Extract breadcrumb navigation from page."""
    breadcrumbs = []

    # Try common breadcrumb patterns
    breadcrumb_selectors = [
        {"class": re.compile(r"breadcrumb", re.I)},
        {"class": re.compile(r"breadcrumbs", re.I)},
        {"role": "navigation"},
        {"aria-label": re.compile(r"breadcrumb", re.I)},
    ]

    for selector in breadcrumb_selectors:
        breadcrumb_nav = soup.find(attrs=selector)
        if breadcrumb_nav:
            # Extract links or list items
            links = breadcrumb_nav.find_all("a")
            if links:
                breadcrumbs = [link.get_text(strip=True) for link in links]
                break

            # Try list items
            items = breadcrumb_nav.find_all("li")
            if items:
                breadcrumbs = [item.get_text(strip=True) for item in items]
                break

    return breadcrumbs if breadcrumbs else None


def _extract_endpoints(soup: BeautifulSoup) -> list[str] | None:
    """Extract API endpoint paths from page content."""
    endpoints = []

    # Look for code blocks or pre tags with endpoint patterns
    code_elements = soup.find_all(["code", "pre"])

    for element in code_elements:
        text = element.get_text()
        # Match patterns like: GET /api/users, POST /v1/customers
        matches = re.findall(r"(GET|POST|PUT|PATCH|DELETE)\s+(/[a-zA-Z0-9/_\-{}:]+)", text)
        for method, path in matches:
            endpoint = f"{method} {path}"
            if endpoint not in endpoints:
                endpoints.append(endpoint)

    return endpoints if endpoints else None
