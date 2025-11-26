"""URL manipulation and validation utilities"""

from urllib.parse import urljoin, urlparse, urlunparse


def normalize_url(url: str) -> str:
    """
    Normalize a URL by standardizing format.
    
    - Ensures https:// scheme if no scheme provided
    - Removes trailing slashes
    - Removes fragment identifiers
    - Lowercases domain
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL string
        
    Examples:
        >>> normalize_url("example.com/docs/")
        "https://example.com/docs"
        >>> normalize_url("HTTP://EXAMPLE.COM/Docs#section")
        "https://example.com/Docs"
    """
    url = url.strip()

    # Add scheme if missing (case-insensitive check)
    if not url.lower().startswith(("http://", "https://")):
        url = f"https://{url}"

    parsed = urlparse(url)

    # Normalize components
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path != "/" else ""

    # Rebuild without fragment and query (for base URLs)
    normalized = urlunparse((
        scheme,
        netloc,
        path,
        "",  # params
        parsed.query,
        ""   # fragment
    ))

    return normalized


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL has valid scheme and netloc
        
    Examples:
        >>> is_valid_url("https://example.com")
        True
        >>> is_valid_url("not a url")
        False
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def guess_openapi_paths(base_url: str) -> list[str]:
    """
    Generate common OpenAPI/Swagger specification paths.
    
    Args:
        base_url: Base URL of the API documentation
        
    Returns:
        List of possible OpenAPI spec URLs to try
        
    Examples:
        >>> guess_openapi_paths("https://api.example.com")
        ['https://api.example.com/openapi.json', ...]
    """
    base_url = normalize_url(base_url)

    common_paths = [
        "/openapi.json",
        "/openapi.yaml",
        "/swagger.json",
        "/swagger.yaml",
        "/api-docs",
        "/api-docs.json",
        "/v1/openapi.json",
        "/v2/openapi.json",
        "/v3/openapi.json",
        "/docs/openapi.json",
        "/api/openapi.json",
        "/api/swagger.json",
        "/.well-known/openapi.json",
    ]

    return [f"{base_url}{path}" for path in common_paths]


def extract_domain(url: str) -> str:
    """
    Extract the domain (netloc) from a URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        Domain string (scheme + netloc)
        
    Examples:
        >>> extract_domain("https://api.example.com/docs/endpoint")
        "https://api.example.com"
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def is_documentation_url(url: str) -> bool:
    """
    Check if a URL looks like documentation.
    
    Uses heuristics to identify documentation URLs based on common patterns.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears to be documentation
        
    Examples:
        >>> is_documentation_url("https://api.example.com/docs")
        True
        >>> is_documentation_url("https://example.com/login")
        False
    """
    url_lower = url.lower()

    doc_indicators = [
        "/docs",
        "/documentation",
        "/api",
        "/reference",
        "/guide",
        "/developer",
        "/api-reference",
        "/swagger",
        "/openapi",
        "/rest",
        "/graphql",
        "developers.",
        "docs.",
        "api.",
    ]

    return any(indicator in url_lower for indicator in doc_indicators)


def build_full_url(base_url: str, path: str) -> str:
    """
    Build a full URL from base URL and relative path.
    
    Handles various path formats (absolute, relative, with/without leading slash).
    
    Args:
        base_url: Base URL
        path: Path to append (can be relative or absolute)
        
    Returns:
        Complete URL
        
    Examples:
        >>> build_full_url("https://example.com", "/docs/api")
        "https://example.com/docs/api"
        >>> build_full_url("https://example.com/v1", "users")
        "https://example.com/v1/users"
    """
    # If path is already a full URL, return it
    if path.startswith(("http://", "https://")):
        return path

    return urljoin(base_url, path)


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs are from the same domain.
    
    Args:
        url1: First URL
        url2: Second URL
        
    Returns:
        True if both URLs have the same domain
        
    Examples:
        >>> is_same_domain("https://api.example.com/v1", "https://api.example.com/v2")
        True
        >>> is_same_domain("https://api.example.com", "https://docs.example.com")
        False
    """
    try:
        domain1 = urlparse(url1).netloc.lower()
        domain2 = urlparse(url2).netloc.lower()
        return domain1 == domain2
    except Exception:
        return False


def get_base_path(url: str) -> str:
    """
    Get the base path from a URL (path without last segment).
    
    Useful for finding parent documentation pages.
    
    Args:
        url: URL to extract base path from
        
    Returns:
        URL with base path
        
    Examples:
        >>> get_base_path("https://example.com/docs/api/users")
        "https://example.com/docs/api"
        >>> get_base_path("https://example.com/docs")
        "https://example.com"
    """
    parsed = urlparse(url)
    path_segments = parsed.path.rstrip("/").split("/")

    if len(path_segments) > 1:
        base_path = "/".join(path_segments[:-1])
    else:
        base_path = ""

    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        base_path,
        "",
        "",
        ""
    ))


def is_external_url(base_url: str, target_url: str) -> bool:
    """
    Check if target URL is external to base URL domain.
    
    Args:
        base_url: Reference URL
        target_url: URL to check
        
    Returns:
        True if target is external to base domain
        
    Examples:
        >>> is_external_url("https://example.com", "https://other.com")
        True
        >>> is_external_url("https://example.com", "/docs/api")
        False
    """
    # Relative URLs are not external
    if not target_url.startswith(("http://", "https://")):
        return False

    return not is_same_domain(base_url, target_url)
