"""Tests for URL helper utilities"""

from curlinator.utils.url_helpers import (
    build_full_url,
    extract_domain,
    get_base_path,
    guess_openapi_paths,
    is_documentation_url,
    is_external_url,
    is_same_domain,
    is_valid_url,
    normalize_url,
)


class TestNormalizeUrl:
    """Tests for normalize_url function"""

    def test_adds_https_scheme_if_missing(self) -> None:
        """Test that https:// is added if no scheme provided"""
        assert normalize_url("example.com") == "https://example.com"
        assert normalize_url("api.example.com/docs") == "https://api.example.com/docs"

    def test_removes_trailing_slash(self) -> None:
        """Test that trailing slashes are removed"""
        assert normalize_url("https://example.com/") == "https://example.com"
        assert normalize_url("https://example.com/docs/") == "https://example.com/docs"

    def test_removes_fragment(self) -> None:
        """Test that URL fragments are removed"""
        assert (
            normalize_url("https://example.com/docs#section")
            == "https://example.com/docs"
        )

    def test_lowercases_domain(self) -> None:
        """Test that domain is lowercased"""
        assert normalize_url("https://EXAMPLE.COM") == "https://example.com"
        assert (
            normalize_url("https://API.EXAMPLE.COM/Docs")
            == "https://api.example.com/Docs"
        )

    def test_preserves_path_case(self) -> None:
        """Test that path case is preserved"""
        assert (
            normalize_url("https://example.com/API/Docs")
            == "https://example.com/API/Docs"
        )

    def test_preserves_query_parameters(self) -> None:
        """Test that query parameters are preserved"""
        assert (
            normalize_url("https://example.com/docs?version=v1")
            == "https://example.com/docs?version=v1"
        )

    def test_handles_http_scheme(self) -> None:
        """Test that http:// scheme is preserved"""
        assert normalize_url("http://example.com") == "http://example.com"

    def test_handles_complex_url(self) -> None:
        """Test normalization of complex URL"""
        input_url = "HTTP://API.EXAMPLE.COM:8080/docs/API/#section"
        expected = "http://api.example.com:8080/docs/API"
        assert normalize_url(input_url) == expected


class TestIsValidUrl:
    """Tests for is_valid_url function"""

    def test_valid_https_url(self) -> None:
        """Test that valid HTTPS URLs are recognized"""
        assert is_valid_url("https://example.com") is True
        assert is_valid_url("https://api.example.com/docs") is True

    def test_valid_http_url(self) -> None:
        """Test that valid HTTP URLs are recognized"""
        assert is_valid_url("http://example.com") is True

    def test_url_with_port(self) -> None:
        """Test URLs with port numbers"""
        assert is_valid_url("https://example.com:8080") is True

    def test_invalid_url_without_scheme(self) -> None:
        """Test that URLs without scheme are invalid"""
        assert is_valid_url("example.com") is False

    def test_invalid_url_without_domain(self) -> None:
        """Test that URLs without domain are invalid"""
        assert is_valid_url("https://") is False

    def test_invalid_random_string(self) -> None:
        """Test that random strings are invalid"""
        assert is_valid_url("not a url") is False
        assert is_valid_url("") is False

    def test_invalid_malformed_url(self) -> None:
        """Test that malformed URLs are invalid"""
        assert is_valid_url("ht!tp://example.com") is False


class TestGuessOpenapiPaths:
    """Tests for guess_openapi_paths function"""

    def test_returns_multiple_paths(self) -> None:
        """Test that multiple OpenAPI path variations are returned"""
        paths = guess_openapi_paths("https://api.example.com")
        assert len(paths) > 5
        assert all(isinstance(p, str) for p in paths)

    def test_includes_common_paths(self) -> None:
        """Test that common OpenAPI paths are included"""
        paths = guess_openapi_paths("https://api.example.com")
        assert "https://api.example.com/openapi.json" in paths
        assert "https://api.example.com/swagger.json" in paths
        assert "https://api.example.com/api-docs" in paths

    def test_includes_versioned_paths(self) -> None:
        """Test that versioned API paths are included"""
        paths = guess_openapi_paths("https://api.example.com")
        assert "https://api.example.com/v1/openapi.json" in paths
        assert "https://api.example.com/v2/openapi.json" in paths

    def test_normalizes_input_url(self) -> None:
        """Test that input URL is normalized"""
        paths1 = guess_openapi_paths("api.example.com/")
        paths2 = guess_openapi_paths("https://api.example.com")
        assert paths1 == paths2

    def test_includes_yaml_variants(self) -> None:
        """Test that YAML format paths are included"""
        paths = guess_openapi_paths("https://api.example.com")
        assert "https://api.example.com/openapi.yaml" in paths
        assert "https://api.example.com/swagger.yaml" in paths


class TestExtractDomain:
    """Tests for extract_domain function"""

    def test_extracts_domain_from_url(self) -> None:
        """Test basic domain extraction"""
        assert extract_domain("https://example.com/docs") == "https://example.com"

    def test_extracts_domain_with_subdomain(self) -> None:
        """Test domain extraction with subdomain"""
        assert extract_domain("https://api.example.com/v1/users") == "https://api.example.com"

    def test_extracts_domain_with_port(self) -> None:
        """Test domain extraction with port"""
        assert extract_domain("https://example.com:8080/docs") == "https://example.com:8080"

    def test_preserves_scheme(self) -> None:
        """Test that URL scheme is preserved"""
        assert extract_domain("http://example.com/docs") == "http://example.com"
        assert extract_domain("https://example.com/docs") == "https://example.com"


class TestIsDocumentationUrl:
    """Tests for is_documentation_url function"""

    def test_recognizes_docs_urls(self) -> None:
        """Test recognition of /docs URLs"""
        assert is_documentation_url("https://example.com/docs") is True
        assert is_documentation_url("https://example.com/docs/api") is True

    def test_recognizes_api_urls(self) -> None:
        """Test recognition of API URLs"""
        assert is_documentation_url("https://example.com/api") is True
        assert is_documentation_url("https://api.example.com") is True

    def test_recognizes_developer_urls(self) -> None:
        """Test recognition of developer URLs"""
        assert is_documentation_url("https://developers.example.com") is True
        assert is_documentation_url("https://example.com/developer") is True

    def test_recognizes_reference_urls(self) -> None:
        """Test recognition of reference URLs"""
        assert is_documentation_url("https://example.com/reference") is True
        assert is_documentation_url("https://example.com/api-reference") is True

    def test_recognizes_swagger_urls(self) -> None:
        """Test recognition of Swagger/OpenAPI URLs"""
        assert is_documentation_url("https://example.com/swagger") is True
        assert is_documentation_url("https://example.com/openapi") is True

    def test_rejects_non_documentation_urls(self) -> None:
        """Test that non-documentation URLs are rejected"""
        assert is_documentation_url("https://example.com/login") is False
        assert is_documentation_url("https://example.com/about") is False
        assert is_documentation_url("https://blog.example.com") is False

    def test_case_insensitive(self) -> None:
        """Test that detection is case-insensitive"""
        assert is_documentation_url("https://example.com/DOCS") is True
        assert is_documentation_url("https://API.example.com") is True


class TestBuildFullUrl:
    """Tests for build_full_url function"""

    def test_builds_url_with_absolute_path(self) -> None:
        """Test building URL with absolute path"""
        result = build_full_url("https://example.com", "/docs/api")
        assert result == "https://example.com/docs/api"

    def test_builds_url_with_relative_path(self) -> None:
        """Test building URL with relative path"""
        result = build_full_url("https://example.com/v1", "users")
        assert result == "https://example.com/users"

    def test_handles_base_with_path(self) -> None:
        """Test building URL when base already has path"""
        result = build_full_url("https://example.com/api/v1/", "users")
        assert result == "https://example.com/api/v1/users"

    def test_returns_full_url_unchanged(self) -> None:
        """Test that full URLs are returned unchanged"""
        full_url = "https://other.com/docs"
        result = build_full_url("https://example.com", full_url)
        assert result == full_url

    def test_handles_parent_path(self) -> None:
        """Test building URL with parent path reference"""
        result = build_full_url("https://example.com/docs/api/", "../guide")
        assert result == "https://example.com/docs/guide"


class TestIsSameDomain:
    """Tests for is_same_domain function"""

    def test_same_domain_returns_true(self) -> None:
        """Test that same domains are recognized"""
        assert is_same_domain(
            "https://example.com/docs",
            "https://example.com/api"
        ) is True

    def test_different_domains_returns_false(self) -> None:
        """Test that different domains are recognized"""
        assert is_same_domain(
            "https://example.com",
            "https://other.com"
        ) is False

    def test_different_subdomains_returns_false(self) -> None:
        """Test that different subdomains are different"""
        assert is_same_domain(
            "https://api.example.com",
            "https://docs.example.com"
        ) is False

    def test_case_insensitive_comparison(self) -> None:
        """Test that domain comparison is case-insensitive"""
        assert is_same_domain(
            "https://EXAMPLE.COM",
            "https://example.com"
        ) is True

    def test_different_schemes_same_domain(self) -> None:
        """Test that scheme doesn't affect domain comparison"""
        assert is_same_domain(
            "http://example.com",
            "https://example.com"
        ) is True

    def test_handles_invalid_urls(self) -> None:
        """Test that invalid URLs are handled gracefully"""
        assert is_same_domain("not a url", "https://example.com") is False


class TestGetBasePath:
    """Tests for get_base_path function"""

    def test_removes_last_path_segment(self) -> None:
        """Test that last path segment is removed"""
        result = get_base_path("https://example.com/docs/api/users")
        assert result == "https://example.com/docs/api"

    def test_handles_single_segment_path(self) -> None:
        """Test handling of single path segment"""
        result = get_base_path("https://example.com/docs")
        assert result == "https://example.com"

    def test_handles_root_path(self) -> None:
        """Test handling of root path"""
        result = get_base_path("https://example.com")
        assert result == "https://example.com"

    def test_removes_trailing_slash(self) -> None:
        """Test that trailing slashes are handled"""
        result = get_base_path("https://example.com/docs/api/")
        assert result == "https://example.com/docs"


class TestIsExternalUrl:
    """Tests for is_external_url function"""

    def test_external_domain_returns_true(self) -> None:
        """Test that external domains are recognized"""
        assert is_external_url(
            "https://example.com",
            "https://other.com"
        ) is True

    def test_same_domain_returns_false(self) -> None:
        """Test that same domain is not external"""
        assert is_external_url(
            "https://example.com",
            "https://example.com/docs"
        ) is False

    def test_relative_url_returns_false(self) -> None:
        """Test that relative URLs are not external"""
        assert is_external_url(
            "https://example.com",
            "/docs/api"
        ) is False
        assert is_external_url(
            "https://example.com",
            "docs/api"
        ) is False

    def test_subdomain_is_external(self) -> None:
        """Test that different subdomains are external"""
        assert is_external_url(
            "https://api.example.com",
            "https://docs.example.com"
        ) is True

