"""
Unit tests for input validation utilities.
"""

import pytest
from fastapi import HTTPException

from curlinator.api.utils.validators import (
    validate_url,
    validate_collection_name,
    validate_max_pages,
    validate_max_depth,
    validate_crawl_request,
)


class TestValidateUrl:
    """Tests for URL validation."""
    
    def test_valid_https_url(self):
        """Test that valid HTTPS URLs pass validation."""
        is_valid, error = validate_url("https://example.com/docs")
        assert is_valid is True
        assert error is None
    
    def test_valid_http_url(self):
        """Test that valid HTTP URLs pass validation."""
        is_valid, error = validate_url("http://example.com/api")
        assert is_valid is True
        assert error is None
    
    def test_url_with_port(self):
        """Test that URLs with ports pass validation."""
        is_valid, error = validate_url("https://example.com:8080/docs")
        assert is_valid is True
        assert error is None
    
    def test_url_with_path_and_query(self):
        """Test that URLs with paths and query strings pass validation."""
        is_valid, error = validate_url("https://example.com/docs/api?version=v1")
        assert is_valid is True
        assert error is None
    
    def test_invalid_scheme_ftp(self):
        """Test that FTP URLs are rejected."""
        is_valid, error = validate_url("ftp://example.com/docs")
        assert is_valid is False
        assert "Invalid URL scheme" in error
        assert "http://" in error or "https://" in error
    
    def test_invalid_scheme_file(self):
        """Test that file:// URLs are rejected."""
        is_valid, error = validate_url("file:///etc/passwd")
        assert is_valid is False
        assert "Invalid URL scheme" in error
    
    def test_missing_scheme(self):
        """Test that URLs without scheme are rejected."""
        is_valid, error = validate_url("example.com/docs")
        assert is_valid is False
        assert "Invalid URL scheme" in error
    
    def test_localhost_hostname(self):
        """Test that localhost URLs are rejected."""
        is_valid, error = validate_url("http://localhost:8000/docs")
        assert is_valid is False
        assert "localhost" in error.lower()
        assert "Security error" in error
    
    def test_localhost_ip(self):
        """Test that 127.0.0.1 URLs are rejected."""
        is_valid, error = validate_url("http://127.0.0.1:8000/docs")
        assert is_valid is False
        assert "private ip" in error.lower()

    def test_private_ip_10(self):
        """Test that 10.x.x.x URLs are rejected."""
        is_valid, error = validate_url("http://10.0.0.1/docs")
        assert is_valid is False
        assert "private ip" in error.lower()

    def test_private_ip_192(self):
        """Test that 192.168.x.x URLs are rejected."""
        is_valid, error = validate_url("http://192.168.1.1/docs")
        assert is_valid is False
        assert "private ip" in error.lower()

    def test_private_ip_172(self):
        """Test that 172.16-31.x.x URLs are rejected."""
        is_valid, error = validate_url("http://172.16.0.1/docs")
        assert is_valid is False
        assert "private ip" in error.lower()
    
    def test_local_domain(self):
        """Test that .local domains are rejected."""
        is_valid, error = validate_url("http://myserver.local/docs")
        assert is_valid is False
        assert "internal" in error.lower() or "local" in error.lower()
    
    def test_internal_domain(self):
        """Test that .internal domains are rejected."""
        is_valid, error = validate_url("http://api.internal/docs")
        assert is_valid is False
        assert "internal" in error.lower()
    
    def test_missing_host(self):
        """Test that URLs without host are rejected."""
        is_valid, error = validate_url("https:///docs")
        assert is_valid is False
        assert "domain" in error.lower() or "host" in error.lower()


class TestValidateCollectionName:
    """Tests for collection name validation."""
    
    def test_valid_alphanumeric(self):
        """Test that alphanumeric names pass validation."""
        is_valid, error = validate_collection_name("myapi123")
        assert is_valid is True
        assert error is None
    
    def test_valid_with_hyphens(self):
        """Test that names with hyphens pass validation."""
        is_valid, error = validate_collection_name("my-api-docs")
        assert is_valid is True
        assert error is None
    
    def test_valid_with_underscores(self):
        """Test that names with underscores pass validation."""
        is_valid, error = validate_collection_name("my_api_docs")
        assert is_valid is True
        assert error is None
    
    def test_valid_mixed(self):
        """Test that names with mixed valid characters pass validation."""
        is_valid, error = validate_collection_name("API_Docs-v2")
        assert is_valid is True
        assert error is None
    
    def test_empty_name(self):
        """Test that empty names are rejected."""
        is_valid, error = validate_collection_name("")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_too_short(self):
        """Test that names shorter than 3 characters are rejected."""
        is_valid, error = validate_collection_name("ab")
        assert is_valid is False
        assert "too short" in error.lower()
        assert "3" in error
    
    def test_too_long(self):
        """Test that names longer than 50 characters are rejected."""
        is_valid, error = validate_collection_name("a" * 51)
        assert is_valid is False
        assert "too long" in error.lower()
        assert "50" in error
    
    def test_special_characters_space(self):
        """Test that names with spaces are rejected."""
        is_valid, error = validate_collection_name("my api docs")
        assert is_valid is False
        assert "Invalid collection name" in error
    
    def test_special_characters_dot(self):
        """Test that names with dots are rejected."""
        is_valid, error = validate_collection_name("my.api.docs")
        assert is_valid is False
        assert "Invalid collection name" in error
    
    def test_special_characters_slash(self):
        """Test that names with slashes are rejected."""
        is_valid, error = validate_collection_name("my/api/docs")
        assert is_valid is False
        assert "Invalid collection name" in error
    
    def test_special_characters_at(self):
        """Test that names with @ are rejected."""
        is_valid, error = validate_collection_name("my@api")
        assert is_valid is False
        assert "Invalid collection name" in error


class TestValidateMaxPages:
    """Tests for max_pages validation."""
    
    def test_valid_min_value(self):
        """Test that minimum valid value passes."""
        is_valid, error = validate_max_pages(1)
        assert is_valid is True
        assert error is None
    
    def test_valid_max_value(self):
        """Test that maximum valid value passes."""
        is_valid, error = validate_max_pages(1000)
        assert is_valid is True
        assert error is None
    
    def test_valid_middle_value(self):
        """Test that middle range values pass."""
        is_valid, error = validate_max_pages(50)
        assert is_valid is True
        assert error is None
    
    def test_none_value(self):
        """Test that None is accepted (optional parameter)."""
        is_valid, error = validate_max_pages(None)
        assert is_valid is True
        assert error is None
    
    def test_too_small(self):
        """Test that values less than 1 are rejected."""
        is_valid, error = validate_max_pages(0)
        assert is_valid is False
        assert "too small" in error.lower()
        assert "1" in error
    
    def test_too_large(self):
        """Test that values greater than 1000 are rejected."""
        is_valid, error = validate_max_pages(1001)
        assert is_valid is False
        assert "too large" in error.lower()
        assert "1000" in error
    
    def test_negative_value(self):
        """Test that negative values are rejected."""
        is_valid, error = validate_max_pages(-10)
        assert is_valid is False
        assert "too small" in error.lower()


class TestValidateMaxDepth:
    """Tests for max_depth validation."""
    
    def test_valid_min_value(self):
        """Test that minimum valid value passes."""
        is_valid, error = validate_max_depth(1)
        assert is_valid is True
        assert error is None
    
    def test_valid_max_value(self):
        """Test that maximum valid value passes."""
        is_valid, error = validate_max_depth(10)
        assert is_valid is True
        assert error is None
    
    def test_valid_middle_value(self):
        """Test that middle range values pass."""
        is_valid, error = validate_max_depth(3)
        assert is_valid is True
        assert error is None
    
    def test_none_value(self):
        """Test that None is accepted (optional parameter)."""
        is_valid, error = validate_max_depth(None)
        assert is_valid is True
        assert error is None
    
    def test_too_small(self):
        """Test that values less than 1 are rejected."""
        is_valid, error = validate_max_depth(0)
        assert is_valid is False
        assert "too small" in error.lower()
        assert "1" in error
    
    def test_too_large(self):
        """Test that values greater than 10 are rejected."""
        is_valid, error = validate_max_depth(11)
        assert is_valid is False
        assert "too large" in error.lower()
        assert "10" in error
    
    def test_negative_value(self):
        """Test that negative values are rejected."""
        is_valid, error = validate_max_depth(-5)
        assert is_valid is False
        assert "too small" in error.lower()


class TestValidateCrawlRequest:
    """Tests for complete crawl request validation."""

    def test_valid_request(self):
        """Test that valid request passes without exception."""
        # Should not raise any exception
        validate_crawl_request(
            url="https://example.com/docs",
            max_pages=50,
            max_depth=3
        )

    def test_invalid_url_raises_422(self):
        """Test that invalid URL raises 422 HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_request(
                url="ftp://example.com/docs",
                max_pages=50,
                max_depth=3
            )
        assert exc_info.value.status_code == 422
        assert "url" in str(exc_info.value.detail).lower()

    def test_invalid_max_pages_raises_422(self):
        """Test that invalid max_pages raises 422 HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_request(
                url="https://example.com/docs",
                max_pages=2000,
                max_depth=3
            )
        assert exc_info.value.status_code == 422
        assert "max_pages" in str(exc_info.value.detail).lower()

    def test_invalid_max_depth_raises_422(self):
        """Test that invalid max_depth raises 422 HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_request(
                url="https://example.com/docs",
                max_pages=50,
                max_depth=20
            )
        assert exc_info.value.status_code == 422
        assert "max_depth" in str(exc_info.value.detail).lower()

    def test_localhost_url_raises_422(self):
        """Test that localhost URL raises 422 HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_request(
                url="http://localhost:8000/docs",
                max_pages=50,
                max_depth=3
            )
        assert exc_info.value.status_code == 422
        assert "localhost" in str(exc_info.value.detail).lower()

    def test_private_ip_raises_422(self):
        """Test that private IP raises 422 HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_crawl_request(
                url="http://192.168.1.1/docs",
                max_pages=50,
                max_depth=3
            )
        assert exc_info.value.status_code == 422
        assert "private" in str(exc_info.value.detail).lower()

    def test_optional_parameters(self):
        """Test that optional parameters can be None."""
        # Should not raise any exception
        validate_crawl_request(
            url="https://example.com/docs",
            max_pages=None,
            max_depth=None
        )

