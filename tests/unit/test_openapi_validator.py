"""Tests for OpenAPI validator"""

from curlinator.core import (
    OpenAPIVersion,
    count_endpoints,
    extract_api_info,
    get_openapi_version,
    get_spec_summary,
    has_authentication,
    is_valid_openapi,
    validate_openapi_structure,
)


class TestIsValidOpenapi:
    """Tests for is_valid_openapi function"""

    def test_validates_openapi_3_0(self) -> None:
        """Test validation of OpenAPI 3.0 spec"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
        }
        assert is_valid_openapi(spec) is True

    def test_validates_openapi_3_1(self) -> None:
        """Test validation of OpenAPI 3.1 spec"""
        spec = {
            "openapi": "3.1.0",
            "info": {"title": "Test API", "version": "1.0.0"},
        }
        assert is_valid_openapi(spec) is True

    def test_validates_swagger_2_0(self) -> None:
        """Test validation of Swagger 2.0 spec"""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test API", "version": "1.0.0"},
        }
        assert is_valid_openapi(spec) is True

    def test_rejects_missing_version(self) -> None:
        """Test rejection of spec without version"""
        spec = {"info": {"title": "Test API"}}
        assert is_valid_openapi(spec) is False

    def test_rejects_missing_info(self) -> None:
        """Test rejection of spec without info"""
        spec = {"openapi": "3.0.0"}
        assert is_valid_openapi(spec) is False

    def test_rejects_missing_title(self) -> None:
        """Test rejection of spec without title"""
        spec = {"openapi": "3.0.0", "info": {"version": "1.0.0"}}
        assert is_valid_openapi(spec) is False

    def test_rejects_non_dict(self) -> None:
        """Test rejection of non-dictionary input"""
        assert is_valid_openapi("not a dict") is False
        assert is_valid_openapi([]) is False
        assert is_valid_openapi(None) is False

    def test_rejects_random_dict(self) -> None:
        """Test rejection of random dictionary"""
        spec = {"random": "data", "not": "openapi"}
        assert is_valid_openapi(spec) is False


class TestGetOpenapiVersion:
    """Tests for get_openapi_version function"""

    def test_detects_openapi_3_0(self) -> None:
        """Test detection of OpenAPI 3.0"""
        spec = {"openapi": "3.0.0"}
        assert get_openapi_version(spec) == OpenAPIVersion.OPENAPI_3_0

    def test_detects_openapi_3_0_variants(self) -> None:
        """Test detection of OpenAPI 3.0 variants"""
        for version in ["3.0.0", "3.0.1", "3.0.2", "3.0.3"]:
            spec = {"openapi": version}
            assert get_openapi_version(spec) == OpenAPIVersion.OPENAPI_3_0

    def test_detects_openapi_3_1(self) -> None:
        """Test detection of OpenAPI 3.1"""
        spec = {"openapi": "3.1.0"}
        assert get_openapi_version(spec) == OpenAPIVersion.OPENAPI_3_1

    def test_detects_future_3_x_as_3_1(self) -> None:
        """Test that future 3.x versions default to 3.1"""
        spec = {"openapi": "3.9.0"}
        assert get_openapi_version(spec) == OpenAPIVersion.OPENAPI_3_1

    def test_detects_swagger_2_0(self) -> None:
        """Test detection of Swagger 2.0"""
        spec = {"swagger": "2.0"}
        assert get_openapi_version(spec) == OpenAPIVersion.SWAGGER_2_0

    def test_returns_unknown_for_missing_version(self) -> None:
        """Test returns unknown when version is missing"""
        spec = {"info": {"title": "API"}}
        assert get_openapi_version(spec) == OpenAPIVersion.UNKNOWN

    def test_returns_unknown_for_invalid_version(self) -> None:
        """Test returns unknown for unsupported version"""
        spec = {"openapi": "1.0.0"}
        assert get_openapi_version(spec) == OpenAPIVersion.UNKNOWN


class TestValidateOpenapiStructure:
    """Tests for validate_openapi_structure function"""

    def test_validates_minimal_openapi_3_0(self) -> None:
        """Test validation of minimal valid OpenAPI 3.0 spec"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
        }
        result = validate_openapi_structure(spec)
        assert result.is_valid is True
        assert result.version == OpenAPIVersion.OPENAPI_3_0
        assert len(result.errors) == 0

    def test_validates_minimal_swagger_2_0(self) -> None:
        """Test validation of minimal valid Swagger 2.0 spec"""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
        }
        result = validate_openapi_structure(spec)
        assert result.is_valid is True
        assert result.version == OpenAPIVersion.SWAGGER_2_0
        assert len(result.errors) == 0

    def test_reports_missing_version_field(self) -> None:
        """Test error for missing version field"""
        spec = {"info": {"title": "Test API"}}
        result = validate_openapi_structure(spec)
        assert result.is_valid is False
        assert any("openapi" in err.lower() or "swagger" in err.lower() for err in result.errors)

    def test_reports_missing_info(self) -> None:
        """Test error for missing info"""
        spec = {"openapi": "3.0.0"}
        result = validate_openapi_structure(spec)
        assert result.is_valid is False
        assert any("info" in err.lower() for err in result.errors)

    def test_reports_missing_title(self) -> None:
        """Test error for missing title in info"""
        spec = {
            "openapi": "3.0.0",
            "info": {"version": "1.0.0"},
            "paths": {},
        }
        result = validate_openapi_structure(spec)
        assert result.is_valid is False
        assert any("title" in err.lower() for err in result.errors)

    def test_reports_missing_version_in_info(self) -> None:
        """Test error for missing version in info"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API"},
            "paths": {},
        }
        result = validate_openapi_structure(spec)
        assert result.is_valid is False
        assert any("version" in err.lower() for err in result.errors)

    def test_swagger_2_0_requires_paths(self) -> None:
        """Test that Swagger 2.0 requires paths"""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test API", "version": "1.0.0"},
        }
        result = validate_openapi_structure(spec)
        assert result.is_valid is False
        assert any("paths" in err.lower() for err in result.errors)

    def test_openapi_3_warns_about_empty_spec(self) -> None:
        """Test warning for OpenAPI 3.x without paths/components"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
        }
        result = validate_openapi_structure(spec)
        assert result.is_valid is False
        # Should warn about missing paths/components/webhooks

    def test_rejects_non_dict_input(self) -> None:
        """Test rejection of non-dict input"""
        result = validate_openapi_structure("not a dict")
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestExtractApiInfo:
    """Tests for extract_api_info function"""

    def test_extracts_basic_info(self) -> None:
        """Test extraction of basic API info"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0",
                "description": "A test API",
            },
        }
        info = extract_api_info(spec)
        assert info["title"] == "Test API"
        assert info["version"] == "1.0.0"
        assert info["description"] == "A test API"

    def test_extracts_base_url_from_openapi_3(self) -> None:
        """Test extraction of base URL from OpenAPI 3.x"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "API", "version": "1.0"},
            "servers": [{"url": "https://api.example.com/v1"}],
        }
        info = extract_api_info(spec)
        assert info["base_url"] == "https://api.example.com/v1"

    def test_extracts_base_url_from_swagger_2(self) -> None:
        """Test extraction of base URL from Swagger 2.0"""
        spec = {
            "swagger": "2.0",
            "info": {"title": "API", "version": "1.0"},
            "host": "api.example.com",
            "basePath": "/v1",
            "schemes": ["https"],
        }
        info = extract_api_info(spec)
        assert info["base_url"] == "https://api.example.com/v1"

    def test_handles_missing_fields(self) -> None:
        """Test handling of missing fields"""
        spec = {"openapi": "3.0.0", "info": {}}
        info = extract_api_info(spec)
        assert info["title"] == ""
        assert info["version"] == ""
        assert info["description"] == ""
        assert info["base_url"] == ""

    def test_handles_non_dict_input(self) -> None:
        """Test handling of non-dict input"""
        info = extract_api_info("not a dict")
        assert info["title"] == ""
        assert info["version"] == ""


class TestCountEndpoints:
    """Tests for count_endpoints function"""

    def test_counts_single_endpoint(self) -> None:
        """Test counting single endpoint"""
        spec = {
            "paths": {
                "/users": {
                    "get": {"summary": "Get users"},
                }
            }
        }
        assert count_endpoints(spec) == 1

    def test_counts_multiple_methods(self) -> None:
        """Test counting multiple methods on same path"""
        spec = {
            "paths": {
                "/users": {
                    "get": {},
                    "post": {},
                }
            }
        }
        assert count_endpoints(spec) == 2

    def test_counts_multiple_paths(self) -> None:
        """Test counting multiple paths"""
        spec = {
            "paths": {
                "/users": {"get": {}, "post": {}},
                "/posts": {"get": {}},
            }
        }
        assert count_endpoints(spec) == 3

    def test_counts_all_http_methods(self) -> None:
        """Test counting all HTTP methods"""
        spec = {
            "paths": {
                "/resource": {
                    "get": {},
                    "post": {},
                    "put": {},
                    "patch": {},
                    "delete": {},
                    "head": {},
                    "options": {},
                }
            }
        }
        assert count_endpoints(spec) == 7

    def test_returns_zero_for_empty_paths(self) -> None:
        """Test returns 0 for empty paths"""
        spec = {"paths": {}}
        assert count_endpoints(spec) == 0

    def test_returns_zero_for_missing_paths(self) -> None:
        """Test returns 0 when paths is missing"""
        spec = {"openapi": "3.0.0"}
        assert count_endpoints(spec) == 0

    def test_handles_non_dict_input(self) -> None:
        """Test handling of non-dict input"""
        assert count_endpoints("not a dict") == 0
        assert count_endpoints(None) == 0


class TestHasAuthentication:
    """Tests for has_authentication function"""

    def test_detects_swagger_2_security_definitions(self) -> None:
        """Test detection of Swagger 2.0 security definitions"""
        spec = {
            "swagger": "2.0",
            "securityDefinitions": {
                "api_key": {"type": "apiKey", "name": "api_key", "in": "header"}
            },
        }
        assert has_authentication(spec) is True

    def test_detects_openapi_3_security_schemes(self) -> None:
        """Test detection of OpenAPI 3.x security schemes"""
        spec = {
            "openapi": "3.0.0",
            "components": {
                "securitySchemes": {
                    "bearerAuth": {"type": "http", "scheme": "bearer"}
                }
            },
        }
        assert has_authentication(spec) is True

    def test_detects_global_security_requirement(self) -> None:
        """Test detection of global security requirement"""
        spec = {
            "openapi": "3.0.0",
            "security": [{"bearerAuth": []}],
        }
        assert has_authentication(spec) is True

    def test_returns_false_for_no_auth(self) -> None:
        """Test returns False when no authentication"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "API", "version": "1.0"},
            "paths": {},
        }
        assert has_authentication(spec) is False

    def test_handles_non_dict_input(self) -> None:
        """Test handling of non-dict input"""
        assert has_authentication("not a dict") is False
        assert has_authentication(None) is False


class TestGetSpecSummary:
    """Tests for get_spec_summary function"""

    def test_summarizes_valid_spec(self) -> None:
        """Test summary of valid spec"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0",
                "description": "A test API",
            },
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {"get": {}, "post": {}},
            },
            "components": {
                "securitySchemes": {"apiKey": {"type": "apiKey"}}
            },
        }
        summary = get_spec_summary(spec)

        assert summary["is_valid"] is True
        assert summary["version"] == "3.0"
        assert summary["title"] == "Test API"
        assert summary["api_version"] == "1.0.0"
        assert summary["description"] == "A test API"
        assert summary["base_url"] == "https://api.example.com"
        assert summary["endpoint_count"] == 2
        assert summary["has_authentication"] is True
        assert len(summary["validation_errors"]) == 0

    def test_summarizes_invalid_spec(self) -> None:
        """Test summary of invalid spec"""
        spec = {"openapi": "3.0.0"}
        summary = get_spec_summary(spec)

        assert summary["is_valid"] is False
        assert len(summary["validation_errors"]) > 0

    def test_includes_all_fields(self) -> None:
        """Test that summary includes all expected fields"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "API", "version": "1.0"},
            "paths": {},
        }
        summary = get_spec_summary(spec)

        expected_keys = [
            "is_valid",
            "version",
            "title",
            "api_version",
            "description",
            "base_url",
            "endpoint_count",
            "has_authentication",
            "validation_errors",
        ]

        for key in expected_keys:
            assert key in summary

