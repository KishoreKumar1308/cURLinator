"""Tests for OpenAPI/Swagger specification detection and parsing utilities.

Tests the openapi_detector module that:
- Detects OpenAPI specs from Swagger UI pages
- Checks 30+ common spec paths
- Parses OpenAPI 3.x and Swagger 2.0 specs
- Converts specs to LlamaIndex Document objects
- Handles metadata flattening for Chroma compatibility
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from llama_index.core.schema import Document

from curlinator.utils.openapi_detector import (
    _create_auth_document,
    _create_endpoint_documents,
    _create_overview_document,
    _detect_from_swagger_ui,
    _get_base_url,
    _is_valid_openapi_spec,
    detect_openapi_spec,
    parse_openapi_to_documents,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client"""
    with patch('curlinator.utils.openapi_detector.HTTPClient') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def sample_openapi_3_spec():
    """Sample OpenAPI 3.0 specification"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Pet Store API",
            "version": "1.0.0",
            "description": "A sample Pet Store API",
            "contact": {
                "name": "API Support",
                "email": "support@petstore.com"
            }
        },
        "servers": [
            {"url": "https://api.petstore.com/v1"}
        ],
        "paths": {
            "/pets": {
                "get": {
                    "summary": "List all pets",
                    "description": "Returns a list of all pets",
                    "operationId": "listPets",
                    "tags": ["pets"],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum number of items to return",
                            "required": False,
                            "schema": {"type": "integer"}
                        }
                    ],
                    "responses": {
                        "200": {"description": "Successful response"},
                        "500": {"description": "Internal server error"}
                    }
                },
                "post": {
                    "summary": "Create a pet",
                    "description": "Creates a new pet",
                    "operationId": "createPet",
                    "tags": ["pets", "admin"],
                    "requestBody": {
                        "description": "Pet object to create",
                        "required": True
                    },
                    "responses": {
                        "201": {"description": "Pet created"},
                        "400": {"description": "Invalid input"}
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "api_key": {
                    "type": "apiKey",
                    "name": "X-API-Key",
                    "in": "header",
                    "description": "API key authentication"
                },
                "oauth2": {
                    "type": "oauth2",
                    "description": "OAuth 2.0 authentication"
                }
            }
        }
    }


@pytest.fixture
def sample_swagger_2_spec():
    """Sample Swagger 2.0 specification"""
    return {
        "swagger": "2.0",
        "info": {
            "title": "User API",
            "version": "2.0.0",
            "description": "User management API"
        },
        "host": "api.example.com",
        "basePath": "/v2",
        "schemes": ["https"],
        "paths": {
            "/users": {
                "get": {
                    "summary": "List users",
                    "description": "Get all users",
                    "responses": {
                        "200": {"description": "Success"}
                    }
                }
            }
        },
        "securityDefinitions": {
            "bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header"
            }
        }
    }


@pytest.fixture
def swagger_ui_html():
    """Sample Swagger UI HTML with spec URL"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Swagger UI</title></head>
    <body>
        <div id="swagger-ui"></div>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: "https://petstore3.swagger.io/api/v3/openapi.json",
                    dom_id: '#swagger-ui',
                });
            }
        </script>
    </body>
    </html>
    """


@pytest.fixture
def swagger_ui_html_relative():
    """Swagger UI HTML with relative spec URL"""
    return """
    <script>
        SwaggerUIBundle({
            url: "/api/v3/openapi.json",
            dom_id: '#swagger-ui'
        });
    </script>
    """


@pytest.fixture
def swagger_ui_html_json_format():
    """Swagger UI HTML with JSON format spec URL"""
    return """
    <script>
        var config = {
            "url": "/docs/swagger.yaml"
        };
    </script>
    """


@pytest.fixture
def swagger_ui_html_const_pattern():
    """Swagger UI HTML with const variable pattern"""
    return """
    <script>
        const specUrl = "https://api.example.com/openapi.yaml";
        const ui = SwaggerUIBundle({ url: specUrl });
    </script>
    """


# ============================================================================
# Test _detect_from_swagger_ui
# ============================================================================

class TestDetectFromSwaggerUI:
    """Tests for Swagger UI spec URL detection"""

    @pytest.mark.asyncio
    async def test_detects_spec_from_swagger_ui_bundle_pattern(
        self, mock_http_client, swagger_ui_html
    ):
        """Test detection of spec URL from SwaggerUIBundle pattern"""
        response = MagicMock()
        response.text = swagger_ui_html
        mock_http_client.get = AsyncMock(return_value=response)

        result = await _detect_from_swagger_ui("https://petstore3.swagger.io")

        assert result == "https://petstore3.swagger.io/api/v3/openapi.json"

    @pytest.mark.asyncio
    async def test_detects_spec_from_relative_url(
        self, mock_http_client, swagger_ui_html_relative
    ):
        """Test detection and conversion of relative spec URL to absolute"""
        response = MagicMock()
        response.text = swagger_ui_html_relative
        mock_http_client.get = AsyncMock(return_value=response)

        result = await _detect_from_swagger_ui("https://api.example.com/docs")

        assert result == "https://api.example.com/api/v3/openapi.json"

    @pytest.mark.asyncio
    async def test_detects_spec_from_json_format(
        self, mock_http_client, swagger_ui_html_json_format
    ):
        """Test detection from JSON format pattern"""
        response = MagicMock()
        response.text = swagger_ui_html_json_format
        mock_http_client.get = AsyncMock(return_value=response)

        result = await _detect_from_swagger_ui("https://api.example.com")

        assert result == "https://api.example.com/docs/swagger.yaml"

    @pytest.mark.asyncio
    async def test_detects_spec_from_const_pattern(
        self, mock_http_client, swagger_ui_html_const_pattern
    ):
        """Test detection from const/var/let variable pattern"""
        response = MagicMock()
        response.text = swagger_ui_html_const_pattern
        mock_http_client.get = AsyncMock(return_value=response)

        result = await _detect_from_swagger_ui("https://api.example.com")

        assert result == "https://api.example.com/openapi.yaml"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_spec_found(self, mock_http_client):
        """Test returns None when no spec URL found in HTML"""
        response = MagicMock()
        response.text = "<html><body>No spec here</body></html>"
        mock_http_client.get = AsyncMock(return_value=response)

        result = await _detect_from_swagger_ui("https://example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_http_error_gracefully(self, mock_http_client):
        """Test handles HTTP errors gracefully"""
        mock_http_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

        result = await _detect_from_swagger_ui("https://example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_generic_exception_gracefully(self, mock_http_client):
        """Test handles generic exceptions gracefully"""
        mock_http_client.get = AsyncMock(side_effect=Exception("Unexpected error"))

        result = await _detect_from_swagger_ui("https://example.com")

        assert result is None


# ============================================================================
# Test _is_valid_openapi_spec
# ============================================================================

class TestIsValidOpenapiSpec:
    """Tests for OpenAPI spec validation"""

    def test_validates_openapi_3_0_spec(self, sample_openapi_3_spec):
        """Test validates OpenAPI 3.0 specification"""
        assert _is_valid_openapi_spec(sample_openapi_3_spec) is True

    def test_validates_openapi_3_1_spec(self):
        """Test validates OpenAPI 3.1 specification"""
        spec = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {}
        }
        assert _is_valid_openapi_spec(spec) is True

    def test_validates_swagger_2_0_spec(self, sample_swagger_2_spec):
        """Test validates Swagger 2.0 specification"""
        assert _is_valid_openapi_spec(sample_swagger_2_spec) is True

    def test_rejects_spec_without_openapi_or_swagger_field(self):
        """Test rejects spec without openapi or swagger field"""
        spec = {
            "info": {"title": "Test"},
            "paths": {}
        }
        assert _is_valid_openapi_spec(spec) is False

    def test_rejects_spec_without_info_field(self):
        """Test rejects spec without info field"""
        spec = {
            "openapi": "3.0.0",
            "paths": {}
        }
        assert _is_valid_openapi_spec(spec) is False

    def test_rejects_spec_without_paths_field(self):
        """Test rejects spec without paths field"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"}
        }
        assert _is_valid_openapi_spec(spec) is False

    def test_rejects_non_dict_spec(self):
        """Test rejects non-dictionary spec"""
        assert _is_valid_openapi_spec("not a dict") is False
        assert _is_valid_openapi_spec([]) is False
        assert _is_valid_openapi_spec(None) is False

    def test_rejects_invalid_openapi_version(self):
        """Test rejects invalid OpenAPI version"""
        spec = {
            "openapi": "2.0.0",  # Invalid - should be 3.x
            "info": {"title": "Test"},
            "paths": {}
        }
        assert _is_valid_openapi_spec(spec) is False

    def test_rejects_invalid_swagger_version(self):
        """Test rejects invalid Swagger version"""
        spec = {
            "swagger": "3.0",  # Invalid - should be 2.0
            "info": {"title": "Test"},
            "paths": {}
        }
        assert _is_valid_openapi_spec(spec) is False


# ============================================================================
# Test detect_openapi_spec
# ============================================================================

class TestDetectOpenapiSpec:
    """Tests for OpenAPI spec detection from URLs"""

    @pytest.mark.asyncio
    async def test_detects_spec_from_swagger_ui_first(
        self, mock_http_client, swagger_ui_html, sample_openapi_3_spec
    ):
        """Test tries Swagger UI detection before common paths"""
        # Mock Swagger UI page response
        swagger_response = MagicMock()
        swagger_response.text = swagger_ui_html

        # Mock spec response
        spec_response = MagicMock()
        spec_response.headers = {"content-type": "application/json"}
        spec_response.json.return_value = sample_openapi_3_spec

        mock_http_client.get = AsyncMock(side_effect=[swagger_response, spec_response])

        result = await detect_openapi_spec("https://petstore3.swagger.io")

        # Should return the spec URL found in Swagger UI
        assert result == "https://petstore3.swagger.io/api/v3/openapi.json"

    @pytest.mark.asyncio
    async def test_detects_spec_from_direct_url(
        self, mock_http_client, sample_openapi_3_spec
    ):
        """Test detects spec when URL points directly to spec"""
        # First call: Swagger UI detection fails
        swagger_response = MagicMock()
        swagger_response.text = "<html>Not Swagger UI</html>"

        # Second call: Direct URL is valid spec
        spec_response = MagicMock()
        spec_response.headers = {"content-type": "application/json"}
        spec_response.json.return_value = sample_openapi_3_spec

        mock_http_client.get = AsyncMock(side_effect=[swagger_response, spec_response])

        result = await detect_openapi_spec("https://api.example.com/openapi.json")

        assert result == "https://api.example.com/openapi.json"

    @pytest.mark.asyncio
    async def test_detects_spec_from_common_path(
        self, mock_http_client, sample_openapi_3_spec
    ):
        """Test detects spec from common paths like /api/v2/openapi.json"""
        # Mock responses: Swagger UI fails, direct URL fails, common path succeeds
        swagger_response = MagicMock()
        swagger_response.text = "<html>Not Swagger UI</html>"

        # Add successful response for /api/v2/openapi.json (11th path in common_paths)
        spec_response = MagicMock()
        spec_response.headers = {"content-type": "application/json"}
        spec_response.json.return_value = sample_openapi_3_spec

        # Mock to fail for first 10 paths, succeed on 11th
        mock_http_client.get = AsyncMock(side_effect=[
            swagger_response,  # Swagger UI detection
            httpx.HTTPError("Not found"),  # Direct URL (path 0)
            httpx.HTTPError("Not found"),  # /openapi.json (path 1)
            httpx.HTTPError("Not found"),  # /openapi.yaml (path 2)
            httpx.HTTPError("Not found"),  # /swagger.json (path 3)
            httpx.HTTPError("Not found"),  # /swagger.yaml (path 4)
            httpx.HTTPError("Not found"),  # /api-docs (path 5)
            httpx.HTTPError("Not found"),  # /api/openapi.json (path 6)
            httpx.HTTPError("Not found"),  # /api/openapi.yaml (path 7)
            httpx.HTTPError("Not found"),  # /api/swagger.json (path 8)
            httpx.HTTPError("Not found"),  # /api/swagger.yaml (path 9)
            httpx.HTTPError("Not found"),  # /api/v1/openapi.json (path 10)
            spec_response,  # /api/v2/openapi.json (path 11) succeeds
        ])

        result = await detect_openapi_spec("https://api.example.com")

        assert result == "https://api.example.com/api/v2/openapi.json"

    @pytest.mark.asyncio
    async def test_detects_yaml_spec(self, mock_http_client):
        """Test detects YAML format specs"""
        swagger_response = MagicMock()
        swagger_response.text = "<html>Not Swagger UI</html>"

        # Create a valid YAML spec that will pass validation
        yaml_content = """
openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
paths:
  /test:
    get:
      summary: Test endpoint
"""

        spec_response = MagicMock()
        spec_response.headers = {"content-type": "application/x-yaml"}
        spec_response.text = yaml_content

        # The implementation will call yaml.safe_load(response.text)
        # So we don't need to mock json(), just provide valid YAML text

        # /openapi.yaml is the 2nd path in common_paths (index 1)
        mock_http_client.get = AsyncMock(side_effect=[
            swagger_response,  # Swagger UI detection
            httpx.HTTPError("Not found"),  # Direct URL (path 0)
            httpx.HTTPError("Not found"),  # /openapi.json (path 1)
            spec_response,  # /openapi.yaml (path 2) succeeds
        ])

        result = await detect_openapi_spec("https://api.example.com")

        assert result == "https://api.example.com/openapi.yaml"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_spec_found(self, mock_http_client):
        """Test returns None when no spec found at any location"""
        swagger_response = MagicMock()
        swagger_response.text = "<html>Not Swagger UI</html>"

        # All requests fail
        mock_http_client.get = AsyncMock(side_effect=[
            swagger_response,
            *[httpx.HTTPError("Not found") for _ in range(30)]
        ])

        result = await detect_openapi_spec("https://api.example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_validates_spec_before_returning(self, mock_http_client):
        """Test validates spec is valid OpenAPI before returning URL"""
        swagger_response = MagicMock()
        swagger_response.text = "<html>Not Swagger UI</html>"

        # Invalid spec (missing required fields)
        invalid_spec_response = MagicMock()
        invalid_spec_response.headers = {"content-type": "application/json"}
        invalid_spec_response.json.return_value = {"invalid": "spec"}

        mock_http_client.get = AsyncMock(side_effect=[
            swagger_response,
            invalid_spec_response,
            httpx.HTTPError("Not found"),
        ])

        result = await detect_openapi_spec("https://api.example.com")

        # Should not return URL for invalid spec
        assert result is None


# ============================================================================
# Test parse_openapi_to_documents
# ============================================================================

class TestParseOpenapiToDocuments:
    """Tests for parsing OpenAPI specs to Document objects"""

    @pytest.mark.asyncio
    async def test_parses_openapi_3_spec_to_documents(
        self, mock_http_client, sample_openapi_3_spec
    ):
        """Test parses OpenAPI 3.0 spec and creates documents"""
        spec_response = MagicMock()
        spec_response.json.return_value = sample_openapi_3_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        documents = await parse_openapi_to_documents(
            "https://api.petstore.com/openapi.json"
        )

        # Should create: 1 overview + 2 endpoints + 1 auth = 4 documents
        assert len(documents) >= 3
        assert all(isinstance(doc, Document) for doc in documents)

    @pytest.mark.asyncio
    async def test_parses_swagger_2_spec_to_documents(
        self, mock_http_client, sample_swagger_2_spec
    ):
        """Test parses Swagger 2.0 spec and creates documents"""
        spec_response = MagicMock()
        spec_response.json.return_value = sample_swagger_2_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        documents = await parse_openapi_to_documents(
            "https://api.example.com/swagger.json"
        )

        # Should create: 1 overview + 1 endpoint + 1 auth = 3 documents
        assert len(documents) >= 2
        assert all(isinstance(doc, Document) for doc in documents)

    @pytest.mark.asyncio
    async def test_creates_overview_document(
        self, mock_http_client, sample_openapi_3_spec
    ):
        """Test creates overview document with API info"""
        spec_response = MagicMock()
        spec_response.json.return_value = sample_openapi_3_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        documents = await parse_openapi_to_documents(
            "https://api.petstore.com/openapi.json"
        )

        overview_docs = [d for d in documents if d.metadata.get("type") == "api_overview"]
        assert len(overview_docs) == 1

        overview = overview_docs[0]
        assert "Pet Store API" in overview.text
        assert "1.0.0" in overview.text
        assert overview.metadata["api_title"] == "Pet Store API"
        assert overview.metadata["api_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_creates_endpoint_documents(
        self, mock_http_client, sample_openapi_3_spec
    ):
        """Test creates separate document for each endpoint"""
        spec_response = MagicMock()
        spec_response.json.return_value = sample_openapi_3_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        documents = await parse_openapi_to_documents(
            "https://api.petstore.com/openapi.json"
        )

        endpoint_docs = [d for d in documents if d.metadata.get("type") == "api_endpoint"]

        # Should have 2 endpoints: GET /pets and POST /pets
        assert len(endpoint_docs) == 2

        # Check GET endpoint
        get_doc = next(d for d in endpoint_docs if d.metadata["method"] == "GET")
        assert get_doc.metadata["endpoint"] == "/pets"
        assert "List all pets" in get_doc.text
        assert "limit" in get_doc.text  # Parameter

        # Check POST endpoint
        post_doc = next(d for d in endpoint_docs if d.metadata["method"] == "POST")
        assert post_doc.metadata["endpoint"] == "/pets"
        assert "Create a pet" in post_doc.text

    @pytest.mark.asyncio
    async def test_flattens_tags_metadata_for_chroma(
        self, mock_http_client, sample_openapi_3_spec
    ):
        """Test converts tags list to comma-separated string for Chroma compatibility"""
        spec_response = MagicMock()
        spec_response.json.return_value = sample_openapi_3_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        documents = await parse_openapi_to_documents(
            "https://api.petstore.com/openapi.json"
        )

        endpoint_docs = [d for d in documents if d.metadata.get("type") == "api_endpoint"]

        # GET /pets has tags: ["pets"]
        get_doc = next(d for d in endpoint_docs if d.metadata["method"] == "GET")
        assert get_doc.metadata["tags"] == "pets"
        assert isinstance(get_doc.metadata["tags"], str)

        # POST /pets has tags: ["pets", "admin"]
        post_doc = next(d for d in endpoint_docs if d.metadata["method"] == "POST")
        assert post_doc.metadata["tags"] == "pets, admin"
        assert isinstance(post_doc.metadata["tags"], str)

    @pytest.mark.asyncio
    async def test_creates_auth_document(
        self, mock_http_client, sample_openapi_3_spec
    ):
        """Test creates authentication document"""
        spec_response = MagicMock()
        spec_response.json.return_value = sample_openapi_3_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        documents = await parse_openapi_to_documents(
            "https://api.petstore.com/openapi.json"
        )

        auth_docs = [d for d in documents if d.metadata.get("type") == "authentication"]
        assert len(auth_docs) == 1

        auth_doc = auth_docs[0]
        assert "api_key" in auth_doc.text
        assert "oauth2" in auth_doc.text
        assert "X-API-Key" in auth_doc.text

    @pytest.mark.asyncio
    async def test_parses_yaml_spec(self, mock_http_client):
        """Test parses YAML format specs"""
        yaml_spec = """
openapi: 3.0.0
info:
  title: YAML API
  version: 1.0.0
paths:
  /test:
    get:
      summary: Test endpoint
      responses:
        '200':
          description: Success
"""
        spec_response = MagicMock()
        spec_response.text = yaml_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        documents = await parse_openapi_to_documents(
            "https://api.example.com/openapi.yaml"
        )

        assert len(documents) >= 1
        overview = next(d for d in documents if d.metadata.get("type") == "api_overview")
        assert "YAML API" in overview.text

    @pytest.mark.asyncio
    async def test_raises_error_for_invalid_spec(self, mock_http_client):
        """Test raises ValueError for invalid spec"""
        invalid_spec = {"invalid": "spec"}
        spec_response = MagicMock()
        spec_response.json.return_value = invalid_spec
        mock_http_client.get = AsyncMock(return_value=spec_response)

        with pytest.raises(ValueError, match="Invalid OpenAPI specification"):
            await parse_openapi_to_documents("https://api.example.com/openapi.json")

    @pytest.mark.asyncio
    async def test_raises_error_for_http_failure(self, mock_http_client):
        """Test raises HTTPError when spec cannot be fetched"""
        mock_http_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

        with pytest.raises(httpx.HTTPError):
            await parse_openapi_to_documents("https://api.example.com/openapi.json")


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions"""

    def test_create_overview_document(self, sample_openapi_3_spec):
        """Test creates overview document with all info"""
        doc = _create_overview_document(
            sample_openapi_3_spec,
            "https://api.petstore.com/openapi.json",
            "3.0.0"
        )

        assert isinstance(doc, Document)
        assert "Pet Store API" in doc.text
        assert "1.0.0" in doc.text
        assert "A sample Pet Store API" in doc.text
        assert "API Support" in doc.text
        assert "support@petstore.com" in doc.text
        assert doc.metadata["type"] == "api_overview"
        assert doc.metadata["api_title"] == "Pet Store API"

    def test_create_endpoint_documents(self, sample_openapi_3_spec):
        """Test creates endpoint documents"""
        docs = _create_endpoint_documents(
            sample_openapi_3_spec,
            "https://api.petstore.com/openapi.json",
            is_openapi_3=True
        )

        assert len(docs) == 2
        assert all(d.metadata["type"] == "api_endpoint" for d in docs)

        # Check metadata structure
        for doc in docs:
            assert "endpoint" in doc.metadata
            assert "method" in doc.metadata
            assert "operation_id" in doc.metadata
            assert "tags" in doc.metadata
            assert isinstance(doc.metadata["tags"], str)  # Flattened

    def test_create_auth_document_openapi_3(self, sample_openapi_3_spec):
        """Test creates auth document for OpenAPI 3.x"""
        doc = _create_auth_document(
            sample_openapi_3_spec,
            "https://api.petstore.com/openapi.json",
            is_openapi_3=True
        )

        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.metadata["type"] == "authentication"
        assert "api_key" in doc.text
        assert "apiKey" in doc.text

    def test_create_auth_document_swagger_2(self, sample_swagger_2_spec):
        """Test creates auth document for Swagger 2.0"""
        doc = _create_auth_document(
            sample_swagger_2_spec,
            "https://api.example.com/swagger.json",
            is_openapi_3=False
        )

        assert doc is not None
        assert isinstance(doc, Document)
        assert "bearer" in doc.text
        assert "Authorization" in doc.text

    def test_create_auth_document_returns_none_when_no_auth(self):
        """Test returns None when no auth schemes defined"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test"},
            "paths": {}
        }
        doc = _create_auth_document(spec, "https://api.example.com", is_openapi_3=True)

        assert doc is None

    def test_get_base_url_openapi_3(self, sample_openapi_3_spec):
        """Test extracts base URL from OpenAPI 3.x servers"""
        base_url = _get_base_url(sample_openapi_3_spec)

        assert base_url == "https://api.petstore.com/v1"

    def test_get_base_url_swagger_2(self, sample_swagger_2_spec):
        """Test extracts base URL from Swagger 2.0 host/basePath"""
        base_url = _get_base_url(sample_swagger_2_spec)

        assert base_url == "https://api.example.com/v2"

    def test_get_base_url_returns_unknown_when_missing(self):
        """Test returns 'Unknown' when base URL cannot be determined"""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test"},
            "paths": {}
        }
        base_url = _get_base_url(spec)

        assert base_url == "Unknown"

