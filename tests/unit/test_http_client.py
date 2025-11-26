"""Tests for HTTP client service"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from curlinator.services import HTTPClient


@pytest.fixture
def http_client() -> HTTPClient:
    """Create HTTP client for testing"""
    return HTTPClient(timeout=5, max_retries=2)


class TestHTTPClientInitialization:
    """Tests for HTTPClient initialization"""

    def test_creates_client_with_defaults(self) -> None:
        """Test client creation with default settings"""
        client = HTTPClient()
        assert client.max_retries == 3
        assert "User-Agent" in client.default_headers
        assert client.timeout > 0

    def test_creates_client_with_custom_timeout(self) -> None:
        """Test client creation with custom timeout"""
        client = HTTPClient(timeout=10)
        assert client.timeout == 10

    def test_creates_client_with_custom_headers(self) -> None:
        """Test client creation with custom headers"""
        custom_headers = {"X-Custom-Header": "test"}
        client = HTTPClient(headers=custom_headers)
        assert "X-Custom-Header" in client.default_headers
        assert client.default_headers["X-Custom-Header"] == "test"

    def test_creates_client_with_custom_retries(self) -> None:
        """Test client creation with custom retry count"""
        client = HTTPClient(max_retries=5)
        assert client.max_retries == 5


class TestHTTPClientGet:
    """Tests for GET requests"""

    @pytest.mark.asyncio
    async def test_get_successful_request(self, http_client: HTTPClient) -> None:
        """Test successful GET request"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "response text"
        mock_response.is_success = True
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await http_client.get("https://example.com")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_with_custom_headers(self, http_client: HTTPClient) -> None:
        """Test GET request with custom headers"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            custom_headers = {"Authorization": "Bearer token"}
            await http_client.get("https://example.com", headers=custom_headers)

            # Verify get was called
            assert mock_client.get.called

    @pytest.mark.asyncio
    async def test_get_with_query_params(self, http_client: HTTPClient) -> None:
        """Test GET request with query parameters"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            params = {"key": "value", "page": 1}
            await http_client.get("https://example.com", params=params)

            assert mock_client.get.called


class TestHTTPClientHead:
    """Tests for HEAD requests"""

    @pytest.mark.asyncio
    async def test_head_returns_true_for_existing_url(self, http_client: HTTPClient) -> None:
        """Test HEAD request returns True for existing URL"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await http_client.head("https://example.com")
            assert result is True

    @pytest.mark.asyncio
    async def test_head_returns_false_for_nonexistent_url(self, http_client: HTTPClient) -> None:
        """Test HEAD request returns False for non-existent URL"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.head = AsyncMock(side_effect=httpx.HTTPError("Not found"))
            mock_client_class.return_value = mock_client

            result = await http_client.head("https://example.com/nonexistent")
            assert result is False

    @pytest.mark.asyncio
    async def test_head_handles_timeout(self, http_client: HTTPClient) -> None:
        """Test HEAD request handles timeout gracefully"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.head = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_class.return_value = mock_client

            result = await http_client.head("https://slow-server.com")
            assert result is False


class TestHTTPClientPost:
    """Tests for POST requests"""

    @pytest.mark.asyncio
    async def test_post_with_json_data(self, http_client: HTTPClient) -> None:
        """Test POST request with JSON data"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            json_data = {"key": "value"}
            response = await http_client.post("https://example.com", json=json_data)
            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_post_with_form_data(self, http_client: HTTPClient) -> None:
        """Test POST request with form data"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            form_data = {"username": "test", "password": "pass"}
            response = await http_client.post("https://example.com", data=form_data)
            assert response.status_code == 200


class TestHTTPClientFetchText:
    """Tests for fetch_text method"""

    @pytest.mark.asyncio
    async def test_fetch_text_returns_content(self, http_client: HTTPClient) -> None:
        """Test fetch_text returns text content"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "<html>Content</html>"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            text = await http_client.fetch_text("https://example.com")
            assert text == "<html>Content</html>"


class TestHTTPClientFetchJson:
    """Tests for fetch_json method"""

    @pytest.mark.asyncio
    async def test_fetch_json_returns_dict(self, http_client: HTTPClient) -> None:
        """Test fetch_json returns parsed JSON"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"openapi": "3.0.0", "info": {"title": "API"}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            data = await http_client.fetch_json("https://example.com/openapi.json")
            assert data["openapi"] == "3.0.0"
            assert "info" in data

    @pytest.mark.asyncio
    async def test_fetch_json_handles_invalid_json(self, http_client: HTTPClient) -> None:
        """Test fetch_json handles invalid JSON gracefully"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(ValueError):
                await http_client.fetch_json("https://example.com/invalid.json")


class TestHTTPClientGetContentType:
    """Tests for get_content_type method"""

    @pytest.mark.asyncio
    async def test_get_content_type_returns_header(self, http_client: HTTPClient) -> None:
        """Test get_content_type returns content-type header"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            content_type = await http_client.get_content_type("https://example.com")
            assert content_type == "application/json"

    @pytest.mark.asyncio
    async def test_get_content_type_returns_none_on_error(self, http_client: HTTPClient) -> None:
        """Test get_content_type returns None on error"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.head = AsyncMock(side_effect=httpx.HTTPError("Error"))
            mock_client_class.return_value = mock_client

            content_type = await http_client.get_content_type("https://example.com")
            assert content_type is None


class TestHTTPClientIsReachable:
    """Tests for is_reachable method"""

    @pytest.mark.asyncio
    async def test_is_reachable_returns_true(self, http_client: HTTPClient) -> None:
        """Test is_reachable returns True for reachable URL"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.is_success = True

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await http_client.is_reachable("https://example.com")
            assert result is True

    @pytest.mark.asyncio
    async def test_is_reachable_returns_false(self, http_client: HTTPClient) -> None:
        """Test is_reachable returns False for unreachable URL"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.head = AsyncMock(side_effect=httpx.NetworkError("Unreachable"))
            mock_client_class.return_value = mock_client

            result = await http_client.is_reachable("https://unreachable.com")
            assert result is False
