"""HTTP client service for making API requests and fetching documentation"""

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from curlinator.config import get_settings


class HTTPClient:
    """
    HTTP client for making requests to APIs and documentation sites.

    Features:
    - Async support
    - Automatic retries on failures
    - Timeout handling
    - Custom headers support
    """

    def __init__(
        self,
        timeout: int | None = None,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds (default from settings)
            max_retries: Maximum number of retry attempts
            headers: Default headers to include in all requests
        """
        self.settings = get_settings()
        self.timeout = timeout or (self.settings.crawler_timeout / 1000)
        self.max_retries = max_retries
        self.default_headers = headers or {
            "User-Agent": "cURLinator/0.1.0 (API Documentation Crawler)",
            "Accept": "*/*",
        }

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        follow_redirects: bool = True,
    ) -> httpx.Response:
        """
        Make a GET request.

        Args:
            url: URL to request
            headers: Additional headers to include
            params: Query parameters
            follow_redirects: Whether to follow redirects

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: On request failure
        """
        merged_headers = {**self.default_headers, **(headers or {})}

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        )
        async def _make_request() -> httpx.Response:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=follow_redirects,
            ) as client:
                response = await client.get(url, headers=merged_headers, params=params)
                response.raise_for_status()
                return response

        return await _make_request()

    async def head(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = True,
    ) -> bool:
        """
        Check if a URL exists using HEAD request.

        Args:
            url: URL to check
            headers: Additional headers
            follow_redirects: Whether to follow redirects

        Returns:
            True if URL exists (2xx status), False otherwise
        """
        merged_headers = {**self.default_headers, **(headers or {})}

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=follow_redirects,
            ) as client:
                response = await client.head(url, headers=merged_headers)
                return response.is_success
        except Exception:
            return False

    async def post(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """
        Make a POST request.

        Args:
            url: URL to request
            data: Form data to send
            json: JSON data to send
            headers: Additional headers

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: On request failure
        """
        merged_headers = {**self.default_headers, **(headers or {})}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                data=data,
                json=json,
                headers=merged_headers,
            )
            response.raise_for_status()
            return response

    async def fetch_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        """
        Fetch URL and return text content.

        Args:
            url: URL to fetch
            headers: Additional headers

        Returns:
            Response text content

        Raises:
            httpx.HTTPError: On request failure
        """
        response = await self.get(url, headers=headers)
        return response.text

    async def fetch_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch URL and parse JSON response.

        Useful for fetching OpenAPI/Swagger specifications.

        Args:
            url: URL to fetch
            headers: Additional headers

        Returns:
            Parsed JSON as dictionary

        Raises:
            httpx.HTTPError: On request failure
            ValueError: If response is not valid JSON
        """
        response = await self.get(url, headers=headers)
        return response.json()

    async def download_file(
        self,
        url: str,
        file_path: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Download a file from URL to local path.

        Args:
            url: URL to download from
            file_path: Local path to save file
            headers: Additional headers

        Raises:
            httpx.HTTPError: On request failure
            IOError: On file write failure
        """
        response = await self.get(url, headers=headers)

        with open(file_path, "wb") as f:
            f.write(response.content)

    async def get_content_type(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> str | None:
        """
        Get the content type of a URL without downloading the full content.

        Args:
            url: URL to check
            headers: Additional headers

        Returns:
            Content-Type header value or None
        """
        merged_headers = {**self.default_headers, **(headers or {})}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.head(url, headers=merged_headers)
                return response.headers.get("content-type")
        except Exception:
            return None

    async def is_reachable(self, url: str) -> bool:
        """
        Check if a URL is reachable.

        Args:
            url: URL to check

        Returns:
            True if URL is reachable, False otherwise
        """
        return await self.head(url)

    def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        # httpx.AsyncClient handles cleanup automatically with context managers
        pass
