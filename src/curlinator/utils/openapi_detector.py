"""
OpenAPI/Swagger specification detection and parsing utilities.

This module provides functions to detect and parse OpenAPI/Swagger specifications
from URLs and convert them into LlamaIndex Document objects for RAG indexing.
"""

import logging
from typing import Any
from urllib.parse import urljoin

import httpx
import yaml
from llama_index.core.schema import Document

from curlinator.services import HTTPClient

logger = logging.getLogger(__name__)


async def _detect_from_swagger_ui(url: str) -> str | None:
    """
    Attempt to extract OpenAPI spec URL from Swagger UI page.

    Swagger UI pages typically load the spec URL via JavaScript.
    Common patterns:
    - SwaggerUIBundle({ url: "..." })
    - const specUrl = "..."
    - "url":"..."

    Args:
        url: URL of potential Swagger UI page

    Returns:
        Spec URL if found, None otherwise
    """
    import re

    http_client = HTTPClient()

    try:
        # Fetch the page
        response = await http_client.get(url)
        html_content = response.text

        # Pattern 1: SwaggerUIBundle({ url: "..." })
        # Matches: url: "https://example.com/spec.json"
        pattern1 = r'url:\s*["\']([^"\']+\.(?:json|yaml|yml))["\']'
        match = re.search(pattern1, html_content, re.IGNORECASE)
        if match:
            spec_url = match.group(1)
            # Make absolute URL if relative
            if not spec_url.startswith(("http://", "https://")):
                spec_url = urljoin(url, spec_url)
            logger.info(f"🔍 Found spec URL in Swagger UI: {spec_url}")
            return spec_url

        # Pattern 2: "url":"..." (JSON format)
        pattern2 = r'"url"\s*:\s*"([^"]+\.(?:json|yaml|yml))"'
        match = re.search(pattern2, html_content, re.IGNORECASE)
        if match:
            spec_url = match.group(1)
            if not spec_url.startswith(("http://", "https://")):
                spec_url = urljoin(url, spec_url)
            logger.info(f"🔍 Found spec URL in Swagger UI: {spec_url}")
            return spec_url

        # Pattern 3: const/var specUrl = "..."
        pattern3 = r'(?:const|var|let)\s+(?:specUrl|definitionURL|spec_url)\s*=\s*["\']([^"\']+\.(?:json|yaml|yml))["\']'
        match = re.search(pattern3, html_content, re.IGNORECASE)
        if match:
            spec_url = match.group(1)
            if not spec_url.startswith(("http://", "https://")):
                spec_url = urljoin(url, spec_url)
            logger.info(f"🔍 Found spec URL in Swagger UI: {spec_url}")
            return spec_url

        return None

    except Exception as e:
        logger.debug(f"Error detecting spec from Swagger UI: {e}")
        return None


async def detect_openapi_spec(url: str) -> str | None:
    """
    Detect if a URL points to an OpenAPI/Swagger specification.

    Tries common OpenAPI spec locations and validates the content.
    Also attempts to extract spec URL from Swagger UI pages.

    Args:
        url: Base URL or direct spec URL to check

    Returns:
        URL of the OpenAPI spec if found, None otherwise

    Example:
        >>> spec_url = await detect_openapi_spec("https://api.stripe.com/docs")
        >>> if spec_url:
        ...     print(f"Found spec at: {spec_url}")
    """
    # First, try to detect spec URL from Swagger UI page
    swagger_ui_spec = await _detect_from_swagger_ui(url)
    if swagger_ui_spec:
        return swagger_ui_spec

    # Common OpenAPI spec paths to try
    # Organized by likelihood and common patterns
    common_paths = [
        "",  # Direct URL
        # Standard paths
        "/openapi.json",
        "/openapi.yaml",
        "/swagger.json",
        "/swagger.yaml",
        "/api-docs",
        # API subdirectory paths
        "/api/openapi.json",
        "/api/openapi.yaml",
        "/api/swagger.json",
        "/api/swagger.yaml",
        # Versioned API paths (common in Swagger UI deployments)
        "/api/v1/openapi.json",
        "/api/v2/openapi.json",
        "/api/v3/openapi.json",
        "/api/v1/swagger.json",
        "/api/v2/swagger.json",
        "/api/v3/swagger.json",
        # Version-only paths
        "/v1/openapi.json",
        "/v2/openapi.json",
        "/v3/openapi.json",
        "/v1/swagger.json",
        "/v2/swagger.json",
        "/v3/swagger.json",
        # Docs subdirectory paths
        "/docs/openapi.json",
        "/docs/swagger.json",
        "/docs/openapi.yaml",
        "/docs/swagger.yaml",
        # Additional common patterns
        "/api/v1/openapi.yaml",
        "/api/v2/openapi.yaml",
        "/api/v3/openapi.yaml",
        "/api/v31/openapi.json",  # OpenAPI 3.1
        "/api/v30/openapi.json",  # OpenAPI 3.0
    ]

    http_client = HTTPClient()

    for path in common_paths:
        try:
            # Construct full URL
            if path == "":
                spec_url = url
            else:
                spec_url = urljoin(url, path)

            # Try to fetch the spec
            response = await http_client.get(spec_url)

            # Check content type
            content_type = response.headers.get("content-type", "").lower()

            # Try to parse as JSON or YAML
            spec_data = None
            if "json" in content_type or spec_url.endswith(".json"):
                try:
                    spec_data = response.json()
                except Exception:
                    continue
            elif "yaml" in content_type or spec_url.endswith((".yaml", ".yml")):
                try:
                    spec_data = yaml.safe_load(response.text)
                except Exception:
                    continue
            else:
                # Try JSON first, then YAML
                try:
                    spec_data = response.json()
                except Exception:
                    try:
                        spec_data = yaml.safe_load(response.text)
                    except Exception:
                        continue

            # Validate it's an OpenAPI/Swagger spec
            if _is_valid_openapi_spec(spec_data):
                logger.info(f"✅ Found OpenAPI spec at: {spec_url}")
                return spec_url

        except httpx.HTTPError:
            continue
        except Exception as e:
            logger.debug(f"Error checking {spec_url}: {e}")
            continue

    logger.info(f"❌ No OpenAPI spec found at {url}")
    return None


def _is_valid_openapi_spec(spec_data: Any) -> bool:
    """
    Validate if data is a valid OpenAPI/Swagger specification.

    Args:
        spec_data: Parsed JSON/YAML data

    Returns:
        True if valid OpenAPI/Swagger spec, False otherwise
    """
    if not isinstance(spec_data, dict):
        return False

    # Check for OpenAPI 3.x
    if "openapi" in spec_data:
        version = spec_data.get("openapi", "")
        if isinstance(version, str) and version.startswith("3."):
            return "info" in spec_data and "paths" in spec_data

    # Check for Swagger 2.0
    if "swagger" in spec_data:
        version = spec_data.get("swagger", "")
        if version == "2.0":
            return "info" in spec_data and "paths" in spec_data

    return False


async def parse_openapi_to_documents(spec_url: str) -> list[Document]:
    """
    Parse OpenAPI specification and convert to LlamaIndex Document objects.

    Creates separate documents for:
    - API overview (title, description, version)
    - Each endpoint (path + method combination)
    - Authentication schemes
    - Data models/schemas

    Args:
        spec_url: URL of the OpenAPI specification

    Returns:
        List of LlamaIndex Document objects with metadata

    Raises:
        httpx.HTTPError: If spec cannot be fetched
        ValueError: If spec is invalid or cannot be parsed

    Example:
        >>> documents = await parse_openapi_to_documents("https://api.stripe.com/openapi.json")
        >>> print(f"Created {len(documents)} documents from OpenAPI spec")
    """
    http_client = HTTPClient()

    try:
        # Fetch the spec
        response = await http_client.get(spec_url)

        # Parse as JSON or YAML
        if spec_url.endswith((".yaml", ".yml")):
            spec_data = yaml.safe_load(response.text)
        else:
            spec_data = response.json()

        # Validate spec
        if not _is_valid_openapi_spec(spec_data):
            raise ValueError(f"Invalid OpenAPI specification at {spec_url}")

        # Determine version
        version = spec_data.get("openapi") or spec_data.get("swagger")
        is_openapi_3 = version.startswith("3.") if isinstance(version, str) else False

        documents = []

        # 1. Create API overview document
        overview_doc = _create_overview_document(spec_data, spec_url, version)
        documents.append(overview_doc)

        # 2. Create endpoint documents
        endpoint_docs = _create_endpoint_documents(spec_data, spec_url, is_openapi_3)
        documents.extend(endpoint_docs)

        # 3. Create authentication document
        auth_doc = _create_auth_document(spec_data, spec_url, is_openapi_3)
        if auth_doc:
            documents.append(auth_doc)

        # 4. Create schema documents (for important models)
        schema_docs = _create_schema_documents(spec_data, spec_url, is_openapi_3)
        documents.extend(schema_docs)

        logger.info(f"✅ Created {len(documents)} documents from OpenAPI spec")
        return documents

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch OpenAPI spec from {spec_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse OpenAPI spec: {e}")
        raise ValueError(f"Failed to parse OpenAPI spec: {e}")


def _create_overview_document(spec_data: dict, spec_url: str, version: str) -> Document:
    """Create overview document from OpenAPI spec."""
    info = spec_data.get("info", {})

    text = f"""API Overview

Title: {info.get("title", "Unknown API")}
Version: {info.get("version", "Unknown")}
OpenAPI Version: {version}

Description:
{info.get("description", "No description available")}

Base URL: {_get_base_url(spec_data)}
"""

    # Add contact info if available
    if "contact" in info:
        contact = info["contact"]
        text += f"\nContact: {contact.get('name', '')} ({contact.get('email', '')})"

    return Document(
        text=text,
        metadata={
            "source": spec_url,
            "type": "api_overview",
            "api_title": info.get("title", "Unknown API"),
            "api_version": info.get("version", "Unknown"),
            "openapi_version": version,
        },
    )


def _create_endpoint_documents(
    spec_data: dict, spec_url: str, is_openapi_3: bool
) -> list[Document]:
    """Create documents for each API endpoint."""
    documents = []
    paths = spec_data.get("paths", {})

    for path, path_item in paths.items():
        # Each HTTP method is a separate endpoint
        for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
            if method not in path_item:
                continue

            operation = path_item[method]

            # Build endpoint documentation text
            text = f"""Endpoint: {method.upper()} {path}

Summary: {operation.get("summary", "No summary")}

Description:
{operation.get("description", "No description available")}
"""

            # Add parameters
            params = operation.get("parameters", [])
            if params:
                text += "\n\nParameters:\n"
                for param in params:
                    param_in = param.get("in", "unknown")
                    param_name = param.get("name", "unknown")
                    param_required = " (required)" if param.get("required") else ""
                    param_desc = param.get("description", "")
                    text += f"- {param_name} ({param_in}){param_required}: {param_desc}\n"

            # Add request body (OpenAPI 3.x)
            if is_openapi_3 and "requestBody" in operation:
                text += "\n\nRequest Body:\n"
                req_body = operation["requestBody"]
                text += f"{req_body.get('description', 'No description')}\n"

            # Add responses
            responses = operation.get("responses", {})
            if responses:
                text += "\n\nResponses:\n"
                for status_code, response in responses.items():
                    text += f"- {status_code}: {response.get('description', 'No description')}\n"

            # Convert tags list to comma-separated string for Chroma compatibility
            tags = operation.get("tags", [])
            tags_str = ", ".join(tags) if tags else ""

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": spec_url,
                        "type": "api_endpoint",
                        "endpoint": path,
                        "method": method.upper(),
                        "operation_id": operation.get("operationId", f"{method}_{path}"),
                        "tags": tags_str,  # Store as comma-separated string
                    },
                )
            )

    return documents


def _create_auth_document(spec_data: dict, spec_url: str, is_openapi_3: bool) -> Document | None:
    """Create authentication document."""
    if is_openapi_3:
        security_schemes = spec_data.get("components", {}).get("securitySchemes", {})
    else:
        security_schemes = spec_data.get("securityDefinitions", {})

    if not security_schemes:
        return None

    text = "Authentication\n\n"

    for name, scheme in security_schemes.items():
        scheme_type = scheme.get("type", "unknown")
        text += f"\n{name} ({scheme_type}):\n"
        text += f"{scheme.get('description', 'No description')}\n"

        if scheme_type == "apiKey":
            text += f"- Location: {scheme.get('in', 'unknown')}\n"
            text += f"- Parameter name: {scheme.get('name', 'unknown')}\n"
        elif scheme_type == "oauth2":
            text += "- OAuth 2.0 authentication\n"

    return Document(
        text=text,
        metadata={
            "source": spec_url,
            "type": "authentication",
        },
    )


def _create_schema_documents(spec_data: dict, spec_url: str, is_openapi_3: bool) -> list[Document]:
    """Create documents for important data schemas/models."""
    # For now, return empty list - can be expanded later
    # This would parse components/schemas (OpenAPI 3) or definitions (Swagger 2)
    return []


def _get_base_url(spec_data: dict) -> str:
    """Extract base URL from OpenAPI spec."""
    # OpenAPI 3.x
    servers = spec_data.get("servers", [])
    if servers and len(servers) > 0:
        return servers[0].get("url", "")

    # Swagger 2.0
    schemes = spec_data.get("schemes", ["https"])
    host = spec_data.get("host", "")
    base_path = spec_data.get("basePath", "")

    if host:
        return f"{schemes[0]}://{host}{base_path}"

    return "Unknown"
