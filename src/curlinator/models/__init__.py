"""Data models for cURLinator"""

from .api_spec import (
    APISpecification,
    APIEndpoint,
    APIParameter,
    AuthMethod,
    HTTPMethod,
    ParameterLocation,
)
from .documentation import (
    DocumentationSource,
    DocumentationPage,
    CodeExample,
    APISectionSummary,
    PageSummary,
    OpenAPIInfo,
    CrawlStatistics,
)

__all__ = [
    "APISpecification",
    "APIEndpoint",
    "APIParameter",
    "AuthMethod",
    "HTTPMethod",
    "ParameterLocation",
    "DocumentationSource",
    "DocumentationPage",
    "CodeExample",
    "APISectionSummary",
    "PageSummary",
    "OpenAPIInfo",
    "CrawlStatistics",
]

