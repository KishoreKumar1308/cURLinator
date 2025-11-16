"""Core business logic and domain operations"""

from .openapi_validator import (
    OpenAPIVersion,
    ValidationResult,
    is_valid_openapi,
    get_openapi_version,
    validate_openapi_structure,
    extract_api_info,
    count_endpoints,
    has_authentication,
    get_spec_summary,
)

__all__ = [
    "OpenAPIVersion",
    "ValidationResult",
    "is_valid_openapi",
    "get_openapi_version",
    "validate_openapi_structure",
    "extract_api_info",
    "count_endpoints",
    "has_authentication",
    "get_spec_summary",
]
