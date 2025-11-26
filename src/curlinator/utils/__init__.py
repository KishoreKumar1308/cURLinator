"""Utility functions"""

from .contextual_enrichment import (
    enrich_document_with_context,
    enrich_documents_batch,
    generate_contextual_prefix,
)
from .openapi_detector import (
    detect_openapi_spec,
    parse_openapi_to_documents,
)
from .page_classifier import (
    classify_page_type,
    extract_page_metadata,
)
from .url_helpers import (
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

__all__ = [
    # URL helpers
    "normalize_url",
    "is_valid_url",
    "guess_openapi_paths",
    "extract_domain",
    "is_documentation_url",
    "build_full_url",
    "is_same_domain",
    "get_base_path",
    "is_external_url",
    # OpenAPI detection
    "detect_openapi_spec",
    "parse_openapi_to_documents",
    # Page classification
    "classify_page_type",
    "extract_page_metadata",
    # Contextual enrichment
    "enrich_document_with_context",
    "generate_contextual_prefix",
    "enrich_documents_batch",
]

