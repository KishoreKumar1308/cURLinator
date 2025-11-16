"""Utility functions"""

from .url_helpers import (
    normalize_url,
    is_valid_url,
    guess_openapi_paths,
    extract_domain,
    is_documentation_url,
    build_full_url,
    is_same_domain,
    get_base_path,
    is_external_url,
)

from .openapi_detector import (
    detect_openapi_spec,
    parse_openapi_to_documents,
)

from .page_classifier import (
    classify_page_type,
    extract_page_metadata,
)

from .contextual_enrichment import (
    enrich_document_with_context,
    generate_contextual_prefix,
    enrich_documents_batch,
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

