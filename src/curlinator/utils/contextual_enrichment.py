"""
Contextual enrichment utilities for improving RAG retrieval accuracy.

Based on Anthropic's research showing that adding contextual prefixes to chunks
before embedding can reduce retrieval failures by 35-49%.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

import logging
from urllib.parse import urlparse

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def enrich_document_with_context(document: Document, site_context: str) -> Document:
    """
    Add contextual prefix to document text before embedding.

    This improves retrieval accuracy by providing context about where the
    document came from and what it's about. The original text is preserved
    and the context is prepended.

    Based on Anthropic's "Contextual Retrieval" approach which showed:
    - 35% reduction in retrieval failures with contextual embeddings
    - 49% reduction when combined with BM25 hybrid search

    Args:
        document: LlamaIndex Document to enrich
        site_context: Overall context about the documentation site
                     (e.g., "Stripe API documentation")

    Returns:
        New Document with contextual prefix added to text

    Example:
        >>> doc = Document(text="Create a customer...", metadata={"url": "..."})
        >>> enriched = enrich_document_with_context(doc, "Stripe API")
        >>> print(enriched.text)
        # "This document is from the Stripe API documentation...
        #  Create a customer..."
    """
    # Generate contextual prefix
    context_prefix = generate_contextual_prefix(document, site_context)

    # Create new document with enriched text
    enriched_text = f"{context_prefix}\n\n{document.text}"

    # Preserve all metadata and add enrichment flag
    enriched_metadata = document.metadata.copy()
    enriched_metadata["contextually_enriched"] = True
    enriched_metadata["original_text_length"] = len(document.text)

    return Document(
        text=enriched_text,
        metadata=enriched_metadata,
        id_=document.id_,
        embedding=document.embedding,
        excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
        excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
    )


def generate_contextual_prefix(document: Document, site_context: str) -> str:
    """
    Generate contextual prefix based on document metadata.

    Creates a concise context statement that helps the embedding model
    understand where this content came from and what it's about.

    Args:
        document: Document to generate context for
        site_context: Overall site context (e.g., "Stripe API documentation")

    Returns:
        Contextual prefix string

    Example:
        >>> doc = Document(
        ...     text="...",
        ...     metadata={
        ...         "url": "https://stripe.com/docs/api/customers/create",
        ...         "type": "api_endpoint",
        ...         "endpoint": "/v1/customers",
        ...         "method": "POST"
        ...     }
        ... )
        >>> prefix = generate_contextual_prefix(doc, "Stripe API")
        >>> print(prefix)
        # "This document is from the Stripe API documentation,
        #  specifically the API endpoint section for POST /v1/customers."
    """
    metadata = document.metadata

    # Start with base context
    context_parts = [f"This document is from the {site_context}"]

    # Add document type context
    doc_type = metadata.get("type", "")
    if doc_type:
        context_parts.append(_get_type_context(doc_type))

    # Add specific context based on type
    if doc_type == "api_endpoint":
        endpoint_context = _get_endpoint_context(metadata)
        if endpoint_context:
            context_parts.append(endpoint_context)

    elif doc_type == "api_overview":
        api_title = metadata.get("api_title", "")
        if api_title:
            context_parts.append(f"providing an overview of the {api_title}")

    elif doc_type == "authentication":
        context_parts.append("explaining authentication and authorization methods")

    # Add page type context if available
    page_type = metadata.get("page_type", "")
    if page_type and page_type != "unknown":
        page_context = _get_page_type_context(page_type)
        if page_context:
            context_parts.append(page_context)

    # Add title context if available
    title = metadata.get("title", "")
    if title and title != "Untitled":
        context_parts.append(f'titled "{title}"')

    # Add URL path context
    url = metadata.get("url", "")
    if url:
        url_context = _get_url_context(url)
        if url_context:
            context_parts.append(url_context)

    # Combine all parts into a coherent sentence
    if len(context_parts) == 1:
        return f"{context_parts[0]}."
    elif len(context_parts) == 2:
        return f"{context_parts[0]}, {context_parts[1]}."
    else:
        # Join with commas and "and" for the last item
        main_parts = ", ".join(context_parts[1:-1])
        return f"{context_parts[0]}, {main_parts}, and {context_parts[-1]}."


def _get_type_context(doc_type: str) -> str:
    """Get context description for document type."""
    type_descriptions = {
        "api_endpoint": "specifically an API endpoint reference",
        "api_overview": "providing an API overview",
        "authentication": "covering authentication",
        "guide": "from a guide section",
        "tutorial": "from a tutorial",
        "quickstart": "from the quickstart guide",
        "sdk": "covering SDK/library usage",
        "webhook": "explaining webhooks",
        "error": "documenting error handling",
        "changelog": "from the changelog",
    }
    return type_descriptions.get(doc_type, f"from the {doc_type} section")


def _get_endpoint_context(metadata: dict) -> str | None:
    """Get context for API endpoint documents."""
    method = metadata.get("method", "")
    endpoint = metadata.get("endpoint", "")

    if method and endpoint:
        return f"for the {method} {endpoint} endpoint"
    elif endpoint:
        return f"for the {endpoint} endpoint"

    return None


def _get_page_type_context(page_type: str) -> str | None:
    """Get context for page type."""
    page_type_contexts = {
        "api_reference": "in the API reference section",
        "guide": "from a how-to guide",
        "tutorial": "from a step-by-step tutorial",
        "overview": "providing an overview",
        "authentication": "explaining authentication",
        "quickstart": "from the getting started guide",
        "sdk": "covering SDK usage",
        "webhook": "about webhooks",
        "error": "about error handling",
        "changelog": "from the changelog",
    }
    return page_type_contexts.get(page_type)


def _get_url_context(url: str) -> str | None:
    """Extract meaningful context from URL path."""
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return None

        # Extract last meaningful segment
        segments = [s for s in path.split("/") if s and not s.isdigit()]

        if segments:
            # Get last 2 segments for context
            relevant_segments = segments[-2:] if len(segments) >= 2 else segments
            path_context = " > ".join(relevant_segments)
            return f"located at {path_context}"

        return None
    except Exception:
        return None


def enrich_documents_batch(
    documents: list[Document], site_context: str, verbose: bool = False
) -> list[Document]:
    """
    Enrich multiple documents with contextual prefixes.

    Convenience function for batch processing.

    Args:
        documents: List of documents to enrich
        site_context: Overall site context
        verbose: Whether to log progress

    Returns:
        List of enriched documents

    Example:
        >>> docs = [doc1, doc2, doc3]
        >>> enriched = enrich_documents_batch(docs, "Stripe API", verbose=True)
        >>> print(f"Enriched {len(enriched)} documents")
    """
    enriched_documents = []

    for i, doc in enumerate(documents):
        try:
            enriched_doc = enrich_document_with_context(doc, site_context)
            enriched_documents.append(enriched_doc)

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Enriched {i + 1}/{len(documents)} documents")

        except Exception as e:
            logger.warning(f"Failed to enrich document {i}: {e}")
            # Keep original document if enrichment fails
            enriched_documents.append(doc)

    if verbose:
        logger.info(f"✅ Enriched {len(enriched_documents)}/{len(documents)} documents")

    return enriched_documents
