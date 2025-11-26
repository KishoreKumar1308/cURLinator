"""cURLinator - AI-powered API documentation crawler and query engine

This is the core library that provides:
- DocumentationAgent: Crawls and processes API documentation
- ChatAgent: RAG-based querying and cURL command generation
- Utilities: OpenAPI detection, page classification, contextual enrichment
"""

from curlinator.agents import ChatAgent, DocumentationAgent

__version__ = "0.1.0"

__all__ = [
    "DocumentationAgent",
    "ChatAgent",
]
