"""
Pydantic models for crawl endpoints.
"""

from enum import Enum
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
from datetime import datetime


class EmbeddingProvider(str, Enum):
    """Embedding model provider options."""

    LOCAL = "local"  # HuggingFace BAAI/bge-small-en-v1.5 (free, slower, ~90MB download)
    OPENAI = "openai"  # OpenAI text-embedding-3-small (fast, costs money)
    GEMINI = "gemini"  # Google Gemini gemini-embedding-001 (fast, costs money)
    AUTO = "auto"  # Automatically choose based on available API keys


class CrawlRequest(BaseModel):
    """Request model for crawl endpoint."""

    url: HttpUrl = Field(
        ...,
        description="Base URL to crawl",
        examples=["https://stripe.com/docs/api"]
    )
    max_pages: Optional[int] = Field(
        50,
        ge=1,
        le=1000,
        description="Maximum number of pages to crawl"
    )
    max_depth: Optional[int] = Field(
        3,
        ge=1,
        le=5,
        description="Maximum crawl depth from base URL"
    )
    embedding_provider: EmbeddingProvider = Field(
        EmbeddingProvider.AUTO,
        description=(
            "Embedding model provider to use for vector indexing. "
            "AUTO selects based on available API keys. "
            "LOCAL uses free HuggingFace model (slower, ~90MB download). "
            "OPENAI/GEMINI use API-based embeddings (faster, costs money)."
        )
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://stripe.com/docs/api",
                    "max_pages": 50,
                    "max_depth": 3,
                    "embedding_provider": "auto"
                },
                {
                    "url": "https://httpbin.org/",
                    "max_pages": 10,
                    "max_depth": 1,
                    "embedding_provider": "local"
                }
            ]
        }
    }


class CrawlResponse(BaseModel):
    """Response model for crawl endpoint."""

    crawl_id: str = Field(..., description="Unique identifier for this crawl")
    status: str = Field(..., description="Crawl status (in_progress, completed, failed, cancelled)")
    pages_crawled: int = Field(default=0, description="Number of pages successfully crawled")
    pages_indexed: int = Field(default=0, description="Number of pages indexed in vector store")
    collection_name: str = Field(..., description="Chroma collection name for querying")
    message: str = Field(..., description="Human-readable status message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "crawl_id": "123e4567-e89b-12d3-a456-426614174000",
                    "status": "in_progress",
                    "pages_crawled": 0,
                    "pages_indexed": 0,
                    "collection_name": "crawl_123e4567-e89b-12d3-a456-426614174000",
                    "message": "Crawl started. Check status at GET /api/v1/crawl/{crawl_id}/status"
                }
            ]
        }
    }


class CrawlProgressResponse(BaseModel):
    """Response model for crawl progress/status endpoint."""

    crawl_id: str = Field(..., description="Unique identifier for this crawl")
    collection_name: str = Field(..., description="Chroma collection name")
    status: str = Field(..., description="Current crawl status")
    progress: dict = Field(..., description="Progress metrics")
    started_at: datetime = Field(..., description="When the crawl started")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="When the crawl completed")
    estimated_completion_at: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "crawl_id": "123e4567-e89b-12d3-a456-426614174000",
                    "collection_name": "crawl_123e4567-e89b-12d3-a456-426614174000",
                    "status": "in_progress",
                    "progress": {
                        "pages_crawled": 45,
                        "pages_indexed": 40,
                        "pages_total_estimate": 200,
                        "current_batch": 3,
                        "total_batches_estimate": 10,
                        "percent_complete": 22.5
                    },
                    "started_at": "2025-11-19T10:30:00Z",
                    "updated_at": "2025-11-19T10:32:15Z",
                    "completed_at": None,
                    "estimated_completion_at": "2025-11-19T10:40:00Z",
                    "error_message": None
                }
            ]
        }
    }

