"""
Crawl endpoint for documentation indexing.
"""

import uuid
import logging
import asyncio
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from curlinator.api.models.crawl import CrawlRequest, CrawlResponse
from curlinator.api.database import get_db
from curlinator.api.db.models import User, DocumentationCollection
from curlinator.api.auth import get_current_user
from curlinator.api.utils.embeddings import get_embedding_model
from curlinator.api.utils.validators import validate_crawl_request
from curlinator.api.middleware import limiter
from curlinator.api.error_codes import (
    create_error_response,
    CRAWL_TIMEOUT,
    CRAWL_NO_DOCUMENTS,
    CRAWL_FAILED,
    DATABASE_INTEGRITY_ERROR,
    DATABASE_QUERY_FAILED
)
from curlinator.agents.documentation_agent import DocumentationAgent
from curlinator.agents.chat_agent import ChatAgent
from curlinator.api.metrics import (
    crawl_operations_total,
    crawl_duration_seconds,
    crawl_pages_total,
    crawl_pages_per_operation,
    vectorstore_documents_indexed_total,
    vectorstore_operations_total
)
import time

router = APIRouter(prefix="/api/v1", tags=["crawl"])
logger = logging.getLogger(__name__)

# Timeout for crawl operations (in seconds)
CRAWL_TIMEOUT_SECONDS = 600  # 10 minutes


@router.post("/crawl", response_model=CrawlResponse)
@limiter.limit("5/hour")
async def crawl_documentation(
    http_request: Request,
    request: CrawlRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Crawl API documentation and create vector index.

    Requires authentication.

    This endpoint:
    1. Validates the URL and parameters
    2. Uses DocumentationAgent to crawl the documentation (with timeout)
    3. Creates a ChatAgent with the crawled documents
    4. Saves collection to database with owner tracking
    5. Returns a collection_name for future queries

    Args:
        request: CrawlRequest with url, max_pages, max_depth
        current_user: Authenticated user (from JWT token)
        db: Database session

    Returns:
        CrawlResponse with crawl_id, status, pages_crawled, collection_name

    Raises:
        HTTPException:
            - 422 if validation fails
            - 400 if no documents found
            - 408 if crawl times out
            - 500 if unexpected error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(http_request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    crawl_id = None
    collection_name = None

    try:
        # Validate request parameters
        validate_crawl_request(
            url=str(request.url),
            max_pages=request.max_pages,
            max_depth=request.max_depth
        )

        crawl_id = str(uuid.uuid4())
        collection_name = f"crawl_{crawl_id}"

        log_adapter.info(
            f"Starting crawl: URL={request.url}, max_pages={request.max_pages}, "
            f"max_depth={request.max_depth}, user={current_user.email}, "
            f"embedding_provider={request.embedding_provider}"
        )

        # Start timer for crawl duration
        crawl_start_time = time.time()

        # Get embedding model based on request
        try:
            embed_model, provider_name, model_name = get_embedding_model(request.embedding_provider)
            log_adapter.info(f"Initialized embedding model: {provider_name}/{model_name}")
        except Exception as e:
            log_adapter.error(f"Failed to initialize embedding model: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    error_code="CRAWL_EMBEDDING_FAILED",
                    message=str(e),
                    suggestion="Check that required API keys are set if using OPENAI or GEMINI providers"
                )
            )

        # Initialize DocumentationAgent
        doc_agent = DocumentationAgent(
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            verbose=True,
        )

        # Execute crawl with timeout (async method)
        try:
            documents = await asyncio.wait_for(
                doc_agent.execute(str(request.url)),
                timeout=CRAWL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(f"Crawl timed out after {CRAWL_TIMEOUT_SECONDS} seconds")
            crawl_operations_total.labels(status="timeout").inc()
            crawl_duration_seconds.observe(time.time() - crawl_start_time)
            raise HTTPException(
                status_code=408,
                detail=create_error_response(
                    error_code=CRAWL_TIMEOUT,
                    message=f"Crawl operation timed out after {CRAWL_TIMEOUT_SECONDS // 60} minutes",
                    suggestion="Try reducing max_pages or max_depth parameters, or crawl a smaller documentation site"
                )
            )
        except Exception as e:
            logger.error(f"Crawl execution failed: {str(e)}", exc_info=True)
            crawl_operations_total.labels(status="failure").inc()
            crawl_duration_seconds.observe(time.time() - crawl_start_time)
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    error_code=CRAWL_FAILED,
                    message=f"Failed to crawl documentation: {str(e)}",
                    suggestion="Verify the URL is accessible and contains valid documentation"
                )
            )

        if not documents:
            crawl_operations_total.labels(status="failure").inc()
            crawl_duration_seconds.observe(time.time() - crawl_start_time)
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    error_code=CRAWL_NO_DOCUMENTS,
                    message="The crawl completed but found no documents to index",
                    suggestion="Verify the URL points to documentation pages with readable content"
                )
            )

        # Track successful crawl
        pages_crawled = len(documents)
        logger.info(f"Crawled {pages_crawled} documents")
        crawl_pages_total.labels(status="success").inc(pages_crawled)
        crawl_pages_per_operation.observe(pages_crawled)

        # Create ChatAgent with documents (builds index with specified embedding model)
        try:
            vectorstore_operations_total.labels(operation="index", status="in_progress").inc()

            chat_agent = ChatAgent(
                documents=documents,
                collection_name=collection_name,
                embed_model=embed_model,
                verbose=True,
            )

            # Track successful indexing
            vectorstore_documents_indexed_total.inc(pages_crawled)
            vectorstore_operations_total.labels(operation="index", status="success").inc()
            logger.info(f"Created collection: {collection_name} with {provider_name} embeddings")

        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}", exc_info=True)
            vectorstore_operations_total.labels(operation="index", status="failure").inc()
            crawl_operations_total.labels(status="failure").inc()
            crawl_duration_seconds.observe(time.time() - crawl_start_time)
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    error_code=CRAWL_FAILED,
                    message=f"Failed to create vector index: {str(e)}",
                    suggestion="This may be a temporary issue. Please try again."
                )
            )

        # Save collection to database with embedding metadata
        try:
            parsed_url = urlparse(str(request.url))
            domain = parsed_url.netloc

            collection = DocumentationCollection(
                name=collection_name,
                url=str(request.url),
                domain=domain,
                pages_crawled=len(documents),
                owner_id=current_user.id,
                is_public=False,
                embedding_provider=provider_name,
                embedding_model=model_name,
            )

            db.add(collection)
            db.commit()
            db.refresh(collection)

            logger.info(f"Saved collection to database: {collection.id}")
            logger.info(f"Embedding metadata: {provider_name} / {model_name}")

        except IntegrityError as e:
            logger.error(f"Database integrity error: {str(e)}", exc_info=True)
            db.rollback()

            # Try to clean up the vector store collection
            try:
                import chromadb
                from curlinator.config import get_settings
                settings = get_settings()
                chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
                chroma_client.delete_collection(collection_name)
                logger.info(f"Cleaned up vector store collection: {collection_name}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up vector store: {cleanup_error}")

            raise HTTPException(
                status_code=409,
                detail=create_error_response(
                    error_code=DATABASE_INTEGRITY_ERROR,
                    message="A collection with this name already exists in the database",
                    suggestion="This is an internal error. Please try again."
                )
            )
        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}", exc_info=True)
            db.rollback()

            # Try to clean up the vector store collection
            try:
                import chromadb
                from curlinator.config import get_settings
                settings = get_settings()
                chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
                chroma_client.delete_collection(collection_name)
                logger.info(f"Cleaned up vector store collection: {collection_name}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up vector store: {cleanup_error}")

            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    error_code=DATABASE_QUERY_FAILED,
                    message="Failed to save collection to database",
                    suggestion="This may be a temporary issue. Please try again."
                )
            )

        # Track successful crawl operation
        crawl_duration = time.time() - crawl_start_time
        crawl_duration_seconds.observe(crawl_duration)
        crawl_operations_total.labels(status="success").inc()

        return CrawlResponse(
            crawl_id=crawl_id,
            status="completed",
            pages_crawled=len(documents),
            collection_name=collection_name,
            message=f"Successfully crawled {len(documents)} pages"
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error during crawl: {str(e)}", exc_info=True)

        # Rollback database transaction
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Failed to rollback database: {rollback_error}")

        # Try to clean up vector store if collection was created
        if collection_name:
            try:
                import chromadb
                from curlinator.config import get_settings
                settings = get_settings()
                chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
                chroma_client.delete_collection(collection_name)
                logger.info(f"Cleaned up vector store collection: {collection_name}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up vector store: {cleanup_error}")

        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                error_code=CRAWL_FAILED,
                message=f"An unexpected error occurred: {str(e)}",
                suggestion="Please try again. If the problem persists, contact support."
            )
        )

