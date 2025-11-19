"""
Crawl endpoint for documentation indexing with incremental batch processing.
"""

import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from curlinator.api.models.crawl import CrawlRequest, CrawlResponse, CrawlProgressResponse
from curlinator.api.database import get_db
from curlinator.api.db.models import User, UserSettings, CrawlJob, CrawlStatus
from curlinator.api.auth import get_current_user
from curlinator.api.utils.embeddings import get_embedding_model
from curlinator.api.utils.validators import validate_crawl_request
from curlinator.api.middleware import limiter
from curlinator.api.error_codes import (
    create_error_response,
    CRAWL_FAILED,
    DATABASE_QUERY_FAILED
)
from curlinator.api.services.incremental_crawler import execute_incremental_crawl, calculate_batch_size

router = APIRouter(prefix="/api/v1", tags=["crawl"])
logger = logging.getLogger(__name__)

# Timeout for crawl operations (in seconds)
CRAWL_TIMEOUT_SECONDS = 600  # 10 minutes


@router.post("/crawl", response_model=CrawlResponse)
@limiter.limit("5/hour")
async def crawl_documentation(
    request: Request,
    body: CrawlRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Crawl API documentation with incremental batch processing and real-time indexing.

    **NEW BEHAVIOR (Incremental Crawling):**
    - Returns immediately with status="in_progress"
    - Crawls and indexes in batches (10-30 pages per batch) in background
    - Users can start querying after first batch (~10 seconds)
    - Check progress at GET /api/v1/crawl/{crawl_id}/status

    **Workflow:**
    1. Validates URL and parameters
    2. Creates CrawlJob in database
    3. Starts background crawl task
    4. Returns immediately with crawl_id and collection_name
    5. Background task crawls in batches and indexes incrementally
    6. Users can query collection while crawl is still in progress

    Args:
        request: Starlette Request object (for rate limiting)
        body: CrawlRequest with url, max_pages, max_depth, embedding_provider
        background_tasks: FastAPI BackgroundTasks for async processing
        current_user: Authenticated user (from JWT token)
        db: Database session

    Returns:
        CrawlResponse with:
        - crawl_id: Unique identifier for this crawl
        - status: "in_progress" (crawl running in background)
        - pages_crawled: 0 (will be updated as crawl progresses)
        - pages_indexed: 0 (will be updated as batches are indexed)
        - collection_name: Chroma collection name for querying
        - message: Instructions to check status endpoint

    Raises:
        HTTPException:
            - 422 if validation fails
            - 402 if free tier user tries to use API-based embeddings
            - 500 if initialization fails
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    try:
        # Validate request parameters
        validate_crawl_request(
            url=str(body.url),
            max_pages=body.max_pages,
            max_depth=body.max_depth
        )

        crawl_id = str(uuid.uuid4())
        collection_name = f"crawl_{crawl_id}"

        # Load user settings for freemium logic
        user_settings = db.query(UserSettings).filter(UserSettings.user_id == current_user.id).first()
        if not user_settings:
            # Create default settings if not exists
            user_settings = UserSettings(
                user_id=current_user.id,
                preferred_embedding_provider="local",
                default_max_pages=50,
                default_max_depth=3,
                free_messages_used=0,
                free_messages_limit=10,
                last_message_reset_date=datetime.now(timezone.utc),
            )
            db.add(user_settings)
            db.commit()
            db.refresh(user_settings)

        # Check if user has any API key configured
        has_api_key = bool(
            user_settings.user_openai_api_key_encrypted or
            user_settings.user_anthropic_api_key_encrypted or
            user_settings.user_gemini_api_key_encrypted
        )

        # Enforce local embeddings for free tier users
        if not has_api_key:
            if body.embedding_provider.upper() != "LOCAL":
                log_adapter.warning(f"Free tier user attempted to use {body.embedding_provider} embeddings")
                raise HTTPException(
                    status_code=402,
                    detail={
                        "error": "API-based embeddings require API key",
                        "message": "Free tier users must use local embeddings. API-based embeddings (OpenAI, Gemini) require you to add your own API key.",
                        "current_provider": body.embedding_provider,
                        "allowed_provider": "local",
                        "upgrade_options": {
                            "byok": True,
                            "paid_credits": False
                        },
                        "suggestion": "Set embedding_provider to 'local' or add your API key in Settings (PATCH /api/v1/settings) to use OpenAI or Gemini embeddings."
                    }
                )

        log_adapter.info(
            f"Starting incremental crawl: URL={body.url}, max_pages={body.max_pages}, "
            f"max_depth={body.max_depth}, user={current_user.email}, "
            f"embedding_provider={body.embedding_provider}"
        )

        # Get embedding model based on request
        try:
            embed_model, provider_name, model_name = get_embedding_model(body.embedding_provider)
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

        # Calculate batch size for progress tracking
        batch_size = calculate_batch_size(body.max_pages)
        total_batches_estimate = (body.max_pages + batch_size - 1) // batch_size

        # Create CrawlJob in database
        try:
            crawl_job = CrawlJob(
                crawl_id=crawl_id,
                user_id=current_user.id,
                collection_id=None,  # Will be set after first batch is indexed
                collection_name=collection_name,
                url=str(body.url),
                max_pages=body.max_pages,
                max_depth=body.max_depth,
                embedding_provider=provider_name,
                embedding_model=model_name,
                status=CrawlStatus.IN_PROGRESS,
                pages_crawled=0,
                pages_indexed=0,
                current_batch=0,
                total_batches_estimate=total_batches_estimate,
                batch_size=batch_size,
                started_at=datetime.now(timezone.utc),
            )

            db.add(crawl_job)
            db.commit()
            db.refresh(crawl_job)

            log_adapter.info(f"Created CrawlJob: {crawl_id}, batch_size={batch_size}, estimated_batches={total_batches_estimate}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to create CrawlJob: {str(e)}", exc_info=True)
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    error_code=DATABASE_QUERY_FAILED,
                    message="Failed to create crawl job in database",
                    suggestion="This may be a temporary issue. Please try again."
                )
            )

        # Create a session factory for background task
        # Background tasks can't use the same session as the request
        from curlinator.api.database import SessionLocal

        def db_session_factory():
            """Context manager for creating database sessions in background task."""
            return SessionLocal()

        # Start background crawl task
        background_tasks.add_task(
            execute_incremental_crawl,
            crawl_id=crawl_id,
            url=str(body.url),
            max_pages=body.max_pages,
            max_depth=body.max_depth,
            collection_name=collection_name,
            embed_model=embed_model,
            provider_name=provider_name,
            model_name=model_name,
            user_id=current_user.id,
            db_session_factory=db_session_factory,
        )

        log_adapter.info(f"Started background crawl task for {crawl_id}")

        # Return immediately with in_progress status
        return CrawlResponse(
            crawl_id=crawl_id,
            status="in_progress",
            pages_crawled=0,
            pages_indexed=0,
            collection_name=collection_name,
            message=f"Crawl started in background. Check status at GET /api/v1/crawl/{crawl_id}/status. You can start querying after the first batch is indexed (~10 seconds)."
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any unexpected errors during initialization
        logger.error(f"Unexpected error during crawl initialization: {str(e)}", exc_info=True)

        # Rollback database transaction
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Failed to rollback database: {rollback_error}")

        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                error_code=CRAWL_FAILED,
                message=f"An unexpected error occurred during crawl initialization: {str(e)}",
                suggestion="Please try again. If the problem persists, contact support."
            )
        )

@router.get("/crawl/{crawl_id}/status", response_model=CrawlProgressResponse)
async def get_crawl_status(
    crawl_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get real-time progress of an incremental crawl job.

    Returns detailed progress information including:
    - Current status (in_progress, completed, failed, cancelled)
    - Pages crawled and indexed
    - Current batch and estimated total batches
    - Estimated completion time
    - Error message (if failed)

    Args:
        crawl_id: Unique crawl job identifier
        current_user: Authenticated user (from JWT token)
        db: Database session

    Returns:
        CrawlProgressResponse with detailed progress metrics

    Raises:
        HTTPException:
            - 404 if crawl job not found or user doesn't have access
            - 500 if database error occurs
    """
    try:
        # Query crawl job and verify ownership
        crawl_job = db.query(CrawlJob).filter(
            CrawlJob.crawl_id == crawl_id,
            CrawlJob.user_id == current_user.id
        ).first()

        if not crawl_job:
            logger.warning(f"Crawl job not found: {crawl_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Crawl job not found",
                    "message": f"Crawl job '{crawl_id}' not found or you don't have access to it",
                    "suggestion": "Verify the crawl_id is correct and belongs to your account"
                }
            )

        # Calculate progress percentage
        percent_complete = 0.0
        if crawl_job.total_batches_estimate and crawl_job.total_batches_estimate > 0:
            percent_complete = (crawl_job.current_batch / crawl_job.total_batches_estimate) * 100

        # Build progress response
        return CrawlProgressResponse(
            crawl_id=crawl_job.crawl_id,
            collection_name=crawl_job.collection_name,
            status=crawl_job.status.value,
            progress={
                "pages_crawled": crawl_job.pages_crawled,
                "pages_indexed": crawl_job.pages_indexed,
                "pages_total_estimate": crawl_job.max_pages,
                "current_batch": crawl_job.current_batch,
                "total_batches_estimate": crawl_job.total_batches_estimate,
                "percent_complete": round(percent_complete, 1)
            },
            started_at=crawl_job.started_at,
            updated_at=crawl_job.updated_at,
            completed_at=crawl_job.completed_at,
            estimated_completion_at=crawl_job.estimated_completion_at,
            error_message=crawl_job.error_message
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error getting crawl status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to retrieve crawl status from database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error getting crawl status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )
