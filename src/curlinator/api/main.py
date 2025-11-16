"""
cURLinator FastAPI Application

Main entry point for the cURLinator API backend.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from curlinator.api.middleware import (
    setup_cors,
    setup_logging,
    setup_rate_limiting,
    setup_rate_limit_headers,
    setup_request_logging,
    setup_metrics
)
from curlinator.api.routes import health, crawl, chat, auth, collections, metrics
from curlinator.config import get_settings
from curlinator.config.settings import validate_environment

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Handles startup and shutdown events.
    """
    # Startup
    # Validate environment variables
    try:
        validate_environment()
    except ValueError as e:
        logger.error(str(e))
        raise

    settings = get_settings()

    # Initialize Sentry for error tracking
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.sentry_environment or settings.environment,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            integrations=[
                FastApiIntegration(),
                StarletteIntegration(),
                SqlalchemyIntegration(),
            ],
            # Send default PII (Personally Identifiable Information)
            send_default_pii=False,
            # Attach stack traces to messages
            attach_stacktrace=True,
            # Enable performance monitoring
            enable_tracing=True,
            # Set release version (optional - can be set via env var)
            release=f"curlinator@1.0.0",
        )
        logger.info(f"✅ Sentry initialized for environment: {settings.sentry_environment or settings.environment}")
    else:
        logger.info("ℹ️  Sentry DSN not configured - error tracking disabled")

    # Configure LLM (if API key is available)
    if settings.openai_api_key:
        llm_kwargs = {
            "model": settings.default_model_openai,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_api_base:
            llm_kwargs["api_base"] = settings.openai_api_base

        Settings.llm = OpenAI(**llm_kwargs)
        logger.info(f"✅ Configured LLM: {settings.default_model_openai}")
    else:
        logger.warning("⚠️  No LLM API key found in environment")

    # NOTE: Embedding models are configured per-request in the crawl endpoint
    # to support multiple providers and avoid global state conflicts

    logger.info("🚀 cURLinator API started successfully")
    logger.info("📚 API documentation available at /docs")

    yield

    # Shutdown
    logger.info("👋 cURLinator API shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="cURLinator API",
    version="1.0.0",
    description="AI-powered API documentation exploration and cURL command generation",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Setup middleware
setup_cors(app)
setup_logging(app)
setup_rate_limiting(app)
setup_rate_limit_headers(app)  # Add rate limit headers to responses
setup_metrics(app)  # Add Prometheus metrics collection
setup_request_logging(app)  # Add request logging with correlation IDs

# Include routers
app.include_router(health.router)
app.include_router(metrics.router)  # Metrics endpoint (no auth required)
app.include_router(auth.router)
app.include_router(crawl.router)
app.include_router(chat.router)
app.include_router(collections.router)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle HTTP exceptions with consistent JSON response.

    Preserves structured error format if detail is a dict,
    otherwise converts simple string errors to structured format.
    """
    # Capture 5xx errors in Sentry (server errors)
    if exc.status_code >= 500:
        sentry_sdk.capture_exception(exc)

    # If detail is already a dict (structured error), use it as-is
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        # Convert simple string errors to structured format
        content = {
            "error": "Error",
            "message": exc.detail,
            "suggestion": "Please check your request and try again."
        }

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed error messages."""
    # Convert errors to JSON-serializable format
    errors = []
    for error in exc.errors():
        error_dict = {
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"],
        }
        # Convert ValueError context to string if present
        if "ctx" in error and "error" in error["ctx"]:
            error_dict["ctx"] = {"error": str(error["ctx"]["error"])}
        elif "ctx" in error:
            error_dict["ctx"] = error["ctx"]

        if "input" in error:
            error_dict["input"] = error["input"]

        errors.append(error_dict)

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": errors
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions.

    Captures the exception in Sentry and returns a generic error response.
    """
    # Capture exception in Sentry
    sentry_sdk.capture_exception(exc)

    # Log the error
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Return generic error response (don't expose internal details)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Our team has been notified.",
            "suggestion": "Please try again later or contact support if the issue persists."
        },
    )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "curlinator.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

