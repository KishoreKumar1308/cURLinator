"""
Middleware configuration for FastAPI application.

Includes CORS, logging, rate limiting, request tracking, and metrics middleware.
"""

import os
import logging
import time
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for FastAPI app.
    
    Allows requests from:
    - Local development (localhost:3000, localhost:5173)
    - Vercel deployments (configured via ALLOWED_ORIGINS env var)
    
    Args:
        app: FastAPI application instance
    """
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173"
    ).split(",")
    
    # Strip whitespace from origins
    allowed_origins = [origin.strip() for origin in allowed_origins]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    logging.info(f"CORS configured with origins: {allowed_origins}")


def setup_logging(app: FastAPI) -> None:
    """
    Configure structured logging for the application.

    Args:
        app: FastAPI application instance
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure structured logging with correlation ID support
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add a default filter to add correlation_id to all log records
    class CorrelationIdFilter(logging.Filter):
        """Add correlation_id to log records if not present."""
        def filter(self, record):
            if not hasattr(record, 'correlation_id'):
                record.correlation_id = 'N/A'
            return True

    # Add filter to root logger
    logging.root.addFilter(CorrelationIdFilter())

    logging.info(f"Logging configured with level: {log_level}")


# Create limiter instance
# Disable rate limiting in test mode
if os.getenv("TESTING") == "true":
    # Create a no-op limiter for testing
    class NoOpLimiter:
        """No-op limiter that doesn't actually limit requests in tests."""
        def limit(self, *args, **kwargs):
            """Decorator that does nothing."""
            def decorator(func):
                return func
            return decorator

    limiter = NoOpLimiter()
else:
    limiter = Limiter(key_func=get_remote_address)


def setup_rate_limiting(app: FastAPI) -> None:
    """
    Configure rate limiting for API endpoints.

    Prevents abuse by limiting the number of requests per IP address:
    - Crawl endpoint: 5 requests per hour (expensive operation)
    - Chat endpoint: 60 requests per minute
    - Auth endpoints: 10 requests per minute (prevent brute force)

    Args:
        app: FastAPI application instance
    """
    app.state.limiter = limiter

    # Custom rate limit exceeded handler with structured error response
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded errors with structured response."""
        # Extract rate limit info from exception
        retry_after = getattr(exc, 'retry_after', None)

        response = Response(
            status_code=429,
            content='{"error": "Rate limit exceeded", "message": "Too many requests. Please try again later.", "suggestion": "Wait a few minutes before making more requests."}',
            media_type="application/json"
        )

        # Add rate limit headers
        if retry_after:
            response.headers["Retry-After"] = str(int(retry_after))

        return response

    logging.info("Rate limiting configured")


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add rate limit headers to responses.

    Adds the following headers to rate-limited endpoints:
    - X-RateLimit-Limit: Maximum number of requests allowed in the time window
    - X-RateLimit-Remaining: Number of requests remaining in the current window
    - X-RateLimit-Reset: Unix timestamp when the rate limit resets
    """

    # Map of endpoint patterns to their rate limits
    RATE_LIMITS = {
        "/api/v1/crawl": {"limit": 5, "period": 3600},  # 5/hour
        "/api/v1/chat": {"limit": 60, "period": 60},  # 60/minute
        "/api/v1/auth/register": {"limit": 10, "period": 60},  # 10/minute
        "/api/v1/auth/login": {"limit": 10, "period": 60},  # 10/minute
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add rate limit headers."""
        # Check if this endpoint has rate limiting
        path = request.url.path
        rate_limit_info = self.RATE_LIMITS.get(path)

        # Process request
        response = await call_next(request)

        # Add rate limit headers if this is a rate-limited endpoint
        if rate_limit_info:
            limit = rate_limit_info["limit"]
            period = rate_limit_info["period"]

            # Calculate reset time (current time + period)
            reset_time = int(time.time()) + period

            # Add headers
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Reset"] = str(reset_time)

            # If we're at the limit or exceeded it, set remaining to 0
            if response.status_code == 429:
                response.headers["X-RateLimit-Remaining"] = "0"
            else:
                # We don't have access to the exact count from slowapi,
                # so we indicate it's available (limit - 1 as a conservative estimate)
                response.headers["X-RateLimit-Remaining"] = str(limit - 1)

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging with correlation IDs.

    Logs:
    - Request method, path, client IP, correlation ID
    - Response status code, processing time
    - Authentication attempts and failures
    - Rate limit hits
    - Validation errors
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Generate correlation ID for request tracing
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        # Extract request details
        client_ip = get_remote_address(request)
        method = request.method
        path = request.url.path

        # Start timer
        start_time = time.time()

        # Create logger adapter with correlation ID
        log_adapter = logging.LoggerAdapter(
            logger,
            {'correlation_id': correlation_id}
        )

        # Log incoming request
        log_adapter.info(
            f"Request started: {method} {path} from {client_ip}"
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}s"

            # Log response
            status_code = response.status_code
            log_level = logging.INFO

            # Use WARNING for client errors, ERROR for server errors
            if 400 <= status_code < 500:
                log_level = logging.WARNING
            elif status_code >= 500:
                log_level = logging.ERROR

            log_adapter.log(
                log_level,
                f"Request completed: {method} {path} - Status: {status_code} - Time: {process_time:.3f}s"
            )

            # Log specific events based on status code
            if status_code == 401:
                log_adapter.warning(f"Authentication failed: {method} {path}")
            elif status_code == 403:
                log_adapter.warning(f"Authorization failed: {method} {path}")
            elif status_code == 429:
                log_adapter.warning(f"Rate limit exceeded: {method} {path} from {client_ip}")
            elif status_code == 422:
                log_adapter.warning(f"Validation error: {method} {path}")

            return response

        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Log exception
            log_adapter.error(
                f"Request failed: {method} {path} - Error: {str(e)} - Time: {process_time:.3f}s",
                exc_info=True
            )
            raise


def setup_rate_limit_headers(app: FastAPI) -> None:
    """
    Configure rate limit headers middleware.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(RateLimitHeadersMiddleware)
    logging.info("Rate limit headers middleware configured")


def setup_request_logging(app: FastAPI) -> None:
    """
    Configure request logging middleware.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(RequestLoggingMiddleware)
    logging.info("Request logging middleware configured")


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting Prometheus metrics on HTTP requests.

    Tracks:
    - Request count by method, endpoint, and status code
    - Request duration by method and endpoint
    - Requests in progress by method and endpoint
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        # Import metrics here to avoid circular imports
        from curlinator.api.metrics import (
            http_requests_total,
            http_request_duration_seconds,
            http_requests_in_progress
        )

        # Extract request details
        method = request.method
        # Normalize path to remove IDs and dynamic segments for better metric grouping
        path = self._normalize_path(request.url.path)

        # Skip metrics collection for the /metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        # Track requests in progress
        http_requests_in_progress.labels(method=method, endpoint=path).inc()

        # Start timer
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code

            # Record metrics
            duration = time.time() - start_time
            http_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)
            http_requests_total.labels(method=method, endpoint=path, status_code=status_code).inc()

            return response

        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            http_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)
            http_requests_total.labels(method=method, endpoint=path, status_code=500).inc()
            raise

        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=path).dec()

    def _normalize_path(self, path: str) -> str:
        """
        Normalize URL path for metric grouping.

        Replaces dynamic segments (UUIDs, collection names, etc.) with placeholders
        to avoid high cardinality in metrics.

        Examples:
            /api/v1/collections/my_collection -> /api/v1/collections/{name}
            /api/v1/sessions/550e8400-e29b-41d4-a716-446655440000 -> /api/v1/sessions/{id}
        """
        import re

        # Replace UUIDs with {id}
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path,
            flags=re.IGNORECASE
        )

        # Replace collection names (after /collections/)
        path = re.sub(r'/collections/[^/]+(?=/|$)', '/collections/{name}', path)

        # Replace email addresses (after /shares/)
        path = re.sub(r'/shares/[^/]+@[^/]+', '/shares/{email}', path)

        return path


def setup_metrics(app: FastAPI) -> None:
    """
    Configure metrics collection middleware.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(MetricsMiddleware)
    logging.info("Metrics middleware configured")
