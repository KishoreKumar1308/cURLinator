"""
Prometheus metrics endpoint.

Exposes application metrics in Prometheus format for scraping.
"""

import os
import logging
from fastapi import APIRouter, Response, Header, HTTPException, status
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(tags=["metrics"])
logger = logging.getLogger(__name__)

# Metrics token for Prometheus scraping (set in environment)
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")


@router.get("/metrics")
async def metrics(authorization: str = Header(None)):
    """
    Prometheus metrics endpoint.

    Returns application metrics in Prometheus text format for scraping.
    Requires authentication via Bearer token or metrics token.

    Authentication options:
    1. Bearer token (JWT) - For authenticated users
    2. Metrics token - For Prometheus scraping (set METRICS_TOKEN env var)

    Args:
        authorization: Authorization header (Bearer token or metrics token)

    Returns:
        Response: Metrics in Prometheus text format

    Raises:
        HTTPException: If authentication fails
    """
    logger.debug("Metrics endpoint called")

    # Check if authorization header is provided
    if not authorization:
        logger.warning("Metrics endpoint accessed without authorization")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide Bearer token or metrics token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if it's a metrics token (for Prometheus)
    if METRICS_TOKEN and authorization == f"Bearer {METRICS_TOKEN}":
        logger.debug("Metrics endpoint accessed with valid metrics token")
    # Check if it's a Bearer token (JWT) - we'll validate it's a valid token
    elif authorization.startswith("Bearer "):
        # For now, just check that it's a Bearer token
        # In Phase 2, we'll add admin role check
        logger.debug("Metrics endpoint accessed with Bearer token")
    else:
        logger.warning("Metrics endpoint accessed with invalid authorization")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate Prometheus metrics in text format
    metrics_data = generate_latest()

    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

