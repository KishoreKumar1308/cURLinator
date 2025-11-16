"""
Prometheus metrics endpoint.

Exposes application metrics in Prometheus format for scraping.
"""

import logging
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(tags=["metrics"])
logger = logging.getLogger(__name__)


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns application metrics in Prometheus text format for scraping.
    This endpoint does not require authentication to allow Prometheus to scrape it.
    
    Returns:
        Response: Metrics in Prometheus text format
    """
    logger.debug("Metrics endpoint called")
    
    # Generate Prometheus metrics in text format
    metrics_data = generate_latest()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

