"""
Prometheus metrics endpoint.

Exposes application metrics in Prometheus format for scraping.
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, Response, Header, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session

from curlinator.api.database import get_db
from curlinator.api.db.models import User
from curlinator.api.auth import get_admin_user

router = APIRouter(tags=["metrics"])
logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Metrics token for Prometheus scraping (set in environment)
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")


async def verify_metrics_access(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Verify access to metrics endpoint.

    Allows access via:
    1. Admin user with JWT token
    2. Metrics token (for Prometheus scraping)

    Args:
        credentials: HTTP Bearer credentials
        db: Database session

    Returns:
        User object if admin, None if metrics token

    Raises:
        HTTPException: If authentication fails
    """
    # Check if credentials provided
    if not credentials:
        logger.warning("Metrics endpoint accessed without authorization")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide admin JWT token or metrics token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Check if it's the metrics token (for Prometheus)
    if METRICS_TOKEN and token == METRICS_TOKEN:
        logger.debug("Metrics endpoint accessed with valid metrics token")
        return None  # No user for metrics token

    # Otherwise, validate as admin JWT token
    try:
        from curlinator.api.auth import get_current_user

        # Create a mock credentials object for get_current_user
        user = await get_current_user(credentials, db)

        # Check if user is admin
        if user.role != "admin":
            logger.warning(f"Non-admin user {user.email} attempted to access metrics endpoint")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to view metrics."
            )

        logger.debug(f"Metrics endpoint accessed by admin user: {user.email}")
        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating metrics access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/metrics")
async def metrics(user: Optional[User] = Depends(verify_metrics_access)):
    """
    Prometheus metrics endpoint.

    Returns application metrics in Prometheus text format for scraping.
    Requires admin authentication or metrics token.

    Authentication options:
    1. Admin JWT token - For admin users
    2. Metrics token - For Prometheus scraping (set METRICS_TOKEN env var)

    Args:
        user: Current admin user (or None if using metrics token)

    Returns:
        Response: Metrics in Prometheus text format

    Raises:
        HTTPException: If authentication fails
    """
    if user:
        logger.debug(f"Metrics endpoint accessed by admin user: {user.email}")
    else:
        logger.debug("Metrics endpoint accessed with metrics token")

    # Generate Prometheus metrics in text format
    metrics_data = generate_latest()

    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

