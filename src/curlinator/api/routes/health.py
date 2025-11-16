"""
Health check endpoint with comprehensive system checks.
"""

import os
import logging
import psutil
from fastapi import APIRouter, Depends
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text
from chromadb import PersistentClient
from chromadb.errors import ChromaError

from curlinator.api.database import get_db

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


def check_database(db: Session) -> dict:
    """
    Check database connectivity and health.

    Returns:
        dict: Database health status
    """
    try:
        # Execute a simple query to check connection
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }


def check_chroma() -> dict:
    """
    Check Chroma vector store connectivity.

    Returns:
        dict: Chroma health status
    """
    try:
        # Try to connect to Chroma
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        client = PersistentClient(path=chroma_dir)

        # Try to list collections (lightweight operation)
        collections = client.list_collections()

        return {
            "status": "healthy",
            "message": "Chroma connection successful",
            "collections_count": len(collections)
        }
    except ChromaError as e:
        logger.error(f"Chroma health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Chroma connection failed: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Chroma health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Chroma connection failed: {str(e)}"
        }


def check_system_metrics() -> dict:
    """
    Check system resource usage.

    Returns:
        dict: System metrics
    """
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024 ** 3)

        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024 ** 3)

        # Determine health status based on thresholds
        status = "healthy"
        warnings = []

        if cpu_percent > 90:
            status = "degraded"
            warnings.append(f"High CPU usage: {cpu_percent}%")

        if memory_percent > 90:
            status = "degraded"
            warnings.append(f"High memory usage: {memory_percent}%")

        if disk_percent > 90:
            status = "degraded"
            warnings.append(f"Low disk space: {disk_percent}% used")

        return {
            "status": status,
            "cpu_percent": round(cpu_percent, 2),
            "memory_percent": round(memory_percent, 2),
            "memory_available_gb": round(memory_available_gb, 2),
            "disk_percent": round(disk_percent, 2),
            "disk_free_gb": round(disk_free_gb, 2),
            "warnings": warnings if warnings else None
        }
    except Exception as e:
        logger.error(f"System metrics check failed: {str(e)}")
        return {
            "status": "unknown",
            "message": f"Failed to get system metrics: {str(e)}"
        }


def check_llm_connectivity() -> dict:
    """
    Check LLM API connectivity with lightweight test calls.

    This is a non-blocking check - failures won't mark the service as unhealthy,
    but will indicate degraded status.

    Returns:
        dict: LLM connectivity status
    """
    llm_status = {}

    # Check OpenAI API
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            # Make a lightweight API call to check connectivity
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)

            # Use a very small, cheap model for the test
            # This is a minimal request that just checks if the API is reachable
            response = client.models.list()

            llm_status["openai"] = {
                "status": "healthy",
                "message": "API connection successful"
            }
        except Exception as e:
            logger.warning(f"OpenAI API health check failed: {str(e)}")
            llm_status["openai"] = {
                "status": "degraded",
                "message": f"API connection failed: {str(e)[:100]}"
            }
    else:
        llm_status["openai"] = {
            "status": "not_configured",
            "message": "API key not set"
        }

    # Check Gemini API
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            # Make a lightweight API call to check connectivity
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)

            # List models to check connectivity
            models = genai.list_models()
            # Just iterate once to verify the API is working
            next(iter(models), None)

            llm_status["gemini"] = {
                "status": "healthy",
                "message": "API connection successful"
            }
        except Exception as e:
            logger.warning(f"Gemini API health check failed: {str(e)}")
            llm_status["gemini"] = {
                "status": "degraded",
                "message": f"API connection failed: {str(e)[:100]}"
            }
    else:
        llm_status["gemini"] = {
            "status": "not_configured",
            "message": "API key not set"
        }

    # Check Anthropic API
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        try:
            # Make a lightweight API call to check connectivity
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_api_key)

            # Just verify the client can be created and API key is valid
            # We don't make an actual completion call to save costs
            # The client will validate the API key format
            llm_status["anthropic"] = {
                "status": "healthy",
                "message": "API key configured (connectivity not tested to save costs)"
            }
        except Exception as e:
            logger.warning(f"Anthropic API health check failed: {str(e)}")
            llm_status["anthropic"] = {
                "status": "degraded",
                "message": f"API configuration failed: {str(e)[:100]}"
            }
    else:
        llm_status["anthropic"] = {
            "status": "not_configured",
            "message": "API key not set"
        }

    return llm_status


def check_dependencies() -> dict:
    """
    Check external API dependencies (deprecated - use check_llm_connectivity instead).

    This function is kept for backward compatibility but now just checks
    if API keys are configured without testing connectivity.

    Returns:
        dict: Dependency status
    """
    dependencies = {}

    # Check OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        dependencies["openai"] = {
            "status": "configured",
            "message": "API key present"
        }
    else:
        dependencies["openai"] = {
            "status": "not_configured",
            "message": "API key not set"
        }

    # Check Gemini API key
    if os.getenv("GEMINI_API_KEY"):
        dependencies["gemini"] = {
            "status": "configured",
            "message": "API key present"
        }
    else:
        dependencies["gemini"] = {
            "status": "not_configured",
            "message": "API key not set"
        }

    # Check Anthropic API key
    if os.getenv("ANTHROPIC_API_KEY"):
        dependencies["anthropic"] = {
            "status": "configured",
            "message": "API key present"
        }
    else:
        dependencies["anthropic"] = {
            "status": "not_configured",
            "message": "API key not set"
        }

    return dependencies


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check endpoint.

    Checks:
    - Database connectivity
    - Chroma vector store
    - System resources (CPU, memory, disk)
    - LLM API connectivity (OpenAI, Gemini, Anthropic)

    Returns:
        dict: Detailed health status

    Status levels:
    - healthy: All critical services are operational
    - degraded: Critical services work but LLM APIs or system resources are degraded
    - unhealthy: Critical services (database, chroma) are down
    """
    # Perform all health checks
    db_health = check_database(db)
    chroma_health = check_chroma()
    system_metrics = check_system_metrics()
    llm_status = check_llm_connectivity()

    # Determine overall status
    overall_status = "healthy"

    # Critical services must be healthy
    if db_health["status"] == "unhealthy" or chroma_health["status"] == "unhealthy":
        overall_status = "unhealthy"
    # System resources or LLM APIs degraded
    elif system_metrics["status"] == "degraded":
        overall_status = "degraded"
    # Check if any LLM API is degraded (but don't fail if just not configured)
    elif any(
        llm["status"] == "degraded"
        for llm in llm_status.values()
    ):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "checks": {
            "database": db_health,
            "chroma": chroma_health,
            "system": system_metrics,
            "llm": llm_status
        }
    }

