"""
User settings API endpoints.
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from curlinator.api.database import get_db
from curlinator.api.auth import get_current_user
from curlinator.api.db.models import User, UserSettings
from curlinator.api.schemas.settings import (
    UserSettingsResponse,
    UserSettingsUpdate,
    ValidateAPIKeyRequest,
    ValidateAPIKeyResponse,
)
from curlinator.api.utils.encryption import encrypt_api_key
from curlinator.api.utils.llm_factory import validate_api_key
from curlinator.api.middleware import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


def get_or_create_user_settings(user_id: str, db: Session) -> UserSettings:
    """Get or create user settings."""
    settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
    
    if not settings:
        # Create default settings
        settings = UserSettings(
            user_id=user_id,
            preferred_embedding_provider="local",
            default_max_pages=50,
            default_max_depth=3,
            free_messages_used=0,
            free_messages_limit=10,
            last_message_reset_date=datetime.now(timezone.utc),
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)
    
    return settings


@router.get("", response_model=UserSettingsResponse)
async def get_user_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get current user's settings.
    
    Returns user preferences for LLM provider, embedding provider, crawl defaults,
    and usage tracking information.
    """
    settings = get_or_create_user_settings(current_user.id, db)
    
    # Calculate free messages remaining
    free_messages_remaining = max(0, settings.free_messages_limit - settings.free_messages_used)
    
    return UserSettingsResponse(
        preferred_llm_provider=settings.preferred_llm_provider,
        preferred_llm_model=settings.preferred_llm_model,
        has_openai_key=bool(settings.user_openai_api_key_encrypted),
        has_anthropic_key=bool(settings.user_anthropic_api_key_encrypted),
        has_gemini_key=bool(settings.user_gemini_api_key_encrypted),
        preferred_embedding_provider=settings.preferred_embedding_provider,
        default_max_pages=settings.default_max_pages,
        default_max_depth=settings.default_max_depth,
        free_messages_used=settings.free_messages_used,
        free_messages_limit=settings.free_messages_limit,
        free_messages_remaining=free_messages_remaining,
    )


@router.patch("", response_model=UserSettingsResponse)
async def update_user_settings(
    body: UserSettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update user settings.
    
    Allows users to update their LLM preferences, API keys, embedding preferences,
    and crawl defaults. API keys are encrypted before storage.
    """
    settings = get_or_create_user_settings(current_user.id, db)
    
    # Update LLM preferences
    if body.preferred_llm_provider is not None:
        settings.preferred_llm_provider = body.preferred_llm_provider
    
    if body.preferred_llm_model is not None:
        settings.preferred_llm_model = body.preferred_llm_model
    
    # Update API keys (encrypt before storing)
    api_key_updated = False
    
    if body.openai_api_key is not None:
        if body.openai_api_key == "":
            settings.user_openai_api_key_encrypted = None
        else:
            settings.user_openai_api_key_encrypted = encrypt_api_key(body.openai_api_key)
        api_key_updated = True
    
    if body.anthropic_api_key is not None:
        if body.anthropic_api_key == "":
            settings.user_anthropic_api_key_encrypted = None
        else:
            settings.user_anthropic_api_key_encrypted = encrypt_api_key(body.anthropic_api_key)
        api_key_updated = True
    
    if body.gemini_api_key is not None:
        if body.gemini_api_key == "":
            settings.user_gemini_api_key_encrypted = None
        else:
            settings.user_gemini_api_key_encrypted = encrypt_api_key(body.gemini_api_key)
        api_key_updated = True
    
    if api_key_updated:
        settings.api_key_last_updated_at = datetime.now(timezone.utc)
    
    # Update embedding preferences
    if body.preferred_embedding_provider is not None:
        settings.preferred_embedding_provider = body.preferred_embedding_provider
    
    # Update crawl defaults
    if body.default_max_pages is not None:
        settings.default_max_pages = body.default_max_pages
    
    if body.default_max_depth is not None:
        settings.default_max_depth = body.default_max_depth
    
    settings.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(settings)
    
    # Return updated settings
    free_messages_remaining = max(0, settings.free_messages_limit - settings.free_messages_used)
    
    return UserSettingsResponse(
        preferred_llm_provider=settings.preferred_llm_provider,
        preferred_llm_model=settings.preferred_llm_model,
        has_openai_key=bool(settings.user_openai_api_key_encrypted),
        has_anthropic_key=bool(settings.user_anthropic_api_key_encrypted),
        has_gemini_key=bool(settings.user_gemini_api_key_encrypted),
        preferred_embedding_provider=settings.preferred_embedding_provider,
        default_max_pages=settings.default_max_pages,
        default_max_depth=settings.default_max_depth,
        free_messages_used=settings.free_messages_used,
        free_messages_limit=settings.free_messages_limit,
        free_messages_remaining=free_messages_remaining,
    )


@router.post("/validate-api-key", response_model=ValidateAPIKeyResponse)
@limiter.limit("5/minute")
async def validate_user_api_key(
    request: Request,
    body: ValidateAPIKeyRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Validate an API key before saving.
    
    Makes a test API call to verify the key is valid.
    Rate limited to 5 requests per minute to prevent abuse.
    """
    is_valid, error = validate_api_key(body.provider, body.api_key)
    
    if is_valid:
        return ValidateAPIKeyResponse(
            valid=True,
            provider=body.provider,
        )
    else:
        return ValidateAPIKeyResponse(
            valid=False,
            provider=body.provider,
            error=error,
        )

