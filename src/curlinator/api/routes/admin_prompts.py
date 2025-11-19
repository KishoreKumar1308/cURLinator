"""
Admin endpoints for system prompt customization.

These endpoints require admin role and provide functionality for:
- Setting system-wide default prompt
- Setting per-user custom prompts for A/B testing
- Viewing current prompts configuration
- Resetting user prompts to system default
"""

import logging
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from curlinator.api.database import get_db
from curlinator.api.db.models import User, UserSettings, SystemConfig
from curlinator.api.auth import get_admin_user
from curlinator.api.middleware import limiter
from curlinator.api.schemas.prompts import (
    SystemPromptUpdate,
    UserPromptUpdate,
    SystemPromptResponse,
    UserPromptInfo,
    PromptsOverviewResponse,
    PromptUpdateSuccessResponse,
    UserPromptDeleteResponse,
)
from curlinator.agents.chat_agent import ChatAgent

router = APIRouter(prefix="/api/v1/admin", tags=["admin", "prompts"])
logger = logging.getLogger(__name__)


@router.patch("/system-prompt", response_model=PromptUpdateSuccessResponse)
@limiter.limit("30/minute")
async def update_system_prompt(
    request: Request,
    body: SystemPromptUpdate,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Set system-wide default prompt (admin only).

    This prompt will be used for all users who don't have a custom prompt.

    Args:
        request: Starlette Request object (for rate limiting)
        body: SystemPromptUpdate with new prompt text
        admin_user: Current admin user
        db: Database session

    Returns:
        PromptUpdateSuccessResponse with success message and preview
    """
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    try:
        log_adapter.info(f"Admin {admin_user.email} updating system-wide prompt")

        # Check if system_prompt config exists
        system_prompt_config = db.query(SystemConfig).filter(
            SystemConfig.config_key == "system_prompt"
        ).first()

        if system_prompt_config:
            # Update existing
            system_prompt_config.config_value = body.prompt
            system_prompt_config.description = body.description
            system_prompt_config.updated_at = datetime.now(timezone.utc)
            system_prompt_config.updated_by_user_id = admin_user.id
            log_adapter.info("Updated existing system prompt configuration")
        else:
            # Create new
            system_prompt_config = SystemConfig(
                config_key="system_prompt",
                config_value=body.prompt,
                description=body.description,
                updated_at=datetime.now(timezone.utc),
                updated_by_user_id=admin_user.id
            )
            db.add(system_prompt_config)
            log_adapter.info("Created new system prompt configuration")

        db.commit()
        db.refresh(system_prompt_config)

        # Create preview (first 100 chars)
        prompt_preview = body.prompt[:100] + "..." if len(body.prompt) > 100 else body.prompt

        log_adapter.info(f"System prompt updated successfully by {admin_user.email}")

        return PromptUpdateSuccessResponse(
            message="System-wide prompt updated successfully",
            prompt_preview=prompt_preview,
            updated_at=system_prompt_config.updated_at,
            updated_by=admin_user.email
        )

    except SQLAlchemyError as e:
        db.rollback()
        log_adapter.error(f"Database error updating system prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update system prompt due to database error"
        )


@router.patch("/users/{user_id}/prompt", response_model=PromptUpdateSuccessResponse)
@limiter.limit("30/minute")
async def update_user_prompt(
    request: Request,
    user_id: str,
    body: UserPromptUpdate,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Set per-user custom prompt for A/B testing (admin only).

    This allows admins to assign different prompt variants to specific users
    for experimentation and A/B testing.

    Args:
        request: Starlette Request object (for rate limiting)
        user_id: User ID to assign custom prompt to
        body: UserPromptUpdate with custom prompt text and variant name
        admin_user: Current admin user
        db: Database session

    Returns:
        PromptUpdateSuccessResponse with success message and preview
    """
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    try:
        log_adapter.info(f"Admin {admin_user.email} updating custom prompt for user {user_id}")

        # Check if user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            log_adapter.warning(f"User {user_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found"
            )

        # Get or create user settings
        user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
        if not user_settings:
            user_settings = UserSettings(
                user_id=user_id,
                preferred_embedding_provider="local",
                default_max_pages=50,
                default_max_depth=3,
                free_messages_used=0,
                free_messages_limit=10,
                last_message_reset_date=datetime.now(timezone.utc),
            )
            db.add(user_settings)
            log_adapter.info(f"Created new settings for user {user_id}")

        # Update custom prompt
        user_settings.custom_system_prompt = body.prompt
        user_settings.prompt_variant_name = body.variant_name
        user_settings.prompt_updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(user_settings)

        # Create preview (first 100 chars)
        prompt_preview = body.prompt[:100] + "..." if len(body.prompt) > 100 else body.prompt

        log_adapter.info(
            f"Custom prompt updated for user {user.email} "
            f"(variant: {body.variant_name or 'none'}) by {admin_user.email}"
        )

        return PromptUpdateSuccessResponse(
            message=f"Custom prompt set for user {user.email}",
            prompt_preview=prompt_preview,
            updated_at=user_settings.prompt_updated_at,
            updated_by=admin_user.email
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        log_adapter.error(f"Database error updating user prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user prompt due to database error"
        )


@router.get("/prompts", response_model=PromptsOverviewResponse)
@limiter.limit("60/minute")
async def get_prompts_overview(
    request: Request,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    View current system-wide prompt and list users with custom prompts (admin only).

    Provides an overview of the current prompt configuration including:
    - System-wide default prompt
    - List of users with custom prompts for A/B testing

    Args:
        request: Starlette Request object (for rate limiting)
        admin_user: Current admin user
        db: Database session

    Returns:
        PromptsOverviewResponse with system prompt and users with custom prompts
    """
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    try:
        log_adapter.info(f"Admin {admin_user.email} viewing prompts overview")

        # Get system-wide prompt
        system_prompt_config = db.query(SystemConfig).filter(
            SystemConfig.config_key == "system_prompt"
        ).first()

        if system_prompt_config:
            # Get admin who updated it
            updated_by_user = db.query(User).filter(
                User.id == system_prompt_config.updated_by_user_id
            ).first()

            system_prompt = SystemPromptResponse(
                prompt=system_prompt_config.config_value,
                description=system_prompt_config.description,
                updated_at=system_prompt_config.updated_at,
                updated_by_email=updated_by_user.email if updated_by_user else None,
                is_default=False
            )
        else:
            # Use hardcoded default from ChatAgent
            default_prompt = ChatAgent._get_default_system_prompt(None)
            system_prompt = SystemPromptResponse(
                prompt=default_prompt,
                description="Hardcoded default prompt from ChatAgent",
                updated_at=None,
                updated_by_email=None,
                is_default=True
            )

        # Get users with custom prompts
        users_with_custom_prompts = db.query(UserSettings, User).join(
            User, UserSettings.user_id == User.id
        ).filter(
            UserSettings.custom_system_prompt.isnot(None)
        ).all()

        user_prompt_infos = []
        for user_settings, user in users_with_custom_prompts:
            prompt_preview = (
                user_settings.custom_system_prompt[:100] + "..."
                if len(user_settings.custom_system_prompt) > 100
                else user_settings.custom_system_prompt
            )
            user_prompt_infos.append(
                UserPromptInfo(
                    user_id=user.id,
                    user_email=user.email,
                    variant_name=user_settings.prompt_variant_name,
                    prompt_preview=prompt_preview,
                    updated_at=user_settings.prompt_updated_at
                )
            )

        log_adapter.info(
            f"Prompts overview retrieved: {len(user_prompt_infos)} users with custom prompts"
        )

        return PromptsOverviewResponse(
            system_prompt=system_prompt,
            users_with_custom_prompts=user_prompt_infos,
            total_users_with_custom_prompts=len(user_prompt_infos)
        )

    except SQLAlchemyError as e:
        log_adapter.error(f"Database error retrieving prompts overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prompts overview due to database error"
        )


@router.delete("/users/{user_id}/prompt", response_model=UserPromptDeleteResponse)
@limiter.limit("30/minute")
async def delete_user_prompt(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Reset user prompt to system default (admin only).

    Removes the custom prompt for a specific user, causing them to use
    the system-wide default prompt instead.

    Args:
        request: Starlette Request object (for rate limiting)
        user_id: User ID to reset prompt for
        admin_user: Current admin user
        db: Database session

    Returns:
        UserPromptDeleteResponse with success message
    """
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    try:
        log_adapter.info(f"Admin {admin_user.email} resetting custom prompt for user {user_id}")

        # Check if user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            log_adapter.warning(f"User {user_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found"
            )

        # Get user settings
        user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
        if not user_settings or not user_settings.custom_system_prompt:
            log_adapter.warning(f"User {user.email} does not have a custom prompt")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user.email} does not have a custom prompt to delete"
            )

        # Reset custom prompt fields
        user_settings.custom_system_prompt = None
        user_settings.prompt_variant_name = None
        user_settings.prompt_updated_at = None

        db.commit()

        log_adapter.info(f"Custom prompt reset for user {user.email} by {admin_user.email}")

        return UserPromptDeleteResponse(
            message=f"Custom prompt removed for user {user.email}. User will now use system default.",
            user_id=user.id,
            user_email=user.email
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        log_adapter.error(f"Database error deleting user prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user prompt due to database error"
        )

