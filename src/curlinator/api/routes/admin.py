"""
Admin endpoints for user management and system administration.

These endpoints require admin role and provide functionality for:
- User management (list, view, delete, activate/deactivate)
- Password reset (admin can reset any user's password)
- System administration
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, EmailStr, Field

from curlinator.api.database import get_db
from curlinator.api.db.models import User, DocumentationCollection, ChatSession
from curlinator.api.auth import get_admin_user, get_password_hash
from curlinator.api.middleware import limiter
from curlinator.api.error_codes import (
    create_error_response,
    DATABASE_QUERY_FAILED,
    RESOURCE_NOT_FOUND,
)
from curlinator.api.metrics import db_queries_total, db_errors_total

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
logger = logging.getLogger(__name__)


# Pydantic models for admin endpoints
class AdminUserResponse(BaseModel):
    """Admin view of user information."""
    
    id: str
    email: str
    is_active: bool
    is_anonymous: bool
    role: str
    created_at: str
    updated_at: Optional[str] = None
    collections_count: int = 0
    sessions_count: int = 0
    
    model_config = {
        "from_attributes": True
    }


class AdminUserListResponse(BaseModel):
    """Response for listing users."""
    
    id: str
    email: str
    is_active: bool
    role: str
    created_at: str
    collections_count: int = 0
    
    model_config = {
        "from_attributes": True
    }


class AdminUpdateUserStatusRequest(BaseModel):
    """Request to update user active status."""
    
    is_active: bool = Field(..., description="Whether the user should be active")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"is_active": False}
            ]
        }
    }


class AdminResetPasswordRequest(BaseModel):
    """Request to reset user password (admin only)."""
    
    new_password: str = Field(
        ...,
        min_length=8,
        description="New password (minimum 8 characters)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"new_password": "NewSecurePass123"}
            ]
        }
    }


@router.get("/users", response_model=List[AdminUserListResponse])
@limiter.limit("60/minute")
async def list_users(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    List all users (admin only).
    
    Returns a list of all users with basic information and counts.
    
    Args:
        request: Starlette Request object (for rate limiting)
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (default: 100, max: 100)
        admin_user: Current admin user
        db: Database session
        
    Returns:
        List of AdminUserListResponse objects
        
    Raises:
        HTTPException: If database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})
    
    log_adapter.info(f"Admin {admin_user.email} listing users (skip={skip}, limit={limit})")
    
    try:
        # Limit max results
        limit = min(limit, 100)
        
        # Query users
        users = db.query(User).offset(skip).limit(limit).all()
        
        # Build response with counts
        response = []
        for user in users:
            collections_count = db.query(DocumentationCollection).filter(
                DocumentationCollection.owner_id == user.id
            ).count()
            
            response.append(AdminUserListResponse(
                id=user.id,
                email=user.email,
                is_active=user.is_active,
                role=user.role,
                created_at=user.created_at.isoformat() if user.created_at else None,
                collections_count=collections_count
            ))
        
        db_queries_total.labels(operation="select", table="users", status="success").inc()
        log_adapter.info(f"Admin {admin_user.email} retrieved {len(response)} users")
        
        return response
        
    except SQLAlchemyError as e:
        log_adapter.error(f"Database error listing users: {str(e)}")
        db_errors_total.labels(operation="select", table="users", error_type="sqlalchemy").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="Failed to retrieve users due to database error.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )


@router.get("/users/{user_id}", response_model=AdminUserResponse)
@limiter.limit("60/minute")
async def get_user(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed user information (admin only).
    
    Args:
        request: Starlette Request object (for rate limiting)
        user_id: User ID to retrieve
        admin_user: Current admin user
        db: Database session
        
    Returns:
        AdminUserResponse with detailed user information
        
    Raises:
        HTTPException: If user not found or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})
    
    log_adapter.info(f"Admin {admin_user.email} retrieving user {user_id}")
    
    try:
        # Query user
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            log_adapter.warning(f"User {user_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_code=RESOURCE_NOT_FOUND,
                    message=f"User with ID '{user_id}' not found.",
                    suggestion="Please verify the user ID and try again."
                )
            )
        
        # Get counts
        collections_count = db.query(DocumentationCollection).filter(
            DocumentationCollection.owner_id == user.id
        ).count()
        
        sessions_count = db.query(ChatSession).filter(
            ChatSession.user_id == user.id
        ).count()
        
        db_queries_total.labels(operation="select", table="users", status="success").inc()
        log_adapter.info(f"Admin {admin_user.email} retrieved user {user.email}")
        
        return AdminUserResponse(
            id=user.id,
            email=user.email,
            is_active=user.is_active,
            is_anonymous=user.is_anonymous,
            role=user.role,
            created_at=user.created_at.isoformat() if user.created_at else None,
            updated_at=user.updated_at.isoformat() if user.updated_at else None,
            collections_count=collections_count,
            sessions_count=sessions_count
        )
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        log_adapter.error(f"Database error retrieving user: {str(e)}")
        db_errors_total.labels(operation="select", table="users", error_type="sqlalchemy").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="Failed to retrieve user due to database error.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )


@router.patch("/users/{user_id}/status")
@limiter.limit("30/minute")
async def update_user_status(
    request: Request,
    user_id: str,
    body: AdminUpdateUserStatusRequest,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update user active status (admin only).

    Allows admin to activate or deactivate user accounts.

    Args:
        request: Starlette Request object (for rate limiting)
        user_id: User ID to update
        body: AdminUpdateUserStatusRequest with is_active flag
        admin_user: Current admin user
        db: Database session

    Returns:
        Success message with updated user info

    Raises:
        HTTPException: If user not found or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Admin {admin_user.email} updating status for user {user_id} to {body.is_active}")

    try:
        # Query user
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            log_adapter.warning(f"User {user_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_code=RESOURCE_NOT_FOUND,
                    message=f"User with ID '{user_id}' not found.",
                    suggestion="Please verify the user ID and try again."
                )
            )

        # Prevent admin from deactivating themselves
        if user.id == admin_user.id and not body.is_active:
            log_adapter.warning(f"Admin {admin_user.email} attempted to deactivate themselves")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    error_code="CANNOT_DEACTIVATE_SELF",
                    message="You cannot deactivate your own account.",
                    suggestion="Please ask another admin to deactivate your account if needed."
                )
            )

        # Update status
        old_status = user.is_active
        user.is_active = body.is_active
        db.commit()

        db_queries_total.labels(operation="update", table="users", status="success").inc()
        log_adapter.info(f"Admin {admin_user.email} updated user {user.email} status from {old_status} to {body.is_active}")

        return {
            "message": f"User {'activated' if body.is_active else 'deactivated'} successfully",
            "user_id": user.id,
            "email": user.email,
            "is_active": user.is_active
        }

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        log_adapter.error(f"Database error updating user status: {str(e)}")
        db.rollback()
        db_errors_total.labels(operation="update", table="users", error_type="sqlalchemy").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="Failed to update user status due to database error.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )


@router.patch("/users/{user_id}/password")
@limiter.limit("10/minute")
async def admin_reset_password(
    request: Request,
    user_id: str,
    body: AdminResetPasswordRequest,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Reset user password (admin only).

    Allows admin to reset any user's password without requiring the current password.
    This is useful for helping users who have forgotten their passwords.

    Args:
        request: Starlette Request object (for rate limiting)
        user_id: User ID whose password to reset
        body: AdminResetPasswordRequest with new password
        admin_user: Current admin user
        db: Database session

    Returns:
        Success message

    Raises:
        HTTPException: If user not found or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Admin {admin_user.email} resetting password for user {user_id}")

    try:
        # Query user
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            log_adapter.warning(f"User {user_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_code=RESOURCE_NOT_FOUND,
                    message=f"User with ID '{user_id}' not found.",
                    suggestion="Please verify the user ID and try again."
                )
            )

        # Update password
        user.hashed_password = get_password_hash(body.new_password)
        db.commit()

        db_queries_total.labels(operation="update", table="users", status="success").inc()
        log_adapter.info(f"Admin {admin_user.email} reset password for user {user.email}")

        return {
            "message": "Password reset successfully",
            "user_id": user.id,
            "email": user.email
        }

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        log_adapter.error(f"Database error resetting password: {str(e)}")
        db.rollback()
        db_errors_total.labels(operation="update", table="users", error_type="sqlalchemy").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="Failed to reset password due to database error.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )


@router.delete("/users/{user_id}", status_code=204)
@limiter.limit("10/minute")
async def delete_user(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete user account (admin only).

    Permanently deletes a user and all associated data:
    - User record
    - All owned collections (database + Chroma vector store)
    - All chat sessions and messages
    - All collection shares

    This action cannot be undone.

    Args:
        request: Starlette Request object (for rate limiting)
        user_id: User ID to delete
        admin_user: Current admin user
        db: Database session

    Returns:
        204 No Content on success

    Raises:
        HTTPException: If user not found or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Admin {admin_user.email} deleting user {user_id}")

    try:
        # Query user
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            log_adapter.warning(f"User {user_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_code=RESOURCE_NOT_FOUND,
                    message=f"User with ID '{user_id}' not found.",
                    suggestion="Please verify the user ID and try again."
                )
            )

        # Prevent admin from deleting themselves
        if user.id == admin_user.id:
            log_adapter.warning(f"Admin {admin_user.email} attempted to delete themselves")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    error_code="CANNOT_DELETE_SELF",
                    message="You cannot delete your own account.",
                    suggestion="Please ask another admin to delete your account if needed."
                )
            )

        # Delete user's collections from Chroma vector store
        from curlinator.api.chroma_client import chroma_client

        collections = db.query(DocumentationCollection).filter(
            DocumentationCollection.owner_id == user.id
        ).all()

        for collection in collections:
            try:
                chroma_client.delete_collection(name=collection.name)
                log_adapter.info(f"Deleted Chroma collection: {collection.name}")
            except Exception as e:
                log_adapter.warning(f"Failed to delete Chroma collection {collection.name}: {str(e)}")

        # Delete user from database (cascade will handle related records)
        user_email = user.email
        db.delete(user)
        db.commit()

        db_queries_total.labels(operation="delete", table="users", status="success").inc()
        log_adapter.info(f"Admin {admin_user.email} deleted user {user_email} (ID: {user_id})")

        return None  # 204 No Content

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        log_adapter.error(f"Database error deleting user: {str(e)}")
        db.rollback()
        db_errors_total.labels(operation="delete", table="users", error_type="sqlalchemy").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="Failed to delete user due to database error.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )

