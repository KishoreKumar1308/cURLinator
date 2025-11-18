"""
Authentication endpoints for user registration and login.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from curlinator.api.database import get_db
from curlinator.api.db.models import User
from curlinator.api.models.auth import RegisterRequest, LoginRequest, AuthResponse, ChangePasswordRequest, DeleteAccountRequest
from curlinator.api.auth import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_user,
    verify_password,
)
from curlinator.api.middleware import limiter
from curlinator.api.error_codes import (
    create_error_response,
    RESOURCE_ALREADY_EXISTS,
    DATABASE_INTEGRITY_ERROR,
    DATABASE_QUERY_FAILED,
    AUTH_INVALID_CREDENTIALS,
    VALIDATION_INVALID_EMAIL
)
from curlinator.api.metrics import (
    auth_attempts_total,
    auth_tokens_created_total,
    db_queries_total,
    db_errors_total
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = logging.getLogger(__name__)


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def register(request: Request, body: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user.

    Creates a new user account with email and password.
    Returns a JWT access token for immediate authentication.

    Args:
        request: Starlette Request object (for rate limiting)
        body: RegisterRequest with email and password
        db: Database session

    Returns:
        AuthResponse with access_token and user info

    Raises:
        HTTPException: If email already registered or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Registration attempt for email: {body.email}")

    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == body.email).first()
        if existing_user:
            log_adapter.warning(f"Registration failed: Email already exists - {body.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    error_code=RESOURCE_ALREADY_EXISTS,
                    message=f"An account with email '{body.email}' already exists.",
                    suggestion="Please use a different email address or try logging in."
                )
            )

        # Create new user
        user = User(
            email=body.email,
            hashed_password=get_password_hash(body.password),
            is_active=True,
            is_anonymous=False,
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        # Track successful database insert
        db_queries_total.labels(operation="insert", table="users", status="success").inc()

        log_adapter.info(f"User registered successfully: {user.email} (ID: {user.id})")

        # Create access token
        access_token = create_access_token(data={"sub": user.id})

        # Track token creation and successful registration
        auth_tokens_created_total.inc()
        auth_attempts_total.labels(endpoint="register", status="success").inc()

        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "email": user.email,
                "is_active": user.is_active,
                "role": user.role,
            }
        )

    except HTTPException:
        # Track failed registration
        auth_attempts_total.labels(endpoint="register", status="failure").inc()
        # Re-raise HTTP exceptions (like email already registered)
        raise

    except IntegrityError as e:
        db.rollback()
        db_errors_total.labels(error_type="integrity").inc()
        auth_attempts_total.labels(endpoint="register", status="failure").inc()
        log_adapter.error(f"Database integrity error during registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=create_error_response(
                error_code=DATABASE_INTEGRITY_ERROR,
                message="Email address is already registered.",
                suggestion="Please use a different email address or try logging in."
            )
        )

    except SQLAlchemyError as e:
        db.rollback()
        db_errors_total.labels(error_type="query").inc()
        auth_attempts_total.labels(endpoint="register", status="failure").inc()
        log_adapter.error(f"Database error during registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="An error occurred while creating your account. Please try again later.",
                suggestion="If the problem persists, please contact support."
            )
        )

    except Exception as e:
        db.rollback()
        log_adapter.error(f"Unexpected error during registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="An unexpected error occurred. Please try again later.",
                suggestion="If the problem persists, please contact support."
            )
        )


@router.post("/login", response_model=AuthResponse)
@limiter.limit("10/minute")
async def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    """
    Login an existing user.

    Authenticates user with email and password.
    Returns a JWT access token.

    Args:
        request: Starlette Request object (for rate limiting)
        body: LoginRequest with email and password
        db: Database session

    Returns:
        AuthResponse with access_token and user info

    Raises:
        HTTPException: If credentials are invalid or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Login attempt for email: {body.email}")
    try:
        user = authenticate_user(db, body.email, body.password)

        if not user:
            log_adapter.warning(f"Login failed: Invalid credentials for {body.email}")
            auth_attempts_total.labels(endpoint="login", status="failure").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=create_error_response(
                    error_code=AUTH_INVALID_CREDENTIALS,
                    message="Invalid email or password.",
                    suggestion="Please check your credentials and try again."
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )

        log_adapter.info(f"User logged in successfully: {user.email} (ID: {user.id})")

        # Create access token
        access_token = create_access_token(data={"sub": user.id})

        # Track successful login
        auth_tokens_created_total.inc()
        auth_attempts_total.labels(endpoint="login", status="success").inc()
        db_queries_total.labels(operation="select", table="users", status="success").inc()

        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "email": user.email,
                "is_active": user.is_active,
                "role": user.role,
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like invalid credentials)
        raise

    except SQLAlchemyError as e:
        db_errors_total.labels(error_type="query").inc()
        auth_attempts_total.labels(endpoint="login", status="failure").inc()
        log_adapter.error(f"Database error during login: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="An error occurred while processing your login. Please try again later.",
                suggestion="If the problem persists, please contact support."
            )
        )

    except Exception as e:
        log_adapter.error(f"Unexpected error during login: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="An unexpected error occurred. Please try again later.",
                suggestion="If the problem persists, please contact support."
            )
        )


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.

    Requires authentication via JWT token.

    Args:
        current_user: Current authenticated user

    Returns:
        User information
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_active": current_user.is_active,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


@router.patch("/password")
@limiter.limit("10/minute")
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user password.

    Requires authentication via JWT token.
    User must provide current password for verification.

    Args:
        request: Starlette Request object (for rate limiting)
        body: ChangePasswordRequest with current and new password
        current_user: Current authenticated user
        db: Database session

    Returns:
        Success message

    Raises:
        HTTPException: If current password is incorrect or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Password change attempt for user: {current_user.email} (ID: {current_user.id})")

    try:
        # Verify current password
        if not verify_password(body.current_password, current_user.hashed_password):
            log_adapter.warning(f"Password change failed: Incorrect current password for {current_user.email}")
            auth_attempts_total.labels(endpoint="change_password", status="failure").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=create_error_response(
                    error_code=AUTH_INVALID_CREDENTIALS,
                    message="Current password is incorrect.",
                    suggestion="Please verify your current password and try again."
                )
            )

        # Check if new password is same as current password
        if verify_password(body.new_password, current_user.hashed_password):
            log_adapter.warning(f"Password change failed: New password same as current for {current_user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    error_code="PASSWORD_UNCHANGED",
                    message="New password must be different from current password.",
                    suggestion="Please choose a different password."
                )
            )

        # Update password
        current_user.hashed_password = get_password_hash(body.new_password)
        db.commit()

        # Track successful password change
        db_queries_total.labels(operation="update", table="users", status="success").inc()
        auth_attempts_total.labels(endpoint="change_password", status="success").inc()

        log_adapter.info(f"Password changed successfully for user: {current_user.email} (ID: {current_user.id})")

        return {
            "message": "Password changed successfully",
            "user_id": current_user.id
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except SQLAlchemyError as e:
        log_adapter.error(f"Database error during password change: {str(e)}")
        db.rollback()
        db_errors_total.labels(operation="update", table="users", error_type="sqlalchemy").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="Failed to update password due to database error.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )
    except Exception as e:
        log_adapter.error(f"Unexpected error during password change: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred while changing password.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )


@router.delete("/me", status_code=204)
@limiter.limit("5/minute")
async def delete_account(
    request: Request,
    body: DeleteAccountRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete user account (self-service).

    Permanently deletes the authenticated user's account and all associated data:
    - User record
    - All owned collections (database + Chroma vector store)
    - All chat sessions and messages
    - All collection shares

    This action cannot be undone. Requires password confirmation for security.

    Args:
        request: Starlette Request object (for rate limiting)
        body: DeleteAccountRequest with password confirmation
        current_user: Current authenticated user
        db: Database session

    Returns:
        204 No Content on success

    Raises:
        HTTPException: If password is incorrect or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Account deletion attempt for user: {current_user.email} (ID: {current_user.id})")

    try:
        # Verify password for confirmation
        if not verify_password(body.password, current_user.hashed_password):
            log_adapter.warning(f"Account deletion failed: Incorrect password for {current_user.email}")
            auth_attempts_total.labels(endpoint="delete_account", status="failure").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=create_error_response(
                    error_code=AUTH_INVALID_CREDENTIALS,
                    message="Password is incorrect.",
                    suggestion="Please verify your password and try again."
                )
            )

        # Import here to avoid circular dependency
        import chromadb
        from curlinator.api.db.models import DocumentationCollection
        from curlinator.config import get_settings

        # Delete user's collections from Chroma vector store
        collections = db.query(DocumentationCollection).filter(
            DocumentationCollection.owner_id == current_user.id
        ).all()

        if collections:
            try:
                settings = get_settings()
                chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)

                for collection in collections:
                    try:
                        chroma_client.delete_collection(name=collection.name)
                        log_adapter.info(f"Deleted Chroma collection: {collection.name}")
                    except Exception as e:
                        log_adapter.warning(f"Failed to delete Chroma collection {collection.name}: {str(e)}")
            except Exception as e:
                log_adapter.warning(f"Failed to connect to Chroma client: {str(e)}")

        # Delete user from database (cascade will handle related records)
        user_email = current_user.email
        user_id = current_user.id
        db.delete(current_user)
        db.commit()

        # Track successful deletion
        db_queries_total.labels(operation="delete", table="users", status="success").inc()
        auth_attempts_total.labels(endpoint="delete_account", status="success").inc()

        log_adapter.info(f"User account deleted successfully: {user_email} (ID: {user_id})")

        return None  # 204 No Content

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except SQLAlchemyError as e:
        log_adapter.error(f"Database error during account deletion: {str(e)}")
        db.rollback()
        db_errors_total.labels(operation="delete", table="users", error_type="sqlalchemy").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code=DATABASE_QUERY_FAILED,
                message="Failed to delete account due to database error.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )
    except Exception as e:
        log_adapter.error(f"Unexpected error during account deletion: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred while deleting account.",
                suggestion="Please try again later or contact support if the issue persists."
            )
        )

