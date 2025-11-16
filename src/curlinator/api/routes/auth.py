"""
Authentication endpoints for user registration and login.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from curlinator.api.database import get_db
from curlinator.api.db.models import User
from curlinator.api.models.auth import RegisterRequest, LoginRequest, AuthResponse
from curlinator.api.auth import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_user,
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
async def register(http_request: Request, request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user.

    Creates a new user account with email and password.
    Returns a JWT access token for immediate authentication.

    Args:
        request: RegisterRequest with email and password
        db: Database session

    Returns:
        AuthResponse with access_token and user info

    Raises:
        HTTPException: If email already registered or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(http_request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Registration attempt for email: {request.email}")

    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            log_adapter.warning(f"Registration failed: Email already exists - {request.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    error_code=RESOURCE_ALREADY_EXISTS,
                    message=f"An account with email '{request.email}' already exists.",
                    suggestion="Please use a different email address or try logging in."
                )
            )

        # Create new user
        user = User(
            email=request.email,
            hashed_password=get_password_hash(request.password),
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
async def login(http_request: Request, request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login an existing user.

    Authenticates user with email and password.
    Returns a JWT access token.

    Args:
        request: LoginRequest with email and password
        db: Database session

    Returns:
        AuthResponse with access_token and user info

    Raises:
        HTTPException: If credentials are invalid or database error occurs
    """
    # Get correlation ID from request state
    correlation_id = getattr(http_request.state, 'correlation_id', 'N/A')
    log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

    log_adapter.info(f"Login attempt for email: {request.email}")
    try:
        user = authenticate_user(db, request.email, request.password)

        if not user:
            log_adapter.warning(f"Login failed: Invalid credentials for {request.email}")
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
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }

