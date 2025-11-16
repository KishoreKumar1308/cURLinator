"""
Pydantic models for authentication endpoints.
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Dict


class RegisterRequest(BaseModel):
    """Request model for user registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ...,
        min_length=8,
        description="User password (minimum 8 characters, must contain uppercase, lowercase, and digit)"
    )

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Validate password strength requirements.

        Password must:
        - Be at least 8 characters long (enforced by Field min_length)
        - Contain at least one uppercase letter
        - Contain at least one lowercase letter
        - Contain at least one digit

        Args:
            v: Password string to validate

        Returns:
            The validated password

        Raises:
            ValueError: If password doesn't meet strength requirements
        """
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email": "user@example.com",
                    "password": "SecurePass123"
                }
            ]
        }
    }


class LoginRequest(BaseModel):
    """Request model for user login."""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email": "user@example.com",
                    "password": "securepassword123"
                }
            ]
        }
    }


class UserResponse(BaseModel):
    """User information response."""
    
    id: str
    email: str
    is_active: bool
    created_at: str
    
    model_config = {
        "from_attributes": True
    }


class AuthResponse(BaseModel):
    """Response model for authentication endpoints."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: Dict = Field(..., description="User information")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "user": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "email": "user@example.com"
                    }
                }
            ]
        }
    }

