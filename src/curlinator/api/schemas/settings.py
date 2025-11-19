"""
Pydantic schemas for user settings endpoints.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class UserSettingsResponse(BaseModel):
    """Response model for user settings."""

    model_config = ConfigDict(from_attributes=True)

    preferred_llm_provider: Optional[str] = None
    preferred_llm_model: Optional[str] = None
    has_openai_key: bool = False
    has_anthropic_key: bool = False
    has_gemini_key: bool = False
    preferred_embedding_provider: str = "local"
    default_max_pages: int = 50
    default_max_depth: int = 3
    free_messages_used: int = 0
    free_messages_limit: int = 10
    free_messages_remaining: int = 10


class UserSettingsUpdate(BaseModel):
    """Request model for updating user settings."""

    preferred_llm_provider: Optional[Literal["openai", "anthropic", "gemini"]] = None
    preferred_llm_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    preferred_embedding_provider: Optional[Literal["local", "openai", "gemini"]] = None
    default_max_pages: Optional[int] = Field(None, ge=1, le=1000)
    default_max_depth: Optional[int] = Field(None, ge=1, le=10)

    @field_validator('openai_api_key', 'anthropic_api_key', 'gemini_api_key', mode='before')
    @classmethod
    def validate_api_key_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key format. Keep empty string as empty string (don't convert to None)."""
        if v is None:
            return None
        if v == "":
            return ""  # Keep empty string to signal removal
        
        # Basic validation - just check it's not empty and has reasonable length
        if len(v) < 10:
            raise ValueError("API key is too short")
        
        return v


class ValidateAPIKeyRequest(BaseModel):
    """Request model for validating an API key."""
    
    provider: Literal["openai", "anthropic", "gemini"]
    api_key: str = Field(..., min_length=10)


class ValidateAPIKeyResponse(BaseModel):
    """Response model for API key validation."""
    
    valid: bool
    provider: str
    error: Optional[str] = None

