"""
Pydantic schemas for system prompt customization endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime


class SystemPromptUpdate(BaseModel):
    """Request model for updating system-wide prompt."""

    prompt: str = Field(
        ...,
        description="System-wide default prompt for ChatAgent",
        min_length=1,
        max_length=10000
    )
    description: Optional[str] = Field(
        None,
        description="Description of this prompt configuration",
        max_length=500
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and sanitize prompt text."""
        # Strip leading/trailing whitespace
        v = v.strip()
        
        # Ensure prompt is not empty after stripping
        if not v:
            raise ValueError("Prompt cannot be empty or only whitespace")
        
        # Check length
        if len(v) > 10000:
            raise ValueError("Prompt exceeds maximum length of 10,000 characters")
        
        return v


class UserPromptUpdate(BaseModel):
    """Request model for updating per-user custom prompt."""

    prompt: str = Field(
        ...,
        description="User-specific custom prompt for A/B testing",
        min_length=1,
        max_length=10000
    )
    variant_name: Optional[str] = Field(
        None,
        description="Label for A/B testing tracking (e.g., 'variant_a', 'concise_prompt')",
        max_length=100
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and sanitize prompt text."""
        # Strip leading/trailing whitespace
        v = v.strip()
        
        # Ensure prompt is not empty after stripping
        if not v:
            raise ValueError("Prompt cannot be empty or only whitespace")
        
        # Check length
        if len(v) > 10000:
            raise ValueError("Prompt exceeds maximum length of 10,000 characters")
        
        return v


class SystemPromptResponse(BaseModel):
    """Response model for system-wide prompt."""

    prompt: str = Field(..., description="Current system-wide default prompt")
    description: Optional[str] = Field(None, description="Description of this prompt")
    updated_at: Optional[datetime] = Field(None, description="When prompt was last updated")
    updated_by_email: Optional[str] = Field(None, description="Email of admin who last updated the prompt")
    is_default: bool = Field(..., description="Whether this is the hardcoded default prompt")


class UserPromptInfo(BaseModel):
    """Information about a user with custom prompt."""

    user_id: str = Field(..., description="User ID")
    user_email: str = Field(..., description="User email")
    variant_name: Optional[str] = Field(None, description="A/B testing variant label")
    prompt_preview: str = Field(..., description="First 100 characters of custom prompt")
    updated_at: Optional[datetime] = Field(None, description="When custom prompt was last updated")


class PromptsOverviewResponse(BaseModel):
    """Response model for prompts overview."""

    system_prompt: SystemPromptResponse = Field(..., description="Current system-wide prompt")
    users_with_custom_prompts: List[UserPromptInfo] = Field(
        default=[],
        description="List of users with custom prompts for A/B testing"
    )
    total_users_with_custom_prompts: int = Field(
        default=0,
        description="Total count of users with custom prompts"
    )


class PromptUpdateSuccessResponse(BaseModel):
    """Success response for prompt update operations."""

    message: str = Field(..., description="Success message")
    prompt_preview: str = Field(..., description="First 100 characters of updated prompt")
    updated_at: datetime = Field(..., description="Timestamp of update")
    updated_by: str = Field(..., description="Email of admin who made the update")


class UserPromptDeleteResponse(BaseModel):
    """Response for deleting user custom prompt."""

    message: str = Field(..., description="Success message")
    user_id: str = Field(..., description="User ID")
    user_email: str = Field(..., description="User email")

