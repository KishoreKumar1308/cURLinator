"""
Pydantic models for collection management and sharing.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from curlinator.api.db.models import CollectionVisibility, SharePermission


class CollectionResponse(BaseModel):
    """Response model for collection details."""
    
    id: str
    name: str
    url: str
    domain: str
    pages_crawled: int
    visibility: CollectionVisibility
    owner_id: str
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = {"from_attributes": True}


class ShareCollectionRequest(BaseModel):
    """Request model for sharing a collection with another user."""
    
    user_email: str = Field(..., description="Email of the user to share with")
    permission: SharePermission = Field(
        default=SharePermission.VIEW,
        description="Permission level: VIEW (metadata only) or CHAT (can query)"
    )


class UpdateShareRequest(BaseModel):
    """Request model for updating share permission."""
    
    permission: SharePermission = Field(
        ...,
        description="New permission level: VIEW or CHAT"
    )


class CollectionShareResponse(BaseModel):
    """Response model for collection share information."""
    
    id: str
    collection_id: str
    collection_name: str
    user_id: str
    user_email: str
    permission: SharePermission
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = {"from_attributes": True}


class UpdateVisibilityRequest(BaseModel):
    """Request model for updating collection visibility."""
    
    visibility: CollectionVisibility = Field(
        ...,
        description="Visibility level: PRIVATE or PUBLIC"
    )

