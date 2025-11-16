"""
Collection management endpoints.
"""

import logging
from typing import List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import or_
import chromadb

from curlinator.api.database import get_db
from curlinator.api.db.models import (
    User,
    DocumentationCollection,
    CollectionShare,
    CollectionVisibility,
    SharePermission,
)
from curlinator.api.auth import get_current_user
from curlinator.config import get_settings
from curlinator.api.models.collection import (
    ShareCollectionRequest,
    CollectionShareResponse,
    UpdateShareRequest,
    UpdateVisibilityRequest,
)

router = APIRouter(prefix="/api/v1", tags=["collections"])
logger = logging.getLogger(__name__)


# Helper functions
def get_accessible_collection(
    db: Session,
    collection_name: str,
    user: User,
    required_permission: Optional[SharePermission] = None,
) -> Tuple[DocumentationCollection, Optional[SharePermission]]:
    """
    Get a collection that the user has access to.

    Access is granted if:
    1. User is the owner
    2. Collection is shared with the user (with appropriate permission)
    3. Collection is public

    Args:
        db: Database session
        collection_name: Name of the collection
        user: Current user
        required_permission: Minimum permission level required (None, VIEW, or CHAT)

    Returns:
        Tuple of (collection, user_permission)
        - collection: The DocumentationCollection object
        - user_permission: The user's permission level (None for owner/public, or SharePermission)

    Raises:
        HTTPException: If collection not found or user doesn't have required access
    """
    # Query collection
    collection = db.query(DocumentationCollection).filter(
        DocumentationCollection.name == collection_name
    ).first()

    if not collection:
        logger.warning(f"Collection not found: {collection_name}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "RESOURCE_NOT_FOUND",
                "message": f"Collection '{collection_name}' not found",
                "suggestion": "Use GET /api/v1/collections to see available collections"
            }
        )

    # Check if user is the owner
    if collection.owner_id == user.id:
        logger.info(f"User {user.email} is owner of collection {collection_name}")
        return collection, None  # Owner has full access

    # Check if collection is public
    if collection.visibility == CollectionVisibility.PUBLIC:
        logger.info(f"Collection {collection_name} is public, granting access to {user.email}")
        # Public collections grant CHAT permission to everyone
        return collection, SharePermission.CHAT

    # Check if collection is shared with the user
    share = db.query(CollectionShare).filter(
        CollectionShare.collection_id == collection.id,
        CollectionShare.user_id == user.id
    ).first()

    if share:
        # Check if user has required permission
        if required_permission:
            if required_permission == SharePermission.CHAT and share.permission == SharePermission.VIEW:
                logger.warning(
                    f"User {user.email} has VIEW permission but CHAT required for {collection_name}"
                )
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "PERMISSION_DENIED",
                        "message": f"You only have VIEW permission for collection '{collection_name}'",
                        "suggestion": "Ask the collection owner to grant you CHAT permission"
                    }
                )

        logger.info(f"User {user.email} has {share.permission} access to collection {collection_name}")
        return collection, share.permission

    # No access
    logger.warning(f"User {user.email} has no access to collection {collection_name}")
    raise HTTPException(
        status_code=404,
        detail={
            "error": "RESOURCE_NOT_FOUND",
            "message": f"Collection '{collection_name}' not found or you don't have access to it",
            "suggestion": "Use GET /api/v1/collections to see your available collections"
        }
    )


# Response models
from pydantic import BaseModel, ConfigDict
from datetime import datetime


class CollectionSummary(BaseModel):
    """Summary of a documentation collection."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    url: str
    embedding_provider: str
    embedding_model: str
    pages_crawled: int
    created_at: datetime


class CollectionDetail(BaseModel):
    """Detailed information about a documentation collection."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    url: str
    domain: str
    embedding_provider: str
    embedding_model: str
    pages_crawled: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    owner_id: str
    document_count: int  # Number of documents in vector store


@router.get("/collections", response_model=List[CollectionSummary])
async def list_collections(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
):
    """
    List all collections accessible to the authenticated user.

    This includes:
    - Collections owned by the user
    - Collections shared with the user
    - Public collections

    This endpoint requires JWT authentication.

    Args:
        db: Database session dependency
        current_user: Authenticated user from JWT token
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (default: 100, max: 100)

    Returns:
        List of CollectionSummary objects

    Raises:
        HTTPException: If user not authenticated
    """
    try:
        logger.info(f"Listing collections for user: {current_user.email}")

        # Validate pagination parameters
        if limit > 100:
            limit = 100
        if skip < 0:
            skip = 0

        # Query collections accessible to the user:
        # 1. Owned by user
        # 2. Shared with user
        # 3. Public collections
        collections = db.query(DocumentationCollection).filter(
            or_(
                DocumentationCollection.owner_id == current_user.id,
                DocumentationCollection.visibility == CollectionVisibility.PUBLIC,
                DocumentationCollection.id.in_(
                    db.query(CollectionShare.collection_id).filter(
                        CollectionShare.user_id == current_user.id
                    )
                )
            )
        ).order_by(
            DocumentationCollection.created_at.desc()
        ).offset(skip).limit(limit).all()

        logger.info(f"Found {len(collections)} accessible collections for user {current_user.email}")

        return collections
        
    except SQLAlchemyError as e:
        logger.error(f"Database error while listing collections: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to retrieve collections from database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while listing collections: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )


@router.get("/collections/{collection_name}", response_model=CollectionDetail)
async def get_collection(
    collection_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed information about a specific collection.

    User must have access to the collection (owner, shared, or public).

    This endpoint requires JWT authentication.

    Args:
        collection_name: Name of the collection
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        CollectionDetail object with full collection information

    Raises:
        HTTPException: If collection not found or user doesn't have access
    """
    try:
        logger.info(f"Getting collection details: {collection_name} for user: {current_user.email}")

        # Check if user has access to the collection
        collection, _ = get_accessible_collection(db, collection_name, current_user)
        
        # Get document count from Chroma vector store
        try:
            settings = get_settings()
            chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
            chroma_collection = chroma_client.get_collection(name=collection_name)
            document_count = chroma_collection.count()
            logger.info(f"Collection {collection_name} has {document_count} documents in vector store")
        except Exception as e:
            logger.warning(f"Failed to get document count from vector store: {str(e)}")
            document_count = 0
        
        # Build response
        response = CollectionDetail(
            id=str(collection.id),
            name=collection.name,
            url=collection.url,
            domain=collection.domain,
            embedding_provider=collection.embedding_provider,
            embedding_model=collection.embedding_model,
            pages_crawled=collection.pages_crawled,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
            owner_id=str(collection.owner_id),
            document_count=document_count,
        )
        
        return response
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while getting collection details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to retrieve collection details from database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while getting collection details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )


@router.delete("/collections/{collection_name}", status_code=204)
async def delete_collection(
    collection_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a collection from both database and vector store.

    Only the collection owner can delete a collection.

    This endpoint requires JWT authentication.

    Args:
        collection_name: Name of the collection to delete
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        204 No Content on successful deletion

    Raises:
        HTTPException: If collection not found or user is not the owner
    """
    try:
        logger.info(f"Deleting collection: {collection_name} for user: {current_user.email}")

        # Query collection - must be owner to delete
        collection = db.query(DocumentationCollection).filter(
            DocumentationCollection.name == collection_name,
            DocumentationCollection.owner_id == current_user.id
        ).first()

        if not collection:
            logger.warning(f"Collection not found for deletion: {collection_name} for user: {current_user.email}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' not found or you are not the owner",
                    "suggestion": "Only collection owners can delete collections"
                }
            )
        
        # Delete from Chroma vector store
        try:
            settings = get_settings()
            chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection from vector store: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection from vector store (may not exist): {str(e)}")
            # Continue with database deletion even if vector store deletion fails
        
        # Delete from database
        try:
            db.delete(collection)
            db.commit()
            logger.info(f"Deleted collection from database: {collection_name}")
        except SQLAlchemyError as e:
            logger.error(f"Database error while deleting collection: {str(e)}", exc_info=True)
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database error",
                    "message": "Failed to delete collection from database",
                    "suggestion": "The collection may have been partially deleted. Please try again or contact support."
                }
            )

        return None  # 204 No Content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error while deleting collection: {str(e)}", exc_info=True)
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Failed to rollback database: {rollback_error}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred while deleting collection: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )


# Collection Sharing Endpoints

@router.post("/collections/{collection_name}/share", response_model=CollectionShareResponse, status_code=201)
async def share_collection(
    collection_name: str,
    request: ShareCollectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Share a collection with another user.

    Only the collection owner can share a collection.

    Args:
        collection_name: Name of the collection to share
        request: Share request with user email and permission level
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        CollectionShareResponse with share details

    Raises:
        HTTPException: If collection not found, user is not owner, or target user not found
    """
    try:
        logger.info(f"Sharing collection {collection_name} with {request.user_email}")

        # Get collection - must be owner
        collection = db.query(DocumentationCollection).filter(
            DocumentationCollection.name == collection_name,
            DocumentationCollection.owner_id == current_user.id
        ).first()

        if not collection:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' not found or you are not the owner",
                    "suggestion": "Only collection owners can share collections"
                }
            )

        # Find target user by email
        target_user = db.query(User).filter(User.email == request.user_email).first()
        if not target_user:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"User with email '{request.user_email}' not found",
                    "suggestion": "Ensure the email address is correct"
                }
            )

        # Cannot share with yourself
        if target_user.id == current_user.id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "VALIDATION_ERROR",
                    "message": "Cannot share collection with yourself",
                    "suggestion": "You already have full access as the owner"
                }
            )

        # Check if already shared
        existing_share = db.query(CollectionShare).filter(
            CollectionShare.collection_id == collection.id,
            CollectionShare.user_id == target_user.id
        ).first()

        if existing_share:
            # Update existing share
            existing_share.permission = request.permission
            db.commit()
            db.refresh(existing_share)
            logger.info(f"Updated share permission for {request.user_email} to {request.permission}")
            share = existing_share
        else:
            # Create new share
            share = CollectionShare(
                collection_id=collection.id,
                user_id=target_user.id,
                permission=request.permission
            )
            db.add(share)
            db.commit()
            db.refresh(share)
            logger.info(f"Created new share for {request.user_email} with {request.permission} permission")

        # Build response
        response = CollectionShareResponse(
            id=share.id,
            collection_id=collection.id,
            collection_name=collection.name,
            user_id=target_user.id,
            user_email=target_user.email,
            permission=share.permission,
            created_at=share.created_at,
            updated_at=share.updated_at,
        )

        return response

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while sharing collection: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to share collection",
                "suggestion": "Please try again"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while sharing collection: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SYSTEM_ERROR",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again"
            }
        )


@router.get("/collections/{collection_name}/shares", response_model=List[CollectionShareResponse])
async def list_collection_shares(
    collection_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all users a collection is shared with.

    Only the collection owner can view shares.

    Args:
        collection_name: Name of the collection
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        List of CollectionShareResponse objects

    Raises:
        HTTPException: If collection not found or user is not the owner
    """
    try:
        logger.info(f"Listing shares for collection {collection_name}")

        # Get collection - must be owner
        collection = db.query(DocumentationCollection).filter(
            DocumentationCollection.name == collection_name,
            DocumentationCollection.owner_id == current_user.id
        ).first()

        if not collection:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' not found or you are not the owner",
                    "suggestion": "Only collection owners can view shares"
                }
            )

        # Get all shares for this collection with eager loading of user data
        # This avoids N+1 query problem by using joinedload
        shares = db.query(CollectionShare).options(
            joinedload(CollectionShare.user)
        ).filter(
            CollectionShare.collection_id == collection.id
        ).all()

        # Build response
        response = []
        for share in shares:
            if share.user:
                response.append(CollectionShareResponse(
                    id=share.id,
                    collection_id=collection.id,
                    collection_name=collection.name,
                    user_id=share.user.id,
                    user_email=share.user.email,
                    permission=share.permission,
                    created_at=share.created_at,
                    updated_at=share.updated_at,
                ))

        logger.info(f"Found {len(response)} shares for collection {collection_name}")
        return response

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while listing shares: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to retrieve shares",
                "suggestion": "Please try again"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while listing shares: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SYSTEM_ERROR",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again"
            }
        )


@router.patch("/collections/{collection_name}/shares/{user_email}", response_model=CollectionShareResponse)
async def update_share_permission(
    collection_name: str,
    user_email: str,
    request: UpdateShareRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update the permission level for a shared collection.

    Only the collection owner can update share permissions.

    Args:
        collection_name: Name of the collection
        user_email: Email of the user whose permission to update
        request: Update request with new permission level
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        CollectionShareResponse with updated share details

    Raises:
        HTTPException: If collection not found, user is not owner, or share not found
    """
    try:
        logger.info(f"Updating share permission for {user_email} on collection {collection_name}")

        # Get collection - must be owner
        collection = db.query(DocumentationCollection).filter(
            DocumentationCollection.name == collection_name,
            DocumentationCollection.owner_id == current_user.id
        ).first()

        if not collection:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' not found or you are not the owner",
                    "suggestion": "Only collection owners can update share permissions"
                }
            )

        # Find target user
        target_user = db.query(User).filter(User.email == user_email).first()
        if not target_user:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"User with email '{user_email}' not found",
                    "suggestion": "Ensure the email address is correct"
                }
            )

        # Find share
        share = db.query(CollectionShare).filter(
            CollectionShare.collection_id == collection.id,
            CollectionShare.user_id == target_user.id
        ).first()

        if not share:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' is not shared with '{user_email}'",
                    "suggestion": "Use POST /api/v1/collections/{collection_name}/share to share the collection first"
                }
            )

        # Update permission
        share.permission = request.permission
        db.commit()
        db.refresh(share)

        logger.info(f"Updated share permission for {user_email} to {request.permission}")

        # Build response
        response = CollectionShareResponse(
            id=share.id,
            collection_id=collection.id,
            collection_name=collection.name,
            user_id=target_user.id,
            user_email=target_user.email,
            permission=share.permission,
            created_at=share.created_at,
            updated_at=share.updated_at,
        )

        return response

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while updating share: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to update share permission",
                "suggestion": "Please try again"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while updating share: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SYSTEM_ERROR",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again"
            }
        )


@router.delete("/collections/{collection_name}/shares/{user_email}", status_code=204)
async def revoke_share(
    collection_name: str,
    user_email: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Revoke a user's access to a shared collection.

    Only the collection owner can revoke shares.

    Args:
        collection_name: Name of the collection
        user_email: Email of the user whose access to revoke
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        204 No Content on successful revocation

    Raises:
        HTTPException: If collection not found, user is not owner, or share not found
    """
    try:
        logger.info(f"Revoking share for {user_email} on collection {collection_name}")

        # Get collection - must be owner
        collection = db.query(DocumentationCollection).filter(
            DocumentationCollection.name == collection_name,
            DocumentationCollection.owner_id == current_user.id
        ).first()

        if not collection:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' not found or you are not the owner",
                    "suggestion": "Only collection owners can revoke shares"
                }
            )

        # Find target user
        target_user = db.query(User).filter(User.email == user_email).first()
        if not target_user:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"User with email '{user_email}' not found",
                    "suggestion": "Ensure the email address is correct"
                }
            )

        # Find and delete share
        share = db.query(CollectionShare).filter(
            CollectionShare.collection_id == collection.id,
            CollectionShare.user_id == target_user.id
        ).first()

        if not share:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' is not shared with '{user_email}'",
                    "suggestion": "Use GET /api/v1/collections/{collection_name}/shares to see current shares"
                }
            )

        db.delete(share)
        db.commit()

        logger.info(f"Revoked share for {user_email} on collection {collection_name}")
        return None  # 204 No Content

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while revoking share: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to revoke share",
                "suggestion": "Please try again"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while revoking share: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SYSTEM_ERROR",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again"
            }
        )


@router.patch("/collections/{collection_name}/visibility", response_model=CollectionDetail)
async def update_collection_visibility(
    collection_name: str,
    request: UpdateVisibilityRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update the visibility of a collection (PRIVATE or PUBLIC).

    Only the collection owner can update visibility.

    Args:
        collection_name: Name of the collection
        request: Update request with new visibility level
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        CollectionDetail with updated collection information

    Raises:
        HTTPException: If collection not found or user is not the owner
    """
    try:
        logger.info(f"Updating visibility for collection {collection_name} to {request.visibility}")

        # Get collection - must be owner
        collection = db.query(DocumentationCollection).filter(
            DocumentationCollection.name == collection_name,
            DocumentationCollection.owner_id == current_user.id
        ).first()

        if not collection:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RESOURCE_NOT_FOUND",
                    "message": f"Collection '{collection_name}' not found or you are not the owner",
                    "suggestion": "Only collection owners can update visibility"
                }
            )

        # Update visibility
        collection.visibility = request.visibility
        # Also update deprecated is_public field for backward compatibility
        collection.is_public = (request.visibility == CollectionVisibility.PUBLIC)
        db.commit()
        db.refresh(collection)

        logger.info(f"Updated visibility for collection {collection_name} to {request.visibility}")

        # Get document count from Chroma vector store
        try:
            settings = get_settings()
            chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
            chroma_collection = chroma_client.get_collection(name=collection_name)
            document_count = chroma_collection.count()
        except Exception as e:
            logger.warning(f"Failed to get document count from vector store: {str(e)}")
            document_count = 0

        # Build response
        response = CollectionDetail(
            id=str(collection.id),
            name=collection.name,
            url=collection.url,
            domain=collection.domain,
            embedding_provider=collection.embedding_provider,
            embedding_model=collection.embedding_model,
            pages_crawled=collection.pages_crawled,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
            owner_id=str(collection.owner_id),
            document_count=document_count,
        )

        return response

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while updating visibility: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Failed to update collection visibility",
                "suggestion": "Please try again"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while updating visibility: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SYSTEM_ERROR",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again"
            }
        )

