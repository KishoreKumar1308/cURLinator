"""
Pydantic models for chat endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str = Field(
        ...,
        description="Message role: 'user' or 'assistant'",
        pattern="^(user|assistant)$"
    )
    content: str = Field(..., description="Message content")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "user",
                    "content": "How do I create a customer?"
                }
            ]
        }
    }


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    collection_name: str = Field(
        ...,
        description="Chroma collection name to query (from crawl response)"
    )
    message: str = Field(
        ...,
        description="User query",
        min_length=1
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Chat session ID for conversation history. If not provided, a new session will be created."
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="(Deprecated) Previous conversation messages for context. Use session_id instead for server-side history management."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "crawl_123e4567-e89b-12d3-a456-426614174000",
                    "message": "How do I create a customer?",
                    "session_id": "550e8400-e29b-41d4-a716-446655440000"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="AI-generated response")
    curl_command: Optional[str] = Field(
        None,
        description="Extracted cURL command (if applicable)"
    )
    sources: List[Dict[str, Any]] = Field(
        default=[],
        description="Source documents used to generate response"
    )
    session_id: str = Field(..., description="Chat session ID for this conversation")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "To create a customer, use the POST /v1/customers endpoint...",
                    "curl_command": "curl https://api.stripe.com/v1/customers -u sk_test_xxx: -d email=customer@example.com",
                    "sources": [
                        {
                            "rank": 1,
                            "score": 0.85,
                            "text": "Create a customer object...",
                            "url": "https://stripe.com/docs/api/customers/create"
                        }
                    ],
                    "session_id": "550e8400-e29b-41d4-a716-446655440000"
                }
            ]
        }
    }



class SessionMessage(BaseModel):
    """Message in a chat session."""

    id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    curl_command: Optional[str] = Field(None, description="cURL command (for assistant messages)")
    created_at: datetime = Field(..., description="Message creation timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "msg_123",
                    "role": "user",
                    "content": "How do I create a customer?",
                    "curl_command": None,
                    "created_at": "2024-01-15T10:30:00Z"
                }
            ]
        }
    }


class SessionResponse(BaseModel):
    """Response model for session endpoints."""

    id: str = Field(..., description="Session ID")
    collection_name: str = Field(..., description="Collection name")
    collection_id: str = Field(..., description="Collection ID")
    message_count: int = Field(..., description="Number of messages in session")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Session last update timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "collection_name": "crawl_123",
                    "collection_id": "col_456",
                    "message_count": 4,
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-01-15T10:30:00Z"
                }
            ]
        }
    }


class SessionDetailResponse(BaseModel):
    """Detailed response model for session with messages."""

    id: str = Field(..., description="Session ID")
    collection_name: str = Field(..., description="Collection name")
    collection_id: str = Field(..., description="Collection ID")
    messages: List[SessionMessage] = Field(..., description="Session messages")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Session last update timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "collection_name": "crawl_123",
                    "collection_id": "col_456",
                    "messages": [
                        {
                            "id": "msg_1",
                            "role": "user",
                            "content": "How do I create a customer?",
                            "curl_command": None,
                            "created_at": "2024-01-15T10:00:00Z"
                        }
                    ],
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-01-15T10:30:00Z"
                }
            ]
        }
    }


