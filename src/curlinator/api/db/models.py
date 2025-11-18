"""
SQLAlchemy database models.
"""

import uuid
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from curlinator.api.database import Base


class CollectionVisibility(str, enum.Enum):
    """Visibility options for documentation collections."""
    PRIVATE = "private"  # Only owner can access
    PUBLIC = "public"    # Anyone can view and chat


class SharePermission(str, enum.Enum):
    """Permission levels for shared collections."""
    VIEW = "view"  # Can view collection metadata only
    CHAT = "chat"  # Can view and chat with collection


class User(Base):
    """User model for authentication and ownership tracking."""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_anonymous = Column(Boolean, default=False)
    role = Column(String, nullable=False, default="user")  # "user" or "admin"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    collections = relationship("DocumentationCollection", back_populates="owner", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    shared_collections = relationship("CollectionShare", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class DocumentationCollection(Base):
    """Documentation collection model for tracking indexed documentation."""

    __tablename__ = "documentation_collections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)  # Chroma collection name
    url = Column(String, nullable=False)
    domain = Column(String, nullable=False, index=True)
    pages_crawled = Column(Integer, default=0)
    is_public = Column(Boolean, default=False)  # Deprecated: use visibility instead
    visibility = Column(SQLEnum(CollectionVisibility), nullable=False, default=CollectionVisibility.PRIVATE)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)

    # Embedding model metadata (for consistency when loading index)
    embedding_provider = Column(String, nullable=False, default="local")  # "local", "openai", "gemini"
    embedding_model = Column(String, nullable=False, default="BAAI/bge-small-en-v1.5")  # Specific model name

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="collections")
    chat_sessions = relationship("ChatSession", back_populates="collection", cascade="all, delete-orphan")
    shares = relationship("CollectionShare", back_populates="collection", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DocumentationCollection(id={self.id}, name={self.name}, domain={self.domain})>"


class ChatSession(Base):
    """Chat session model for tracking conversations."""
    
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    collection_id = Column(String, ForeignKey("documentation_collections.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    collection = relationship("DocumentationCollection", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, collection_id={self.collection_id})>"


class ChatMessage(Base):
    """Chat message model for storing conversation history."""

    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    curl_command = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role={self.role}, session_id={self.session_id})>"


class CollectionShare(Base):
    """Collection sharing model for managing access permissions."""

    __tablename__ = "collection_shares"
    __table_args__ = (
        UniqueConstraint('collection_id', 'user_id', name='unique_collection_user_share'),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    collection_id = Column(String, ForeignKey("documentation_collections.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    permission = Column(SQLEnum(SharePermission), nullable=False, default=SharePermission.VIEW)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    collection = relationship("DocumentationCollection", back_populates="shares")
    user = relationship("User", back_populates="shared_collections")

    def __repr__(self):
        return f"<CollectionShare(id={self.id}, collection_id={self.collection_id}, user_id={self.user_id}, permission={self.permission})>"

