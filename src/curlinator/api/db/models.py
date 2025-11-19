"""
SQLAlchemy database models.
"""

import uuid
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum, UniqueConstraint, JSON
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


class CrawlStatus(str, enum.Enum):
    """Status values for crawl jobs."""
    IN_PROGRESS = "in_progress"  # Crawl is actively running
    COMPLETED = "completed"      # Crawl finished successfully
    FAILED = "failed"            # Crawl failed with error
    CANCELLED = "cancelled"      # User cancelled the crawl


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


class UserSettings(Base):
    """User settings model for LLM preferences, API keys, and usage tracking."""

    __tablename__ = "user_settings"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)

    # LLM Configuration
    preferred_llm_provider = Column(String, nullable=True)  # "openai", "anthropic", "gemini"
    preferred_llm_model = Column(String, nullable=True)  # Specific model name
    user_openai_api_key_encrypted = Column(String, nullable=True)
    user_anthropic_api_key_encrypted = Column(String, nullable=True)
    user_gemini_api_key_encrypted = Column(String, nullable=True)

    # Embedding Configuration
    preferred_embedding_provider = Column(String, default="local")  # "local", "openai", "gemini"

    # Crawl Defaults
    default_max_pages = Column(Integer, default=50)
    default_max_depth = Column(Integer, default=3)

    # Usage Tracking (Daily Reset)
    free_messages_used = Column(Integer, default=0)
    free_messages_limit = Column(Integer, default=10)
    last_message_reset_date = Column(DateTime(timezone=True), server_default=func.now())

    # System Prompt Customization (Admin-only A/B Testing)
    custom_system_prompt = Column(Text, nullable=True)  # User-specific prompt override
    prompt_variant_name = Column(String, nullable=True)  # Label for A/B testing tracking
    prompt_updated_at = Column(DateTime(timezone=True), nullable=True)  # When custom prompt was last updated

    # Audit Trail
    api_key_last_updated_at = Column(DateTime(timezone=True), nullable=True)
    api_key_last_validated_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    user = relationship("User", backref="settings", uselist=False)

    def __repr__(self):
        return f"<UserSettings(user_id={self.user_id}, provider={self.preferred_llm_provider})>"


class SystemConfig(Base):
    """System-wide configuration model for admin-managed settings."""

    __tablename__ = "system_config"

    config_key = Column(String, primary_key=True)  # e.g., "system_prompt"
    config_value = Column(Text, nullable=False)  # The actual configuration value
    description = Column(Text, nullable=True)  # Description of this configuration
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    updated_by_user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Admin who made the change

    # Relationship
    updated_by = relationship("User")

    def __repr__(self):
        return f"<SystemConfig(config_key={self.config_key})>"


class CrawlJob(Base):
    """Crawl job model for tracking incremental crawling progress."""

    __tablename__ = "crawl_jobs"

    crawl_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    collection_id = Column(String, ForeignKey("documentation_collections.id"), nullable=True)
    collection_name = Column(String, nullable=False)

    # Crawl configuration
    url = Column(String, nullable=False)
    max_pages = Column(Integer, nullable=False)
    max_depth = Column(Integer, nullable=False)
    embedding_provider = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)

    # Progress tracking
    status = Column(SQLEnum(CrawlStatus), nullable=False, default=CrawlStatus.IN_PROGRESS)
    pages_crawled = Column(Integer, default=0)
    pages_indexed = Column(Integer, default=0)
    current_batch = Column(Integer, default=0)
    total_batches_estimate = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=False)

    # Error tracking
    error_message = Column(Text, nullable=True)
    failed_urls = Column(JSON, nullable=True)  # Stored as JSON array for SQLite compatibility
    retry_count = Column(Integer, default=0)

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    estimated_completion_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User")
    collection = relationship("DocumentationCollection")

    def __repr__(self):
        return f"<CrawlJob(crawl_id={self.crawl_id}, status={self.status}, pages_indexed={self.pages_indexed}/{self.max_pages})>"

