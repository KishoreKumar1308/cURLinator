"""Shared fixtures and configuration for integration tests"""

import pytest
import os
import logging
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set TESTING environment variable BEFORE any imports
# This must be done before importing the app to disable rate limiting
os.environ["TESTING"] = "true"
os.environ["JWT_SECRET"] = "test-secret-key-for-testing-only"
os.environ["API_KEY_ENCRYPTION_KEY"] = "WUclFDmhiJNfbBDocG1gWPkRMvpKABdZwdYqRxC3LTI="

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini

from curlinator.config import get_settings
from curlinator.api.utils.llm_validation import is_valid_api_key
from curlinator.api.main import app
from curlinator.api.database import Base, get_db
from curlinator.api.db.models import User, DocumentationCollection
from curlinator.api.auth import get_password_hash, create_access_token

logger = logging.getLogger(__name__)

# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _clear_settings_cache():
    """Clear the settings cache to force reload after environment changes"""
    get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for all integration tests"""
    # Configure local embedding model to avoid API dependencies
    # This runs once for the entire test session
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="./data/models"
    )

    # Configure LLM if a VALID API key is available
    # This is needed for QueryFusionRetriever and other LlamaIndex components
    # Skip LLM initialization if only test/placeholder keys are present
    # Try all providers in order: OpenAI, Anthropic, Gemini
    settings = get_settings()
    llm_configured = False

    # Try OpenAI first (most common)
    if is_valid_api_key(settings.openai_api_key, "openai"):
        try:
            Settings.llm = OpenAI(
                model=settings.default_model_openai,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base
            )
            llm_configured = True
            # Override default provider for tests to use OpenAI
            os.environ["DEFAULT_LLM_PROVIDER"] = "openai"
            _clear_settings_cache()  # Clear cache to reload settings with new provider
            logger.info(f"✅ Configured OpenAI LLM for tests: {settings.default_model_openai}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI LLM: {e}")

    # Try Anthropic if OpenAI not available
    if not llm_configured and is_valid_api_key(settings.anthropic_api_key, "anthropic"):
        try:
            Settings.llm = Anthropic(
                model=settings.default_model_anthropic,
                api_key=settings.anthropic_api_key
            )
            llm_configured = True
            # Override default provider for tests to use Anthropic
            os.environ["DEFAULT_LLM_PROVIDER"] = "anthropic"
            _clear_settings_cache()  # Clear cache to reload settings with new provider
            logger.info(f"✅ Configured Anthropic LLM for tests: {settings.default_model_anthropic}")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic LLM: {e}")

    # Try Gemini if neither OpenAI nor Anthropic available
    if not llm_configured and is_valid_api_key(settings.gemini_api_key, "gemini"):
        try:
            Settings.llm = Gemini(
                model=settings.default_model_gemini,
                api_key=settings.gemini_api_key
            )
            llm_configured = True
            # Override default provider for tests to use Gemini
            os.environ["DEFAULT_LLM_PROVIDER"] = "gemini"
            _clear_settings_cache()  # Clear cache to reload settings with new provider
            logger.info(f"✅ Configured Gemini LLM for tests: {settings.default_model_gemini}")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini LLM: {e}")

    if not llm_configured:
        logger.warning(
            "⚠️  No valid LLM API key found - LLM will not be initialized. "
            "Tests that require LLM will be skipped or use mocks."
        )

    yield

    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        is_active=True,
        role="user"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def test_user_id(test_user):
    """Get test user ID."""
    return test_user.id


@pytest.fixture(scope="function")
def auth_headers(test_user):
    """Create authentication headers for test user."""
    token = create_access_token(data={"sub": test_user.id})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def test_collection(db_session, test_user):
    """Create a test collection."""
    collection = DocumentationCollection(
        name="test_collection",
        url="https://docs.example.com",
        domain="docs.example.com",
        pages_crawled=10,
        owner_id=test_user.id,
        embedding_provider="local",
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    db_session.add(collection)
    db_session.commit()
    db_session.refresh(collection)
    return collection


def _has_valid_llm_api_key():
    """
    Check if a VALID API key is available for LLM tests.

    Returns True if at least one valid (non-test/placeholder) API key is configured.
    Used by pytest.mark.skipif to skip tests that require real LLM API access.
    """
    from curlinator.config import get_settings

    settings = get_settings()
    has_key = bool(
        is_valid_api_key(settings.openai_api_key, "openai") or
        is_valid_api_key(settings.anthropic_api_key, "anthropic") or
        is_valid_api_key(settings.gemini_api_key, "gemini")
    )

    return has_key


@pytest.fixture(scope="session")
def check_api_key():
    """Check if a VALID API key is available for LLM tests"""
    return _has_valid_llm_api_key()


# Pytest marker for tests that require a real LLM API key
requires_llm = pytest.mark.skipif(
    not _has_valid_llm_api_key(),
    reason="Requires valid LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY). "
           "Skipping in CI to avoid API costs and rate limits. Run locally with valid API key to test."
)

