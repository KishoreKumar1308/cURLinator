"""
Database configuration and session management.
"""

import os
import logging
import time
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import Pool
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# Get database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/curlinator_dev"
)

# Fix Railway's postgres:// to postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Get retry configuration from environment
DATABASE_MAX_RETRIES = int(os.getenv("DATABASE_MAX_RETRIES", "5"))


def create_engine_with_retry(database_url: str, max_retries: int = DATABASE_MAX_RETRIES):
    """
    Create SQLAlchemy engine with retry logic.

    Retries connection with exponential backoff if database is not available.
    This is especially important for Railway deployments where the database
    may start after the API.

    Args:
        database_url: Database connection URL
        max_retries: Maximum number of retry attempts (default: 5)

    Returns:
        SQLAlchemy engine

    Raises:
        OperationalError: If connection fails after all retries
    """
    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        try:
            # Create engine with connection pooling
            engine = create_engine(
                database_url,
                pool_pre_ping=True,  # Verify connections before using
                pool_size=10,
                max_overflow=20,
            )

            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            if retry_count > 0:
                logger.info(f"✅ Database connection established after {retry_count} retries")
            else:
                logger.info("✅ Database connection established")

            return engine

        except OperationalError as e:
            last_error = e
            retry_count += 1

            if retry_count <= max_retries:
                # Calculate exponential backoff: 1s, 2s, 4s, 8s, 16s
                wait_time = 2 ** (retry_count - 1)
                logger.warning(
                    f"⚠️  Database connection failed (attempt {retry_count}/{max_retries}). "
                    f"Retrying in {wait_time}s... Error: {str(e)}"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"❌ Database connection failed after {max_retries} retries. "
                    f"Last error: {str(e)}"
                )
                raise

    # This should never be reached, but just in case
    if last_error:
        raise last_error
    raise OperationalError("Failed to connect to database", None, None)


# Create engine with retry logic
engine = create_engine_with_retry(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()


def get_db():
    """
    Dependency for database sessions.

    Yields a database session and ensures it's closed after use.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Database Metrics Instrumentation
# ============================================================================

def setup_database_metrics():
    """
    Set up database connection pool metrics.

    This function should be called after the engine is created to start
    collecting metrics on database connection pool usage.
    """
    try:
        from curlinator.api.metrics import db_connections_active

        @event.listens_for(Pool, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Track when a connection is created."""
            # Update active connections gauge
            pool = engine.pool
            db_connections_active.set(pool.checkedout())

        @event.listens_for(Pool, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Track when a connection is checked out from the pool."""
            pool = engine.pool
            db_connections_active.set(pool.checkedout())

        @event.listens_for(Pool, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Track when a connection is returned to the pool."""
            pool = engine.pool
            db_connections_active.set(pool.checkedout())

        logger.info("Database metrics instrumentation configured")
    except ImportError:
        logger.warning("Metrics module not available, skipping database metrics setup")


# Set up database metrics
setup_database_metrics()

