"""add_performance_indexes

Revision ID: 1094e6e8dab4
Revises: ada6cce15156
Create Date: 2025-11-15 23:15:56.245895

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1094e6e8dab4'
down_revision = 'ada6cce15156'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add indexes for foreign keys and frequently queried columns."""

    # Indexes on foreign keys for better join performance
    op.create_index('ix_documentation_collections_owner_id', 'documentation_collections', ['owner_id'])
    op.create_index('ix_chat_sessions_collection_id', 'chat_sessions', ['collection_id'])
    op.create_index('ix_chat_sessions_user_id', 'chat_sessions', ['user_id'])
    op.create_index('ix_chat_messages_session_id', 'chat_messages', ['session_id'])
    op.create_index('ix_collection_shares_collection_id', 'collection_shares', ['collection_id'])
    op.create_index('ix_collection_shares_user_id', 'collection_shares', ['user_id'])

    # Indexes on timestamp columns used for sorting
    op.create_index('ix_documentation_collections_created_at', 'documentation_collections', ['created_at'])
    op.create_index('ix_chat_sessions_updated_at', 'chat_sessions', ['updated_at'])

    # Composite index for common query patterns
    # Query: Get all sessions for a user in a specific collection
    op.create_index('ix_chat_sessions_user_collection', 'chat_sessions', ['user_id', 'collection_id'])


def downgrade() -> None:
    """Remove performance indexes."""

    # Drop composite indexes
    op.drop_index('ix_chat_sessions_user_collection', table_name='chat_sessions')

    # Drop timestamp indexes
    op.drop_index('ix_chat_sessions_updated_at', table_name='chat_sessions')
    op.drop_index('ix_documentation_collections_created_at', table_name='documentation_collections')

    # Drop foreign key indexes
    op.drop_index('ix_collection_shares_user_id', table_name='collection_shares')
    op.drop_index('ix_collection_shares_collection_id', table_name='collection_shares')
    op.drop_index('ix_chat_messages_session_id', table_name='chat_messages')
    op.drop_index('ix_chat_sessions_user_id', table_name='chat_sessions')
    op.drop_index('ix_chat_sessions_collection_id', table_name='chat_sessions')
    op.drop_index('ix_documentation_collections_owner_id', table_name='documentation_collections')

