"""add_user_settings_table

Revision ID: 97fc2b4ef6f7
Revises: ac85656aacd1
Create Date: 2025-11-18 22:29:10.878761

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '97fc2b4ef6f7'
down_revision = 'ac85656aacd1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create user_settings table
    op.create_table(
        'user_settings',
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('preferred_llm_provider', sa.String(), nullable=True),
        sa.Column('preferred_llm_model', sa.String(), nullable=True),
        sa.Column('user_openai_api_key_encrypted', sa.String(), nullable=True),
        sa.Column('user_anthropic_api_key_encrypted', sa.String(), nullable=True),
        sa.Column('user_gemini_api_key_encrypted', sa.String(), nullable=True),
        sa.Column('preferred_embedding_provider', sa.String(), nullable=False, server_default='local'),
        sa.Column('default_max_pages', sa.Integer(), nullable=False, server_default='50'),
        sa.Column('default_max_depth', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('free_messages_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('free_messages_limit', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('last_message_reset_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('api_key_last_updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('api_key_last_validated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id')
    )

    # Create default settings for existing users
    op.execute("""
        INSERT INTO user_settings (user_id, preferred_embedding_provider, default_max_pages, default_max_depth, free_messages_used, free_messages_limit, last_message_reset_date, created_at)
        SELECT id, 'local', 50, 3, 0, 10, now(), now()
        FROM users
        WHERE id NOT IN (SELECT user_id FROM user_settings)
    """)


def downgrade() -> None:
    op.drop_table('user_settings')

