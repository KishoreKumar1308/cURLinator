"""add_system_prompt_customization

Revision ID: 61a08c281a85
Revises: 97fc2b4ef6f7
Create Date: 2025-11-19 12:34:48.354436

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '61a08c281a85'
down_revision = '97fc2b4ef6f7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add prompt customization fields to user_settings table
    op.add_column('user_settings', sa.Column('custom_system_prompt', sa.Text(), nullable=True))
    op.add_column('user_settings', sa.Column('prompt_variant_name', sa.String(), nullable=True))
    op.add_column('user_settings', sa.Column('prompt_updated_at', sa.DateTime(timezone=True), nullable=True))

    # Create system_config table
    op.create_table(
        'system_config',
        sa.Column('config_key', sa.String(), nullable=False),
        sa.Column('config_value', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_by_user_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['updated_by_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('config_key')
    )


def downgrade() -> None:
    # Drop system_config table
    op.drop_table('system_config')

    # Remove prompt customization fields from user_settings table
    op.drop_column('user_settings', 'prompt_updated_at')
    op.drop_column('user_settings', 'prompt_variant_name')
    op.drop_column('user_settings', 'custom_system_prompt')

