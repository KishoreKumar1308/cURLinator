"""add_user_role_field

Revision ID: ac85656aacd1
Revises: 1094e6e8dab4
Create Date: 2025-11-18 14:35:46.527428

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ac85656aacd1'
down_revision = '1094e6e8dab4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add role column to users table."""
    # Add role column with default 'user'
    op.add_column('users', sa.Column('role', sa.String(), nullable=False, server_default='user'))


def downgrade() -> None:
    """Remove role column from users table."""
    op.drop_column('users', 'role')

