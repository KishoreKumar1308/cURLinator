"""Add crawl_jobs table for incremental crawling

Revision ID: add_crawl_jobs_001
Revises: <previous_revision>
Create Date: 2025-11-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_crawl_jobs_001'
down_revision = None  # TODO: Update with actual previous revision
branch_labels = None
depends_on = None


def upgrade():
    # Create crawl_jobs table
    op.create_table(
        'crawl_jobs',
        sa.Column('crawl_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('collection_id', sa.String(), sa.ForeignKey('documentation_collections.id'), nullable=True),
        sa.Column('collection_name', sa.String(), nullable=False),
        
        # Crawl configuration
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('max_pages', sa.Integer(), nullable=False),
        sa.Column('max_depth', sa.Integer(), nullable=False),
        sa.Column('embedding_provider', sa.String(), nullable=False),
        sa.Column('embedding_model', sa.String(), nullable=False),
        
        # Progress tracking
        sa.Column('status', sa.String(), nullable=False),  # 'in_progress', 'completed', 'failed', 'cancelled'
        sa.Column('pages_crawled', sa.Integer(), default=0),
        sa.Column('pages_indexed', sa.Integer(), default=0),
        sa.Column('current_batch', sa.Integer(), default=0),
        sa.Column('total_batches_estimate', sa.Integer(), nullable=True),
        sa.Column('batch_size', sa.Integer(), nullable=False),
        
        # Error tracking
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('failed_urls', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('retry_count', sa.Integer(), default=0),
        
        # Timestamps
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('estimated_completion_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.PrimaryKeyConstraint('crawl_id'),
        sa.Index('idx_crawl_jobs_user_id', 'user_id'),
        sa.Index('idx_crawl_jobs_status', 'status'),
        sa.Index('idx_crawl_jobs_created_at', 'started_at'),
    )


def downgrade():
    op.drop_table('crawl_jobs')

