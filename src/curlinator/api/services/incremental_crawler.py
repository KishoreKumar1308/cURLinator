"""
Incremental Crawler Service

Handles background crawling with batch processing and real-time indexing.
Uses FastAPI BackgroundTasks for MVP (no Celery/Redis required).

Key Features:
- Batch processing (10-30 pages per batch)
- Real-time progress tracking in database
- Incremental indexing (users can query after first batch)
- Error handling and retry logic
- Progress estimation

Architecture:
1. Crawl pages in batches using DocumentationAgent
2. Index each batch immediately using ChatAgent.update_index()
3. Update CrawlJob progress in database after each batch
4. Continue until max_pages reached or crawl complete
"""

import logging
import asyncio
import time
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from llama_index.core.schema import Document

from curlinator.api.db.models import CrawlJob, CrawlStatus, DocumentationCollection
from curlinator.agents.documentation_agent import DocumentationAgent
from curlinator.agents.chat_agent import ChatAgent
from curlinator.api.metrics import (
    crawl_operations_total,
    crawl_duration_seconds,
    crawl_pages_total,
    crawl_pages_per_operation,
    vectorstore_operations_total,
    vectorstore_documents_indexed_total,
)

logger = logging.getLogger(__name__)


def calculate_batch_size(max_pages: int) -> int:
    """
    Calculate optimal batch size based on total pages to crawl.
    
    Strategy:
    - Small crawls (1-30 pages): 10 pages per batch
    - Medium crawls (31-100 pages): 20 pages per batch
    - Large crawls (101+ pages): 30 pages per batch
    
    Args:
        max_pages: Maximum pages to crawl
        
    Returns:
        Batch size (10-30 pages)
    """
    if max_pages <= 30:
        return 10
    elif max_pages <= 100:
        return 20
    else:
        return 30


async def execute_incremental_crawl(
    crawl_id: str,
    url: str,
    max_pages: int,
    max_depth: int,
    collection_name: str,
    embed_model: any,
    provider_name: str,
    model_name: str,
    user_id: str,
    db_session_factory: callable,
) -> None:
    """
    Execute incremental crawl in background with batch processing.
    
    This function runs as a FastAPI BackgroundTask and:
    1. Crawls pages in batches
    2. Indexes each batch immediately
    3. Updates progress in database
    4. Allows users to query after first batch
    
    Args:
        crawl_id: Unique crawl job ID
        url: Base URL to crawl
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
        collection_name: Chroma collection name
        embed_model: Embedding model instance
        provider_name: Embedding provider name (for DB)
        model_name: Embedding model name (for DB)
        user_id: User ID who initiated crawl
        db_session_factory: Factory function to create DB sessions
    """
    logger.info(f"[{crawl_id}] Starting incremental crawl for {url}")
    
    crawl_start_time = time.time()
    batch_size = calculate_batch_size(max_pages)
    total_batches_estimate = (max_pages + batch_size - 1) // batch_size  # Ceiling division
    
    # Track overall crawl state
    all_documents: List[Document] = []
    pages_crawled = 0
    pages_indexed = 0
    current_batch = 0
    chat_agent: Optional[ChatAgent] = None
    collection_id: Optional[str] = None
    
    try:
        # Initialize DocumentationAgent
        doc_agent = DocumentationAgent(
            max_pages=max_pages,
            max_depth=max_depth,
            verbose=True,
        )

        logger.info(f"[{crawl_id}] Batch size: {batch_size}, Estimated batches: {total_batches_estimate}")

        # Initialize crawl state for batch crawling
        logger.info(f"[{crawl_id}] Initializing batch crawl...")
        crawl_state = doc_agent.initialize_crawl_state(url)
        is_complete = False

        # Crawl and index in batches
        while not is_complete and pages_crawled < max_pages:
            current_batch += 1

            logger.info(f"[{crawl_id}] Starting batch {current_batch}/{total_batches_estimate}...")

            # Crawl one batch
            batch_docs, crawl_state, is_complete = await doc_agent.execute_batch(
                base_url=url,
                batch_size=batch_size,
                crawl_state=crawl_state,
            )

            if not batch_docs:
                logger.warning(f"[{crawl_id}] No documents in batch {current_batch}, continuing...")
                continue

            pages_crawled += len(batch_docs)
            logger.info(f"[{crawl_id}] Batch {current_batch} crawled: {len(batch_docs)} pages (total: {pages_crawled})")

            # Index the batch immediately
            try:
                if current_batch == 1:
                    # First batch: Create ChatAgent and build initial index
                    logger.info(f"[{crawl_id}] Creating initial index with first batch...")
                    vectorstore_operations_total.labels(operation="index", status="in_progress").inc()
                    
                    chat_agent = ChatAgent(
                        documents=batch_docs,
                        collection_name=collection_name,
                        embed_model=embed_model,
                        verbose=True,
                    )
                    
                    pages_indexed = len(batch_docs)
                    vectorstore_documents_indexed_total.inc(pages_indexed)
                    vectorstore_operations_total.labels(operation="index", status="success").inc()
                    
                    logger.info(f"[{crawl_id}] ✅ First batch indexed ({pages_indexed} pages)")
                    
                    # Create DocumentationCollection in database after first batch
                    with db_session_factory() as db:
                        from urllib.parse import urlparse
                        parsed_url = urlparse(url)
                        domain = parsed_url.netloc
                        
                        collection = DocumentationCollection(
                            name=collection_name,
                            url=url,
                            domain=domain,
                            pages_crawled=pages_crawled,  # Total pages from full crawl
                            owner_id=user_id,
                            is_public=False,
                            embedding_provider=provider_name,
                            embedding_model=model_name,
                        )
                        
                        db.add(collection)
                        db.commit()
                        db.refresh(collection)
                        collection_id = collection.id
                        
                        # Update crawl job with collection_id
                        crawl_job = db.query(CrawlJob).filter(CrawlJob.crawl_id == crawl_id).first()
                        if crawl_job:
                            crawl_job.collection_id = collection_id
                            db.commit()
                        
                        logger.info(f"[{crawl_id}] Created collection in database: {collection_id}")
                
                else:
                    # Subsequent batches: Update existing index
                    logger.info(f"[{crawl_id}] Appending batch {current_batch} to existing index...")
                    vectorstore_operations_total.labels(operation="update", status="in_progress").inc()
                    
                    chat_agent.update_index(batch_docs)
                    
                    pages_indexed += len(batch_docs)
                    vectorstore_documents_indexed_total.inc(len(batch_docs))
                    vectorstore_operations_total.labels(operation="update", status="success").inc()
                    
                    logger.info(f"[{crawl_id}] ✅ Batch {current_batch} indexed ({len(batch_docs)} pages)")
                
                # Update progress in database after each batch
                with db_session_factory() as db:
                    crawl_job = db.query(CrawlJob).filter(CrawlJob.crawl_id == crawl_id).first()
                    if crawl_job:
                        crawl_job.pages_crawled = pages_crawled
                        crawl_job.pages_indexed = pages_indexed
                        crawl_job.current_batch = current_batch
                        crawl_job.updated_at = datetime.now(timezone.utc)

                        # Estimate completion time based on current progress
                        elapsed_time = time.time() - crawl_start_time
                        if current_batch > 0:
                            time_per_batch = elapsed_time / current_batch
                            remaining_batches = total_batches_estimate - current_batch
                            estimated_remaining_time = time_per_batch * remaining_batches
                            crawl_job.estimated_completion_at = datetime.now(timezone.utc) + timedelta(seconds=estimated_remaining_time)

                        db.commit()
                        logger.info(f"[{crawl_id}] Updated progress: {pages_indexed}/{pages_crawled} pages indexed")
            
            except Exception as batch_error:
                logger.error(f"[{crawl_id}] Error processing batch {current_batch}: {batch_error}", exc_info=True)
                # Continue with next batch instead of failing entire crawl
                continue
        
        # Cleanup WebDriver
        logger.info(f"[{crawl_id}] Cleaning up WebDriver...")
        doc_agent.cleanup_crawl_state(crawl_state)

        # Mark crawl as completed
        with db_session_factory() as db:
            crawl_job = db.query(CrawlJob).filter(CrawlJob.crawl_id == crawl_id).first()
            if crawl_job:
                crawl_job.status = CrawlStatus.COMPLETED
                crawl_job.pages_crawled = pages_crawled
                crawl_job.pages_indexed = pages_indexed
                crawl_job.completed_at = datetime.now(timezone.utc)
                crawl_job.estimated_completion_at = None  # Clear estimate
                db.commit()

        # Track successful crawl
        crawl_duration = time.time() - crawl_start_time
        crawl_duration_seconds.observe(crawl_duration)
        crawl_operations_total.labels(status="success").inc()
        crawl_pages_total.labels(status="success").inc(pages_crawled)
        crawl_pages_per_operation.observe(pages_crawled)

        logger.info(f"[{crawl_id}] ✅ Crawl completed: {pages_indexed}/{pages_crawled} pages indexed in {crawl_duration:.1f}s")

    except Exception as e:
        logger.error(f"[{crawl_id}] Crawl failed: {e}", exc_info=True)

        # Cleanup WebDriver if it exists
        try:
            if 'crawl_state' in locals() and crawl_state:
                logger.info(f"[{crawl_id}] Cleaning up WebDriver after error...")
                doc_agent.cleanup_crawl_state(crawl_state)
        except Exception as cleanup_error:
            logger.warning(f"[{crawl_id}] Error during cleanup: {cleanup_error}")

        # Mark crawl as failed
        with db_session_factory() as db:
            crawl_job = db.query(CrawlJob).filter(CrawlJob.crawl_id == crawl_id).first()
            if crawl_job:
                crawl_job.status = CrawlStatus.FAILED
                crawl_job.error_message = str(e)
                crawl_job.completed_at = datetime.now(timezone.utc)
                db.commit()

        # Track failed crawl
        crawl_operations_total.labels(status="failure").inc()
        crawl_duration_seconds.observe(time.time() - crawl_start_time)

