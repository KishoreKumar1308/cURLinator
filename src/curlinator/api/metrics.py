"""
Prometheus metrics for cURLinator API.

This module defines and manages all Prometheus metrics for monitoring
the cURLinator application, including:
- HTTP request metrics (count, latency, errors)
- Database operation metrics
- Vector store operation metrics
- Crawling operation metrics
"""

import logging
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)

# ============================================================================
# HTTP Request Metrics
# ============================================================================

http_requests_total = Counter(
    'curlinator_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'curlinator_http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_requests_in_progress = Gauge(
    'curlinator_http_requests_in_progress',
    'Number of HTTP requests currently being processed',
    ['method', 'endpoint']
)

# ============================================================================
# Authentication Metrics
# ============================================================================

auth_attempts_total = Counter(
    'curlinator_auth_attempts_total',
    'Total number of authentication attempts',
    ['endpoint', 'status']  # status: success, failure
)

auth_tokens_created_total = Counter(
    'curlinator_auth_tokens_created_total',
    'Total number of JWT tokens created'
)

# ============================================================================
# Database Metrics
# ============================================================================

db_queries_total = Counter(
    'curlinator_db_queries_total',
    'Total number of database queries',
    ['operation', 'table', 'status']  # operation: select, insert, update, delete
)

db_query_duration_seconds = Histogram(
    'curlinator_db_query_duration_seconds',
    'Database query latency in seconds',
    ['operation', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

db_connections_active = Gauge(
    'curlinator_db_connections_active',
    'Number of active database connections'
)

db_errors_total = Counter(
    'curlinator_db_errors_total',
    'Total number of database errors',
    ['error_type']  # error_type: integrity, query, connection
)

# ============================================================================
# Vector Store (Chroma) Metrics
# ============================================================================

vectorstore_operations_total = Counter(
    'curlinator_vectorstore_operations_total',
    'Total number of vector store operations',
    ['operation', 'status']  # operation: index, query, delete
)

vectorstore_operation_duration_seconds = Histogram(
    'curlinator_vectorstore_operation_duration_seconds',
    'Vector store operation latency in seconds',
    ['operation'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

vectorstore_documents_indexed_total = Counter(
    'curlinator_vectorstore_documents_indexed_total',
    'Total number of documents indexed in vector store'
)

vectorstore_queries_total = Counter(
    'curlinator_vectorstore_queries_total',
    'Total number of vector store queries',
    ['collection']
)

vectorstore_collections_total = Gauge(
    'curlinator_vectorstore_collections_total',
    'Total number of vector store collections'
)

# ============================================================================
# Crawling Metrics
# ============================================================================

crawl_operations_total = Counter(
    'curlinator_crawl_operations_total',
    'Total number of crawl operations',
    ['status']  # status: success, failure, timeout
)

crawl_duration_seconds = Histogram(
    'curlinator_crawl_duration_seconds',
    'Crawl operation duration in seconds',
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

crawl_pages_total = Counter(
    'curlinator_crawl_pages_total',
    'Total number of pages crawled',
    ['status']  # status: success, failure
)

crawl_pages_per_operation = Histogram(
    'curlinator_crawl_pages_per_operation',
    'Number of pages crawled per operation',
    buckets=(1, 5, 10, 25, 50, 100, 200)
)

# ============================================================================
# Chat/Query Metrics
# ============================================================================

chat_queries_total = Counter(
    'curlinator_chat_queries_total',
    'Total number of chat queries',
    ['collection', 'status']  # status: success, failure
)

chat_query_duration_seconds = Histogram(
    'curlinator_chat_query_duration_seconds',
    'Chat query processing duration in seconds',
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

chat_sessions_active = Gauge(
    'curlinator_chat_sessions_active',
    'Number of active chat sessions'
)

chat_messages_total = Counter(
    'curlinator_chat_messages_total',
    'Total number of chat messages',
    ['role']  # role: user, assistant
)

# ============================================================================
# Collection Sharing Metrics
# ============================================================================

collection_shares_total = Counter(
    'curlinator_collection_shares_total',
    'Total number of collection shares',
    ['operation']  # operation: created, updated, revoked
)

collection_visibility_changes_total = Counter(
    'curlinator_collection_visibility_changes_total',
    'Total number of collection visibility changes',
    ['from_visibility', 'to_visibility']
)

# ============================================================================
# Application Info
# ============================================================================

app_info = Info(
    'curlinator_app',
    'cURLinator application information'
)

# Set application info
app_info.info({
    'version': '1.0.0',
    'name': 'cURLinator API'
})

# ============================================================================
# Helper Functions and Decorators
# ============================================================================

def track_time(metric: Histogram, labels: dict = None):
    """
    Decorator to track execution time of a function.
    
    Args:
        metric: Prometheus Histogram metric to record duration
        labels: Dictionary of labels to apply to the metric
    
    Example:
        @track_time(db_query_duration_seconds, {'operation': 'select', 'table': 'users'})
        def get_user(user_id):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def increment_counter(metric: Counter, labels: dict = None, amount: float = 1):
    """
    Helper function to increment a counter metric.
    
    Args:
        metric: Prometheus Counter metric to increment
        labels: Dictionary of labels to apply to the metric
        amount: Amount to increment by (default: 1)
    """
    if labels:
        metric.labels(**labels).inc(amount)
    else:
        metric.inc(amount)


def set_gauge(metric: Gauge, value: float, labels: dict = None):
    """
    Helper function to set a gauge metric value.
    
    Args:
        metric: Prometheus Gauge metric to set
        value: Value to set
        labels: Dictionary of labels to apply to the metric
    """
    if labels:
        metric.labels(**labels).set(value)
    else:
        metric.set(value)


logger.info("Prometheus metrics initialized")

