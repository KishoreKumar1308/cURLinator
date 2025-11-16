"""
Standardized error codes for API responses.

Error codes follow the format: CATEGORY_SPECIFIC_ERROR
- AUTH_*: Authentication and authorization errors
- VALIDATION_*: Input validation errors
- RESOURCE_*: Resource-related errors (not found, conflict, etc.)
- CRAWL_*: Crawling operation errors
- CHAT_*: Chat/query operation errors
- DATABASE_*: Database operation errors
- RATE_LIMIT_*: Rate limiting errors
- SYSTEM_*: System/server errors
"""

# Authentication & Authorization Errors (AUTH_*)
AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
AUTH_TOKEN_INVALID = "AUTH_TOKEN_INVALID"
AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
AUTH_NOT_AUTHENTICATED = "AUTH_NOT_AUTHENTICATED"
AUTH_NOT_AUTHORIZED = "AUTH_NOT_AUTHORIZED"

# Validation Errors (VALIDATION_*)
VALIDATION_INVALID_EMAIL = "VALIDATION_INVALID_EMAIL"
VALIDATION_WEAK_PASSWORD = "VALIDATION_WEAK_PASSWORD"
VALIDATION_INVALID_URL = "VALIDATION_INVALID_URL"
VALIDATION_INVALID_PARAMETER = "VALIDATION_INVALID_PARAMETER"
VALIDATION_MISSING_FIELD = "VALIDATION_MISSING_FIELD"

# Resource Errors (RESOURCE_*)
RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
RESOURCE_FORBIDDEN = "RESOURCE_FORBIDDEN"

# Crawl Operation Errors (CRAWL_*)
CRAWL_TIMEOUT = "CRAWL_TIMEOUT"
CRAWL_NO_DOCUMENTS = "CRAWL_NO_DOCUMENTS"
CRAWL_FAILED = "CRAWL_FAILED"
CRAWL_INVALID_URL = "CRAWL_INVALID_URL"
CRAWL_EMBEDDING_FAILED = "CRAWL_EMBEDDING_FAILED"

# Chat/Query Errors (CHAT_*)
CHAT_COLLECTION_NOT_FOUND = "CHAT_COLLECTION_NOT_FOUND"
CHAT_QUERY_FAILED = "CHAT_QUERY_FAILED"
CHAT_INVALID_COLLECTION = "CHAT_INVALID_COLLECTION"

# Database Errors (DATABASE_*)
DATABASE_CONNECTION_FAILED = "DATABASE_CONNECTION_FAILED"
DATABASE_QUERY_FAILED = "DATABASE_QUERY_FAILED"
DATABASE_INTEGRITY_ERROR = "DATABASE_INTEGRITY_ERROR"

# Rate Limiting Errors (RATE_LIMIT_*)
RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

# System Errors (SYSTEM_*)
SYSTEM_INTERNAL_ERROR = "SYSTEM_INTERNAL_ERROR"
SYSTEM_SERVICE_UNAVAILABLE = "SYSTEM_SERVICE_UNAVAILABLE"
SYSTEM_CONFIGURATION_ERROR = "SYSTEM_CONFIGURATION_ERROR"


def create_error_response(
    error_code: str,
    message: str,
    suggestion: str = None
) -> dict:
    """
    Create a standardized error response.
    
    Args:
        error_code: Standardized error code (e.g., AUTH_INVALID_CREDENTIALS)
        message: Human-readable error message
        suggestion: Optional suggestion to fix the error
        
    Returns:
        dict: Standardized error response
    """
    response = {
        "error_code": error_code,
        "error": error_code.replace("_", " ").title(),
        "message": message
    }
    
    if suggestion:
        response["suggestion"] = suggestion
    
    return response

