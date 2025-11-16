"""
Input validation utilities for API endpoints.
"""

import re
import ipaddress
from urllib.parse import urlparse
from typing import Tuple, Optional
from fastapi import HTTPException


# Validation constants
COLLECTION_NAME_MIN_LENGTH = 3
COLLECTION_NAME_MAX_LENGTH = 50
COLLECTION_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

MAX_PAGES_MIN = 1
MAX_PAGES_MAX = 1000
MAX_DEPTH_MIN = 1
MAX_DEPTH_MAX = 10

# Private IP ranges and localhost
PRIVATE_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),  # Link-local addresses
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
    ipaddress.ip_network('fe80::/10'),
]

# Cloud metadata endpoints (SSRF protection)
BLOCKED_IPS = [
    '169.254.169.254',  # AWS, Azure, GCP metadata
    '169.254.170.2',    # AWS ECS metadata
    'fd00:ec2::254',    # AWS IPv6 metadata
]


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a URL is properly formatted and safe to crawl.
    
    Args:
        url: The URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Checks:
        - URL starts with http:// or https://
        - URL has a valid domain/host
        - URL is not pointing to localhost or private IP addresses
        - URL is not using dangerous protocols
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            return False, (
                f"Invalid URL scheme '{parsed.scheme}'. "
                "URL must start with 'http://' or 'https://'. "
                "Example: https://example.com/docs"
            )
        
        # Check if host exists
        if not parsed.netloc:
            return False, (
                "Invalid URL: missing domain/host. "
                "URL must include a domain name. "
                "Example: https://example.com/docs"
            )
        
        # Extract hostname (without port)
        hostname = parsed.hostname
        if not hostname:
            return False, (
                "Invalid URL: could not extract hostname. "
                "Please provide a valid URL with a domain name."
            )
        
        # Check for localhost
        if hostname.lower() in ['localhost', '0.0.0.0']:
            return False, (
                "Security error: Cannot crawl localhost URLs. "
                "Please provide a publicly accessible URL."
            )

        # Check for cloud metadata endpoints (SSRF protection)
        if hostname in BLOCKED_IPS:
            return False, (
                f"Security error: Cannot crawl cloud metadata endpoint '{hostname}'. "
                "This IP address is blocked for security reasons."
            )

        # Check for private IP addresses
        try:
            ip = ipaddress.ip_address(hostname)

            # Additional check for cloud metadata IPs
            if str(ip) in BLOCKED_IPS:
                return False, (
                    f"Security error: Cannot crawl cloud metadata endpoint '{hostname}'. "
                    "This IP address is blocked for security reasons."
                )

            for private_range in PRIVATE_IP_RANGES:
                if ip in private_range:
                    return False, (
                        f"Security error: Cannot crawl private IP address '{hostname}'. "
                        "Please provide a publicly accessible URL."
                    )
        except ValueError:
            # Not an IP address, which is fine (it's a domain name)
            pass
        
        # Check for suspicious patterns
        if '.local' in hostname.lower() or '.internal' in hostname.lower():
            return False, (
                "Security error: Cannot crawl internal/local domains. "
                "Please provide a publicly accessible URL."
            )
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}. Please provide a valid URL."


def validate_collection_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a collection name meets requirements.
    
    Args:
        name: The collection name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Requirements:
        - Length between 3 and 50 characters
        - Only alphanumeric characters, hyphens, and underscores
        - No special characters that could cause issues
    """
    if not name:
        return False, "Collection name cannot be empty."
    
    if len(name) < COLLECTION_NAME_MIN_LENGTH:
        return False, (
            f"Collection name too short. "
            f"Must be at least {COLLECTION_NAME_MIN_LENGTH} characters. "
            f"Example: 'my-api-docs'"
        )
    
    if len(name) > COLLECTION_NAME_MAX_LENGTH:
        return False, (
            f"Collection name too long. "
            f"Must be at most {COLLECTION_NAME_MAX_LENGTH} characters. "
            f"Current length: {len(name)}"
        )
    
    if not COLLECTION_NAME_PATTERN.match(name):
        return False, (
            "Invalid collection name format. "
            "Collection name must contain only letters, numbers, hyphens (-), and underscores (_). "
            "Examples: 'stripe-api', 'my_docs_v2', 'api-collection-1'"
        )
    
    return True, None


def validate_max_pages(max_pages: Optional[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate max_pages parameter.
    
    Args:
        max_pages: Maximum number of pages to crawl
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_pages is None:
        return True, None
    
    if not isinstance(max_pages, int):
        return False, f"max_pages must be an integer, got {type(max_pages).__name__}"
    
    if max_pages < MAX_PAGES_MIN:
        return False, (
            f"max_pages too small. "
            f"Must be at least {MAX_PAGES_MIN}. "
            f"Recommended: 10-100 for most documentation sites."
        )
    
    if max_pages > MAX_PAGES_MAX:
        return False, (
            f"max_pages too large. "
            f"Must be at most {MAX_PAGES_MAX}. "
            f"Current value: {max_pages}. "
            f"Recommended: 10-100 for most documentation sites."
        )
    
    return True, None


def validate_max_depth(max_depth: Optional[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate max_depth parameter.
    
    Args:
        max_depth: Maximum crawl depth from base URL
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_depth is None:
        return True, None
    
    if not isinstance(max_depth, int):
        return False, f"max_depth must be an integer, got {type(max_depth).__name__}"
    
    if max_depth < MAX_DEPTH_MIN:
        return False, (
            f"max_depth too small. "
            f"Must be at least {MAX_DEPTH_MIN}. "
            f"Recommended: 2-5 for most documentation sites."
        )
    
    if max_depth > MAX_DEPTH_MAX:
        return False, (
            f"max_depth too large. "
            f"Must be at most {MAX_DEPTH_MAX}. "
            f"Current value: {max_depth}. "
            f"Recommended: 2-5 for most documentation sites."
        )
    
    return True, None


def validate_crawl_request(url: str, max_pages: Optional[int] = None, max_depth: Optional[int] = None) -> None:
    """
    Validate all crawl request parameters and raise HTTPException if invalid.
    
    Args:
        url: The URL to crawl
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawl depth
        
    Raises:
        HTTPException: 422 Unprocessable Entity if validation fails
    """
    # Validate URL
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid URL",
                "message": error_msg,
                "field": "url"
            }
        )
    
    # Validate max_pages
    is_valid, error_msg = validate_max_pages(max_pages)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid max_pages parameter",
                "message": error_msg,
                "field": "max_pages",
                "valid_range": f"{MAX_PAGES_MIN}-{MAX_PAGES_MAX}"
            }
        )
    
    # Validate max_depth
    is_valid, error_msg = validate_max_depth(max_depth)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid max_depth parameter",
                "message": error_msg,
                "field": "max_depth",
                "valid_range": f"{MAX_DEPTH_MIN}-{MAX_DEPTH_MAX}"
            }
        )

