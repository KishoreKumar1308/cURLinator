"""
Encryption utilities for securing user API keys.
"""

import logging
from typing import Optional
from cryptography.fernet import Fernet

from curlinator.config.settings import get_settings

logger = logging.getLogger(__name__)


def get_encryption_key() -> bytes:
    """
    Get the encryption key from settings.
    
    Returns:
        bytes: The encryption key
        
    Raises:
        ValueError: If encryption key is not configured
    """
    settings = get_settings()
    
    if not settings.api_key_encryption_key:
        raise ValueError(
            "API_KEY_ENCRYPTION_KEY is not configured. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    
    # Ensure the key is bytes
    key = settings.api_key_encryption_key
    if isinstance(key, str):
        key = key.encode()
    
    return key


def encrypt_api_key(api_key: str) -> str:
    """
    Encrypt an API key using Fernet encryption.
    
    Args:
        api_key: The plaintext API key to encrypt
        
    Returns:
        str: The encrypted API key (base64 encoded)
        
    Raises:
        ValueError: If encryption key is not configured
    """
    if not api_key:
        return ""
    
    try:
        encryption_key = get_encryption_key()
        fernet = Fernet(encryption_key)
        encrypted = fernet.encrypt(api_key.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Failed to encrypt API key: {e}")
        raise ValueError("Failed to encrypt API key") from e


def decrypt_api_key(encrypted_api_key: str) -> Optional[str]:
    """
    Decrypt an encrypted API key.
    
    Args:
        encrypted_api_key: The encrypted API key (base64 encoded)
        
    Returns:
        str: The decrypted API key, or None if decryption fails
        
    Raises:
        ValueError: If encryption key is not configured
    """
    if not encrypted_api_key:
        return None
    
    try:
        encryption_key = get_encryption_key()
        fernet = Fernet(encryption_key)
        decrypted = fernet.decrypt(encrypted_api_key.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Failed to decrypt API key: {e}")
        return None

