#!/usr/bin/env python3
"""
macOS Keychain API key manager for CBI-V15
"""
import subprocess
import logging

logger = logging.getLogger(__name__)

def get_api_key(key_name: str) -> str | None:
    """
    Retrieve API key from macOS Keychain
    
    Args:
        key_name: Name of the key (e.g., "DATABENTO_API_KEY")
        
    Returns:
        API key string or None if not found
    """
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s", key_name,
                "-w"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        logger.debug(f"Key {key_name} not found in Keychain")
        return None
    except FileNotFoundError:
        logger.warning("security command not found (not on macOS?)")
        return None
    except Exception as e:
        logger.error(f"Error retrieving key {key_name}: {e}")
        return None

def store_api_key(key_name: str, service: str, key_value: str) -> bool:
    """
    Store API key in macOS Keychain
    
    Args:
        key_name: Name of the key
        service: Service name (e.g., "databento")
        key_value: The API key value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-a", service,
                "-s", key_name,
                "-w", key_value,
                "-U"
            ],
            check=True,
            capture_output=True
        )
        logger.info(f"Stored {key_name} in Keychain")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to store {key_name}: {e}")
        return False
    except FileNotFoundError:
        logger.warning("security command not found (not on macOS?)")
        return False

