#!/usr/bin/env python3
"""
macOS Keychain and GCP Secret Manager API key manager for CBI-V15
"""
import subprocess
import logging
import os

logger = logging.getLogger(__name__)

# Key name mapping: Keychain name -> Secret Manager name
SECRET_MANAGER_MAP = {
    "DATABENTO_API_KEY": "databento-api-key",
    "FRED_API_KEY": "fred-api-key",
    "SCRAPECREATORS_API_KEY": "scrapecreators-api-key",
    "GLIDE_API_KEY": "glide-api-key",
    "GLIDE_BEARER_TOKEN": "glide-api-key",
}

def get_api_key(key_name: str, project_id: str = "cbi-v15") -> str | None:
    """
    Retrieve API key from multiple sources in order:
    1. Environment variable
    2. macOS Keychain
    3. GCP Secret Manager
    
    Args:
        key_name: Name of the key (e.g., "DATABENTO_API_KEY")
        project_id: GCP project ID (default: cbi-v15)
        
    Returns:
        API key string or None if not found
    """
    # 1. Check environment variable first
    env_value = os.getenv(key_name)
    if env_value:
        logger.debug(f"Found {key_name} in environment")
        return env_value
    
    # 2. Try macOS Keychain
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
            check=True,
            timeout=5
        )
        key_value = result.stdout.strip()
        if key_value:
            logger.debug(f"Found {key_name} in Keychain")
            return key_value
    except subprocess.CalledProcessError:
        logger.debug(f"Key {key_name} not found in Keychain")
    except FileNotFoundError:
        logger.debug("security command not found (not on macOS?)")
    except Exception as e:
        logger.debug(f"Error checking Keychain for {key_name}: {e}")
    
    # 3. Try GCP Secret Manager
    secret_name = SECRET_MANAGER_MAP.get(key_name)
    if secret_name:
        try:
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            
            response = client.access_secret_version(request={"name": secret_path})
            key_value = response.payload.data.decode("UTF-8").strip()
            
            if key_value:
                logger.debug(f"Found {key_name} in Secret Manager")
                return key_value
        except ImportError:
            logger.debug("google-cloud-secret-manager not installed")
        except Exception as e:
            logger.debug(f"Error checking Secret Manager for {key_name}: {e}")
    
    logger.debug(f"Key {key_name} not found in any location")
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
