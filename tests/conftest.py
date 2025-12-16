"""
Pytest configuration and fixtures for CBI-V15 tests.
"""
import pytest
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture
def project_root():
    """Return project root path."""
    return PROJECT_ROOT

@pytest.fixture
def env_vars():
    """Load environment variables for tests."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    return os.environ

