# Utilities

## Purpose

Shared utilities used across the application - API clients, credential management.

## What Belongs Here

- `keychain_manager.py` - macOS Keychain integration
- `openai_client.py` - OpenAI API client for orchestration/tools
- Database connection utilities
- Common helpers

## What Does NOT Belong Here

- Training-specific utilities (→ `src/training/utils/`)
- Feature-specific utilities (→ `src/features/`)

## Current Files

- `keychain_manager.py` - Secure credential storage via macOS Keychain
- `openai_client.py` - OpenAI client for CBI-V15 orchestration and tools
