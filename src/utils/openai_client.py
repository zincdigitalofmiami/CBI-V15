#!/usr/bin/env python3
"""
OpenAI client utilities for CBI-V15.
Fetches the API key from env or macOS Keychain and exposes a simple chat helper.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from .keychain_manager import get_api_key

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")


def resolve_api_key() -> Optional[str]:
    """Return OpenAI API key from env or Keychain."""
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    keychain_key = get_api_key("OPENAI_API_KEY")
    if keychain_key:
        return keychain_key

    logger.warning("OPENAI_API_KEY not set in env or Keychain")
    return None


def get_client(api_key: Optional[str] = None):
    """Instantiate an OpenAI client with the provided or resolved key."""
    if OpenAI is None:
        raise ImportError("openai package not installed. Install with `pip install openai>=1.30.0`.")

    key = api_key or resolve_api_key()
    if not key:
        raise ValueError("OpenAI API key missing; set OPENAI_API_KEY in env or store in Keychain.")

    return OpenAI(api_key=key)


def get_default_model(model: Optional[str] = None) -> str:
    """Get the default model name (env override, otherwise gpt-5.1)."""
    return model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def run_chat(prompt: str, system: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
    """Execute a simple chat completion and return the assistant text."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=get_default_model(model),
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content  # type: ignore[return-value]
