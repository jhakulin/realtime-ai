"""
Provider Configuration Helper for Multi-Provider Support

This module centralizes provider-specific settings (API keys, models, voices, endpoints)
to make samples simple and focused on their specific functionality.

Usage:
    from provider_config import get_provider_config, list_available_providers

    # Auto-detect provider from environment variables
    config = get_provider_config()

    # Or specify a provider explicitly
    config = get_provider_config("grok")

    # Use config in RealtimeAIOptions
    options = RealtimeAIOptions(
        api_key=config["api_key"],
        model=config["model"],
        voice=config["voice"],
        ...
    )
    client = RealtimeAIClient(options, stream_options, handler, provider=config["provider"])
"""

import os
from typing import Optional

# Provider registry with all configuration
PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-realtime-preview",
        "default_voice": "alloy",
        "voices": ["alloy", "echo", "shimmer", "sage", "ash", "coral", "ballad", "verse"],
    },
    "grok": {
        "env_key": "XAI_API_KEY",
        "default_model": "grok-3-fast",
        "default_voice": "cove",
        "voices": ["cove", "ember", "juniper", "sage"],
    },
    "gemini": {
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash-exp",
        "default_voice": "Puck",
        "voices": ["Puck", "Charon", "Kore", "Fenrir", "Aoede"],
    },
}

# Priority order for auto-detection
PROVIDER_PRIORITY = ["openai", "grok", "gemini"]


def get_provider_config(provider: Optional[str] = None) -> Optional[dict]:
    """
    Get provider configuration, either by auto-detecting from environment variables
    or using a specified provider.

    Args:
        provider: Optional provider name ("openai", "grok", "gemini").
                  If None, auto-detects based on available API keys.

    Returns:
        Dictionary with provider configuration:
        {
            "provider": str,      # Provider name
            "api_key": str,       # API key value
            "model": str,         # Default model for provider
            "voice": str,         # Default voice for provider
            "voices": list[str],  # Available voices for provider
        }
        Returns None if no valid configuration found.
    """
    if provider:
        # Use specified provider
        if provider not in PROVIDERS:
            return None
        config = PROVIDERS[provider]
        api_key = os.getenv(config["env_key"])
        if not api_key:
            return None
        return {
            "provider": provider,
            "api_key": api_key,
            "model": config["default_model"],
            "voice": config["default_voice"],
            "voices": config["voices"],
        }

    # Auto-detect provider based on available API keys
    for provider_name in PROVIDER_PRIORITY:
        config = PROVIDERS[provider_name]
        api_key = os.getenv(config["env_key"])
        if api_key:
            return {
                "provider": provider_name,
                "api_key": api_key,
                "model": config["default_model"],
                "voice": config["default_voice"],
                "voices": config["voices"],
            }

    return None


def list_available_providers() -> list[str]:
    """
    Returns list of providers with valid API keys configured.

    Returns:
        List of provider names that have their API keys set in environment variables.
    """
    available = []
    for provider_name, config in PROVIDERS.items():
        if os.getenv(config["env_key"]):
            available.append(provider_name)
    return available


def get_provider_env_keys() -> dict[str, str]:
    """
    Returns a mapping of provider names to their environment variable keys.

    Useful for error messages that guide users on which env vars to set.

    Returns:
        Dictionary mapping provider name to env var name.
    """
    return {name: config["env_key"] for name, config in PROVIDERS.items()}
