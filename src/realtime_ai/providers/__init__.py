"""
Provider abstraction for multi-provider support.

This package contains the base provider interface and implementations
for different realtime AI providers (OpenAI, Gemini, Grok, Nova).
"""

from realtime_ai.providers.base_provider import BaseProvider

__all__ = ["BaseProvider"]
