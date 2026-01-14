"""
Provider factory for creating provider instances.

The factory maintains a registry of available providers and can instantiate
them by name.
"""

from typing import Type, Dict, List
from realtime_ai.providers.base_provider import BaseProvider
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions


class ProviderFactory:
    """
    Factory for creating provider instances.

    Providers register themselves with the factory and can then be
    instantiated by name.
    """

    # Class-level registry of providers
    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider implementation.

        Args:
            name: Provider name (e.g., 'openai', 'gemini', 'grok', 'nova')
            provider_class: Provider class (must inherit from BaseProvider)

        Raises:
            ValueError: If provider class doesn't inherit from BaseProvider
        """
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseProvider"
            )

        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(cls, provider_name: str, options: RealtimeAIOptions) -> BaseProvider:
        """
        Create a provider instance by name.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'gemini')
            options: Configuration options for the provider

        Returns:
            BaseProvider: Initialized provider instance

        Raises:
            ValueError: If provider name is unknown
        """
        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            available = ", ".join(cls.list_providers())
            raise ValueError(
                f"Unknown provider: '{provider_name}'. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(options)

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List[str]: List of provider names
        """
        return sorted(cls._providers.keys())

    @classmethod
    def is_registered(cls, provider_name: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            provider_name: Provider name to check

        Returns:
            bool: True if provider is registered
        """
        return provider_name.lower() in cls._providers

    @classmethod
    def unregister(cls, provider_name: str) -> None:
        """
        Unregister a provider (mainly for testing).

        Args:
            provider_name: Provider name to unregister
        """
        provider_name = provider_name.lower()
        if provider_name in cls._providers:
            del cls._providers[provider_name]

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered providers (mainly for testing).

        Warning: This will remove all providers including built-in ones.
        """
        cls._providers.clear()


# ============================================================================
# Auto-Register Built-in Providers
# ============================================================================

# Always available: OpenAI
from realtime_ai.providers.openai_provider import OpenAIProvider
ProviderFactory.register("openai", OpenAIProvider)

# Grok (xAI) - OpenAI-compatible API
from realtime_ai.providers.grok_provider import GrokProvider
ProviderFactory.register("grok", GrokProvider)

# Gemini (Google) - Event synthesis provider
from realtime_ai.providers.gemini_provider import GeminiProvider
ProviderFactory.register("gemini", GeminiProvider)

# Note: Nova provider will be registered when implemented
