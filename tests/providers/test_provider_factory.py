"""Tests for ProviderFactory."""

import pytest
from typing import AsyncIterator, List
from realtime_ai.providers.provider_factory import ProviderFactory
from realtime_ai.providers.base_provider import BaseProvider
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.normalized_events import NormalizedEvent, EventType


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    @property
    def provider_name(self) -> str:
        return "mock"

    async def connect(self) -> None:
        self._is_connected = True

    async def disconnect(self) -> None:
        self._is_connected = False

    async def send_audio(self, audio_data: bytes) -> None:
        pass

    async def send_text(self, text: str, role: str = "user") -> None:
        pass

    async def update_session(self, options: RealtimeAIOptions) -> None:
        pass

    async def generate_response(self, commit_audio: bool = True) -> None:
        pass

    async def cancel_response(self) -> None:
        pass

    async def send_function_result(self, call_id: str, result: str) -> None:
        pass

    async def receive_events(self) -> AsyncIterator[NormalizedEvent]:
        yield NormalizedEvent(
            event_id="mock_1",
            event_type=EventType.SESSION_CREATED,
            timestamp=1234567890.0,
            provider="mock"
        )

    def normalize_incoming_event(self, raw_event: dict) -> List[NormalizedEvent]:
        return []


class AnotherMockProvider(MockProvider):
    """Another mock provider for testing."""

    @property
    def provider_name(self) -> str:
        return "another_mock"


class NotAProvider:
    """Not a provider (doesn't inherit from BaseProvider)."""
    pass


@pytest.fixture
def test_options():
    """Create test RealtimeAIOptions."""
    return RealtimeAIOptions(
        api_key="test_key",
        model="test_model",
        modalities=["text", "audio"],
        instructions="Test instructions"
    )


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear provider registry before and after each test."""
    ProviderFactory.clear_registry()
    yield
    ProviderFactory.clear_registry()


class TestProviderFactoryRegistration:
    """Test provider registration."""

    def test_register_provider(self):
        """Test registering a provider."""
        ProviderFactory.register("mock", MockProvider)

        assert ProviderFactory.is_registered("mock")
        assert "mock" in ProviderFactory.list_providers()

    def test_register_multiple_providers(self):
        """Test registering multiple providers."""
        ProviderFactory.register("mock1", MockProvider)
        ProviderFactory.register("mock2", AnotherMockProvider)

        assert ProviderFactory.is_registered("mock1")
        assert ProviderFactory.is_registered("mock2")
        assert set(ProviderFactory.list_providers()) == {"mock1", "mock2"}

    def test_register_case_insensitive(self):
        """Test that provider names are case-insensitive."""
        ProviderFactory.register("MockProvider", MockProvider)

        assert ProviderFactory.is_registered("mockprovider")
        assert ProviderFactory.is_registered("MOCKPROVIDER")
        assert ProviderFactory.is_registered("MockProvider")

    def test_register_invalid_provider_class(self):
        """Test that registering invalid provider class raises error."""
        with pytest.raises(ValueError, match="must inherit from BaseProvider"):
            ProviderFactory.register("invalid", NotAProvider)

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        ProviderFactory.register("mock", MockProvider)
        assert ProviderFactory.is_registered("mock")

        ProviderFactory.unregister("mock")
        assert not ProviderFactory.is_registered("mock")

    def test_unregister_nonexistent_provider(self):
        """Test unregistering a provider that doesn't exist (should not raise error)."""
        ProviderFactory.unregister("nonexistent")  # Should not raise

    def test_clear_registry(self):
        """Test clearing the entire registry."""
        ProviderFactory.register("mock1", MockProvider)
        ProviderFactory.register("mock2", AnotherMockProvider)

        assert len(ProviderFactory.list_providers()) == 2

        ProviderFactory.clear_registry()

        assert len(ProviderFactory.list_providers()) == 0


class TestProviderFactoryCreation:
    """Test provider instance creation."""

    def test_create_provider(self, test_options):
        """Test creating a provider instance."""
        ProviderFactory.register("mock", MockProvider)

        provider = ProviderFactory.create("mock", test_options)

        assert isinstance(provider, MockProvider)
        assert isinstance(provider, BaseProvider)
        assert provider.provider_name == "mock"
        assert provider.options == test_options

    def test_create_provider_case_insensitive(self, test_options):
        """Test creating provider with case-insensitive name."""
        ProviderFactory.register("mock", MockProvider)

        provider1 = ProviderFactory.create("mock", test_options)
        provider2 = ProviderFactory.create("MOCK", test_options)
        provider3 = ProviderFactory.create("Mock", test_options)

        assert all(isinstance(p, MockProvider) for p in [provider1, provider2, provider3])

    def test_create_unknown_provider(self, test_options):
        """Test creating unknown provider raises error."""
        ProviderFactory.register("mock", MockProvider)

        with pytest.raises(ValueError, match="Unknown provider: 'nonexistent'"):
            ProviderFactory.create("nonexistent", test_options)

    def test_create_error_message_shows_available(self, test_options):
        """Test that error message shows available providers."""
        ProviderFactory.register("mock1", MockProvider)
        ProviderFactory.register("mock2", AnotherMockProvider)

        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create("nonexistent", test_options)

        error_message = str(exc_info.value)
        assert "Available providers:" in error_message
        assert "mock1" in error_message
        assert "mock2" in error_message

    def test_create_multiple_instances(self, test_options):
        """Test creating multiple instances of same provider."""
        ProviderFactory.register("mock", MockProvider)

        provider1 = ProviderFactory.create("mock", test_options)
        provider2 = ProviderFactory.create("mock", test_options)

        # Should create separate instances
        assert provider1 is not provider2
        assert isinstance(provider1, MockProvider)
        assert isinstance(provider2, MockProvider)


class TestProviderFactoryQuery:
    """Test provider factory query methods."""

    def test_list_providers_empty(self):
        """Test listing providers when none registered."""
        providers = ProviderFactory.list_providers()
        assert providers == []

    def test_list_providers(self):
        """Test listing registered providers."""
        ProviderFactory.register("mock1", MockProvider)
        ProviderFactory.register("mock2", AnotherMockProvider)
        ProviderFactory.register("mock3", MockProvider)

        providers = ProviderFactory.list_providers()

        assert len(providers) == 3
        assert set(providers) == {"mock1", "mock2", "mock3"}

    def test_list_providers_sorted(self):
        """Test that provider list is sorted."""
        ProviderFactory.register("zebra", MockProvider)
        ProviderFactory.register("apple", AnotherMockProvider)
        ProviderFactory.register("middle", MockProvider)

        providers = ProviderFactory.list_providers()

        assert providers == ["apple", "middle", "zebra"]

    def test_is_registered_true(self):
        """Test is_registered for registered provider."""
        ProviderFactory.register("mock", MockProvider)

        assert ProviderFactory.is_registered("mock")

    def test_is_registered_false(self):
        """Test is_registered for unregistered provider."""
        assert not ProviderFactory.is_registered("nonexistent")

    def test_is_registered_case_insensitive(self):
        """Test is_registered is case-insensitive."""
        ProviderFactory.register("MockProvider", MockProvider)

        assert ProviderFactory.is_registered("mockprovider")
        assert ProviderFactory.is_registered("MOCKPROVIDER")
        assert ProviderFactory.is_registered("MockProvider")


class TestProviderFactoryEdgeCases:
    """Test edge cases and error handling."""

    def test_register_same_name_twice(self):
        """Test registering two providers with same name (should overwrite)."""
        ProviderFactory.register("mock", MockProvider)
        ProviderFactory.register("mock", AnotherMockProvider)

        # Should use the second registration
        provider = ProviderFactory.create("mock", RealtimeAIOptions(
            api_key="test",
            model="test",
            modalities=["text"],
            instructions="test"
        ))

        assert isinstance(provider, AnotherMockProvider)

    def test_factory_is_singleton(self):
        """Test that factory maintains single registry across instances."""
        ProviderFactory.register("mock", MockProvider)

        # Registry should be accessible from any reference to the class
        assert ProviderFactory.is_registered("mock")
        assert len(ProviderFactory.list_providers()) == 1

    def test_empty_provider_name(self, test_options):
        """Test creating provider with empty name."""
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.create("", test_options)
