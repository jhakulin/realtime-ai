"""Tests for BaseProvider interface."""

import pytest
import pytest_asyncio
from abc import ABC
from typing import AsyncIterator, List
from realtime_ai.providers.base_provider import BaseProvider
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.normalized_events import NormalizedEvent, EventType


@pytest.fixture
def test_options():
    """Create test RealtimeAIOptions."""
    return RealtimeAIOptions(
        api_key="test_key",
        model="test_model",
        modalities=["text", "audio"],
        instructions="Test instructions"
    )


class ConcreteProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""

    def __init__(self, options: RealtimeAIOptions):
        super().__init__(options)
        self.connect_called = False
        self.disconnect_called = False
        self.send_audio_called = False
        self.send_text_called = False

    @property
    def provider_name(self) -> str:
        return "test"

    async def connect(self) -> None:
        self.connect_called = True
        self._is_connected = True

    async def disconnect(self) -> None:
        self.disconnect_called = True
        self._is_connected = False

    async def send_audio(self, audio_data: bytes) -> None:
        self.send_audio_called = True

    async def send_text(self, text: str, role: str = "user") -> None:
        self.send_text_called = True

    async def update_session(self, options: RealtimeAIOptions) -> None:
        self.options = options

    async def generate_response(self, commit_audio: bool = True) -> None:
        pass

    async def cancel_response(self) -> None:
        pass

    async def send_function_result(self, call_id: str, result: str) -> None:
        pass

    async def receive_events(self) -> AsyncIterator[NormalizedEvent]:
        """Yield test events."""
        yield NormalizedEvent(
            event_id="test_1",
            event_type=EventType.SESSION_CREATED,
            timestamp=1234567890.0,
            provider="test"
        )

    def normalize_incoming_event(self, raw_event: dict) -> List[NormalizedEvent]:
        return [
            NormalizedEvent(
                event_id="evt_1",
                event_type=EventType.AUDIO_DELTA,
                timestamp=1234567890.0,
                provider="test",
                raw_event=raw_event
            )
        ]


class IncompleteProvider(BaseProvider):
    """Provider that doesn't implement all abstract methods (for testing)."""

    @property
    def provider_name(self) -> str:
        return "incomplete"


class TestBaseProvider:
    """Test BaseProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self, test_options):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseProvider(test_options)

    def test_incomplete_provider_cannot_instantiate(self, test_options):
        """Test that incomplete provider implementation cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider(test_options)

    def test_concrete_provider_can_instantiate(self, test_options):
        """Test that complete provider implementation can be instantiated."""
        provider = ConcreteProvider(test_options)

        assert isinstance(provider, BaseProvider)
        assert provider.options == test_options
        assert provider.provider_name == "test"

    @pytest.mark.asyncio
    async def test_provider_connection_lifecycle(self, test_options):
        """Test provider connection and disconnection."""
        provider = ConcreteProvider(test_options)

        # Initially not connected
        assert not provider.is_connected
        assert not provider.connect_called

        # Connect
        await provider.connect()
        assert provider.is_connected
        assert provider.connect_called

        # Disconnect
        await provider.disconnect()
        assert not provider.is_connected
        assert provider.disconnect_called

    @pytest.mark.asyncio
    async def test_send_audio(self, test_options):
        """Test sending audio data."""
        provider = ConcreteProvider(test_options)

        audio_data = b"test_audio_data"
        await provider.send_audio(audio_data)

        assert provider.send_audio_called

    @pytest.mark.asyncio
    async def test_send_text(self, test_options):
        """Test sending text message."""
        provider = ConcreteProvider(test_options)

        await provider.send_text("Hello world", role="user")

        assert provider.send_text_called

    @pytest.mark.asyncio
    async def test_update_session(self):
        """Test updating session configuration."""
        options1 = RealtimeAIOptions(
            api_key="test1",
            model="model1",
            modalities=["text"],
            instructions="Test 1"
        )
        provider = ConcreteProvider(options1)

        assert provider.options == options1

        # Update session
        options2 = RealtimeAIOptions(
            api_key="test2",
            model="model2",
            modalities=["text", "audio"],
            instructions="Test 2"
        )
        await provider.update_session(options2)

        assert provider.options == options2

    @pytest.mark.asyncio
    async def test_receive_events_async_iterator(self, test_options):
        """Test receiving events as async iterator."""
        provider = ConcreteProvider(test_options)

        events = []
        async for event in provider.receive_events():
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == EventType.SESSION_CREATED
        assert events[0].provider == "test"

    def test_normalize_incoming_event(self, test_options):
        """Test event normalization."""
        provider = ConcreteProvider(test_options)

        raw_event = {"type": "test.event", "data": "test_data"}
        normalized_events = provider.normalize_incoming_event(raw_event)

        assert len(normalized_events) == 1
        assert normalized_events[0].event_type == EventType.AUDIO_DELTA
        assert normalized_events[0].provider == "test"
        assert normalized_events[0].raw_event == raw_event

    @pytest.mark.asyncio
    async def test_optional_truncate_response(self, test_options):
        """Test optional truncate_response method (default implementation does nothing)."""
        provider = ConcreteProvider(test_options)

        # Should not raise error (default implementation does nothing)
        await provider.truncate_response("item_1", 0, 1000)

    @pytest.mark.asyncio
    async def test_optional_clear_input_audio_buffer(self, test_options):
        """Test optional clear_input_audio_buffer method (default implementation does nothing)."""
        provider = ConcreteProvider(test_options)

        # Should not raise error (default implementation does nothing)
        await provider.clear_input_audio_buffer()


class TestBaseProviderInterface:
    """Test that BaseProvider defines the correct interface."""

    def test_has_required_methods(self):
        """Test that BaseProvider has all required abstract methods."""
        required_methods = [
            'connect',
            'disconnect',
            'send_audio',
            'send_text',
            'update_session',
            'generate_response',
            'cancel_response',
            'send_function_result',
            'receive_events',
            'normalize_incoming_event',
        ]

        for method_name in required_methods:
            assert hasattr(BaseProvider, method_name)
            method = getattr(BaseProvider, method_name)
            assert callable(method) or isinstance(method, property)

    def test_has_required_properties(self):
        """Test that BaseProvider has all required properties."""
        assert hasattr(BaseProvider, 'provider_name')
        assert hasattr(BaseProvider, 'is_connected')

    def test_has_optional_methods(self):
        """Test that BaseProvider has optional methods."""
        optional_methods = [
            'truncate_response',
            'clear_input_audio_buffer',
        ]

        for method_name in optional_methods:
            assert hasattr(BaseProvider, method_name)
