"""Tests for OpenAIProvider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from realtime_ai.providers.openai_provider import OpenAIProvider
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.normalized_events import (
    EventType,
    SessionCreatedEvent,
    AudioDeltaEvent,
    TranscriptDeltaEvent,
    FunctionCallEvent,
    ErrorEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
)


@pytest.fixture
def test_options():
    """Create test RealtimeAIOptions."""
    return RealtimeAIOptions(
        api_key="test_key",
        model="gpt-4o-realtime-preview",
        modalities=["text", "audio"],
        instructions="Test instructions"
    )


@pytest.fixture
def mock_service_manager():
    """Create a mock RealtimeAIServiceManager."""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.send_event = AsyncMock()
    mock.update_session = AsyncMock()
    mock.clear_event_queue = AsyncMock()
    mock.get_next_event = AsyncMock(return_value=None)
    return mock


class TestOpenAIProviderBasics:
    """Test basic OpenAIProvider functionality."""

    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    def test_provider_initialization(self, mock_manager_class, test_options):
        """Test provider initialization."""
        provider = OpenAIProvider(test_options)

        assert provider.provider_name == "openai"
        assert provider.options == test_options
        assert not provider.is_connected
        mock_manager_class.assert_called_once_with(test_options)

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_connect(self, mock_manager_class, test_options):
        """Test connecting to OpenAI."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)
        await provider.connect()

        assert provider.is_connected
        mock_manager.connect.assert_called_once()

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_disconnect(self, mock_manager_class, test_options):
        """Test disconnecting from OpenAI."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)
        await provider.connect()
        await provider.disconnect()

        assert not provider.is_connected
        mock_manager.disconnect.assert_called_once()


class TestOpenAIProviderOperations:
    """Test OpenAI provider operations."""

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_send_audio(self, mock_manager_class, test_options):
        """Test sending audio data."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)
        audio_data = b"test_audio_data"

        await provider.send_audio(audio_data)

        # Verify send_event was called with input_audio_buffer.append
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "input_audio_buffer.append"
        assert "audio" in call_args

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_send_text(self, mock_manager_class, test_options):
        """Test sending text message."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)

        await provider.send_text("Hello world", role="user")

        # Verify send_event was called with conversation.item.create
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "conversation.item.create"
        assert call_args["item"]["role"] == "user"
        assert call_args["item"]["content"][0]["text"] == "Hello world"

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_generate_response(self, mock_manager_class, test_options):
        """Test generating a response."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)

        await provider.generate_response(commit_audio=True)

        # Should send two events: commit and response.create
        assert mock_manager.send_event.call_count == 2

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_cancel_response(self, mock_manager_class, test_options):
        """Test cancelling a response."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)

        await provider.cancel_response()

        # Verify response.cancel was sent and queue was cleared
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "response.cancel"
        mock_manager.clear_event_queue.assert_called_once()


class TestOpenAIProviderEventNormalization:
    """Test event normalization from OpenAI to generic format."""

    def test_normalize_session_created(self, test_options):
        """Test normalizing session.created event."""
        provider = OpenAIProvider(test_options)

        openai_event = {
            "event_id": "evt_123",
            "type": "session.created",
            "session": {
                "id": "sess_abc",
                "model": "gpt-4o-realtime-preview"
            }
        }

        normalized = provider.normalize_incoming_event(openai_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], SessionCreatedEvent)
        assert normalized[0].event_type == EventType.SESSION_CREATED
        assert normalized[0].provider == "openai"
        assert normalized[0].session_id == "sess_abc"

    def test_normalize_audio_delta(self, test_options):
        """Test normalizing response.audio.delta event."""
        provider = OpenAIProvider(test_options)

        openai_event = {
            "event_id": "evt_456",
            "type": "response.audio.delta",
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "SGVsbG8="
        }

        normalized = provider.normalize_incoming_event(openai_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], AudioDeltaEvent)
        assert normalized[0].event_type == EventType.AUDIO_DELTA
        assert normalized[0].delta == "SGVsbG8="
        assert normalized[0].response_id == "resp_1"

    def test_normalize_transcript_delta(self, test_options):
        """Test normalizing response.audio_transcript.delta event."""
        provider = OpenAIProvider(test_options)

        openai_event = {
            "event_id": "evt_789",
            "type": "response.audio_transcript.delta",
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello "
        }

        normalized = provider.normalize_incoming_event(openai_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], TranscriptDeltaEvent)
        assert normalized[0].event_type == EventType.TRANSCRIPT_DELTA
        assert normalized[0].delta == "Hello "

    def test_normalize_function_call(self, test_options):
        """Test normalizing response.function_call_arguments.done event."""
        provider = OpenAIProvider(test_options)

        openai_event = {
            "event_id": "evt_func",
            "type": "response.function_call_arguments.done",
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"location":"SF"}'
        }

        normalized = provider.normalize_incoming_event(openai_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], FunctionCallEvent)
        assert normalized[0].call_id == "call_abc"
        assert normalized[0].function_name == "get_weather"
        assert normalized[0].arguments == '{"location":"SF"}'

    def test_normalize_error_event(self, test_options):
        """Test normalizing error event."""
        provider = OpenAIProvider(test_options)

        openai_event = {
            "event_id": "evt_err",
            "type": "error",
            "error": {
                "type": "rate_limit",
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded"
            }
        }

        normalized = provider.normalize_incoming_event(openai_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], ErrorEvent)
        assert normalized[0].error_code == "rate_limit_exceeded"
        assert normalized[0].error_message == "Rate limit exceeded"

    def test_normalize_speech_started(self, test_options):
        """Test normalizing input_audio_buffer.speech_started event."""
        provider = OpenAIProvider(test_options)

        openai_event = {
            "event_id": "evt_speech",
            "type": "input_audio_buffer.speech_started",
            "item_id": "item_1"
        }

        normalized = provider.normalize_incoming_event(openai_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], SpeechStartedEvent)
        assert normalized[0].event_type == EventType.SPEECH_STARTED

    def test_normalize_unknown_event(self, test_options):
        """Test normalizing unknown event type."""
        provider = OpenAIProvider(test_options)

        openai_event = {
            "event_id": "evt_unknown",
            "type": "unknown.event.type"
        }

        normalized = provider.normalize_incoming_event(openai_event)

        # Should return empty list for unknown events
        assert len(normalized) == 0


class TestOpenAIProviderAdvanced:
    """Test advanced OpenAI provider features."""

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_truncate_response(self, mock_manager_class, test_options):
        """Test truncating a response (OpenAI-specific feature)."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)

        await provider.truncate_response("item_1", 0, 5000)

        # Verify conversation.item.truncate was sent
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "conversation.item.truncate"
        assert call_args["item_id"] == "item_1"
        assert call_args["audio_end_ms"] == 5000

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.openai_provider.RealtimeAIServiceManager')
    async def test_clear_input_audio_buffer(self, mock_manager_class, test_options):
        """Test clearing input audio buffer."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = OpenAIProvider(test_options)

        await provider.clear_input_audio_buffer()

        # Verify input_audio_buffer.clear was sent
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "input_audio_buffer.clear"
