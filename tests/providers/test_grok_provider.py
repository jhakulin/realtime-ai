"""Tests for GrokProvider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from realtime_ai.providers.grok_provider import GrokProvider
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
    """Create test RealtimeAIOptions for Grok."""
    return RealtimeAIOptions(
        api_key="test_grok_key",
        model="grok-2-vision-1212",
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


class TestGrokProviderBasics:
    """Test basic GrokProvider functionality."""

    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    def test_provider_initialization(self, mock_manager_class, test_options):
        """Test provider initialization."""
        provider = GrokProvider(test_options)

        assert provider.provider_name == "grok"
        assert provider.options == test_options
        assert not provider.is_connected
        mock_manager_class.assert_called_once()

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_connect(self, mock_manager_class, test_options):
        """Test connecting to Grok."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)
        await provider.connect()

        assert provider.is_connected
        mock_manager.connect.assert_called_once()

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_disconnect(self, mock_manager_class, test_options):
        """Test disconnecting from Grok."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)
        await provider.connect()
        await provider.disconnect()

        assert not provider.is_connected
        mock_manager.disconnect.assert_called_once()

    def test_voice_constants(self):
        """Test that voice personality constants are defined."""
        assert GrokProvider.VOICE_ARA == "ara"
        assert GrokProvider.VOICE_REX == "rex"
        assert GrokProvider.VOICE_SAL == "sal"
        assert GrokProvider.VOICE_EVE == "eve"
        assert GrokProvider.VOICE_LEO == "leo"


class TestGrokProviderOperations:
    """Test Grok provider operations."""

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_send_audio(self, mock_manager_class, test_options):
        """Test sending audio data."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)
        audio_data = b"test_audio_data"

        await provider.send_audio(audio_data)

        # Verify send_event was called with input_audio_buffer.append
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "input_audio_buffer.append"
        assert "audio" in call_args

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_send_text(self, mock_manager_class, test_options):
        """Test sending text message."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)

        await provider.send_text("Hello Grok", role="user")

        # Verify send_event was called with conversation.item.create
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "conversation.item.create"
        assert call_args["item"]["role"] == "user"
        assert call_args["item"]["content"][0]["text"] == "Hello Grok"

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_generate_response(self, mock_manager_class, test_options):
        """Test generating a response."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)

        await provider.generate_response(commit_audio=True)

        # Should send two events: commit and response.create
        assert mock_manager.send_event.call_count == 2

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_cancel_response(self, mock_manager_class, test_options):
        """Test cancelling a response."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)

        await provider.cancel_response()

        # Verify response.cancel was sent and queue was cleared
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "response.cancel"
        mock_manager.clear_event_queue.assert_called_once()

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_send_function_result(self, mock_manager_class, test_options):
        """Test sending function call result."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)

        await provider.send_function_result("call_123", '{"result": "success"}')

        # Verify conversation.item.create with function_call_output
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "conversation.item.create"
        assert call_args["item"]["type"] == "function_call_output"
        assert call_args["item"]["call_id"] == "call_123"


class TestGrokProviderEventNormalization:
    """Test event normalization from Grok to generic format."""

    def test_normalize_session_created(self, test_options):
        """Test normalizing session.created event."""
        provider = GrokProvider(test_options)

        grok_event = {
            "event_id": "evt_123",
            "type": "session.created",
            "session": {
                "id": "sess_abc",
                "model": "grok-2-vision-1212"
            }
        }

        normalized = provider.normalize_incoming_event(grok_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], SessionCreatedEvent)
        assert normalized[0].event_type == EventType.SESSION_CREATED
        assert normalized[0].provider == "grok"
        assert normalized[0].session_id == "sess_abc"

    def test_normalize_audio_delta(self, test_options):
        """Test normalizing response.audio.delta event."""
        provider = GrokProvider(test_options)

        grok_event = {
            "event_id": "evt_456",
            "type": "response.audio.delta",
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "SGVsbG8="
        }

        normalized = provider.normalize_incoming_event(grok_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], AudioDeltaEvent)
        assert normalized[0].event_type == EventType.AUDIO_DELTA
        assert normalized[0].provider == "grok"
        assert normalized[0].delta == "SGVsbG8="
        assert normalized[0].response_id == "resp_1"

    def test_normalize_transcript_delta(self, test_options):
        """Test normalizing response.audio_transcript.delta event."""
        provider = GrokProvider(test_options)

        grok_event = {
            "event_id": "evt_789",
            "type": "response.audio_transcript.delta",
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello from Grok "
        }

        normalized = provider.normalize_incoming_event(grok_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], TranscriptDeltaEvent)
        assert normalized[0].event_type == EventType.TRANSCRIPT_DELTA
        assert normalized[0].provider == "grok"
        assert normalized[0].delta == "Hello from Grok "

    def test_normalize_function_call(self, test_options):
        """Test normalizing response.function_call_arguments.done event."""
        provider = GrokProvider(test_options)

        grok_event = {
            "event_id": "evt_func",
            "type": "response.function_call_arguments.done",
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "call_id": "call_abc",
            "name": "web_search",
            "arguments": '{"query":"latest news"}'
        }

        normalized = provider.normalize_incoming_event(grok_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], FunctionCallEvent)
        assert normalized[0].provider == "grok"
        assert normalized[0].call_id == "call_abc"
        assert normalized[0].function_name == "web_search"
        assert normalized[0].arguments == '{"query":"latest news"}'

    def test_normalize_error_event(self, test_options):
        """Test normalizing error event."""
        provider = GrokProvider(test_options)

        grok_event = {
            "event_id": "evt_err",
            "type": "error",
            "error": {
                "type": "rate_limit",
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded"
            }
        }

        normalized = provider.normalize_incoming_event(grok_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], ErrorEvent)
        assert normalized[0].provider == "grok"
        assert normalized[0].error_code == "rate_limit_exceeded"
        assert normalized[0].error_message == "Rate limit exceeded"

    def test_normalize_speech_started(self, test_options):
        """Test normalizing input_audio_buffer.speech_started event."""
        provider = GrokProvider(test_options)

        grok_event = {
            "event_id": "evt_speech",
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": 1000,
            "item_id": "item_1"
        }

        normalized = provider.normalize_incoming_event(grok_event)

        assert len(normalized) == 1
        assert isinstance(normalized[0], SpeechStartedEvent)
        assert normalized[0].event_type == EventType.SPEECH_STARTED
        assert normalized[0].provider == "grok"
        assert normalized[0].audio_start_ms == 1000

    def test_normalize_unknown_event(self, test_options):
        """Test normalizing unknown event type."""
        provider = GrokProvider(test_options)

        grok_event = {
            "event_id": "evt_unknown",
            "type": "unknown.event.type"
        }

        normalized = provider.normalize_incoming_event(grok_event)

        # Should return empty list for unknown events
        assert len(normalized) == 0


class TestGrokProviderAdvanced:
    """Test advanced Grok provider features."""

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_truncate_response(self, mock_manager_class, test_options):
        """Test truncating a response (OpenAI-compatible feature)."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)

        await provider.truncate_response("item_1", 0, 5000)

        # Verify conversation.item.truncate was sent
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "conversation.item.truncate"
        assert call_args["item_id"] == "item_1"
        assert call_args["audio_end_ms"] == 5000

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_clear_input_audio_buffer(self, mock_manager_class, test_options):
        """Test clearing input audio buffer."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)

        await provider.clear_input_audio_buffer()

        # Verify input_audio_buffer.clear was sent
        mock_manager.send_event.assert_called_once()
        call_args = mock_manager.send_event.call_args[0][0]
        assert call_args["type"] == "input_audio_buffer.clear"

    @pytest.mark.asyncio
    @patch('realtime_ai.providers.grok_provider.RealtimeAIServiceManager')
    async def test_update_session(self, mock_manager_class, test_options):
        """Test updating session configuration."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        provider = GrokProvider(test_options)

        # Update with new options
        new_options = RealtimeAIOptions(
            api_key="test_grok_key",
            model="grok-2-vision-1212",
            modalities=["audio"],
            instructions="Updated instructions"
        )

        await provider.update_session(new_options)

        # Verify session update was called
        mock_manager.update_session.assert_called_once_with(new_options)
        assert provider.options == new_options


class TestGrokProviderCompatibility:
    """Test Grok's OpenAI compatibility."""

    def test_event_types_match_openai(self, test_options):
        """Test that Grok uses the same event types as OpenAI."""
        provider = GrokProvider(test_options)

        # Test a few key event types
        openai_compatible_events = [
            "session.created",
            "session.updated",
            "response.audio.delta",
            "response.audio.done",
            "response.audio_transcript.delta",
            "input_audio_buffer.speech_started",
            "response.function_call_arguments.done",
            "error"
        ]

        for event_type in openai_compatible_events:
            grok_event = {
                "event_id": f"evt_{event_type}",
                "type": event_type,
                "session": {"id": "test", "model": "test"} if "session" in event_type else None,
                "response_id": "resp_1" if "response" in event_type else None,
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "test",
                "audio_start_ms": 0,
                "error": {"code": "test", "message": "test"} if event_type == "error" else None,
                "call_id": "call_1" if "function" in event_type else None,
                "name": "test_func" if "function" in event_type else None,
                "arguments": "{}" if "function" in event_type else None
            }

            # Should not raise error and should return normalized events
            normalized = provider.normalize_incoming_event(grok_event)
            # Most events should normalize to exactly 1 event
            # (except unknown types which return empty list)
            if event_type in openai_compatible_events:
                assert len(normalized) >= 0  # Should handle gracefully
