"""Tests for GeminiProvider and event synthesis."""

import pytest
from realtime_ai.providers.gemini_provider import GeminiProvider
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.normalized_events import (
    EventType,
    SessionCreatedEvent,
    ResponseCreatedEvent,
    AudioDeltaEvent,
    AudioDoneEvent,
    TranscriptDeltaEvent,
    TranscriptDoneEvent,
    ResponseDoneEvent,
    InputTranscriptCompletedEvent,
    FunctionCallEvent,
)


@pytest.fixture
def test_options():
    """Create test RealtimeAIOptions for Gemini."""
    return RealtimeAIOptions(
        api_key="test_gemini_key",
        model="gemini-2.0-flash-exp",
        modalities=["text", "audio"],
        instructions="You are a helpful assistant."
    )


class TestGeminiProviderBasics:
    """Test basic GeminiProvider functionality."""

    def test_provider_initialization(self, test_options):
        """Test provider initialization."""
        provider = GeminiProvider(test_options)

        assert provider.provider_name == "gemini"
        assert provider.options == test_options
        assert not provider.is_connected
        assert not provider._response_active
        assert provider._accumulated_transcript == ""

    @pytest.mark.asyncio
    async def test_connect(self, test_options):
        """Test connecting to Gemini."""
        from unittest.mock import AsyncMock, patch
        provider = GeminiProvider(test_options)

        # Mock the websocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value = mock_ws

            await provider.connect()

            assert provider.is_connected
            # Setup message is sent, but not yet complete (requires setupComplete from server)
            mock_ws.send.assert_called_once()  # Setup message sent

    @pytest.mark.asyncio
    async def test_disconnect(self, test_options):
        """Test disconnecting from Gemini."""
        from unittest.mock import AsyncMock, patch
        provider = GeminiProvider(test_options)

        # Mock the websocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value = mock_ws

            await provider.connect()
            await provider.disconnect()

            assert not provider.is_connected
            assert not provider._response_active
            mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_audio(self, test_options):
        """Test sending audio data (format check)."""
        provider = GeminiProvider(test_options)
        audio_data = b"test_audio_data"

        # Should not raise error (even though websocket not connected)
        await provider.send_audio(audio_data)

    @pytest.mark.asyncio
    async def test_send_text(self, test_options):
        """Test sending text message."""
        provider = GeminiProvider(test_options)

        # Should not raise error
        await provider.send_text("Hello Gemini", role="user")


class TestGeminiEventSynthesis:
    """Test event synthesis - THE CRITICAL FEATURE."""

    def test_setup_complete(self, test_options):
        """Test setupComplete → session.created."""
        provider = GeminiProvider(test_options)

        gemini_message = {"setupComplete": {}}

        events = provider.normalize_incoming_event(gemini_message)

        assert len(events) == 1
        assert isinstance(events[0], SessionCreatedEvent)
        assert events[0].event_type == EventType.SESSION_CREATED
        assert events[0].provider == "gemini"

    def test_server_content_first_message(self, test_options):
        """Test first serverContent synthesizes response.created + deltas."""
        provider = GeminiProvider(test_options)

        # First serverContent with audio + text
        gemini_message = {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {"inlineData": {"data": "SGVsbG8=", "mimeType": "audio/pcm"}},
                        {"text": "Hello there"}
                    ]
                },
                "turnComplete": False
            }
        }

        events = provider.normalize_incoming_event(gemini_message)

        # Should synthesize: response.created + audio.delta + transcript.delta
        assert len(events) == 3
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], AudioDeltaEvent)
        assert isinstance(events[2], TranscriptDeltaEvent)

        # Verify response state
        assert provider._response_active
        assert provider._current_response_id is not None
        assert provider._accumulated_transcript == "Hello there"

    def test_server_content_subsequent_message(self, test_options):
        """Test subsequent serverContent only emits deltas."""
        provider = GeminiProvider(test_options)

        # First message
        provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Hello"}]},
                "turnComplete": False
            }
        })

        # Second message (should NOT create new response)
        events = provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": " there"}]},
                "turnComplete": False
            }
        })

        # Should only emit transcript.delta (no response.created)
        assert len(events) == 1
        assert isinstance(events[0], TranscriptDeltaEvent)
        assert events[0].delta == " there"

        # Accumulated transcript should preserve both
        assert provider._accumulated_transcript == "Hello there"

    def test_server_content_turn_complete(self, test_options):
        """Test turnComplete=True emits done events."""
        provider = GeminiProvider(test_options)

        # First message (starts response)
        provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Hello"}]},
                "turnComplete": False
            }
        })

        # Final message with turnComplete
        events = provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": " world!"}]},
                "turnComplete": True
            }
        })

        # Should emit: transcript.delta + audio.done + transcript.done + response.done
        assert len(events) == 4
        assert isinstance(events[0], TranscriptDeltaEvent)
        assert isinstance(events[1], AudioDoneEvent)
        assert isinstance(events[2], TranscriptDoneEvent)
        assert isinstance(events[3], ResponseDoneEvent)

        # transcript.done should have full accumulated text
        assert events[2].transcript == "Hello world!"

        # Response state should be reset
        assert not provider._response_active
        assert provider._accumulated_transcript == ""

    def test_input_audio_transcription(self, test_options):
        """Test inputAudioTranscription field synthesis."""
        provider = GeminiProvider(test_options)

        gemini_message = {
            "serverContent": {
                "inputAudioTranscription": "user said hello",
                "modelTurn": {"parts": []},
                "turnComplete": False
            }
        }

        events = provider.normalize_incoming_event(gemini_message)

        # Should emit InputTranscriptCompletedEvent
        assert len(events) >= 1
        assert isinstance(events[0], InputTranscriptCompletedEvent)
        assert events[0].transcript == "user said hello"
        assert events[0].provider == "gemini"

    def test_complete_conversation_sequence(self, test_options):
        """Test full conversation sequence with multiple messages."""
        provider = GeminiProvider(test_options)

        # Message 1: Input transcription
        events1 = provider.normalize_incoming_event({
            "serverContent": {
                "inputAudioTranscription": "Hello",
                "turnComplete": False
            }
        })
        assert len(events1) == 1
        assert isinstance(events1[0], InputTranscriptCompletedEvent)

        # Message 2: First response (start)
        events2 = provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Hi"}]},
                "turnComplete": False
            }
        })
        assert len(events2) == 2  # response.created + transcript.delta
        assert isinstance(events2[0], ResponseCreatedEvent)
        assert isinstance(events2[1], TranscriptDeltaEvent)

        # Message 3: Continue response
        events3 = provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": " there!"}]},
                "turnComplete": False
            }
        })
        assert len(events3) == 1  # Only transcript.delta
        assert isinstance(events3[0], TranscriptDeltaEvent)

        # Message 4: End response
        events4 = provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": []},
                "turnComplete": True
            }
        })
        assert len(events4) == 3  # audio.done + transcript.done + response.done
        assert isinstance(events4[0], AudioDoneEvent)
        assert isinstance(events4[1], TranscriptDoneEvent)
        assert isinstance(events4[2], ResponseDoneEvent)
        assert events4[1].transcript == "Hi there!"


class TestGeminiFunctionCalling:
    """Test function calling event synthesis."""

    def test_tool_call_synthesis(self, test_options):
        """Test toolCall → FunctionCallEvent."""
        provider = GeminiProvider(test_options)

        gemini_message = {
            "toolCall": {
                "functionCalls": [
                    {
                        "id": "call_123",
                        "name": "get_weather",
                        "args": {"location": "San Francisco"}
                    }
                ]
            }
        }

        events = provider.normalize_incoming_event(gemini_message)

        assert len(events) == 1
        assert isinstance(events[0], FunctionCallEvent)
        assert events[0].call_id == "call_123"
        assert events[0].function_name == "get_weather"
        assert events[0].provider == "gemini"
        assert "San Francisco" in events[0].arguments

    def test_multiple_function_calls(self, test_options):
        """Test multiple function calls in one message."""
        provider = GeminiProvider(test_options)

        gemini_message = {
            "toolCall": {
                "functionCalls": [
                    {"id": "call_1", "name": "func1", "args": {}},
                    {"id": "call_2", "name": "func2", "args": {}}
                ]
            }
        }

        events = provider.normalize_incoming_event(gemini_message)

        # Should emit one event per function call
        assert len(events) == 2
        assert all(isinstance(e, FunctionCallEvent) for e in events)
        assert events[0].call_id == "call_1"
        assert events[1].call_id == "call_2"

    def test_tool_call_cancellation(self, test_options):
        """Test toolCallCancellation (Gemini-specific)."""
        provider = GeminiProvider(test_options)

        gemini_message = {
            "toolCallCancellation": {
                "ids": ["call_123"]
            }
        }

        events = provider.normalize_incoming_event(gemini_message)

        # Should be ignored (empty list)
        assert len(events) == 0


class TestGeminiStateManagement:
    """Test state management during event synthesis."""

    def test_state_reset_on_disconnect(self, test_options):
        """Test that state resets on disconnect."""
        provider = GeminiProvider(test_options)

        # Start a response
        provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Hello"}]},
                "turnComplete": False
            }
        })

        assert provider._response_active
        assert provider._accumulated_transcript == "Hello"

        # Disconnect
        provider._reset_response_state()

        # State should be cleared
        assert not provider._response_active
        assert provider._accumulated_transcript == ""
        assert provider._current_response_id is None

    def test_state_reset_on_turn_complete(self, test_options):
        """Test that state resets on turnComplete."""
        provider = GeminiProvider(test_options)

        # Start and finish response
        provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Hello"}]},
                "turnComplete": False
            }
        })

        provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": []},
                "turnComplete": True
            }
        })

        # State should be reset
        assert not provider._response_active
        assert provider._accumulated_transcript == ""

    def test_multiple_responses_sequential(self, test_options):
        """Test multiple responses in sequence."""
        provider = GeminiProvider(test_options)

        # First response
        events1 = provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "First"}]},
                "turnComplete": True
            }
        })

        first_response_id = events1[0].response_id

        # Second response (should have different ID)
        events2 = provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Second"}]},
                "turnComplete": True
            }
        })

        second_response_id = events2[0].response_id

        assert first_response_id != second_response_id


class TestGeminiEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_server_content(self, test_options):
        """Test serverContent with no parts."""
        provider = GeminiProvider(test_options)

        gemini_message = {
            "serverContent": {
                "modelTurn": {"parts": []},
                "turnComplete": False
            }
        }

        events = provider.normalize_incoming_event(gemini_message)

        # Should return empty list (no parts to process)
        assert len(events) == 0

    def test_unknown_message_type(self, test_options):
        """Test unknown message type."""
        provider = GeminiProvider(test_options)

        gemini_message = {
            "unknownMessage": {"data": "something"}
        }

        events = provider.normalize_incoming_event(gemini_message)

        # Should return empty list
        assert len(events) == 0

    def test_audio_and_text_in_same_part(self, test_options):
        """Test serverContent with both audio and text."""
        provider = GeminiProvider(test_options)

        gemini_message = {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {"inlineData": {"data": "YXVkaW8=", "mimeType": "audio/pcm"}},
                        {"text": "transcript"}
                    ]
                },
                "turnComplete": False
            }
        }

        events = provider.normalize_incoming_event(gemini_message)

        # Should emit: response.created + audio.delta + transcript.delta
        assert len(events) == 3
        assert isinstance(events[0], ResponseCreatedEvent)
        assert isinstance(events[1], AudioDeltaEvent)
        assert isinstance(events[2], TranscriptDeltaEvent)


class TestGeminiProviderOperations:
    """Test provider operations."""

    @pytest.mark.asyncio
    async def test_update_session(self, test_options):
        """Test updating session configuration."""
        provider = GeminiProvider(test_options)

        new_options = RealtimeAIOptions(
            api_key="test_gemini_key",
            model="gemini-2.0-flash-exp",
            modalities=["audio"],
            instructions="Updated instructions"
        )

        await provider.update_session(new_options)

        assert provider.options == new_options

    @pytest.mark.asyncio
    async def test_generate_response(self, test_options):
        """Test generate_response (no-op for Gemini)."""
        provider = GeminiProvider(test_options)

        # Should not raise error (Gemini doesn't need explicit response generation)
        await provider.generate_response(commit_audio=True)

    @pytest.mark.asyncio
    async def test_cancel_response(self, test_options):
        """Test cancelling response."""
        provider = GeminiProvider(test_options)

        # Start a response
        provider.normalize_incoming_event({
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Hello"}]},
                "turnComplete": False
            }
        })

        assert provider._response_active

        # Cancel
        await provider.cancel_response()

        # State should be reset
        assert not provider._response_active

    @pytest.mark.asyncio
    async def test_send_function_result(self, test_options):
        """Test sending function result."""
        provider = GeminiProvider(test_options)

        # Should not raise error
        await provider.send_function_result("call_123", '{"result": "success"}')

    @pytest.mark.asyncio
    async def test_truncate_response(self, test_options):
        """Test truncate_response (no-op for Gemini)."""
        provider = GeminiProvider(test_options)

        # Should not raise error (Gemini doesn't support truncation)
        await provider.truncate_response("item_1", 0, 5000)

    @pytest.mark.asyncio
    async def test_clear_input_audio_buffer(self, test_options):
        """Test clear_input_audio_buffer (no-op for Gemini)."""
        provider = GeminiProvider(test_options)

        # Should not raise error (Gemini doesn't have explicit buffer management)
        await provider.clear_input_audio_buffer()
