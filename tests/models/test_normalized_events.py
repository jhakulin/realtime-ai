"""Tests for normalized event model."""

import pytest
import time
from realtime_ai.models.normalized_events import (
    EventType,
    NormalizedEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    AudioDeltaEvent,
    AudioDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptDeltaEvent,
    TranscriptDoneEvent,
    InputTranscriptCompletedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    FunctionCallEvent,
    ErrorEvent,
    RateLimit,
    RateLimitsUpdatedEvent,
    AudioBufferCommittedEvent,
    ConversationItemCreatedEvent,
)


class TestEventType:
    """Test EventType enum."""

    def test_event_type_values(self):
        """Test that event types have correct string values."""
        assert EventType.SESSION_CREATED.value == "session.created"
        assert EventType.AUDIO_DELTA.value == "audio.delta"
        assert EventType.FUNCTION_CALL.value == "function.call"
        assert EventType.ERROR.value == "error"

    def test_event_type_membership(self):
        """Test that all expected event types exist."""
        expected_types = {
            "SESSION_CREATED",
            "SESSION_UPDATED",
            "AUDIO_INPUT",
            "SPEECH_STARTED",
            "SPEECH_STOPPED",
            "INPUT_TRANSCRIPT_COMPLETED",
            "TRANSCRIPT_DELTA",
            "TRANSCRIPT_DONE",
            "RESPONSE_CREATED",
            "RESPONSE_CONTENT_PART_ADDED",
            "RESPONSE_OUTPUT_ITEM_ADDED",
            "RESPONSE_OUTPUT_ITEM_DONE",
            "RESPONSE_DONE",
            "AUDIO_DELTA",
            "AUDIO_DONE",
            "FUNCTION_CALL",
            "FUNCTION_RESPONSE",
            "ERROR",
            "AUDIO_BUFFER_COMMITTED",
            "AUDIO_BUFFER_CLEARED",
            "RATE_LIMITS_UPDATED",
            "CONVERSATION_ITEM_CREATED",
        }

        actual_types = {e.name for e in EventType}
        assert expected_types == actual_types


class TestNormalizedEvent:
    """Test base NormalizedEvent class."""

    def test_create_base_event(self):
        """Test creating a base normalized event."""
        event = NormalizedEvent(
            event_id="evt_123",
            event_type=EventType.SESSION_CREATED,
            timestamp=time.time(),
            provider="openai"
        )

        assert event.event_id == "evt_123"
        assert event.event_type == EventType.SESSION_CREATED
        assert event.provider == "openai"
        assert event.raw_event is None

    def test_event_with_raw_data(self):
        """Test event with raw provider data."""
        raw_data = {"type": "session.created", "session": {"id": "sess_123"}}
        event = NormalizedEvent(
            event_id="evt_123",
            event_type=EventType.SESSION_CREATED,
            timestamp=time.time(),
            provider="openai",
            raw_event=raw_data
        )

        assert event.raw_event == raw_data

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = NormalizedEvent(
            event_id="evt_123",
            event_type=EventType.AUDIO_DELTA,
            timestamp=1234567890.0,
            provider="gemini"
        )

        event_dict = event.to_dict()

        assert event_dict["event_id"] == "evt_123"
        assert event_dict["event_type"] == "audio.delta"
        assert event_dict["timestamp"] == 1234567890.0
        assert event_dict["provider"] == "gemini"
        assert event_dict["raw_event"] is None


class TestSessionEvents:
    """Test session-related events."""

    def test_session_created_event(self):
        """Test SessionCreatedEvent."""
        event = SessionCreatedEvent(
            event_id="evt_123",
            event_type=EventType.SESSION_CREATED,
            timestamp=time.time(),
            provider="openai",
            session_id="sess_abc",
            config={"voice": "alloy", "modalities": ["text", "audio"]}
        )

        assert event.session_id == "sess_abc"
        assert event.config["voice"] == "alloy"
        assert event.event_type == EventType.SESSION_CREATED

    def test_session_updated_event(self):
        """Test SessionUpdatedEvent."""
        event = SessionUpdatedEvent(
            event_id="evt_456",
            event_type=EventType.SESSION_UPDATED,
            timestamp=time.time(),
            provider="gemini",
            session_id="sess_xyz",
            config={"temperature": 0.8}
        )

        assert event.session_id == "sess_xyz"
        assert event.config["temperature"] == 0.8


class TestAudioEvents:
    """Test audio-related events."""

    def test_audio_delta_event(self):
        """Test AudioDeltaEvent."""
        event = AudioDeltaEvent(
            event_id="evt_123",
            event_type=EventType.AUDIO_DELTA,
            timestamp=time.time(),
            provider="openai",
            response_id="resp_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            delta="SGVsbG8gd29ybGQ="  # Base64
        )

        assert event.response_id == "resp_1"
        assert event.item_id == "item_1"
        assert event.delta == "SGVsbG8gd29ybGQ="
        assert event.output_index == 0
        assert event.content_index == 0

    def test_audio_done_event(self):
        """Test AudioDoneEvent."""
        event = AudioDoneEvent(
            event_id="evt_456",
            event_type=EventType.AUDIO_DONE,
            timestamp=time.time(),
            provider="grok",
            response_id="resp_2",
            item_id="item_2",
            output_index=0,
            content_index=0
        )

        assert event.response_id == "resp_2"
        assert event.item_id == "item_2"


class TestSpeechEvents:
    """Test speech detection (VAD) events."""

    def test_speech_started_event(self):
        """Test SpeechStartedEvent."""
        event = SpeechStartedEvent(
            event_id="evt_123",
            event_type=EventType.SPEECH_STARTED,
            timestamp=time.time(),
            provider="openai",
            audio_start_ms=1000,
            item_id="item_1"
        )

        assert event.audio_start_ms == 1000
        assert event.item_id == "item_1"

    def test_speech_stopped_event(self):
        """Test SpeechStoppedEvent."""
        event = SpeechStoppedEvent(
            event_id="evt_456",
            event_type=EventType.SPEECH_STOPPED,
            timestamp=time.time(),
            provider="openai",
            audio_end_ms=5000,
            item_id="item_1"
        )

        assert event.audio_end_ms == 5000
        assert event.item_id == "item_1"


class TestTranscriptEvents:
    """Test transcription events."""

    def test_input_transcript_completed(self):
        """Test InputTranscriptCompletedEvent."""
        event = InputTranscriptCompletedEvent(
            event_id="evt_123",
            event_type=EventType.INPUT_TRANSCRIPT_COMPLETED,
            timestamp=time.time(),
            provider="openai",
            item_id="item_1",
            content_index=0,
            transcript="Hello, how are you?"
        )

        assert event.item_id == "item_1"
        assert event.transcript == "Hello, how are you?"
        assert event.content_index == 0

    def test_transcript_delta_event(self):
        """Test TranscriptDeltaEvent."""
        event = TranscriptDeltaEvent(
            event_id="evt_456",
            event_type=EventType.TRANSCRIPT_DELTA,
            timestamp=time.time(),
            provider="gemini",
            response_id="resp_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            delta="Hello "
        )

        assert event.delta == "Hello "
        assert event.response_id == "resp_1"

    def test_transcript_done_event(self):
        """Test TranscriptDoneEvent."""
        event = TranscriptDoneEvent(
            event_id="evt_789",
            event_type=EventType.TRANSCRIPT_DONE,
            timestamp=time.time(),
            provider="nova",
            response_id="resp_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            transcript="Hello world"
        )

        assert event.transcript == "Hello world"
        assert event.response_id == "resp_1"


class TestResponseEvents:
    """Test response-related events."""

    def test_response_created_event(self):
        """Test ResponseCreatedEvent."""
        event = ResponseCreatedEvent(
            event_id="evt_123",
            event_type=EventType.RESPONSE_CREATED,
            timestamp=time.time(),
            provider="openai",
            response_id="resp_1",
            config={"modalities": ["audio", "text"]}
        )

        assert event.response_id == "resp_1"
        assert event.config["modalities"] == ["audio", "text"]

    def test_response_done_event(self):
        """Test ResponseDoneEvent."""
        event = ResponseDoneEvent(
            event_id="evt_456",
            event_type=EventType.RESPONSE_DONE,
            timestamp=time.time(),
            provider="gemini",
            response_id="resp_1",
            status="completed",
            usage={"total_tokens": 150}
        )

        assert event.response_id == "resp_1"
        assert event.status == "completed"
        assert event.usage["total_tokens"] == 150


class TestFunctionEvents:
    """Test function calling events."""

    def test_function_call_event(self):
        """Test FunctionCallEvent."""
        event = FunctionCallEvent(
            event_id="evt_123",
            event_type=EventType.FUNCTION_CALL,
            timestamp=time.time(),
            provider="openai",
            call_id="call_abc",
            function_name="get_weather",
            arguments='{"location": "San Francisco"}',
            response_id="resp_1",
            item_id="item_1"
        )

        assert event.call_id == "call_abc"
        assert event.function_name == "get_weather"
        assert event.arguments == '{"location": "San Francisco"}'
        assert event.response_id == "resp_1"


class TestErrorEvent:
    """Test error events."""

    def test_error_event(self):
        """Test ErrorEvent."""
        event = ErrorEvent(
            event_id="evt_123",
            event_type=EventType.ERROR,
            timestamp=time.time(),
            provider="openai",
            error_code="rate_limit_exceeded",
            error_message="Rate limit exceeded",
            error_type="rate_limit",
            details={"retry_after": 60}
        )

        assert event.error_code == "rate_limit_exceeded"
        assert event.error_message == "Rate limit exceeded"
        assert event.error_type == "rate_limit"
        assert event.details["retry_after"] == 60


class TestRateLimitEvents:
    """Test rate limiting events."""

    def test_rate_limit_dataclass(self):
        """Test RateLimit dataclass."""
        rate_limit = RateLimit(
            name="requests",
            limit=1000,
            remaining=750,
            reset_seconds=3600
        )

        assert rate_limit.name == "requests"
        assert rate_limit.limit == 1000
        assert rate_limit.remaining == 750
        assert rate_limit.reset_seconds == 3600

    def test_rate_limits_updated_event(self):
        """Test RateLimitsUpdatedEvent."""
        rate_limits = [
            RateLimit("requests", 1000, 750, 3600),
            RateLimit("tokens", 100000, 50000, 3600)
        ]

        event = RateLimitsUpdatedEvent(
            event_id="evt_123",
            event_type=EventType.RATE_LIMITS_UPDATED,
            timestamp=time.time(),
            provider="openai",
            rate_limits=rate_limits
        )

        assert len(event.rate_limits) == 2
        assert event.rate_limits[0].name == "requests"
        assert event.rate_limits[1].name == "tokens"


class TestBufferEvents:
    """Test audio buffer events."""

    def test_audio_buffer_committed(self):
        """Test AudioBufferCommittedEvent."""
        event = AudioBufferCommittedEvent(
            event_id="evt_123",
            event_type=EventType.AUDIO_BUFFER_COMMITTED,
            timestamp=time.time(),
            provider="openai",
            previous_item_id="item_0",
            item_id="item_1"
        )

        assert event.previous_item_id == "item_0"
        assert event.item_id == "item_1"


class TestConversationEvents:
    """Test conversation events."""

    def test_conversation_item_created(self):
        """Test ConversationItemCreatedEvent."""
        event = ConversationItemCreatedEvent(
            event_id="evt_123",
            event_type=EventType.CONVERSATION_ITEM_CREATED,
            timestamp=time.time(),
            provider="openai",
            previous_item_id="item_0",
            item={"type": "message", "role": "user", "content": "Hello"}
        )

        assert event.previous_item_id == "item_0"
        assert event.item["type"] == "message"
        assert event.item["role"] == "user"


class TestEventDefaults:
    """Test that events have sensible defaults."""

    def test_audio_delta_defaults(self):
        """Test AudioDeltaEvent with minimal arguments."""
        event = AudioDeltaEvent(
            event_id="evt_123",
            event_type=EventType.AUDIO_DELTA,
            timestamp=time.time(),
            provider="test"
        )

        assert event.response_id == ""
        assert event.item_id == ""
        assert event.output_index == 0
        assert event.content_index == 0
        assert event.delta == ""

    def test_session_created_defaults(self):
        """Test SessionCreatedEvent with minimal arguments."""
        event = SessionCreatedEvent(
            event_id="evt_123",
            event_type=EventType.SESSION_CREATED,
            timestamp=time.time(),
            provider="test"
        )

        assert event.session_id == ""
        assert event.config == {}
