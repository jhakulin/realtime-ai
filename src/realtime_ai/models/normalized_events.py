"""
Normalized event model for multi-provider support.

This module defines provider-agnostic event types that all realtime AI providers
translate to/from. This enables a unified event handling interface regardless of
the underlying provider (OpenAI, Gemini, Grok, Nova).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class EventType(Enum):
    """Normalized event types across all providers."""

    # Session Events
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"

    # Audio Input Events
    AUDIO_INPUT = "audio.input"

    # Speech Detection Events (VAD)
    SPEECH_STARTED = "speech.started"
    SPEECH_STOPPED = "speech.stopped"

    # Transcription Events
    INPUT_TRANSCRIPT_DELTA = "input.transcript.delta"
    INPUT_TRANSCRIPT_COMPLETED = "input.transcript.completed"
    TRANSCRIPT_DELTA = "transcript.delta"
    TRANSCRIPT_DONE = "transcript.done"

    # Response Events
    RESPONSE_CREATED = "response.created"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_DONE = "response.done"

    # Audio Output Events
    AUDIO_DELTA = "audio.delta"
    AUDIO_DONE = "audio.done"

    # Function Calling Events
    FUNCTION_CALL = "function.call"
    FUNCTION_RESPONSE = "function.response"

    # Error Events
    ERROR = "error"

    # Buffer Events
    AUDIO_BUFFER_COMMITTED = "audio.buffer.committed"
    AUDIO_BUFFER_CLEARED = "audio.buffer.cleared"

    # Rate Limiting
    RATE_LIMITS_UPDATED = "rate_limits.updated"

    # Conversation Events
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"


@dataclass
class NormalizedEvent:
    """
    Base class for all normalized events.

    All events across all providers inherit from this base class.
    """
    event_id: str
    event_type: EventType
    timestamp: float
    provider: str  # 'openai', 'gemini', 'grok', 'nova'
    raw_event: Optional[Dict[str, Any]] = None  # Original provider event (for debugging)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "raw_event": self.raw_event
        }


# ============================================================================
# Session Events
# ============================================================================

@dataclass
class SessionCreatedEvent(NormalizedEvent):
    """Session initialized with provider."""
    session_id: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionUpdatedEvent(NormalizedEvent):
    """Session configuration updated."""
    session_id: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Audio Input Events
# ============================================================================

@dataclass
class AudioInputEvent(NormalizedEvent):
    """Audio data sent to provider."""
    audio_data: bytes = b""


@dataclass
class AudioBufferCommittedEvent(NormalizedEvent):
    """Audio buffer committed for processing."""
    previous_item_id: str = ""
    item_id: str = ""


@dataclass
class AudioBufferClearedEvent(NormalizedEvent):
    """Audio buffer cleared."""
    pass


# ============================================================================
# Speech Detection Events (VAD)
# ============================================================================

@dataclass
class SpeechStartedEvent(NormalizedEvent):
    """Voice activity detected (user started speaking)."""
    audio_start_ms: int = 0
    item_id: str = ""


@dataclass
class SpeechStoppedEvent(NormalizedEvent):
    """Voice activity ended (user stopped speaking)."""
    audio_end_ms: int = 0
    item_id: str = ""


# ============================================================================
# Transcription Events
# ============================================================================

@dataclass
class InputTranscriptDeltaEvent(NormalizedEvent):
    """Streaming input audio transcription chunk."""
    item_id: str = ""
    content_index: int = 0
    delta: str = ""


@dataclass
class InputTranscriptCompletedEvent(NormalizedEvent):
    """Input audio transcription completed."""
    item_id: str = ""
    content_index: int = 0
    transcript: str = ""


@dataclass
class TranscriptDeltaEvent(NormalizedEvent):
    """Streaming output transcript chunk."""
    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


@dataclass
class TranscriptDoneEvent(NormalizedEvent):
    """Output transcript completed."""
    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    transcript: str = ""


# ============================================================================
# Response Events
# ============================================================================

@dataclass
class ResponseCreatedEvent(NormalizedEvent):
    """Response generation started."""
    response_id: str = ""
    config: Optional[Dict[str, Any]] = None


@dataclass
class ResponseContentPartAddedEvent(NormalizedEvent):
    """Content part added to response."""
    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    part: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseContentPartDoneEvent(NormalizedEvent):
    """Content part completed."""
    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    part: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseOutputItemAddedEvent(NormalizedEvent):
    """Output item added to response."""
    response_id: str = ""
    output_index: int = 0
    item: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseOutputItemDoneEvent(NormalizedEvent):
    """Output item completed."""
    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    item: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseDoneEvent(NormalizedEvent):
    """Response generation completed."""
    response_id: str = ""
    status: str = "completed"
    usage: Optional[Dict[str, Any]] = None


# ============================================================================
# Audio Output Events
# ============================================================================

@dataclass
class AudioDeltaEvent(NormalizedEvent):
    """Streaming audio output chunk."""
    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    delta: str = ""  # Base64-encoded audio chunk


@dataclass
class AudioDoneEvent(NormalizedEvent):
    """Audio output stream completed."""
    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0


# ============================================================================
# Function Calling Events
# ============================================================================

@dataclass
class FunctionCallEvent(NormalizedEvent):
    """Provider requesting function execution."""
    call_id: str = ""
    function_name: str = ""
    arguments: str = ""  # JSON string of arguments
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    output_index: int = 0


@dataclass
class FunctionResponseEvent(NormalizedEvent):
    """Function execution result."""
    call_id: str = ""
    function_name: str = ""
    result: str = ""  # JSON string of result


# ============================================================================
# Error Events
# ============================================================================

@dataclass
class ErrorEvent(NormalizedEvent):
    """Error occurred during processing."""
    error_code: str = ""
    error_message: str = ""
    error_type: str = ""  # 'server_error', 'rate_limit', 'invalid_request', etc.
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# Rate Limiting Events
# ============================================================================

@dataclass
class RateLimit:
    """Rate limit information."""
    name: str
    limit: int
    remaining: int
    reset_seconds: int


@dataclass
class RateLimitsUpdatedEvent(NormalizedEvent):
    """Rate limit information updated."""
    rate_limits: List[RateLimit] = field(default_factory=list)


# ============================================================================
# Conversation Events
# ============================================================================

@dataclass
class ConversationItemCreatedEvent(NormalizedEvent):
    """Conversation item created."""
    previous_item_id: str = ""
    item: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationItemDeletedEvent(NormalizedEvent):
    """Conversation item deleted."""
    item_id: str = ""
