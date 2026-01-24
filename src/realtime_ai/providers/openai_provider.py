"""
OpenAI provider implementation.

Wraps the existing RealtimeAIServiceManager to provide OpenAI Realtime API
support through the provider interface.
"""

import logging
import time
import uuid
from typing import AsyncIterator, List

from realtime_ai.aio.realtime_ai_service_manager import RealtimeAIServiceManager
from realtime_ai.models.normalized_events import (
    AudioBufferClearedEvent,
    AudioBufferCommittedEvent,
    AudioDeltaEvent,
    AudioDoneEvent,
    ConversationItemCreatedEvent,
    ConversationItemDeletedEvent,
    ErrorEvent,
    EventType,
    FunctionCallEvent,
    InputTranscriptCompletedEvent,
    InputTranscriptDeltaEvent,
    NormalizedEvent,
    RateLimit,
    RateLimitsUpdatedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptDeltaEvent,
    TranscriptDoneEvent,
)
from realtime_ai.models.realtime_ai_events import EventBase
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    Provider implementation for OpenAI Realtime API.

    This wraps the existing RealtimeAIServiceManager to maintain backward
    compatibility while implementing the new provider interface.
    """

    def __init__(self, options: RealtimeAIOptions):
        super().__init__(options)
        self._service_manager = RealtimeAIServiceManager(options)

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"

    # ============================================================================
    # Connection Management
    # ============================================================================

    async def connect(self) -> None:
        """Connect to OpenAI Realtime API."""
        logger.info(
            f"OpenAIProvider: Connecting to OpenAI Realtime API (model={self.options.model})"
        )
        try:
            await self._service_manager.connect()
            self._is_connected = True
            logger.info("OpenAIProvider: Successfully connected to OpenAI Realtime API")
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to connect - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    async def disconnect(self) -> None:
        """Disconnect from OpenAI Realtime API."""
        logger.info("OpenAIProvider: Disconnecting from OpenAI Realtime API")
        try:
            await self._service_manager.disconnect()
            self._is_connected = False
            logger.info(
                "OpenAIProvider: Successfully disconnected from OpenAI Realtime API"
            )
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Error during disconnect - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            self._is_connected = False
            raise

    # ============================================================================
    # Audio Operations
    # ============================================================================

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to OpenAI."""
        # OpenAI expects audio to be sent through the service manager's WebSocket
        # The existing implementation handles this through AudioStreamManager
        # For now, we'll send it as an input_audio_buffer.append event
        import base64

        logger.debug(
            f"OpenAIProvider: Sending audio chunk (size={len(audio_data)} bytes)"
        )
        try:
            event = {
                "event_id": self._generate_event_id(),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode("utf-8"),
            }
            await self._service_manager.send_event(event)
            logger.debug(
                f"OpenAIProvider: Successfully sent audio chunk (size={len(audio_data)} bytes)"
            )
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to send audio - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # ============================================================================
    # Text Operations
    # ============================================================================

    async def send_text(self, text: str, role: str = "user") -> None:
        """Send text message to OpenAI."""
        logger.info(
            f"OpenAIProvider: Sending text message (role={role}, length={len(text)} chars)"
        )
        logger.debug(
            f"OpenAIProvider: Text content: {text[:100]}{'...' if len(text) > 100 else ''}"
        )

        try:
            event = {
                "event_id": self._generate_event_id(),
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": "text" if role == "assistant" else "input_text",
                            "text": text,
                        }
                    ],
                },
            }
            await self._service_manager.send_event(event)
            logger.info(f"OpenAIProvider: Successfully sent text message (role={role})")
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to send text - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # ============================================================================
    # Image Operations
    # ============================================================================

    async def send_image(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> None:
        """
        Send image to OpenAI Realtime API.

        Images are sent via conversation.item.create with input_image content type.

        Args:
            image_data: Raw image bytes (PNG, JPEG, WebP, GIF)
            image_format: Image format ('png', 'jpeg', 'webp', 'gif')
        """
        import base64

        logger.info(
            f"OpenAIProvider: Sending image (size={len(image_data)} bytes, format={image_format})"
        )

        try:
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            data_url = f"data:image/{image_format};base64,{encoded_image}"

            event = {
                "event_id": self._generate_event_id(),
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": data_url,
                        }
                    ],
                },
            }
            await self._service_manager.send_event(event)
            logger.info(
                f"OpenAIProvider: Successfully sent image (size={len(image_data)} bytes)"
            )
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to send image - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # ============================================================================
    # Session Management
    # ============================================================================

    async def update_session(self, options: RealtimeAIOptions) -> None:
        """Update session configuration."""
        logger.info(
            f"OpenAIProvider: Updating session configuration (model={options.model})"
        )
        try:
            await self._service_manager.update_session(options)
            self.options = options
            logger.info("OpenAIProvider: Successfully updated session configuration")
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to update session - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # ============================================================================
    # Response Generation
    # ============================================================================

    async def generate_response(self, commit_audio: bool = True) -> None:
        """Request OpenAI to generate a response."""
        logger.info(
            f"OpenAIProvider: Generating response (commit_audio={commit_audio})"
        )
        try:
            if commit_audio:
                commit_event = {
                    "event_id": self._generate_event_id(),
                    "type": "input_audio_buffer.commit",
                }
                await self._service_manager.send_event(commit_event)
                logger.debug("OpenAIProvider: Committed audio buffer")

            response_create_event = {
                "event_id": self._generate_event_id(),
                "type": "response.create",
                "response": {"modalities": self.options.modalities},
            }
            await self._service_manager.send_event(response_create_event)
            logger.info("OpenAIProvider: Successfully requested response generation")
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to generate response - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    async def cancel_response(self) -> None:
        """Cancel ongoing response generation."""
        logger.info("OpenAIProvider: Cancelling ongoing response")
        try:
            cancel_event = {
                "event_id": self._generate_event_id(),
                "type": "response.cancel",
            }
            await self._service_manager.send_event(cancel_event)
            await self._service_manager.clear_event_queue()
            logger.info("OpenAIProvider: Successfully cancelled response")
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to cancel response - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # ============================================================================
    # Function Calling
    # ============================================================================

    async def send_function_result(self, call_id: str, result: str) -> None:
        """Send function call result back to OpenAI."""
        logger.info(
            f"OpenAIProvider: Sending function result (call_id={call_id}, result_length={len(result)} chars)"
        )
        logger.debug(
            f"OpenAIProvider: Function result content: {result[:200]}{'...' if len(result) > 200 else ''}"
        )

        try:
            # Create the function call output event
            item_create_event = {
                "event_id": self._generate_event_id(),
                "type": "conversation.item.create",
                "item": {
                    "id": str(uuid.uuid4()).replace("-", ""),
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            }
            await self._service_manager.send_event(item_create_event)

            # Trigger response generation
            response_event = {
                "event_id": self._generate_event_id(),
                "type": "response.create",
                "response": {"modalities": self.options.modalities},
            }
            await self._service_manager.send_event(response_event)
            logger.info(
                f"OpenAIProvider: Successfully sent function result (call_id={call_id})"
            )
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to send function result - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # ============================================================================
    # Event Streaming
    # ============================================================================

    async def receive_events(self) -> AsyncIterator[NormalizedEvent]:
        """
        Async generator yielding normalized events from OpenAI.

        This continuously polls the service manager's event queue and
        normalizes OpenAI events to the generic event format.
        """
        while self._is_connected:
            try:
                # Get next event from service manager
                openai_event = await self._service_manager.get_next_event()

                if openai_event:
                    # Normalize OpenAI event to generic format
                    normalized_events = self.normalize_incoming_event(
                        vars(openai_event)
                    )

                    # Yield each normalized event
                    for event in normalized_events:
                        yield event
            except Exception as e:
                logger.error(
                    f"OpenAIProvider: Error receiving events - {type(e).__name__}: {str(e)}",
                    exc_info=True,
                )
                # Don't break the loop, continue receiving events

    # ============================================================================
    # Event Normalization
    # ============================================================================

    def normalize_incoming_event(self, raw_event: dict) -> List[NormalizedEvent]:
        """
        Convert OpenAI event to normalized event(s).

        OpenAI has a 1:1 mapping - each OpenAI event maps to one normalized event.
        """
        event_type = raw_event.get("type")
        event_id = raw_event.get("event_id", self._generate_event_id())
        timestamp = time.time()

        # Session Events
        if event_type == "session.created":
            return [
                SessionCreatedEvent(
                    event_id=event_id,
                    event_type=EventType.SESSION_CREATED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    session_id=raw_event.get("session", {}).get("id", ""),
                    config=raw_event.get("session", {}),
                )
            ]

        elif event_type == "session.updated":
            return [
                SessionUpdatedEvent(
                    event_id=event_id,
                    event_type=EventType.SESSION_UPDATED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    session_id=raw_event.get("session", {}).get("id", ""),
                    config=raw_event.get("session", {}),
                )
            ]

        # Speech Detection Events
        elif event_type == "input_audio_buffer.speech_started":
            return [
                SpeechStartedEvent(
                    event_id=event_id,
                    event_type=EventType.SPEECH_STARTED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    audio_start_ms=0,  # OpenAI doesn't provide this
                    item_id=raw_event.get("item_id", ""),
                )
            ]

        elif event_type == "input_audio_buffer.speech_stopped":
            return [
                SpeechStoppedEvent(
                    event_id=event_id,
                    event_type=EventType.SPEECH_STOPPED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    audio_end_ms=raw_event.get("audio_end_ms", 0),
                    item_id=raw_event.get("item_id", ""),
                )
            ]

        # Audio Buffer Events
        elif event_type == "input_audio_buffer.committed":
            return [
                AudioBufferCommittedEvent(
                    event_id=event_id,
                    event_type=EventType.AUDIO_BUFFER_COMMITTED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    previous_item_id=raw_event.get("previous_item_id", ""),
                    item_id=raw_event.get("item_id", ""),
                )
            ]

        elif event_type == "input_audio_buffer.cleared":
            return [
                AudioBufferClearedEvent(
                    event_id=event_id,
                    event_type=EventType.AUDIO_BUFFER_CLEARED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                )
            ]

        # Transcription Events
        elif event_type == "conversation.item.input_audio_transcription.delta":
            return [
                InputTranscriptDeltaEvent(
                    event_id=event_id,
                    event_type=EventType.INPUT_TRANSCRIPT_DELTA,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    item_id=raw_event.get("item_id", ""),
                    content_index=raw_event.get("content_index", 0),
                    delta=raw_event.get("delta", ""),
                )
            ]

        elif event_type == "conversation.item.input_audio_transcription.completed":
            return [
                InputTranscriptCompletedEvent(
                    event_id=event_id,
                    event_type=EventType.INPUT_TRANSCRIPT_COMPLETED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    item_id=raw_event.get("item_id", ""),
                    content_index=raw_event.get("content_index", 0),
                    transcript=raw_event.get("transcript", ""),
                )
            ]

        elif event_type == "response.audio_transcript.delta":
            return [
                TranscriptDeltaEvent(
                    event_id=event_id,
                    event_type=EventType.TRANSCRIPT_DELTA,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    item_id=raw_event.get("item_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    content_index=raw_event.get("content_index", 0),
                    delta=raw_event.get("delta", ""),
                )
            ]

        elif event_type == "response.audio_transcript.done":
            return [
                TranscriptDoneEvent(
                    event_id=event_id,
                    event_type=EventType.TRANSCRIPT_DONE,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    item_id=raw_event.get("item_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    content_index=raw_event.get("content_index", 0),
                    transcript=raw_event.get("transcript", ""),
                )
            ]

        # Response Events
        elif event_type == "response.created":
            return [
                ResponseCreatedEvent(
                    event_id=event_id,
                    event_type=EventType.RESPONSE_CREATED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response", {}).get("id", ""),
                    config=raw_event.get("response", {}),
                )
            ]

        elif event_type == "response.content_part.added":
            return [
                ResponseContentPartAddedEvent(
                    event_id=event_id,
                    event_type=EventType.RESPONSE_CONTENT_PART_ADDED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    item_id=raw_event.get("item_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    content_index=raw_event.get("content_index", 0),
                    part=raw_event.get("part", {}),
                )
            ]

        elif event_type == "response.content_part.done":
            return [
                ResponseContentPartDoneEvent(
                    event_id=event_id,
                    event_type=EventType.RESPONSE_CONTENT_PART_DONE,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    item_id=raw_event.get("item_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    content_index=raw_event.get("content_index", 0),
                    part=raw_event.get("part", {}),
                )
            ]

        elif event_type == "response.output_item.added":
            return [
                ResponseOutputItemAddedEvent(
                    event_id=event_id,
                    event_type=EventType.RESPONSE_OUTPUT_ITEM_ADDED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    item=raw_event.get("item", {}),
                )
            ]

        elif event_type == "response.output_item.done":
            return [
                ResponseOutputItemDoneEvent(
                    event_id=event_id,
                    event_type=EventType.RESPONSE_OUTPUT_ITEM_DONE,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    item_id=raw_event.get("item_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    item=raw_event.get("item", {}),
                )
            ]

        elif event_type == "response.done":
            return [
                ResponseDoneEvent(
                    event_id=event_id,
                    event_type=EventType.RESPONSE_DONE,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response", {}).get("id", ""),
                    status=raw_event.get("response", {}).get("status", "completed"),
                    usage=raw_event.get("response", {}).get("usage"),
                )
            ]

        # Audio Output Events
        elif event_type == "response.audio.delta":
            return [
                AudioDeltaEvent(
                    event_id=event_id,
                    event_type=EventType.AUDIO_DELTA,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    item_id=raw_event.get("item_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    content_index=raw_event.get("content_index", 0),
                    delta=raw_event.get("delta", ""),
                )
            ]

        elif event_type == "response.audio.done":
            return [
                AudioDoneEvent(
                    event_id=event_id,
                    event_type=EventType.AUDIO_DONE,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    response_id=raw_event.get("response_id", ""),
                    item_id=raw_event.get("item_id", ""),
                    output_index=raw_event.get("output_index", 0),
                    content_index=raw_event.get("content_index", 0),
                )
            ]

        # Function Calling Events
        elif event_type == "response.function_call_arguments.delta":
            # We skip delta events and only process done
            return []

        elif event_type == "response.function_call_arguments.done":
            return [
                FunctionCallEvent(
                    event_id=event_id,
                    event_type=EventType.FUNCTION_CALL,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    call_id=raw_event.get("call_id", ""),
                    function_name=raw_event.get("name", ""),
                    arguments=raw_event.get("arguments", ""),
                    response_id=raw_event.get("response_id"),
                    item_id=raw_event.get("item_id"),
                    output_index=raw_event.get("output_index", 0),
                )
            ]

        # Conversation Events
        elif event_type == "conversation.item.created":
            return [
                ConversationItemCreatedEvent(
                    event_id=event_id,
                    event_type=EventType.CONVERSATION_ITEM_CREATED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    previous_item_id=raw_event.get("previous_item_id", ""),
                    item=raw_event.get("item", {}),
                )
            ]

        elif event_type == "conversation.item.deleted":
            return [
                ConversationItemDeletedEvent(
                    event_id=event_id,
                    event_type=EventType.CONVERSATION_ITEM_DELETED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    item_id=raw_event.get("item_id", ""),
                )
            ]

        # Error Events
        elif event_type == "error":
            error_data = raw_event.get("error", {})
            # Handle both dict and ErrorDetails object
            if hasattr(error_data, "code"):
                error_code = error_data.code
                error_message = error_data.message
                error_type = error_data.type
            else:
                error_code = error_data.get("code", "")
                error_message = error_data.get("message", "")
                error_type = error_data.get("type", "")

            return [
                ErrorEvent(
                    event_id=event_id,
                    event_type=EventType.ERROR,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    error_code=error_code,
                    error_message=error_message,
                    error_type=error_type,
                    details=error_data
                    if isinstance(error_data, dict)
                    else vars(error_data),
                )
            ]

        # Rate Limit Events
        elif event_type == "rate_limits.updated":
            rate_limits_data = raw_event.get("rate_limits", [])
            rate_limits = [
                RateLimit(
                    name=rl.name if hasattr(rl, "name") else rl.get("name", ""),
                    limit=rl.limit if hasattr(rl, "limit") else rl.get("limit", 0),
                    remaining=rl.remaining
                    if hasattr(rl, "remaining")
                    else rl.get("remaining", 0),
                    reset_seconds=rl.reset_seconds
                    if hasattr(rl, "reset_seconds")
                    else rl.get("reset_seconds", 0),
                )
                for rl in rate_limits_data
            ]
            return [
                RateLimitsUpdatedEvent(
                    event_id=event_id,
                    event_type=EventType.RATE_LIMITS_UPDATED,
                    timestamp=timestamp,
                    provider="openai",
                    raw_event=raw_event,
                    rate_limits=rate_limits,
                )
            ]

        # Unknown event type
        else:
            logger.warning(f"OpenAIProvider: Unknown event type: {event_type}")
            return []

    # ============================================================================
    # Optional Advanced Operations
    # ============================================================================

    async def truncate_response(
        self, item_id: str, content_index: int, audio_end_ms: int
    ) -> None:
        """Truncate a response item (OpenAI-specific feature)."""
        truncate_event = {
            "event_id": self._generate_event_id(),
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
        }
        await self._service_manager.send_event(truncate_event)
        logger.debug(f"OpenAIProvider: Truncated item {item_id}")

    async def clear_input_audio_buffer(self) -> None:
        """Clear input audio buffer (OpenAI-specific feature)."""
        clear_event = {
            "event_id": self._generate_event_id(),
            "type": "input_audio_buffer.clear",
        }
        await self._service_manager.send_event(clear_event)
        logger.debug("OpenAIProvider: Cleared input audio buffer")

    async def commit_audio_buffer(self) -> None:
        """
        Commit the input audio buffer without generating a response.

        This triggers:
        - input_audio_buffer.committed event (immediately)
        - conversation.item.created event (user message item created)
        - conversation.item.input_audio_transcription.completed event
          (async, only if input_audio_transcription_enabled=True in session config)

        Note: Transcription runs asynchronously. The transcription event may arrive
        before or after other events. Use item_id to correlate events.

        Use this for push-to-talk scenarios when you need the transcription
        but want to control when the response is generated separately.
        """
        logger.info("OpenAIProvider: Committing audio buffer without response")
        try:
            commit_event = {
                "event_id": self._generate_event_id(),
                "type": "input_audio_buffer.commit",
            }
            await self._service_manager.send_event(commit_event)
            logger.debug("OpenAIProvider: Committed audio buffer (no response)")
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to commit audio buffer - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    async def delete_conversation_item(self, item_id: str) -> None:
        """
        Delete a conversation item from the history.

        Args:
            item_id: The ID of the conversation item to delete.

        This triggers:
        - conversation.item.deleted event on success
        - error event if item doesn't exist
        """
        logger.info(f"OpenAIProvider: Deleting conversation item {item_id}")
        try:
            delete_event = {
                "event_id": self._generate_event_id(),
                "type": "conversation.item.delete",
                "item_id": item_id,
            }
            await self._service_manager.send_event(delete_event)
            logger.debug(f"OpenAIProvider: Deleted conversation item {item_id}")
        except Exception as e:
            logger.error(
                f"OpenAIProvider: Failed to delete conversation item - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        return f"event_{uuid.uuid4()}"
