"""
Grok Provider for xAI's Voice Agent API.

The Grok Voice Agent API is compatible with the OpenAI Realtime API,
making integration straightforward. Key differences:
- Different endpoint: wss://api.x.ai/v1/realtime
- Voice personalities: 5 options (Ara, Rex, Sal, Eve, Leo)
- Built-in tools: Web Search, X Search, Collections
- Additional audio formats: G.711 μ-law and A-law (optional)

Since Grok is OpenAI-compatible, event normalization is identical.
"""

import time
import base64
import logging
from typing import AsyncIterator, List
from realtime_ai.providers.base_provider import BaseProvider
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.aio.realtime_ai_service_manager import RealtimeAIServiceManager
from realtime_ai.models.normalized_events import (
    NormalizedEvent, EventType,
    SessionCreatedEvent, SessionUpdatedEvent,
    AudioDeltaEvent, AudioDoneEvent,
    TranscriptDeltaEvent, TranscriptDoneEvent,
    InputTranscriptCompletedEvent,
    SpeechStartedEvent, SpeechStoppedEvent,
    ResponseCreatedEvent, ResponseDoneEvent,
    ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent,
    FunctionCallEvent, ErrorEvent, RateLimitsUpdatedEvent,
    AudioBufferCommittedEvent, ConversationItemCreatedEvent,
    RateLimit
)

logger = logging.getLogger(__name__)


class GrokProvider(BaseProvider):
    """
    Provider implementation for xAI's Grok Voice Agent API.

    Grok is OpenAI Realtime API compatible, so this implementation
    reuses much of the OpenAI logic with Grok-specific configuration.
    """

    # Grok voice personalities
    VOICE_ARA = "ara"      # Female, warm
    VOICE_REX = "rex"      # Male, professional
    VOICE_SAL = "sal"      # Neutral, smooth
    VOICE_EVE = "eve"      # Female, energetic
    VOICE_LEO = "leo"      # Male, authoritative

    def __init__(self, options: RealtimeAIOptions):
        """
        Initialize Grok provider.

        Args:
            options: Configuration options. Should include xAI API key.
        """
        super().__init__(options)

        # Override endpoint for Grok
        grok_options = self._create_grok_options(options)
        self._service_manager = RealtimeAIServiceManager(grok_options)

    def _create_grok_options(self, options: RealtimeAIOptions) -> RealtimeAIOptions:
        """
        Create Grok-specific options from base options.

        Overrides the endpoint to use xAI's API.
        """
        # Create a new options instance with Grok endpoint
        from dataclasses import replace
        return replace(options, url="wss://api.x.ai/v1/realtime")

    @property
    def provider_name(self) -> str:
        """Returns the provider name."""
        return "grok"

    async def connect(self) -> None:
        """
        Establishes connection to Grok Voice Agent API.

        Uses standard WebSocket connection to wss://api.x.ai/v1/realtime.
        """
        logger.info(f"GrokProvider: Connecting to Grok Voice Agent API (model={self.options.model}, voice={self.options.voice})")
        try:
            await self._service_manager.connect()
            self._is_connected = True
            logger.info("GrokProvider: Successfully connected to Grok Voice Agent API")
        except Exception as e:
            logger.error(f"GrokProvider: Failed to connect - {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """Disconnects from Grok API."""
        logger.info("GrokProvider: Disconnecting from Grok Voice Agent API")
        try:
            await self._service_manager.disconnect()
            self._is_connected = False
            logger.info("GrokProvider: Successfully disconnected from Grok Voice Agent API")
        except Exception as e:
            logger.error(f"GrokProvider: Error during disconnect - {type(e).__name__}: {str(e)}", exc_info=True)
            self._is_connected = False
            raise

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Sends audio data to Grok.

        Supports PCM, G.711 μ-law, and G.711 A-law formats.
        Default: PCM linear16 at 24kHz (OpenAI-compatible).

        Args:
            audio_data: Raw audio bytes
        """
        logger.debug(f"GrokProvider: Sending audio chunk (size={len(audio_data)} bytes)")
        try:
            # Encode to base64 for transmission
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')

            # Send as input_audio_buffer.append event (OpenAI-compatible)
            event = {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio
            }
            await self._service_manager.send_event(event)
            logger.debug(f"GrokProvider: Successfully sent audio chunk (size={len(audio_data)} bytes)")
        except Exception as e:
            logger.error(f"GrokProvider: Failed to send audio - {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def send_text(self, text: str, role: str = "user") -> None:
        """
        Sends text message to Grok.

        Args:
            text: Text content
            role: Message role (user/assistant)
        """
        logger.info(f"GrokProvider: Sending text message (role={role}, length={len(text)} chars)")
        logger.debug(f"GrokProvider: Text content: {text[:100]}{'...' if len(text) > 100 else ''}")

        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            await self._service_manager.send_event(event)
            logger.info(f"GrokProvider: Successfully sent text message (role={role})")
        except Exception as e:
            logger.error(f"GrokProvider: Failed to send text - {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def update_session(self, options: RealtimeAIOptions) -> None:
        """
        Updates the session configuration.

        Supports Grok-specific features:
        - Voice personalities: ara, rex, sal, eve, leo
        - Built-in tools: web_search, x_search, collections

        Args:
            options: Updated configuration options
        """
        logger.info(f"GrokProvider: Updating session configuration (model={options.model}, voice={options.voice})")
        try:
            self.options = options
            await self._service_manager.update_session(options)
            logger.info("GrokProvider: Successfully updated session configuration")
        except Exception as e:
            logger.error(f"GrokProvider: Failed to update session - {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def generate_response(self, commit_audio: bool = True) -> None:
        """
        Triggers response generation from Grok.

        Args:
            commit_audio: Whether to commit buffered audio before response
        """
        logger.info(f"GrokProvider: Generating response (commit_audio={commit_audio})")
        try:
            if commit_audio:
                # Commit buffered audio
                commit_event = {
                    "type": "input_audio_buffer.commit"
                }
                await self._service_manager.send_event(commit_event)
                logger.debug("GrokProvider: Committed audio buffer")

            # Request response generation
            response_event = {
                "type": "response.create",
                "response": {
                    "modalities": self.options.modalities
                }
            }
            await self._service_manager.send_event(response_event)
            logger.info("GrokProvider: Successfully requested response generation")
        except Exception as e:
            logger.error(f"GrokProvider: Failed to generate response - {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def cancel_response(self) -> None:
        """
        Cancels the current response generation.

        Note: Grok does not support the response.cancel event - sending it causes
        the server to close the WebSocket connection. Instead, we just clear the
        local event queue to stop processing the current response.
        """
        logger.info("GrokProvider: Cancelling ongoing response (local only - Grok does not support response.cancel)")
        try:
            # Note: Do NOT send response.cancel to Grok - it closes the connection
            # Just clear the local event queue to stop processing
            await self._service_manager.clear_event_queue()
            logger.info("GrokProvider: Successfully cancelled response")
        except Exception as e:
            logger.error(f"GrokProvider: Failed to cancel response - {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def send_function_result(self, call_id: str, result: str) -> None:
        """
        Sends function call result back to Grok.

        Supports built-in tools:
        - web_search: Current information lookup
        - x_search: X posts and trends
        - collections: Document/RAG queries
        - Custom functions: User-defined JSON schemas

        Args:
            call_id: Function call ID from FunctionCallEvent
            result: Function execution result (JSON string)
        """
        logger.info(f"GrokProvider: Sending function result (call_id={call_id}, result_length={len(result)} chars)")
        logger.debug(f"GrokProvider: Function result content: {result[:200]}{'...' if len(result) > 200 else ''}")

        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result
                }
            }
            await self._service_manager.send_event(event)
            logger.info(f"GrokProvider: Successfully sent function result (call_id={call_id})")
        except Exception as e:
            logger.error(f"GrokProvider: Failed to send function result - {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def receive_events(self) -> AsyncIterator[NormalizedEvent]:
        """
        Receives events from Grok as an async iterator.

        Yields normalized events that are compatible across all providers.

        Yields:
            NormalizedEvent: Provider-agnostic event
        """
        while self._is_connected:
            try:
                raw_event = await self._service_manager.get_next_event()
                if raw_event:
                    # Normalize Grok event to generic format
                    normalized_events = self.normalize_incoming_event(vars(raw_event))
                    for event in normalized_events:
                        yield event
            except Exception as e:
                logger.error(f"GrokProvider: Error receiving events - {type(e).__name__}: {str(e)}", exc_info=True)
                # Don't break the loop, continue receiving events

    def normalize_incoming_event(self, raw_event: dict) -> List[NormalizedEvent]:
        """
        Converts Grok event to normalized format.

        Grok uses slightly different event type names than OpenAI:
        - response.output_audio.delta vs response.audio.delta
        - response.output_audio_transcript.delta vs response.audio_transcript.delta

        Args:
            raw_event: Raw Grok event dictionary

        Returns:
            List of normalized events (usually 1, sometimes 0 for ignored events)
        """
        event_type = raw_event.get("type")
        event_id = raw_event.get("event_id", "")


        # Normalize Grok event types to OpenAI equivalents for matching
        event_type_aliases = {
            "response.output_audio.delta": "response.audio.delta",
            "response.output_audio.done": "response.audio.done",
            "response.output_audio_transcript.delta": "response.audio_transcript.delta",
            "response.output_audio_transcript.done": "response.audio_transcript.done",
            "conversation.item.added": "conversation.item.created",
        }
        event_type = event_type_aliases.get(event_type, event_type)
        timestamp = time.time()

        # Session events
        if event_type == "session.created":
            return [SessionCreatedEvent(
                event_id=event_id,
                event_type=EventType.SESSION_CREATED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                session_id=raw_event.get("session", {}).get("id", ""),
                config=raw_event.get("session", {})
            )]

        elif event_type == "session.updated":
            return [SessionUpdatedEvent(
                event_id=event_id,
                event_type=EventType.SESSION_UPDATED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                session_id=raw_event.get("session", {}).get("id", ""),
                config=raw_event.get("session", {})
            )]

        # Audio events
        elif event_type == "response.audio.delta":
            return [AudioDeltaEvent(
                event_id=event_id,
                event_type=EventType.AUDIO_DELTA,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                content_index=raw_event.get("content_index", 0),
                delta=raw_event.get("delta", "")
            )]

        elif event_type == "response.audio.done":
            return [AudioDoneEvent(
                event_id=event_id,
                event_type=EventType.AUDIO_DONE,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                content_index=raw_event.get("content_index", 0)
            )]

        # Transcript events
        elif event_type == "response.audio_transcript.delta":
            return [TranscriptDeltaEvent(
                event_id=event_id,
                event_type=EventType.TRANSCRIPT_DELTA,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                content_index=raw_event.get("content_index", 0),
                delta=raw_event.get("delta", "")
            )]

        elif event_type == "response.audio_transcript.done":
            return [TranscriptDoneEvent(
                event_id=event_id,
                event_type=EventType.TRANSCRIPT_DONE,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                content_index=raw_event.get("content_index", 0),
                transcript=raw_event.get("transcript", "")
            )]

        elif event_type == "response.text.delta":
            return [TranscriptDeltaEvent(
                event_id=event_id,
                event_type=EventType.TRANSCRIPT_DELTA,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                content_index=raw_event.get("content_index", 0),
                delta=raw_event.get("delta", "")
            )]

        elif event_type == "response.text.done":
            return [TranscriptDoneEvent(
                event_id=event_id,
                event_type=EventType.TRANSCRIPT_DONE,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                content_index=raw_event.get("content_index", 0),
                transcript=raw_event.get("text", "")
            )]

        # Input transcript completed
        elif event_type == "conversation.item.input_audio_transcription.completed":
            return [InputTranscriptCompletedEvent(
                event_id=event_id,
                event_type=EventType.INPUT_TRANSCRIPT_COMPLETED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                item_id=raw_event.get("item_id", ""),
                content_index=raw_event.get("content_index", 0),
                transcript=raw_event.get("transcript", "")
            )]

        # Speech detection events
        elif event_type == "input_audio_buffer.speech_started":
            return [SpeechStartedEvent(
                event_id=event_id,
                event_type=EventType.SPEECH_STARTED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                audio_start_ms=raw_event.get("audio_start_ms", 0),
                item_id=raw_event.get("item_id", "")
            )]

        elif event_type == "input_audio_buffer.speech_stopped":
            return [SpeechStoppedEvent(
                event_id=event_id,
                event_type=EventType.SPEECH_STOPPED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                audio_end_ms=raw_event.get("audio_end_ms", 0),
                item_id=raw_event.get("item_id", "")
            )]

        # Buffer events
        elif event_type == "input_audio_buffer.committed":
            return [AudioBufferCommittedEvent(
                event_id=event_id,
                event_type=EventType.AUDIO_BUFFER_COMMITTED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                previous_item_id=raw_event.get("previous_item_id", ""),
                item_id=raw_event.get("item_id", "")
            )]

        # Conversation events
        elif event_type == "conversation.item.created":
            return [ConversationItemCreatedEvent(
                event_id=event_id,
                event_type=EventType.CONVERSATION_ITEM_CREATED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                previous_item_id=raw_event.get("previous_item_id", ""),
                item=raw_event.get("item", {})
            )]

        # Response events
        elif event_type == "response.created":
            return [ResponseCreatedEvent(
                event_id=event_id,
                event_type=EventType.RESPONSE_CREATED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response", {}).get("id", "")
            )]

        elif event_type == "response.done":
            return [ResponseDoneEvent(
                event_id=event_id,
                event_type=EventType.RESPONSE_DONE,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response", {}).get("id", ""),
                status=raw_event.get("response", {}).get("status", "")
            )]

        elif event_type == "response.output_item.added":
            return [ResponseOutputItemAddedEvent(
                event_id=event_id,
                event_type=EventType.RESPONSE_OUTPUT_ITEM_ADDED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                output_index=raw_event.get("output_index", 0),
                item=raw_event.get("item", {})
            )]

        elif event_type == "response.output_item.done":
            return [ResponseOutputItemDoneEvent(
                event_id=event_id,
                event_type=EventType.RESPONSE_OUTPUT_ITEM_DONE,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                item=raw_event.get("item", {})
            )]

        # Function call events
        elif event_type == "response.function_call_arguments.done":
            return [FunctionCallEvent(
                event_id=event_id,
                event_type=EventType.FUNCTION_CALL,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                response_id=raw_event.get("response_id", ""),
                item_id=raw_event.get("item_id", ""),
                output_index=raw_event.get("output_index", 0),
                call_id=raw_event.get("call_id", ""),
                function_name=raw_event.get("name", ""),
                arguments=raw_event.get("arguments", "")
            )]

        # Error events
        elif event_type == "error":
            error_data = raw_event.get("error", {})
            # Handle both dict and ErrorDetails dataclass
            if hasattr(error_data, 'code'):
                # It's an ErrorDetails dataclass
                error_code = error_data.code or "unknown"
                error_message = error_data.message or ""
                error_type = error_data.type or ""
            else:
                # It's a dict
                error_code = error_data.get("code", "unknown")
                error_message = error_data.get("message", "")
                error_type = error_data.get("type", "")
            return [ErrorEvent(
                event_id=event_id,
                event_type=EventType.ERROR,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                error_code=error_code,
                error_message=error_message,
                error_type=error_type
            )]

        # Rate limit events
        elif event_type == "rate_limits.updated":
            rate_limits_data = raw_event.get("rate_limits", [])
            rate_limits = [
                RateLimit(
                    name=limit.get("name", ""),
                    limit=limit.get("limit", 0),
                    remaining=limit.get("remaining", 0),
                    reset_seconds=limit.get("reset_seconds", 0)
                )
                for limit in rate_limits_data
            ]
            return [RateLimitsUpdatedEvent(
                event_id=event_id,
                event_type=EventType.RATE_LIMITS_UPDATED,
                timestamp=timestamp,
                provider="grok",
                raw_event=raw_event,
                rate_limits=rate_limits
            )]

        # Unknown event type - ignore
        logger.debug(f"GrokProvider: Unhandled event type: {event_type}")
        return []

    # Optional features (Grok-compatible with OpenAI)
    async def truncate_response(self, item_id: str, content_index: int, audio_end_ms: int) -> None:
        """
        Truncates a response (OpenAI-compatible feature).

        Args:
            item_id: Item to truncate
            content_index: Content index to truncate at
            audio_end_ms: Audio end time in milliseconds
        """
        event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms
        }
        await self._service_manager.send_event(event)

    async def clear_input_audio_buffer(self) -> None:
        """
        Clears the input audio buffer.

        Note: Grok always uses server-side VAD and does not properly support
        the input_audio_buffer.clear event. Sending it can cause the connection
        to close. We skip sending this event for Grok.
        """
        logger.debug(
            "GrokProvider: Skipping input_audio_buffer.clear (not supported by Grok)"
        )

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
        """
        logger.info("GrokProvider: Committing audio buffer without response")
        try:
            commit_event = {
                "type": "input_audio_buffer.commit"
            }
            await self._service_manager.send_event(commit_event)
            logger.debug("GrokProvider: Committed audio buffer (no response)")
        except Exception as e:
            logger.error(
                f"GrokProvider: Failed to commit audio buffer - {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    async def delete_conversation_item(self, item_id: str) -> None:
        """
        Delete a conversation item from the history.

        Note: Grok does not support the conversation.item.delete event.
        This method is a no-op. To reset conversation context with Grok,
        use reconnect() to get a fresh session.

        Args:
            item_id: The ID of the conversation item to delete (ignored).
        """
        logger.debug(
            f"GrokProvider: Skipping conversation.item.delete for {item_id} "
            "(not supported by Grok)"
        )

    async def reconnect(self) -> None:
        """
        Reconnect to get a fresh session with no conversation history.

        This is the only way to clear conversation context in Grok,
        since it doesn't support conversation.item.delete.
        """
        logger.info("GrokProvider: Reconnecting for fresh session...")
        await self._service_manager.reconnect()
        logger.info("GrokProvider: Reconnected with fresh session.")
