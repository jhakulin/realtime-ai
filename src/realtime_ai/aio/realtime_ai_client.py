import asyncio
import logging
import uuid

from realtime_ai.aio.audio_stream_manager import AudioStreamManager
from realtime_ai.aio.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models import realtime_ai_events
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.models.normalized_events import (
    AudioBufferClearedEvent,
    AudioBufferCommittedEvent,
    AudioDeltaEvent,
    AudioDoneEvent,
    ConversationItemCreatedEvent,
    ErrorEvent,
    EventType,
    FunctionCallEvent,
    InputTranscriptCompletedEvent,
    InputTranscriptDeltaEvent,
    NormalizedEvent,
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
from realtime_ai.providers.provider_factory import ProviderFactory

logger = logging.getLogger(__name__)


class RealtimeAIClient:
    """
    Manages overall interaction with realtime speech AI providers.

    Supports multiple providers: OpenAI, Gemini, Grok, Nova.
    Defaults to OpenAI for backward compatibility.
    """

    def __init__(
        self,
        options: RealtimeAIOptions,
        stream_options: AudioStreamOptions,
        event_handler: RealtimeAIEventHandler,
        provider: str = "openai",
    ):
        self._options = options
        self._provider_name = provider.lower()

        # Create provider instance from factory
        self._provider = ProviderFactory.create(self._provider_name, options)

        # Create audio stream manager with provider
        self._audio_stream_manager = AudioStreamManager(stream_options, self._provider)

        self._event_handler = event_handler
        self._is_running = False
        self._consume_task = None

    async def start(self):
        """Starts the RealtimeAIClient."""
        if not self._is_running:
            self._is_running = True
            try:
                # Connect to provider
                await self._provider.connect()
                logger.info(
                    f"RealtimeAIClient: Connected to {self._provider_name} provider."
                )

                # Schedule the event consumption coroutine as a background task
                self._consume_task = asyncio.create_task(self._consume_events())
            except Exception as e:
                logger.error(f"RealtimeAIClient: Error during client start: {e}")
                self._is_running = False

    async def stop(self):
        """Stops the RealtimeAIClient gracefully."""
        if self._is_running:
            self._is_running = False
            try:
                # Stop audio stream if available
                if self._audio_stream_manager:
                    await self._audio_stream_manager.stop_stream()

                # Disconnect from provider
                await self._provider.disconnect()
                logger.info("RealtimeAIClient: Provider disconnected.")

                if self._consume_task:
                    # Cancel the consume_events task and wait for it to finish
                    self._consume_task.cancel()
                    try:
                        await self._consume_task
                    except asyncio.CancelledError:
                        logger.info("RealtimeAIClient: consume_events task cancelled.")
            except Exception as e:
                logger.error(f"RealtimeAIClient: Error during client stop: {e}")

    async def send_audio(self, audio_data: bytes):
        """Sends audio data to the audio stream manager for processing."""
        logger.info("RealtimeAIClient: Queuing audio data for streaming.")
        await self._audio_stream_manager.write_audio_buffer(audio_data)

    async def send_text(
        self, text: str, role: str = "user", generate_response: bool = True
    ):
        """Sends text input to the provider."""
        await self._provider.send_text(text, role=role)
        logger.info(
            f"RealtimeAIClient: Sent text input to {self._provider_name} provider."
        )

        # Generate a response if required
        if generate_response:
            await self.generate_response(commit_audio_buffer=False)

    async def send_image(
        self,
        image_data: bytes,
        image_format: str = "png",
        generate_response: bool = True,
    ):
        """
        Sends an image to the provider for vision processing.

        Args:
            image_data: Raw image bytes (PNG, JPEG, WebP, GIF)
            image_format: Image format ('png', 'jpeg', 'webp', 'gif')
            generate_response: Whether to automatically generate a response

        Raises:
            NotImplementedError: If provider doesn't support image input

        Supported providers:
            - OpenAI: Yes
            - Gemini: Yes (JPEG recommended)
            - Grok: No

        Example:
            with open("screenshot.png", "rb") as f:
                await client.send_image(f.read(), image_format="png")
        """
        await self._provider.send_image(image_data, image_format)
        logger.info(
            f"RealtimeAIClient: Sent image to {self._provider_name} provider."
        )

        # Generate a response if required
        if generate_response:
            await self.generate_response(commit_audio_buffer=False)

    async def update_session(self, options: RealtimeAIOptions):
        """Updates the session configuration with the provided options."""
        if self._is_running:
            await self._provider.update_session(options)

        self._options = options
        logger.info(
            f"RealtimeAIClient: Sent session update to {self._provider_name} provider."
        )

    async def generate_response(self, commit_audio_buffer: bool = True):
        """Generates a response from the provider."""
        logger.info("RealtimeAIClient: Generating response.")
        await self._provider.generate_response(commit_audio=commit_audio_buffer)

    async def cancel_response(self):
        """Cancels the current response from the provider."""
        await self._provider.cancel_response()
        logger.info(
            f"RealtimeAIClient: Sent cancel request to {self._provider_name} provider."
        )

    async def truncate_response(
        self, item_id: str, content_index: int, audio_end_ms: int
    ):
        """Truncates a response (provider-specific feature)."""
        await self._provider.truncate_response(item_id, content_index, audio_end_ms)
        logger.info(
            f"RealtimeAIClient: Sent truncate request to {self._provider_name} provider."
        )

    async def clear_input_audio_buffer(self):
        """Clears the input audio buffer (provider-specific feature)."""
        await self._provider.clear_input_audio_buffer()
        logger.info(
            f"RealtimeAIClient: Sent clear buffer request to {self._provider_name} provider."
        )

    async def generate_response_from_function_call(
        self, call_id: str, function_output: str
    ):
        """
        Sends a function call result to the provider and generates a response.

        :param call_id: The ID of the function call.
        :param function_output: The output of the function call.
        """
        # Send function result to provider
        await self._provider.send_function_result(call_id, function_output)
        logger.info(
            f"RealtimeAIClient: Function call output sent to {self._provider_name} provider."
        )

        # Generate response
        await self._provider.generate_response(commit_audio=False)

    async def _consume_events(self):
        """Consume events from the provider asynchronously."""
        logger.info(
            f"RealtimeAIClient: Started consuming events from {self._provider_name} provider."
        )
        try:
            async for normalized_event in self._provider.receive_events():
                if not self._is_running:
                    break

                try:
                    # Convert normalized event to OpenAI format for backward compatibility
                    openai_event = self._to_openai_event(normalized_event)
                    if openai_event:
                        # Schedule the event handler as an independent task
                        asyncio.create_task(self._handle_event(openai_event))
                except Exception as e:
                    logger.error(f"RealtimeAIClient: Error processing event: {e}")
        except asyncio.CancelledError:
            logger.info("RealtimeAIClient: consume_events loop has been cancelled.")
        except Exception as e:
            logger.error(f"RealtimeAIClient: Error in consume_events: {e}")
        finally:
            logger.info("RealtimeAIClient: Stopped consuming events.")

    def _to_openai_event(self, normalized_event: NormalizedEvent) -> EventBase:
        """
        Converts a normalized event to OpenAI EventBase format for backward compatibility.

        This ensures existing event handlers continue to work unchanged.
        """
        event_type = normalized_event.event_type

        # Session events
        if isinstance(normalized_event, SessionCreatedEvent):
            return realtime_ai_events.SessionCreated(
                event_id=normalized_event.event_id,
                type="session.created",
                session=normalized_event.config,
            )
        elif isinstance(normalized_event, SessionUpdatedEvent):
            return realtime_ai_events.SessionUpdated(
                event_id=normalized_event.event_id,
                type="session.updated",
                session=normalized_event.config,
            )

        # Audio events
        elif isinstance(normalized_event, AudioDeltaEvent):
            return realtime_ai_events.ResponseAudioDelta(
                event_id=normalized_event.event_id,
                type="response.audio.delta",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
                delta=normalized_event.delta,
            )
        elif isinstance(normalized_event, AudioDoneEvent):
            return realtime_ai_events.ResponseAudioDone(
                event_id=normalized_event.event_id,
                type="response.audio.done",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
            )

        # Transcript events
        elif isinstance(normalized_event, TranscriptDeltaEvent):
            # Check if it's audio transcript or text transcript
            event_suffix = (
                "audio_transcript"
                if normalized_event.raw_event
                and "audio_transcript" in normalized_event.raw_event.get("type", "")
                else "audio_transcript"
            )
            return realtime_ai_events.ResponseAudioTranscriptDelta(
                event_id=normalized_event.event_id,
                type=f"response.{event_suffix}.delta",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
                delta=normalized_event.delta,
            )
        elif isinstance(normalized_event, TranscriptDoneEvent):
            event_suffix = (
                "audio_transcript"
                if normalized_event.raw_event
                and "audio_transcript" in normalized_event.raw_event.get("type", "")
                else "audio_transcript"
            )
            return realtime_ai_events.ResponseAudioTranscriptDone(
                event_id=normalized_event.event_id,
                type=f"response.{event_suffix}.done",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
                transcript=normalized_event.transcript,
            )
        elif isinstance(normalized_event, InputTranscriptDeltaEvent):
            return realtime_ai_events.ConversationItemInputAudioTranscriptionDelta(
                event_id=normalized_event.event_id,
                type="conversation.item.input_audio_transcription.delta",
                item_id=normalized_event.item_id,
                content_index=normalized_event.content_index,
                delta=normalized_event.delta,
            )
        elif isinstance(normalized_event, InputTranscriptCompletedEvent):
            return realtime_ai_events.ConversationItemInputAudioTranscriptionCompleted(
                event_id=normalized_event.event_id,
                type="conversation.item.input_audio_transcription.completed",
                item_id=normalized_event.item_id,
                content_index=normalized_event.content_index,
                transcript=normalized_event.transcript,
            )

        # Speech detection events
        elif isinstance(normalized_event, SpeechStartedEvent):
            return realtime_ai_events.InputAudioBufferSpeechStarted(
                event_id=normalized_event.event_id,
                type="input_audio_buffer.speech_started",
                audio_start_ms=normalized_event.audio_start_ms,
                item_id=normalized_event.item_id,
            )
        elif isinstance(normalized_event, SpeechStoppedEvent):
            return realtime_ai_events.InputAudioBufferSpeechStopped(
                event_id=normalized_event.event_id,
                type="input_audio_buffer.speech_stopped",
                audio_end_ms=normalized_event.audio_end_ms,
                item_id=normalized_event.item_id,
            )

        # Buffer events
        elif isinstance(normalized_event, AudioBufferCommittedEvent):
            return realtime_ai_events.InputAudioBufferCommitted(
                event_id=normalized_event.event_id,
                type="input_audio_buffer.committed",
                previous_item_id=normalized_event.previous_item_id,
                item_id=normalized_event.item_id,
            )
        elif isinstance(normalized_event, AudioBufferClearedEvent):
            return realtime_ai_events.EventBase(
                event_id=normalized_event.event_id,
                type="audio.buffer.cleared",
            )

        # Conversation events
        elif isinstance(normalized_event, ConversationItemCreatedEvent):
            return realtime_ai_events.ConversationItemCreated(
                event_id=normalized_event.event_id,
                type="conversation.item.created",
                previous_item_id=normalized_event.previous_item_id,
                item=normalized_event.item,
            )

        # Response events
        elif isinstance(normalized_event, ResponseCreatedEvent):
            return realtime_ai_events.ResponseCreated(
                event_id=normalized_event.event_id,
                type="response.created",
                response={"id": normalized_event.response_id},
            )
        elif isinstance(normalized_event, ResponseOutputItemAddedEvent):
            return realtime_ai_events.ResponseOutputItemAdded(
                event_id=normalized_event.event_id,
                type="response.output_item.added",
                response_id=normalized_event.response_id,
                output_index=normalized_event.output_index,
                item=normalized_event.item,
            )
        elif isinstance(normalized_event, ResponseOutputItemDoneEvent):
            return realtime_ai_events.ResponseOutputItemDone(
                event_id=normalized_event.event_id,
                type="response.output_item.done",
                response_id=normalized_event.response_id,
                output_index=normalized_event.output_index,
                item=normalized_event.item,
            )
        elif isinstance(normalized_event, ResponseContentPartAddedEvent):
            return realtime_ai_events.ResponseContentPartAdded(
                event_id=normalized_event.event_id,
                type="response.content_part.added",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
                part=normalized_event.part,
            )
        elif isinstance(normalized_event, ResponseContentPartDoneEvent):
            return realtime_ai_events.ResponseContentPartDone(
                event_id=normalized_event.event_id,
                type="response.content_part.done",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
                part=normalized_event.part,
            )
        elif isinstance(normalized_event, ResponseDoneEvent):
            return realtime_ai_events.ResponseDone(
                event_id=normalized_event.event_id,
                type="response.done",
                response={
                    "id": normalized_event.response_id,
                    "status": normalized_event.status,
                },
            )

        # Function call events
        elif isinstance(normalized_event, FunctionCallEvent):
            return realtime_ai_events.ResponseFunctionCallArgumentsDone(
                event_id=normalized_event.event_id,
                type="response.function_call_arguments.done",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                call_id=normalized_event.call_id,
                arguments=normalized_event.arguments,
            )

        # Error events
        elif isinstance(normalized_event, ErrorEvent):
            error_details = realtime_ai_events.ErrorDetails(
                type=normalized_event.error_type or "unknown",
                code=normalized_event.error_code,
                message=normalized_event.error_message,
                param=None,
                event_id=None,
            )
            return realtime_ai_events.ErrorEvent(
                event_id=normalized_event.event_id, type="error", error=error_details
            )

        # Rate limit events
        elif isinstance(normalized_event, RateLimitsUpdatedEvent):
            rate_limits = [
                realtime_ai_events.RateLimit(
                    name=limit.name,
                    limit=limit.limit,
                    remaining=limit.remaining,
                    reset_seconds=limit.reset_seconds,
                )
                for limit in normalized_event.rate_limits
            ]
            return realtime_ai_events.RateLimitsUpdated(
                event_id=normalized_event.event_id,
                type="rate_limits.updated",
                rate_limits=rate_limits,
            )

        # Unknown event type - return generic EventBase
        else:
            logger.warning(
                f"Unknown normalized event type: {event_type}, creating generic EventBase"
            )
            return EventBase(
                event_id=normalized_event.event_id,
                type=str(event_type.value)
                if hasattr(event_type, "value")
                else str(event_type),
            )

    async def _handle_event(self, event: EventBase):
        """Handles the received event based on its type using the event handler."""
        event_type = event.type
        method_name = f"on_{event_type.replace('.', '_')}"
        handler = getattr(self._event_handler, method_name, None)

        if callable(handler):
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    f"Error in handler {method_name} for event {event_type}: {e}"
                )
        else:
            await self._event_handler.on_unhandled_event(event_type, vars(event))

    @property
    def options(self):
        return self._options

    @property
    def is_running(self):
        return self._is_running
