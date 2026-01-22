import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
import uuid

from realtime_ai.audio_stream_manager import AudioStreamManager
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
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler

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

        self._event_handler = event_handler
        self._is_running = False
        self._lock = threading.Lock()

        # Initialize the consume thread and executor as None
        self._consume_thread = None
        self.executor = None
        self._stop_event = threading.Event()

        # Event loop for async provider operations
        self._provider_loop = None
        self._provider_loop_thread = None

        # Create audio stream manager with provider (will be set up with loop in start())
        self._audio_stream_manager = None
        self._stream_options = stream_options

        # Session ready event - signals when session is initialized
        self._session_ready = threading.Event()

    def start(self, session_ready_timeout: float = 10.0):
        """
        Starts the RealtimeAIClient.

        Args:
            session_ready_timeout: Maximum time to wait for session initialization (seconds).
                                   Set to 0 to skip waiting.
        """
        with self._lock:
            if self._is_running:
                logger.warning("RealtimeAIClient: Client is already running.")
                return

            self._is_running = True
            self._stop_event.clear()
            self._session_ready.clear()
            try:
                # Start provider loop for async operations
                self._start_provider_loop()

                # Connect to provider
                future = asyncio.run_coroutine_threadsafe(
                    self._provider.connect(), self._provider_loop
                )
                future.result(timeout=10)  # Wait for connection

                # Create audio stream manager with provider loop
                self._audio_stream_manager = AudioStreamManager(
                    self._stream_options, self._provider, self._provider_loop
                )

                logger.debug("RealtimeAIClient: Client started.")

                # Initialize and start the ThreadPoolExecutor here
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
                logger.debug("RealtimeAIClient: ThreadPoolExecutor initialized.")

                # Initialize and start the consume thread
                self._consume_thread = threading.Thread(
                    target=self._consume_events,
                    daemon=True,
                    name="RealtimeAIClient_ConsumeThread",
                )
                self._consume_thread.start()
                logger.debug("RealtimeAIClient: Event consumption thread started.")

                # Wait for session to be ready (session.created or session.updated)
                if session_ready_timeout > 0:
                    if self._session_ready.wait(timeout=session_ready_timeout):
                        logger.info("RealtimeAIClient: Session is ready.")
                    else:
                        logger.warning(
                            f"RealtimeAIClient: Session ready timeout after {session_ready_timeout}s. "
                            "Proceeding anyway - first interaction may be delayed."
                        )
            except Exception as e:
                self._is_running = False
                logger.error(f"RealtimeAIClient: Error during client start: {e}")

    def stop(self, timeout: float = 5.0):
        """Stops the RealtimeAIClient gracefully."""
        with self._lock:
            if not self._is_running:
                logger.warning("RealtimeAIClient: Client is already stopped.")
                return

            self._is_running = False

            # Signal stop event
            self._stop_event.set()

            try:
                # Stop audio stream manager if available
                if self._audio_stream_manager:
                    self._audio_stream_manager.stop_stream()

                # Disconnect from provider
                if self._provider_loop:
                    future = asyncio.run_coroutine_threadsafe(
                        self._provider.disconnect(), self._provider_loop
                    )
                    try:
                        future.result(timeout=timeout)
                    except Exception as e:
                        logger.error(
                            f"RealtimeAIClient: Error disconnecting provider: {e}"
                        )

                if self._consume_thread is not None:
                    # Attempt to join the consume thread within the timeout
                    self._consume_thread.join(timeout=timeout)
                    if self._consume_thread.is_alive():
                        logger.warning(
                            "RealtimeAIClient: Consume thread did not terminate within the timeout."
                        )
                    else:
                        logger.debug("RealtimeAIClient: Consume thread terminated.")
                    self._consume_thread = None

                if self.executor is not None:
                    self.executor.shutdown(wait=True)
                    logger.debug("RealtimeAIClient: ThreadPoolExecutor shut down.")
                    self.executor = None

                # Stop provider loop thread
                self._stop_provider_loop()

                logger.info("RealtimeAIClient: Services stopped.")
            except Exception as e:
                logger.error(f"RealtimeAIClient: Error during client stop: {e}")

    def send_audio(self, audio_data: bytes):
        """Sends audio data to the audio stream manager for processing."""
        logger.debug("RealtimeAIClient: Queuing audio data for streaming.")
        self._audio_stream_manager.write_audio_buffer_sync(
            audio_data
        )  # Ensure this is a sync method

    def send_text(self, text: str, role: str = "user", generate_response: bool = True):
        """Sends text input to the provider."""
        # Send text via provider interface
        future = asyncio.run_coroutine_threadsafe(
            self._provider.send_text(text, role), self._provider_loop
        )
        future.result(timeout=5)

        logger.info("RealtimeAIClient: Sent text input to server.")

        # Generate a response if required
        if generate_response:
            self.generate_response(commit_audio_buffer=False)

    def update_session(self, options: RealtimeAIOptions):
        """Updates the session configuration with the provided options."""
        if self._is_running:
            future = asyncio.run_coroutine_threadsafe(
                self._provider.update_session(options), self._provider_loop
            )
            future.result(timeout=5)

        self._options = options
        logger.info("RealtimeAIClient: Sent session update to server.")

    def generate_response(self, commit_audio_buffer: bool = True):
        """Generates a response from the provider."""
        logger.info("RealtimeAIClient: Generating response.")
        future = asyncio.run_coroutine_threadsafe(
            self._provider.generate_response(commit_audio_buffer), self._provider_loop
        )
        future.result(timeout=5)

    def cancel_response(self):
        """Cancels the current response from the provider."""
        future = asyncio.run_coroutine_threadsafe(
            self._provider.cancel_response(), self._provider_loop
        )
        future.result(timeout=5)

        logger.info("Client: Sent response.cancel event to server.")

    def truncate_response(self, item_id: str, content_index: int, audio_end_ms: int):
        """Truncates a response (provider-specific feature)."""
        future = asyncio.run_coroutine_threadsafe(
            self._provider.truncate_response(item_id, content_index, audio_end_ms),
            self._provider_loop,
        )
        future.result(timeout=5)

        logger.info("Client: Sent conversation.item.truncate event to server.")

    def clear_input_audio_buffer(self):
        """Clears the input audio buffer (provider-specific feature)."""
        future = asyncio.run_coroutine_threadsafe(
            self._provider.clear_input_audio_buffer(), self._provider_loop
        )
        future.result(timeout=5)

        logger.info("Client: Sent input_audio_buffer.clear event to server.")

    def generate_response_from_function_call(self, call_id: str, function_output: str):
        """
        Sends a function call result to the provider and generates a response.

        :param call_id: The ID of the function call.
        :param function_output: The output of the function call.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._provider.send_function_result(call_id, function_output),
            self._provider_loop,
        )
        future.result(timeout=5)

        logger.info("Function call output event sent.")

    def _consume_events(self):
        """Consume events from the provider."""
        logger.debug("Consume thread: Started consuming events.")

        # Consume from provider (asynchronous)
        asyncio.run(self._consume_provider_events())

        logger.debug("Consume thread: Stopped consuming events.")

    async def _consume_provider_events(self):
        """Async method to consume events from provider."""
        try:
            async for normalized_event in self._provider.receive_events():
                if self._stop_event.is_set():
                    break

                # Signal session ready on session.created or session.updated
                if isinstance(normalized_event, (SessionCreatedEvent, SessionUpdatedEvent)):
                    if not self._session_ready.is_set():
                        self._session_ready.set()
                        logger.debug("RealtimeAIClient: Session ready event received.")

                # Convert normalized event to OpenAI format for backward compatibility
                openai_event = self._to_openai_event(normalized_event)
                if openai_event and self.executor is not None:
                    self.executor.submit(self._handle_event, openai_event)

                await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"RealtimeAIClient: Error in consume_provider_events: {e}")

    def _handle_event(self, event: EventBase):
        """Handles the received event based on its type using the event handler."""
        event_type = event.type
        method_name = f"on_{event_type.replace('.', '_')}"
        handler = getattr(self._event_handler, method_name, None)

        if callable(handler):
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in handler {method_name} for event {event_type}: {e}"
                )
        else:
            self._event_handler.on_unhandled_event(event_type, vars(event))

    def _start_provider_loop(self):
        """Starts a background event loop for async provider operations."""

        def run_loop():
            self._provider_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._provider_loop)
            self._provider_loop.run_forever()

        self._provider_loop_thread = threading.Thread(
            target=run_loop, daemon=True, name="ProviderEventLoop"
        )
        self._provider_loop_thread.start()

        # Wait for loop to be ready
        timeout = 5
        start_time = time.time()
        while self._provider_loop is None:
            if time.time() - start_time > timeout:
                raise RuntimeError("Provider event loop failed to start")
            time.sleep(0.1)

    def _stop_provider_loop(self):
        """Stops the background event loop."""
        if self._provider_loop is not None:
            self._provider_loop.call_soon_threadsafe(self._provider_loop.stop)
            if self._provider_loop_thread is not None:
                self._provider_loop_thread.join(timeout=2)
            self._provider_loop = None
            self._provider_loop_thread = None

    def _to_openai_event(self, normalized_event: NormalizedEvent) -> EventBase:
        """
        Converts a normalized event back to OpenAI EventBase format.

        This provides backward compatibility with existing event handlers
        that expect OpenAI-formatted events.

        Args:
            normalized_event: Provider-agnostic normalized event

        Returns:
            EventBase: OpenAI-formatted event, or None if conversion not supported
        """
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
            return realtime_ai_events.ResponseAudioTranscriptDelta(
                event_id=normalized_event.event_id,
                type="response.audio_transcript.delta",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
                delta=normalized_event.delta,
            )

        elif isinstance(normalized_event, TranscriptDoneEvent):
            return realtime_ai_events.ResponseAudioTranscriptDone(
                event_id=normalized_event.event_id,
                type="response.audio_transcript.done",
                response_id=normalized_event.response_id,
                item_id=normalized_event.item_id,
                output_index=normalized_event.output_index,
                content_index=normalized_event.content_index,
                transcript=normalized_event.transcript,
            )

        elif isinstance(normalized_event, InputTranscriptCompletedEvent):
            return realtime_ai_events.ConversationItemInputAudioTranscriptionCompleted(
                event_id=normalized_event.event_id,
                type="conversation.item.input_audio_transcription.completed",
                item_id=normalized_event.item_id,
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
                name=normalized_event.function_name,
                arguments=normalized_event.arguments,
            )

        # Error events
        elif isinstance(normalized_event, ErrorEvent):
            error_details = realtime_ai_events.ErrorDetails(
                type=normalized_event.error_type,
                code=normalized_event.error_code,
                message=normalized_event.error_message,
                param=None,
                event_id=None,
            )
            return realtime_ai_events.ErrorEvent(
                event_id=normalized_event.event_id, type="error", error=error_details
            )

        # Speech events
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
                item_id=normalized_event.item_id,
                previous_item_id=normalized_event.previous_item_id,
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

        # Rate limit events
        elif isinstance(normalized_event, RateLimitsUpdatedEvent):
            return realtime_ai_events.RateLimitsUpdated(
                event_id=normalized_event.event_id,
                type="rate_limits.updated",
                rate_limits=[rl.__dict__ for rl in normalized_event.rate_limits],
            )

        # Unknown event type
        logger.warning(
            f"Unknown normalized event type: {type(normalized_event).__name__}"
        )
        return None

    @property
    def options(self):
        return self._options

    @property
    def is_running(self):
        return self._is_running

    # Optional: Ensure that threads are cleaned up if the object is deleted while running
    def __del__(self):
        if self._is_running:
            self.stop()
