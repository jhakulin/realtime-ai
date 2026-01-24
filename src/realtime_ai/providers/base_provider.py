"""
Base provider interface for realtime AI providers.

All realtime AI providers must implement this abstract base class to be
compatible with the RealtimeAIClient.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List
from realtime_ai.models.normalized_events import NormalizedEvent
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions


class BaseProvider(ABC):
    """
    Abstract base class for all realtime AI providers.

    All providers must implement this interface to be compatible
    with the RealtimeAIClient.
    """

    def __init__(self, options: RealtimeAIOptions):
        """
        Initialize provider with configuration.

        Args:
            options: Configuration options (may contain provider-specific fields)
        """
        self.options = options
        self._is_connected = False

    # ============================================================================
    # Connection Management
    # ============================================================================

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to provider service.

        Implementation should:
        - Open WebSocket/HTTP2 connection
        - Authenticate with provider
        - Send initial configuration/setup
        - Set self._is_connected = True

        Raises:
            ConnectionError: If connection fails
            Exception: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to provider service.

        Implementation should:
        - Gracefully close connection
        - Clean up resources
        - Set self._is_connected = False
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if provider is currently connected."""
        return self._is_connected

    # ============================================================================
    # Audio Operations
    # ============================================================================

    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to provider for processing.

        Args:
            audio_data: Raw audio bytes (PCM16 format)

        Implementation should:
        - Convert audio to provider's expected format if needed
        - Stream audio to provider
        - Handle buffering if required
        """
        pass

    # ============================================================================
    # Text Operations
    # ============================================================================

    @abstractmethod
    async def send_text(self, text: str, role: str = "user") -> None:
        """
        Send text message to provider.

        Args:
            text: Text content to send
            role: Message role ('user' or 'assistant')

        Implementation should:
        - Create provider-specific text message event
        - Send to provider
        """
        pass

    # ============================================================================
    # Session Management
    # ============================================================================

    @abstractmethod
    async def update_session(self, options: RealtimeAIOptions) -> None:
        """
        Update session configuration.

        Args:
            options: New configuration options

        Implementation should:
        - Map generic options to provider-specific config
        - Send configuration update to provider
        - Update self.options
        """
        pass

    # ============================================================================
    # Response Generation
    # ============================================================================

    @abstractmethod
    async def generate_response(self, commit_audio: bool = True) -> None:
        """
        Request provider to generate a response.

        Args:
            commit_audio: Whether to commit audio buffer before generating response

        Implementation should:
        - Commit audio buffer if requested (OpenAI-specific concept, optional)
        - Send response generation request to provider

        Note: Some providers may generate responses automatically (no explicit request needed)
        """
        pass

    @abstractmethod
    async def cancel_response(self) -> None:
        """
        Cancel ongoing response generation (for interruptions).

        Implementation should:
        - Send cancellation signal to provider
        - Clear any pending response data
        """
        pass

    # ============================================================================
    # Function Calling
    # ============================================================================

    @abstractmethod
    async def send_function_result(self, call_id: str, result: str) -> None:
        """
        Send function call result back to provider.

        Args:
            call_id: ID of the function call to respond to
            result: Function execution result (JSON string)

        Implementation should:
        - Create provider-specific function result message
        - Send to provider
        - May trigger automatic response generation (provider-dependent)
        """
        pass

    # ============================================================================
    # Event Streaming (Core of Multi-Provider Abstraction)
    # ============================================================================

    @abstractmethod
    def receive_events(self) -> AsyncIterator[NormalizedEvent]:
        """
        Async generator yielding normalized events from provider.

        Note: Implementations should use 'async def' for this method.
        The abstract signature omits 'async' for proper mypy type checking
        of async generators.

        Yields:
            NormalizedEvent: Normalized events (provider-agnostic)

        Implementation should:
        - Receive raw events from provider (WebSocket/HTTP2)
        - Parse provider-specific event format
        - Call normalize_incoming_event() to convert to normalized events
        - Yield each normalized event
        - Continue until connection closes

        Example:
            async for event in provider.receive_events():
                if event.event_type == EventType.AUDIO_DELTA:
                    play_audio(event.delta)
        """
        pass

    # ============================================================================
    # Event Normalization (Provider-Specific Logic)
    # ============================================================================

    @abstractmethod
    def normalize_incoming_event(self, raw_event: dict) -> List[NormalizedEvent]:
        """
        Convert provider-specific event to normalized event(s).

        Args:
            raw_event: Provider's native event format (dict)

        Returns:
            List[NormalizedEvent]: One or more normalized events

        Note: Returns list because some providers (Gemini) may emit one message
              that maps to multiple normalized events (event synthesis).

        Example:
            # OpenAI: 1:1 mapping
            {"type": "response.audio.delta", ...} → [AudioDeltaEvent(...)]

            # Gemini: 1:N mapping (event synthesis)
            {"serverContent": {...}} → [
                ResponseCreatedEvent(...),
                AudioDeltaEvent(...),
                TranscriptDeltaEvent(...),
                ResponseDoneEvent(...)
            ]
        """
        pass

    # ============================================================================
    # Provider Metadata
    # ============================================================================

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the name of the provider (e.g., 'openai', 'gemini', 'grok', 'nova').

        Used for:
        - Factory registration
        - Logging and debugging
        - Event metadata
        """
        pass

    # ============================================================================
    # Optional: Advanced Operations
    # ============================================================================

    async def truncate_response(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int
    ) -> None:
        """
        Truncate a response item (optional, provider-dependent).

        Only required for providers that support precise truncation (like OpenAI).
        Default implementation does nothing.

        Args:
            item_id: ID of the item to truncate
            content_index: Index of the content part to truncate
            audio_end_ms: Audio end time in milliseconds
        """
        pass

    async def clear_input_audio_buffer(self) -> None:
        """
        Clear input audio buffer (optional, provider-dependent).

        Only required for providers with explicit audio buffering (like OpenAI).
        Default implementation does nothing.
        """
        pass

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

        Default implementation does nothing (for providers without explicit buffering).
        """
        pass

    async def delete_conversation_item(self, item_id: str) -> None:
        """
        Delete a conversation item from the history.

        Args:
            item_id: The ID of the conversation item to delete.

        This triggers:
        - conversation.item.deleted event on success
        - error event if item doesn't exist

        Use this to remove specific items from conversation history.
        Default implementation does nothing.
        """
        pass

    async def reconnect(self) -> None:
        """
        Reconnect to get a fresh session with no conversation history.

        Use this when the provider doesn't support conversation.item.delete
        (e.g., Grok) and you need to clear conversation context.

        Default implementation does nothing.
        """
        pass

    # ============================================================================
    # Optional: Image/Vision Operations
    # ============================================================================

    async def send_image(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> None:
        """
        Send an image to the provider for vision processing.

        This is an optional capability - not all providers support images.
        Providers that support images should override this method.

        Args:
            image_data: Raw image bytes (PNG, JPEG, WebP, GIF)
            image_format: Image format ('png', 'jpeg', 'webp', 'gif')

        Raises:
            NotImplementedError: If provider doesn't support image input

        Supported providers:
            - OpenAI: Yes (via conversation.item.create with input_image)
            - Gemini: Yes (via realtime_input.video)
            - Grok: No (audio/text only)
        """
        raise NotImplementedError(
            f"Provider '{self.provider_name}' does not support image input"
        )
