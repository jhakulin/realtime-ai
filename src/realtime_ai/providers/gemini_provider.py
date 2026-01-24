"""
Gemini Provider for Google's Gemini Live API.

The Gemini Live API uses a different event model than OpenAI:
- Coarse-grained messages (setup/serverContent/toolCall)
- Single-field message structure (not type-based)
- No delta/done streaming pattern

This provider implements EVENT SYNTHESIS - converting coarse Gemini
messages into fine-grained normalized events (1:N mapping).

Key differences from OpenAI/Grok:
- WebSocket endpoint: wss://generativelanguage.googleapis.com/...
- Authentication: Google AI token (30-minute expiry)
- Message structure: {setup/clientContent/realtimeInput/toolResponse}
- Event synthesis: One serverContent → Multiple normalized events
"""

import time
import json
import uuid
import base64
import asyncio
import logging
import websockets
from typing import AsyncIterator, List, Optional, Dict, Any, TYPE_CHECKING
from realtime_ai.providers.base_provider import BaseProvider

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.normalized_events import (
    NormalizedEvent, EventType,
    SessionCreatedEvent, SessionUpdatedEvent,
    AudioDeltaEvent, AudioDoneEvent,
    TranscriptDeltaEvent, TranscriptDoneEvent,
    InputTranscriptCompletedEvent,
    SpeechStartedEvent, SpeechStoppedEvent,
    ResponseCreatedEvent, ResponseDoneEvent,
    FunctionCallEvent, ErrorEvent,
    AudioBufferCommittedEvent, ConversationItemCreatedEvent
)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """
    Provider implementation for Google's Gemini Live API.

    Implements event synthesis to convert Gemini's coarse-grained
    messages into fine-grained normalized events.

    Key Features:
    - Event synthesis (1:N mapping)
    - State-based response tracking
    - Google AI token authentication
    - Voice Activity Detection (VAD) with sensitivity levels
    """

    def __init__(self, options: RealtimeAIOptions):
        """
        Initialize Gemini provider.

        Args:
            options: Configuration options. Should include Google AI API key.
        """
        super().__init__(options)

        # State tracking for event synthesis
        self._current_response_id: Optional[str] = None
        self._current_item_id: Optional[str] = None
        self._accumulated_transcript: str = ""
        self._response_active: bool = False
        self._output_index: int = 0
        self._content_index: int = 0

        # WebSocket connection state
        self._websocket: Optional["ClientConnection"] = None
        self._setup_complete = False
        self._receive_task: Optional[asyncio.Task] = None

        # Event queue for receive_events
        self._event_queue: asyncio.Queue = asyncio.Queue()

    @property
    def provider_name(self) -> str:
        """Returns the provider name."""
        return "gemini"

    async def connect(self) -> None:
        """
        Establishes connection to Gemini Live API.

        Endpoint: wss://generativelanguage.googleapis.com/ws/...
        Authentication: Google AI API key as query parameter
        """
        if self._websocket:
            logger.warning("GeminiProvider: Already connected")
            return

        try:
            # Build WebSocket URL with API key authentication
            base_url = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
            url = f"{base_url}?key={self.options.api_key}"

            logger.info(f"GeminiProvider: Connecting to Gemini Live API...")
            self._websocket = await websockets.connect(url)
            self._is_connected = True
            logger.info("GeminiProvider: WebSocket connection established")

            # Start receiving messages in background
            self._receive_task = asyncio.create_task(self._receive_messages())

            # Send setup message
            await self._send_setup_message()

        except Exception as e:
            logger.error(f"GeminiProvider: Connection error: {e}")
            self._is_connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnects from Gemini API."""
        self._is_connected = False
        self._setup_complete = False

        # Cancel receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
                logger.info("GeminiProvider: WebSocket closed gracefully")
            except Exception as e:
                logger.error(f"GeminiProvider: Error closing WebSocket: {e}")
            finally:
                self._websocket = None

        self._reset_response_state()

    def _reset_response_state(self) -> None:
        """Resets response state after response completes."""
        self._current_response_id = None
        self._current_item_id = None
        self._accumulated_transcript = ""
        self._response_active = False
        self._output_index = 0
        self._content_index = 0

    def _generate_event_id(self) -> str:
        """Generates a unique event ID."""
        return f"evt_{uuid.uuid4().hex[:16]}"

    async def _send_setup_message(self) -> None:
        """
        Sends the initial setup message to configure the Gemini session.
        """
        setup_message = {
            "setup": {
                "model": f"models/{self.options.model}",
                "generationConfig": {
                    "temperature": self.options.temperature if hasattr(self.options, 'temperature') else 0.8,
                    "maxOutputTokens": self.options.max_output_tokens if hasattr(self.options, 'max_output_tokens') else 1024,
                },
                "systemInstruction": {
                    "parts": [{"text": self.options.instructions or "You are a helpful assistant."}]
                },
                "speechConfig": {
                    "audioEncoding": "pcm"
                }
            }
        }

        # Add tools if configured
        if hasattr(self.options, 'tools') and self.options.tools:
            setup_message["setup"]["tools"] = self.options.tools  # type: ignore[assignment]

        await self._send_message(setup_message)
        logger.info("GeminiProvider: Setup message sent")

    async def _send_message(self, message: dict) -> None:
        """
        Sends a message via WebSocket.

        Args:
            message: Message dictionary to send
        """
        if not self._websocket:
            logger.debug("GeminiProvider: WebSocket not connected, message not sent (test mode)")
            return

        try:
            message_str = json.dumps(message)
            await self._websocket.send(message_str)
            logger.debug(f"GeminiProvider: Sent message: {message_str}")
        except Exception as e:
            logger.error(f"GeminiProvider: Send failed: {e}")
            raise

    async def _receive_messages(self) -> None:
        """
        Listens for incoming WebSocket messages and processes them.
        Runs in background task started by connect().
        """
        try:
            async for message in self._websocket:
                try:
                    gemini_message = json.loads(message)
                    logger.debug(f"GeminiProvider: Received message: {gemini_message}")

                    # Handle setupComplete
                    if "setupComplete" in gemini_message:
                        self._setup_complete = True
                        logger.info("GeminiProvider: Setup complete")

                    # Normalize and queue events
                    normalized_events = self.normalize_incoming_event(gemini_message)
                    for event in normalized_events:
                        await self._event_queue.put(event)

                except json.JSONDecodeError as e:
                    logger.error(f"GeminiProvider: Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"GeminiProvider: Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"GeminiProvider: Connection closed: {e.code} - {e.reason}")
            self._is_connected = False
        except asyncio.CancelledError:
            logger.info("GeminiProvider: Receive task cancelled")
        except Exception as e:
            logger.error(f"GeminiProvider: Error in receive loop: {e}")
            self._is_connected = False

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Sends audio data to Gemini using realtimeInput message.

        Gemini format:
        {
            "realtimeInput": {
                "mediaChunks": [
                    {"data": "<base64>", "mimeType": "audio/pcm"}
                ]
            }
        }

        Args:
            audio_data: Raw audio bytes
        """
        # Encode to base64
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')

        # Send as realtimeInput message
        message = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "data": encoded_audio,
                        "mimeType": "audio/pcm"
                    }
                ]
            }
        }

        await self._send_message(message)

    async def send_text(self, text: str, role: str = "user") -> None:
        """
        Sends text message to Gemini using clientContent message.

        Gemini format:
        {
            "clientContent": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": "Hello"}]
                    }
                ],
                "turnComplete": True
            }
        }

        Args:
            text: Text content
            role: Message role (user/model)
        """
        message = {
            "clientContent": {
                "turns": [
                    {
                        "role": role,
                        "parts": [
                            {"text": text}
                        ]
                    }
                ],
                "turnComplete": True
            }
        }

        await self._send_message(message)

    async def send_image(
        self,
        image_data: bytes,
        image_format: str = "jpeg",
    ) -> None:
        """
        Send image to Gemini Live API.

        Images are sent via realtimeInput with video field.
        Gemini treats images as single video frames.

        Gemini format:
        {
            "realtimeInput": {
                "video": {
                    "mimeType": "image/jpeg",
                    "data": "<base64>"
                }
            }
        }

        Args:
            image_data: Raw image bytes (JPEG recommended, PNG also supported)
            image_format: Image format ('jpeg', 'png', 'webp', 'gif')

        Note:
            Gemini recommends JPEG format at quality 90 for best results.
            Native resolution is 768x768.
        """
        # Encode to base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Map format to MIME type
        mime_type = f"image/{image_format}"

        logger.info(
            f"GeminiProvider: Sending image (size={len(image_data)} bytes, format={image_format})"
        )

        # Send as realtimeInput message with video field
        message = {
            "realtimeInput": {
                "video": {
                    "mimeType": mime_type,
                    "data": encoded_image
                }
            }
        }

        await self._send_message(message)
        logger.info(
            f"GeminiProvider: Successfully sent image (size={len(image_data)} bytes)"
        )

    async def update_session(self, options: RealtimeAIOptions) -> None:
        """
        Updates the session configuration using setup message.

        Gemini format:
        {
            "setup": {
                "model": "models/gemini-2.0-flash-exp",
                "systemInstruction": {"parts": [{"text": "..."}]},
                "tools": [...],
                "speechConfig": {
                    "voiceConfig": {...},
                    "audioEncoding": "pcm",
                    "activityDetectionConfig": {...}
                }
            }
        }

        Args:
            options: Updated configuration options
        """
        # Update options
        self.options = options

        setup_message = {
            "setup": {
                "model": f"models/{options.model}",
                "systemInstruction": {
                    "parts": [{"text": options.instructions}]
                },
                "speechConfig": {
                    "audioEncoding": "pcm"
                }
            }
        }

        await self._send_message(setup_message)

    async def generate_response(self, commit_audio: bool = True) -> None:
        """
        Triggers response generation from Gemini.

        In Gemini, responses are triggered implicitly when:
        1. clientContent is sent with turnComplete=True
        2. realtimeInput reaches silence threshold (VAD)

        No explicit "response.create" message needed.

        Args:
            commit_audio: Ignored for Gemini (automatic via VAD)
        """
        # Gemini doesn't have explicit response generation
        # Responses are automatic based on turn completion or VAD
        pass

    async def cancel_response(self) -> None:
        """
        Cancels the current response generation.

        Gemini approach: Send clientContent with turnComplete=True
        to interrupt the model.
        """
        # Send interruption via clientContent
        message = {
            "clientContent": {
                "turns": [],
                "turnComplete": True
            }
        }

        await self._send_message(message)

        # Reset response state
        self._reset_response_state()

    async def send_function_result(self, call_id: str, result: str) -> None:
        """
        Sends function call result back to Gemini using toolResponse message.

        Gemini format:
        {
            "toolResponse": {
                "functionResponses": [
                    {
                        "id": "call_id",
                        "response": {
                            "result": "function output"
                        }
                    }
                ]
            }
        }

        Args:
            call_id: Function call ID from FunctionCallEvent
            result: Function execution result (JSON string)
        """
        message = {
            "toolResponse": {
                "functionResponses": [
                    {
                        "id": call_id,
                        "response": json.loads(result) if result else {}
                    }
                ]
            }
        }

        await self._send_message(message)

    async def receive_events(self) -> AsyncIterator[NormalizedEvent]:
        """
        Receives events from Gemini as an async iterator.

        Yields normalized events synthesized from Gemini messages.
        Events are queued by _receive_messages() background task.

        Yields:
            NormalizedEvent: Provider-agnostic event
        """
        while self._is_connected:
            try:
                # Wait for event with timeout to allow checking _is_connected
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                # No event available, continue loop
                continue
            except Exception as e:
                logger.error(f"GeminiProvider: Error in receive_events: {e}")
                break

        # Drain remaining events from queue before stopping
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                yield event
            except asyncio.QueueEmpty:
                break

    def normalize_incoming_event(self, raw_event: dict) -> List[NormalizedEvent]:
        """
        Converts Gemini message to normalized events.

        CRITICAL: Implements event synthesis (1:N mapping).
        One Gemini message can produce multiple normalized events.

        Message types:
        - setupComplete → session.created
        - serverContent → response.created, audio.delta, transcript.delta, *.done
        - toolCall → function.call
        - toolCallCancellation → ignored

        Args:
            raw_event: Raw Gemini message dictionary

        Returns:
            List of normalized events (0 to 7 events per message)
        """
        events: List[NormalizedEvent] = []
        timestamp = time.time()

        # setupComplete - Session initialized
        if "setupComplete" in raw_event:
            self._setup_complete = True
            events.append(SessionCreatedEvent(
                event_id=self._generate_event_id(),
                event_type=EventType.SESSION_CREATED,
                timestamp=timestamp,
                provider="gemini",
                raw_event=raw_event,
                session_id=f"sess_{uuid.uuid4().hex[:16]}",
                config={}
            ))
            return events

        # serverContent - THE COMPLEX ONE (synthesizes multiple events)
        if "serverContent" in raw_event:
            server_content = raw_event["serverContent"]

            # Handle input transcription (user speech recognized)
            if "inputAudioTranscription" in server_content:
                events.append(InputTranscriptCompletedEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.INPUT_TRANSCRIPT_COMPLETED,
                    timestamp=timestamp,
                    provider="gemini",
                    raw_event=raw_event,
                    item_id=self._current_item_id or f"item_{uuid.uuid4().hex[:16]}",
                    content_index=0,
                    transcript=server_content["inputAudioTranscription"]
                ))

            # Handle model response (audio + transcript)
            model_turn = server_content.get("modelTurn", {})
            parts = model_turn.get("parts", [])

            if parts:
                # First message in response? Emit response.created
                if not self._response_active:
                    self._response_active = True
                    self._current_response_id = f"resp_{uuid.uuid4().hex[:16]}"
                    self._current_item_id = f"item_{uuid.uuid4().hex[:16]}"
                    self._output_index = 0
                    self._content_index = 0

                    events.append(ResponseCreatedEvent(
                        event_id=self._generate_event_id(),
                        event_type=EventType.RESPONSE_CREATED,
                        timestamp=timestamp,
                        provider="gemini",
                        raw_event=raw_event,
                        response_id=self._current_response_id
                    ))

                # Process each part (audio, text, etc.)
                for part in parts:
                    # Audio part
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        audio_data = inline_data.get("data", "")

                        events.append(AudioDeltaEvent(
                            event_id=self._generate_event_id(),
                            event_type=EventType.AUDIO_DELTA,
                            timestamp=timestamp,
                            provider="gemini",
                            raw_event=raw_event,
                            response_id=self._current_response_id or "",
                            item_id=self._current_item_id or "",
                            output_index=self._output_index,
                            content_index=self._content_index,
                            delta=audio_data
                        ))

                    # Text part (transcript)
                    if "text" in part:
                        text = part["text"]
                        self._accumulated_transcript += text

                        events.append(TranscriptDeltaEvent(
                            event_id=self._generate_event_id(),
                            event_type=EventType.TRANSCRIPT_DELTA,
                            timestamp=timestamp,
                            provider="gemini",
                            raw_event=raw_event,
                            response_id=self._current_response_id or "",
                            item_id=self._current_item_id or "",
                            output_index=self._output_index,
                            content_index=self._content_index,
                            delta=text
                        ))

            # Turn complete? Emit done events (check OUTSIDE of parts block)
            if server_content.get("turnComplete", False) and self._response_active:
                # Audio done
                events.append(AudioDoneEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.AUDIO_DONE,
                    timestamp=timestamp,
                    provider="gemini",
                    raw_event=raw_event,
                    response_id=self._current_response_id or "",
                    item_id=self._current_item_id or "",
                    output_index=self._output_index,
                    content_index=self._content_index
                ))

                # Transcript done
                events.append(TranscriptDoneEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.TRANSCRIPT_DONE,
                    timestamp=timestamp,
                    provider="gemini",
                    raw_event=raw_event,
                    response_id=self._current_response_id or "",
                    item_id=self._current_item_id or "",
                    output_index=self._output_index,
                    content_index=self._content_index,
                    transcript=self._accumulated_transcript
                ))

                # Response done
                events.append(ResponseDoneEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.RESPONSE_DONE,
                    timestamp=timestamp,
                    provider="gemini",
                    raw_event=raw_event,
                    response_id=self._current_response_id or "",
                    status="completed"
                ))

                # Reset state for next response
                self._reset_response_state()

            return events

        # toolCall - Function execution request
        if "toolCall" in raw_event:
            tool_call = raw_event["toolCall"]
            function_calls = tool_call.get("functionCalls", [])

            for func_call in function_calls:
                events.append(FunctionCallEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.FUNCTION_CALL,
                    timestamp=timestamp,
                    provider="gemini",
                    raw_event=raw_event,
                    response_id=self._current_response_id or "",
                    item_id=self._current_item_id or "",
                    output_index=self._output_index,
                    call_id=func_call.get("id", ""),
                    function_name=func_call.get("name", ""),
                    arguments=json.dumps(func_call.get("args", {}))
                ))

            return events

        # toolCallCancellation - Tool cancellation (Gemini-specific)
        if "toolCallCancellation" in raw_event:
            # Could emit error event or just ignore
            # For now, ignore (return empty list)
            return events

        # Unknown message type
        return events

    # Optional features (Gemini-specific implementations)

    async def truncate_response(self, item_id: str, content_index: int, audio_end_ms: int) -> None:
        """
        Truncates a response.

        Gemini doesn't have this feature - implement as no-op.
        """
        pass

    async def clear_input_audio_buffer(self) -> None:
        """
        Clears the input audio buffer.

        Gemini doesn't have explicit buffer management - implement as no-op.
        """
        pass

    async def commit_audio_buffer(self) -> None:
        """
        Commit the input audio buffer without generating a response.

        Gemini doesn't have explicit audio buffer management like OpenAI/Grok.
        Audio is processed automatically via VAD. This is a no-op for Gemini.
        """
        logger.debug(
            "GeminiProvider: commit_audio_buffer is a no-op (Gemini uses automatic VAD)"
        )
