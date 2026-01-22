"""
Server-Side Streaming Example - No Audio Hardware Required

This example demonstrates how to use the realtime-ai library in a server
environment where audio comes from and goes to WebSocket clients rather
than local microphone/speakers.

Features:
- Local ONNX-based VAD (Silero) - no server-side VAD dependency
- Hardware-free operation - no microphone/speakers needed
- WebSocket bridge between clients and AI providers
- Image/vision support (OpenAI and Gemini providers)

Use Cases:
- Web applications with browser-based audio
- Telephony integrations (Twilio, SIP)
- Multi-user voice applications
- Headless server deployments
- CI/CD testing environments

Architecture:
    Browser/Client <--WebSocket--> This Server <--WebSocket--> AI Provider
                                       |
                                  Local VAD (Silero ONNX)

To run:
    1. Set OPENAI_API_KEY (for OpenAI) or XAI_API_KEY (for Grok) environment variable
    2. python sample_server_side_streaming.py
    3. Connect a WebSocket client to ws://localhost:8765

Client Protocol:
    - Send: Raw PCM16 audio bytes (24kHz, mono)
    - Send JSON messages:
        {"type": "text", "text": "hello"}
        {"type": "image", "data": "<base64>", "format": "png"}
    - Receive JSON messages:
        {"type": "audio", "data": "<base64 encoded audio>"}
        {"type": "transcript", "text": "..."}
        {"type": "status", "message": "..."}
        {"type": "vad", "event": "speech_started|speech_stopped"}
        {"type": "image_sent", "message": "..."}
        {"type": "image_error", "message": "..."}
"""

import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.vad import SileroVoiceActivityDetector, VoiceActivityDetector

from realtime_ai.aio.realtime_ai_client import RealtimeAIClient
from realtime_ai.aio.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.models.realtime_ai_events import (
    ConversationItemCreated,
    ConversationItemInputAudioTranscriptionCompleted,
    ErrorEvent,
    InputAudioBufferCommitted,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    RateLimitsUpdated,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDelta,
    ResponseAudioTranscriptDone,
    ResponseContentPartAdded,
    ResponseContentPartDone,
    ResponseCreated,
    ResponseDone,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    ResponseOutputItemAdded,
    ResponseOutputItemDone,
    SessionCreated,
    SessionUpdated,
)
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Reduce noise from libraries
logging.getLogger("realtime_ai").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Path to Silero VAD model
SCRIPT_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = SCRIPT_DIR / "../resources"
SILERO_MODEL_PATH = RESOURCES_DIR / "silero_vad.onnx"

# Configuration
USE_LOCAL_VAD = True  # Set to False to use server-side VAD instead
USE_SILERO_VAD = True  # Set to False to use simple RMS-based VAD


class LocalVADProcessor:
    """
    Processes audio through local VAD and triggers responses.

    This allows server-side speech detection without relying on
    the AI provider's server-side VAD, giving more control and
    reducing latency.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_size: int = 1024,
        use_silero: bool = True,
        on_speech_start=None,
        on_speech_end=None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self._audio_buffer = bytearray()
        self._is_speaking = False

        # Initialize VAD
        if use_silero and SILERO_MODEL_PATH.exists():
            logger.info(f"Using Silero VAD from {SILERO_MODEL_PATH}")
            self.vad = SileroVoiceActivityDetector(
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                min_speech_duration=0.3,
                min_silence_duration=0.8,  # Slightly faster for real-time
                model_path=str(SILERO_MODEL_PATH),
                threshold=0.5,
            )
        else:
            logger.info("Using simple RMS-based VAD")
            self.vad = VoiceActivityDetector(
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                window_duration=1.0,
                silence_ratio=1.5,
                min_speech_duration=0.3,
                min_silence_duration=0.8,
            )

    def process_audio(self, audio_bytes: bytes) -> None:
        """
        Process incoming audio bytes through VAD.

        Args:
            audio_bytes: Raw PCM16 audio data
        """
        # Add to buffer
        self._audio_buffer.extend(audio_bytes)

        # Process complete chunks
        bytes_per_chunk = self.chunk_size * 2  # 2 bytes per sample (PCM16)

        while len(self._audio_buffer) >= bytes_per_chunk:
            # Extract chunk
            chunk_bytes = bytes(self._audio_buffer[:bytes_per_chunk])
            del self._audio_buffer[:bytes_per_chunk]

            # Convert to numpy array
            audio_chunk = np.frombuffer(chunk_bytes, dtype=np.int16)

            # Process through VAD
            state_changed, is_speech = self.vad.process_audio_chunk(audio_chunk)

            if state_changed:
                if is_speech and not self._is_speaking:
                    self._is_speaking = True
                    logger.info("Local VAD: Speech started")
                    if self.on_speech_start:
                        asyncio.create_task(self._call_async(self.on_speech_start))

                elif not is_speech and self._is_speaking:
                    self._is_speaking = False
                    logger.info("Local VAD: Speech ended")
                    if self.on_speech_end:
                        asyncio.create_task(self._call_async(self.on_speech_end))

    async def _call_async(self, callback):
        """Helper to call sync or async callbacks."""
        if asyncio.iscoroutinefunction(callback):
            await callback()
        else:
            callback()

    def reset(self):
        """Reset VAD state."""
        self._audio_buffer.clear()
        self._is_speaking = False
        self.vad.reset()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


class ServerSideEventHandler(RealtimeAIEventHandler):
    """
    Event handler that bridges AI provider responses to a WebSocket client.

    This handler demonstrates server-side usage WITHOUT any audio hardware.
    All audio is forwarded to/from the connected WebSocket client.
    """

    def __init__(self, client_websocket: WebSocketServerProtocol, use_local_vad: bool = False):
        super().__init__()
        self._client_ws = client_websocket
        self._ai_client: Optional[RealtimeAIClient] = None
        self._is_responding = False
        self._vad_processor: Optional[LocalVADProcessor] = None
        self._use_local_vad = use_local_vad

    def set_ai_client(self, client: RealtimeAIClient):
        """Set the AI client reference for response control."""
        self._ai_client = client

    def set_vad_processor(self, vad: LocalVADProcessor):
        """Set the local VAD processor."""
        self._vad_processor = vad

    async def _send_to_client(self, message: dict):
        """Send a JSON message to the connected client."""
        try:
            await self._client_ws.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Client connection closed while sending message")
        except Exception as e:
            logger.error(f"Error sending to client: {e}")

    # =========================================================================
    # Local VAD Callbacks
    # =========================================================================

    async def on_local_speech_start(self):
        """Called when local VAD detects speech start."""
        logger.info(f"Local VAD speech start - is_responding: {self._is_responding}")

        # Always clear client audio queue when user starts speaking
        # This handles the case where audio is still playing from buffer
        # even after the server response has "completed"
        await self._send_to_client({"type": "clear_audio"})
        logger.info("Sent clear_audio to client")

        await self._send_to_client(
            {"type": "vad", "event": "speech_started", "source": "local"}
        )

        # Cancel ongoing response if still generating
        if self._is_responding and self._ai_client:
            logger.info("User interrupted active response - cancelling")
            await self._ai_client.cancel_response()
            await self._ai_client.clear_input_audio_buffer()
            self._is_responding = False

    async def on_local_speech_end(self):
        """Called when local VAD detects speech end."""
        await self._send_to_client(
            {"type": "vad", "event": "speech_stopped", "source": "local"}
        )

        # Trigger response generation
        if self._ai_client:
            logger.info("Local VAD: Generating response")
            await self._ai_client.generate_response()

    # =========================================================================
    # Audio Events - Forward audio to client
    # =========================================================================

    async def on_response_audio_delta(self, event: ResponseAudioDelta) -> None:
        """Forward audio chunks to the client."""
        self._is_responding = True
        if event.delta:
            await self._send_to_client(
                {
                    "type": "audio",
                    "data": event.delta,
                    "item_id": event.item_id,
                    "content_index": event.content_index,
                }
            )
            logger.debug(f"Forwarded audio chunk for item {event.item_id}")

    async def on_response_audio_done(self, event: ResponseAudioDone) -> None:
        """Notify client that audio stream for this response is complete."""
        await self._send_to_client(
            {
                "type": "audio_done",
                "item_id": event.item_id,
                "content_index": event.content_index,
            }
        )
        logger.debug(f"Audio done for item {event.item_id}")

    # =========================================================================
    # Transcript Events - Forward transcripts to client
    # =========================================================================

    async def on_response_audio_transcript_delta(
        self, event: ResponseAudioTranscriptDelta
    ) -> None:
        """Forward assistant's speech transcript to client in real-time."""
        if event.delta:
            await self._send_to_client(
                {
                    "type": "transcript_delta",
                    "role": "assistant",
                    "text": event.delta,
                    "response_id": event.response_id,
                }
            )

    async def on_response_audio_transcript_done(
        self, event: ResponseAudioTranscriptDone
    ) -> None:
        """Send complete assistant transcript to client."""
        if event.transcript:
            await self._send_to_client(
                {
                    "type": "transcript_done",
                    "role": "assistant",
                    "text": event.transcript,
                }
            )
            logger.info(f"Assistant: {event.transcript}")

    async def on_conversation_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompleted
    ) -> None:
        """Send user's speech transcript to client."""
        if event.transcript:
            await self._send_to_client(
                {"type": "transcript_done", "role": "user", "text": event.transcript}
            )
            logger.info(f"User: {event.transcript}")

    # =========================================================================
    # Session Events
    # =========================================================================

    async def on_session_created(self, event: SessionCreated) -> None:
        """Notify client that AI session is ready."""
        await self._send_to_client(
            {"type": "status", "message": "connected", "session": event.session}
        )
        logger.info("AI session created")

    async def on_session_updated(self, event: SessionUpdated) -> None:
        """Notify client of session configuration changes."""
        await self._send_to_client({"type": "status", "message": "session_updated"})
        logger.debug("AI session updated")

    # =========================================================================
    # Speech Detection Events (Server VAD - used if local VAD disabled)
    # =========================================================================

    async def on_input_audio_buffer_speech_started(
        self, event: InputAudioBufferSpeechStarted
    ) -> None:
        """Handle server-side VAD speech start (if enabled)."""
        # Skip server VAD when using local VAD to avoid conflicts/spurious triggers
        if self._use_local_vad:
            logger.debug(
                f"Server VAD: Ignoring speech_started (using local VAD) at {event.audio_start_ms}ms"
            )
            return

        logger.info(
            f"Server VAD: Speech started at {event.audio_start_ms}ms - is_responding: {self._is_responding}"
        )

        # Always clear client audio queue when user starts speaking
        await self._send_to_client({"type": "clear_audio"})
        logger.info("Sent clear_audio to client")

        await self._send_to_client(
            {
                "type": "vad",
                "event": "speech_started",
                "source": "server",
                "audio_start_ms": event.audio_start_ms,
            }
        )

        # Cancel ongoing response if still generating
        if self._is_responding and self._ai_client:
            logger.info("User interrupted active response - cancelling")
            await self._ai_client.cancel_response()
            await self._ai_client.clear_input_audio_buffer()
            self._is_responding = False

    async def on_input_audio_buffer_speech_stopped(
        self, event: InputAudioBufferSpeechStopped
    ) -> None:
        """Handle server-side VAD speech stop (if enabled)."""
        # Skip server VAD when using local VAD
        if self._use_local_vad:
            logger.debug(
                f"Server VAD: Ignoring speech_stopped (using local VAD) at {event.audio_end_ms}ms"
            )
            return

        await self._send_to_client(
            {
                "type": "vad",
                "event": "speech_stopped",
                "source": "server",
                "audio_end_ms": event.audio_end_ms,
            }
        )
        logger.info(f"Server VAD: Speech stopped at {event.audio_end_ms}ms")

    async def on_input_audio_buffer_committed(
        self, event: InputAudioBufferCommitted
    ) -> None:
        """Audio buffer committed for processing."""
        logger.debug(f"Audio buffer committed: {event.item_id}")

    # =========================================================================
    # Response Lifecycle Events
    # =========================================================================

    async def on_response_created(self, event: ResponseCreated) -> None:
        """AI started generating a response."""
        self._is_responding = True
        await self._send_to_client({"type": "status", "message": "response_started"})
        logger.debug("Response generation started")

    async def on_response_done(self, event: ResponseDone) -> None:
        """AI finished generating response."""
        self._is_responding = False
        status = event.response.get("status", "completed")
        await self._send_to_client(
            {"type": "status", "message": "response_done", "status": status}
        )
        logger.debug(f"Response completed with status: {status}")

    async def on_response_content_part_added(
        self, event: ResponseContentPartAdded
    ) -> None:
        """New content part added to response."""
        logger.debug(f"Content part added: {event.part}")

    async def on_response_content_part_done(
        self, event: ResponseContentPartDone
    ) -> None:
        """Content part completed."""
        logger.debug(f"Content part done: {event.part}")

    async def on_response_output_item_added(
        self, event: ResponseOutputItemAdded
    ) -> None:
        """Output item added to response."""
        logger.debug(f"Output item added: {event.item}")

    async def on_response_output_item_done(self, event: ResponseOutputItemDone) -> None:
        """Output item completed."""
        logger.debug(f"Output item done: {event.item}")

    # =========================================================================
    # Conversation Events
    # =========================================================================

    async def on_conversation_item_created(
        self, event: ConversationItemCreated
    ) -> None:
        """New conversation item created."""
        logger.debug(f"Conversation item created: {event.item}")

    # =========================================================================
    # Error Handling
    # =========================================================================

    async def on_error(self, event: ErrorEvent) -> None:
        """Forward errors to client."""
        error_msg = (
            event.error.message if hasattr(event.error, "message") else str(event.error)
        )
        await self._send_to_client({"type": "error", "message": error_msg})
        logger.error(f"AI Error: {error_msg}")

    # =========================================================================
    # Rate Limits
    # =========================================================================

    async def on_rate_limits_updated(self, event: RateLimitsUpdated) -> None:
        """Log rate limit updates."""
        for rate in event.rate_limits:
            name = rate.name if hasattr(rate, "name") else rate.get("name", "unknown")
            remaining = (
                rate.remaining
                if hasattr(rate, "remaining")
                else rate.get("remaining", 0)
            )
            logger.debug(f"Rate limit - {name}: {remaining} remaining")

    # =========================================================================
    # Function Calling
    # =========================================================================

    async def on_response_function_call_arguments_delta(
        self, event: ResponseFunctionCallArgumentsDelta
    ) -> None:
        """Function call arguments streaming."""
        logger.debug(f"Function args delta: {event.delta}")

    async def on_response_function_call_arguments_done(
        self, event: ResponseFunctionCallArgumentsDone
    ) -> None:
        """Function call complete."""
        logger.info(f"Function call: {event.call_id} with args: {event.arguments}")

    # =========================================================================
    # Catch-all
    # =========================================================================

    async def on_unhandled_event(
        self, event_type: str, event_data: Dict[str, Any]
    ) -> None:
        """Log any unhandled events."""
        logger.warning(f"Unhandled event: {event_type}")


async def handle_client_connection(websocket: WebSocketServerProtocol):
    """
    Handle a single client WebSocket connection.

    Creates an AI client for the connection and bridges audio
    between the client and AI provider, with optional local VAD.
    """
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected from {websocket.remote_address}")

    ai_client = None
    vad_processor = None

    try:
        # Get API key from environment - check both OpenAI and Grok
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "grok"

        if not api_key:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": "Server not configured: OPENAI_API_KEY or XAI_API_KEY not set",
                    }
                )
            )
            return

        # Create event handler for this client connection
        handler = ServerSideEventHandler(websocket, use_local_vad=USE_LOCAL_VAD)

        # Configure AI options
        # When using local VAD, disable server-side VAD (turn_detection=None)
        # Note: Use "gpt-realtime" for image/vision support
        #       Use "gpt-4o-realtime-preview" for audio/text only (no image support)
        # For Grok Voice Agent API, use "grok-3" (or "grok-2-public" for older)
        model = "gpt-realtime" if provider == "openai" else "grok-3"

        options = RealtimeAIOptions(
            api_key=api_key,
            model=model,
            modalities=["audio", "text"],
            instructions="You are a helpful assistant. Keep responses concise.",
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            # Disable server VAD when using local VAD
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            }
            if not USE_LOCAL_VAD
            else None,
        )

        # Audio stream configuration (24kHz, 16-bit mono)
        stream_options = AudioStreamOptions(
            sample_rate=24000, channels=1, bytes_per_sample=2
        )

        # Create AI client with the detected provider
        ai_client = RealtimeAIClient(
            options, stream_options, handler, provider=provider
        )
        handler.set_ai_client(ai_client)

        # Set up local VAD if enabled
        if USE_LOCAL_VAD:
            vad_processor = LocalVADProcessor(
                sample_rate=24000,
                chunk_size=1024,
                use_silero=USE_SILERO_VAD,
                on_speech_start=handler.on_local_speech_start,
                on_speech_end=handler.on_local_speech_end,
            )
            handler.set_vad_processor(vad_processor)
            logger.info(f"Local VAD enabled (Silero: {USE_SILERO_VAD})")
        else:
            logger.info("Using server-side VAD")

        # Connect to AI provider
        await ai_client.start()
        logger.info(f"AI client started for client {client_id}")

        # Process incoming messages from client
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Binary message = raw audio data from client
                    # Forward to AI provider
                    await ai_client.send_audio(message)

                    # Also process through local VAD if enabled
                    if vad_processor:
                        vad_processor.process_audio(message)

                elif isinstance(message, str):
                    # JSON message from client
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "audio":
                        # Base64 encoded audio
                        audio_bytes = base64.b64decode(data["data"])
                        await ai_client.send_audio(audio_bytes)
                        if vad_processor:
                            vad_processor.process_audio(audio_bytes)

                    elif msg_type == "text":
                        # Text input from client
                        text = data.get("text", "")
                        if text:
                            await ai_client.send_text(text)
                            logger.info(f"Sent text to AI: {text}")

                    elif msg_type == "image":
                        # Image input from client
                        image_data = data.get("data", "")
                        image_format = data.get("format", "png")
                        if image_data:
                            try:
                                image_bytes = base64.b64decode(image_data)
                                await ai_client.send_image(
                                    image_bytes, image_format=image_format
                                )
                                logger.info(
                                    f"Sent image to AI ({len(image_bytes)} bytes, format={image_format})"
                                )
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "image_sent",
                                            "message": "Image sent to AI",
                                        }
                                    )
                                )
                            except NotImplementedError as e:
                                logger.warning(f"Image not supported by provider: {e}")
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "image_error",
                                            "message": f"Provider does not support images: {PROVIDER}",
                                        }
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Error sending image: {e}")
                                await websocket.send(
                                    json.dumps(
                                        {"type": "image_error", "message": str(e)}
                                    )
                                )

                    elif msg_type == "generate":
                        # Explicit request to generate response
                        await ai_client.generate_response()

                    elif msg_type == "cancel":
                        # Cancel current response
                        await ai_client.cancel_response()

                    elif msg_type == "config":
                        # Runtime configuration
                        if "use_local_vad" in data:
                            # Could dynamically switch VAD mode
                            pass

                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

            except json.JSONDecodeError:
                logger.warning("Received invalid JSON from client")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
    finally:
        # Clean up
        if vad_processor:
            vad_processor.reset()

        if ai_client:
            try:
                await ai_client.stop()
                logger.info(f"AI client stopped for client {client_id}")
            except Exception as e:
                logger.error(f"Error stopping AI client: {e}")


async def main():
    """Main entry point - starts the WebSocket server."""
    host = os.getenv("SERVER_HOST", "localhost")
    port = int(os.getenv("SERVER_PORT", "8765"))

    logger.info("=" * 60)
    logger.info("Server-Side Streaming Example")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This example demonstrates using realtime-ai WITHOUT")
    logger.info("any audio hardware (no microphone/speakers needed).")
    logger.info("")
    logger.info("Audio flows: Client <-> This Server <-> AI Provider")
    logger.info("")

    # VAD configuration
    if USE_LOCAL_VAD:
        vad_type = "Silero ONNX" if USE_SILERO_VAD else "RMS-based"
        logger.info(f"VAD Mode: LOCAL ({vad_type})")
        if USE_SILERO_VAD:
            if SILERO_MODEL_PATH.exists():
                logger.info(f"Silero model: {SILERO_MODEL_PATH}")
            else:
                logger.warning(f"Silero model not found at {SILERO_MODEL_PATH}")
                logger.warning("Falling back to RMS-based VAD")
    else:
        logger.info("VAD Mode: SERVER (OpenAI server-side VAD)")

    logger.info("")
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    logger.info("")
    logger.info("Client Protocol:")
    logger.info("  - Send raw PCM16 audio as binary WebSocket messages")
    logger.info('  - Or send JSON: {"type": "audio", "data": "<base64>"}')
    logger.info('  - Or send JSON: {"type": "text", "text": "hello"}')
    logger.info(
        '  - Or send JSON: {"type": "image", "data": "<base64>", "format": "png"}'
    )
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("XAI_API_KEY"):
        logger.error("")
        logger.error(
            "ERROR: OPENAI_API_KEY or XAI_API_KEY environment variable not set!"
        )
        logger.error("Please set one before running:")
        logger.error("  For OpenAI: export OPENAI_API_KEY=your-key")
        logger.error("  For Grok: export XAI_API_KEY=your-key")
        logger.error("")
        return

    # Start WebSocket server
    async with websockets.serve(
        handle_client_connection,
        host,
        port,
        ping_interval=30,
        ping_timeout=10,
    ):
        logger.info(f"Server listening on ws://{host}:{port}")
        # Run forever
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
