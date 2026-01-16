# Realtime AI - Server-Side Integration API

A Python library for integrating real-time speech AI providers (OpenAI, Gemini, Grok) into server-side applications. Designed for headless operation without audio hardware dependencies.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Core API](#core-api)
  - [RealtimeAIClient](#realtimeaiclient)
  - [RealtimeAIOptions](#realtimeaioptions)
  - [AudioStreamOptions](#audiostreamoptions)
  - [RealtimeAIEventHandler](#realtimeaieventhandler)
- [Events Reference](#events-reference)
- [Audio Formats](#audio-formats)
- [Image/Vision Input](#imagevision-input)
- [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
- [Provider Support](#provider-support)
- [Server-Side Examples](#server-side-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Overview

The realtime-ai library provides a unified interface for real-time speech-to-speech AI interactions. Key features:

- **Hardware Independent**: No microphone/speaker dependencies - pure stream-based I/O
- **Multi-Provider**: Supports OpenAI, Google Gemini, and xAI Grok
- **Async & Sync**: Both synchronous and asynchronous client implementations
- **Local VAD**: Optional ONNX-based voice activity detection
- **Event-Driven**: Comprehensive event system for all AI interactions

### Use Cases

- WebSocket-based voice assistants
- Telephony integrations (Twilio, SIP)
- Multi-user voice applications
- Automated testing without audio hardware
- Server-side audio processing pipelines

---

## Installation

```bash
pip install realtime-ai

# Or from source
git clone https://github.com/jhakulin/realtime-ai.git
cd realtime-ai
pip install -e .
```

### Dependencies

```
websockets
numpy
onnxruntime  # For local VAD
resampy      # For audio resampling
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│  (WebSocket server, telephony handler, etc.)                │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              send_audio()        Event Callbacks
              send_text()         on_response_audio_delta()
              send_image()        on_response_done()
                    │                   │
                    ▼                   │
┌─────────────────────────────────────────────────────────────┐
│                   RealtimeAIClient                           │
│  - Manages provider connection                              │
│  - Handles audio streaming                                  │
│  - Dispatches events to handler                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Provider Layer                             │
│  OpenAIProvider | GeminiProvider | GrokProvider             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    AI Provider APIs
              (OpenAI, Gemini, Grok WebSocket)
```

---

## Quick Start

### Async Client (Recommended for Servers)

```python
import asyncio
import base64
from realtime_ai.aio.realtime_ai_client import RealtimeAIClient
from realtime_ai.aio.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions

class MyEventHandler(RealtimeAIEventHandler):
    async def on_response_audio_delta(self, event):
        # Receive audio from AI - forward to your client
        audio_bytes = base64.b64decode(event.delta)
        await self.send_to_client(audio_bytes)

    async def on_response_audio_transcript_done(self, event):
        print(f"Assistant: {event.transcript}")

    # Implement other required abstract methods...
    async def on_error(self, event): pass
    async def on_session_created(self, event): pass
    async def on_session_updated(self, event): pass
    async def on_input_audio_buffer_speech_started(self, event): pass
    async def on_input_audio_buffer_speech_stopped(self, event): pass
    async def on_input_audio_buffer_committed(self, event): pass
    async def on_conversation_item_created(self, event): pass
    async def on_conversation_item_input_audio_transcription_completed(self, event): pass
    async def on_response_created(self, event): pass
    async def on_response_content_part_added(self, event): pass
    async def on_response_content_part_done(self, event): pass
    async def on_response_output_item_added(self, event): pass
    async def on_response_output_item_done(self, event): pass
    async def on_response_audio_done(self, event): pass
    async def on_response_audio_transcript_delta(self, event): pass
    async def on_response_done(self, event): pass
    async def on_response_function_call_arguments_delta(self, event): pass
    async def on_response_function_call_arguments_done(self, event): pass
    async def on_rate_limits_updated(self, event): pass

async def main():
    options = RealtimeAIOptions(
        api_key="your-api-key",
        model="gpt-4o-realtime-preview",
        modalities=["audio", "text"],
        instructions="You are a helpful assistant.",
        voice="alloy",
    )

    stream_options = AudioStreamOptions(
        sample_rate=24000,
        channels=1,
        bytes_per_sample=2
    )

    handler = MyEventHandler()
    client = RealtimeAIClient(options, stream_options, handler)

    await client.start()

    # Send audio from your source (WebSocket, file, etc.)
    audio_chunk = get_audio_from_source()
    await client.send_audio(audio_chunk)

    # Or send text
    await client.send_text("Hello, how are you?")

    await client.stop()

asyncio.run(main())
```

### Sync Client

```python
from realtime_ai.realtime_ai_client import RealtimeAIClient
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler

# Same pattern, but with sync methods
client = RealtimeAIClient(options, stream_options, handler)
client.start()
client.send_audio(audio_chunk)
client.send_text("Hello")
client.stop()
```

---

## Core API

### RealtimeAIClient

The main client class for interacting with AI providers.

#### Constructor

```python
# Async version
from realtime_ai.aio.realtime_ai_client import RealtimeAIClient

client = RealtimeAIClient(
    options: RealtimeAIOptions,
    stream_options: AudioStreamOptions,
    event_handler: RealtimeAIEventHandler,
    provider: str = "openai"  # "openai", "gemini", or "grok"
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `await start()` | Connect to provider and start event processing |
| `await stop()` | Disconnect and clean up resources |
| `await send_audio(audio_data: bytes)` | Send raw audio bytes to provider |
| `await send_text(text: str, role: str = "user", generate_response: bool = True)` | Send text message |
| `await send_image(image_data: bytes, image_format: str = "png", generate_response: bool = True)` | Send image for vision processing (OpenAI, Gemini only) |
| `await generate_response(commit_audio_buffer: bool = True)` | Trigger AI response generation |
| `await cancel_response()` | Cancel current AI response |
| `await clear_input_audio_buffer()` | Clear buffered input audio |
| `await update_session(options: RealtimeAIOptions)` | Update session configuration |
| `await generate_response_from_function_call(call_id: str, output: str)` | Send function call result |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `options` | `RealtimeAIOptions` | Current session options |
| `is_running` | `bool` | Whether client is connected and running |

---

### RealtimeAIOptions

Configuration options for the AI session.

```python
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions

options = RealtimeAIOptions(
    # Required
    api_key: str,                    # Provider API key

    # Model configuration
    model: str = "gpt-4o-realtime-preview",
    modalities: List[str] = ["audio", "text"],
    instructions: str = "",          # System prompt
    voice: str = "alloy",            # Voice for audio output

    # Audio formats
    input_audio_format: str = "pcm16",
    output_audio_format: str = "pcm16",
    input_audio_transcription_model: str = "whisper-1",

    # Voice Activity Detection (server-side)
    turn_detection: Optional[dict] = {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
    },
    # Set to None to disable server VAD (use local VAD instead)

    # Function calling
    tools: List[dict] = [],          # Tool definitions
    tool_choice: str = "auto",       # "auto", "none", or "required"

    # Generation parameters
    temperature: float = 0.8,
    max_output_tokens: Optional[int] = None,
)
```

#### Voice Options

| Provider | Available Voices |
|----------|-----------------|
| OpenAI | `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`, `ballad`, `coral`, `sage`, `verse` |
| Gemini | `Puck`, `Charon`, `Kore`, `Fenrir`, `Aoede` |
| Grok | `Chill`, `Grok` |

---

### AudioStreamOptions

Configuration for audio streaming.

```python
from realtime_ai.models.audio_stream_options import AudioStreamOptions

stream_options = AudioStreamOptions(
    sample_rate: int = 24000,    # Hz (24000 recommended for speech)
    channels: int = 1,           # Mono
    bytes_per_sample: int = 2    # 16-bit PCM
)
```

---

### RealtimeAIEventHandler

Abstract base class for handling events from the AI provider. You must implement all abstract methods.

#### Async Version

```python
from realtime_ai.aio.realtime_ai_event_handler import RealtimeAIEventHandler

class MyHandler(RealtimeAIEventHandler):
    # All methods are async and must be implemented

    async def on_error(self, event: ErrorEvent) -> None:
        """Handle errors from the provider."""
        print(f"Error: {event.error.message}")

    async def on_session_created(self, event: SessionCreated) -> None:
        """Called when AI session is established."""
        pass

    async def on_session_updated(self, event: SessionUpdated) -> None:
        """Called when session configuration changes."""
        pass

    async def on_input_audio_buffer_speech_started(
        self, event: InputAudioBufferSpeechStarted
    ) -> None:
        """Server VAD detected user started speaking."""
        pass

    async def on_input_audio_buffer_speech_stopped(
        self, event: InputAudioBufferSpeechStopped
    ) -> None:
        """Server VAD detected user stopped speaking."""
        pass

    async def on_input_audio_buffer_committed(
        self, event: InputAudioBufferCommitted
    ) -> None:
        """Audio buffer was committed for processing."""
        pass

    async def on_conversation_item_created(
        self, event: ConversationItemCreated
    ) -> None:
        """New conversation item was created."""
        pass

    async def on_conversation_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompleted
    ) -> None:
        """User's speech was transcribed."""
        print(f"User said: {event.transcript}")

    async def on_response_created(self, event: ResponseCreated) -> None:
        """AI started generating a response."""
        pass

    async def on_response_content_part_added(
        self, event: ResponseContentPartAdded
    ) -> None:
        """New content part added to response."""
        pass

    async def on_response_content_part_done(
        self, event: ResponseContentPartDone
    ) -> None:
        """Content part finished."""
        pass

    async def on_response_output_item_added(
        self, event: ResponseOutputItemAdded
    ) -> None:
        """Output item added to response."""
        pass

    async def on_response_output_item_done(
        self, event: ResponseOutputItemDone
    ) -> None:
        """Output item completed."""
        pass

    async def on_response_audio_delta(self, event: ResponseAudioDelta) -> None:
        """
        Received audio chunk from AI.
        This is the main method for receiving AI audio output.
        """
        audio_bytes = base64.b64decode(event.delta)
        # Forward to your client, save to file, etc.

    async def on_response_audio_done(self, event: ResponseAudioDone) -> None:
        """AI finished sending audio for this response."""
        pass

    async def on_response_audio_transcript_delta(
        self, event: ResponseAudioTranscriptDelta
    ) -> None:
        """Real-time transcript of AI's speech."""
        print(event.delta, end="", flush=True)

    async def on_response_audio_transcript_done(
        self, event: ResponseAudioTranscriptDone
    ) -> None:
        """Complete transcript of AI's speech."""
        print(f"\nAssistant: {event.transcript}")

    async def on_response_done(self, event: ResponseDone) -> None:
        """AI finished generating the response."""
        status = event.response.get("status")
        # "completed", "cancelled", "failed"

    async def on_response_function_call_arguments_delta(
        self, event: ResponseFunctionCallArgumentsDelta
    ) -> None:
        """Streaming function call arguments."""
        pass

    async def on_response_function_call_arguments_done(
        self, event: ResponseFunctionCallArgumentsDone
    ) -> None:
        """Function call complete - execute and return result."""
        result = execute_function(event.call_id, event.arguments)
        await client.generate_response_from_function_call(
            event.call_id, result
        )

    async def on_rate_limits_updated(self, event: RateLimitsUpdated) -> None:
        """Rate limit information updated."""
        pass

    async def on_unhandled_event(
        self, event_type: str, event_data: dict
    ) -> None:
        """Called for any unhandled event types."""
        pass
```

---

## Events Reference

### Audio Events

| Event | Key Fields | Description |
|-------|------------|-------------|
| `ResponseAudioDelta` | `delta` (base64), `item_id`, `content_index` | Audio chunk from AI |
| `ResponseAudioDone` | `item_id`, `content_index` | Audio stream complete |

### Transcript Events

| Event | Key Fields | Description |
|-------|------------|-------------|
| `ResponseAudioTranscriptDelta` | `delta` | Real-time AI transcript chunk |
| `ResponseAudioTranscriptDone` | `transcript` | Complete AI transcript |
| `ConversationItemInputAudioTranscriptionCompleted` | `transcript` | User's speech transcribed |

### Session Events

| Event | Key Fields | Description |
|-------|------------|-------------|
| `SessionCreated` | `session` | Session established |
| `SessionUpdated` | `session` | Session config changed |

### VAD Events (Server-Side)

| Event | Key Fields | Description |
|-------|------------|-------------|
| `InputAudioBufferSpeechStarted` | `audio_start_ms`, `item_id` | User started speaking |
| `InputAudioBufferSpeechStopped` | `audio_end_ms`, `item_id` | User stopped speaking |
| `InputAudioBufferCommitted` | `item_id` | Audio committed for processing |

### Response Lifecycle

| Event | Key Fields | Description |
|-------|------------|-------------|
| `ResponseCreated` | `response` | Response generation started |
| `ResponseDone` | `response.status` | Response complete/cancelled/failed |

### Function Calling

| Event | Key Fields | Description |
|-------|------------|-------------|
| `ResponseFunctionCallArgumentsDelta` | `delta`, `call_id` | Streaming function args |
| `ResponseFunctionCallArgumentsDone` | `arguments`, `call_id` | Function call ready |

### Error Events

| Event | Key Fields | Description |
|-------|------------|-------------|
| `ErrorEvent` | `error.message`, `error.code` | Error from provider |

---

## Audio Formats

### Supported Formats

| Format | Description | Sample Rate | Bits |
|--------|-------------|-------------|------|
| `pcm16` | Raw PCM, little-endian | 24000 Hz | 16-bit |
| `g711_ulaw` | G.711 μ-law (telephony) | 8000 Hz | 8-bit |
| `g711_alaw` | G.711 A-law (telephony) | 8000 Hz | 8-bit |

### Audio Data Flow

```python
# Sending audio to AI
raw_pcm16_bytes = get_audio_from_source()  # Your audio source
await client.send_audio(raw_pcm16_bytes)

# Receiving audio from AI
async def on_response_audio_delta(self, event):
    # event.delta is base64-encoded PCM16
    audio_bytes = base64.b64decode(event.delta)
    send_to_client(audio_bytes)
```

### Converting Audio Formats

```python
import numpy as np

# Float32 to PCM16
def float32_to_pcm16(float_data: np.ndarray) -> bytes:
    pcm16 = (float_data * 32767).astype(np.int16)
    return pcm16.tobytes()

# PCM16 to Float32
def pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0
```

---

## Image/Vision Input

The library supports sending images for vision processing, enabling multimodal voice+vision applications.

### Supported Providers & Models

| Provider | Image Support | Model Required | Recommended Format |
|----------|--------------|----------------|-------------------|
| OpenAI | ✅ Yes | `gpt-realtime` | PNG, JPEG, WebP, GIF |
| OpenAI | ❌ No | `gpt-4o-realtime-preview` | N/A (audio/text only) |
| Gemini | ✅ Yes | `gemini-2.0-flash-exp` | JPEG (quality 90) |
| Grok | ❌ No | Any | N/A |

**Important:** For OpenAI image support, you must use the `gpt-realtime` model (released August 2025). The older `gpt-4o-realtime-preview` models do NOT support image input.

### Basic Usage

```python
# Send an image file
with open("screenshot.png", "rb") as f:
    image_data = f.read()

await client.send_image(image_data, image_format="png")

# Send with a question (no auto-response, then ask)
await client.send_image(image_data, generate_response=False)
await client.send_text("What do you see in this image?")
```

### Image Formats

| Format | MIME Type | Notes |
|--------|-----------|-------|
| `png` | image/png | Best for screenshots, diagrams |
| `jpeg` | image/jpeg | Best for photos, recommended for Gemini |
| `webp` | image/webp | Good compression |
| `gif` | image/gif | Static images only |

### Provider-Specific Details

**OpenAI:**
- **Requires `gpt-realtime` model** - older `gpt-4o-realtime-preview` does NOT support images
- Images sent via `conversation.item.create` with `input_image` type
- Supports high/low detail modes (affects token usage)
- Maximum size: 20MB recommended

**Gemini:**
- Images sent via `realtimeInput.video` field
- Recommended: JPEG at quality 90
- Native resolution: 768x768 (images resized automatically)

### Example: Screenshot Analysis

```python
import io
from PIL import Image

# Capture or load screenshot
screenshot = Image.open("error_screenshot.png")

# Convert to JPEG for Gemini compatibility
buffer = io.BytesIO()
screenshot.convert("RGB").save(buffer, format="JPEG", quality=90)
jpeg_bytes = buffer.getvalue()

# Send to AI with question
await client.send_image(jpeg_bytes, image_format="jpeg", generate_response=False)
await client.send_text("What error is shown in this screenshot?")
```

### Handling Unsupported Providers

```python
try:
    await client.send_image(image_data)
except NotImplementedError as e:
    # Provider doesn't support images (e.g., Grok)
    print(f"Image not supported: {e}")
    # Fallback: describe the image in text
    await client.send_text("I have a screenshot showing an error message...")
```

---

## Voice Activity Detection (VAD)

### Server-Side VAD (Provider-Based)

```python
options = RealtimeAIOptions(
    api_key="...",
    turn_detection={
        "type": "server_vad",
        "threshold": 0.5,           # 0.0-1.0, higher = less sensitive
        "prefix_padding_ms": 300,   # Audio to include before speech
        "silence_duration_ms": 500, # Silence before "speech ended"
    }
)
```

### Local VAD (Silero ONNX)

For lower latency and provider independence:

```python
from samples.utils.vad import SileroVoiceActivityDetector

vad = SileroVoiceActivityDetector(
    sample_rate=24000,
    chunk_size=1024,
    min_speech_duration=0.3,    # Seconds before "speech started"
    min_silence_duration=0.8,   # Seconds before "speech ended"
    model_path="silero_vad.onnx",
    threshold=0.5,
)

# Process audio chunks
def process_audio(audio_bytes: bytes):
    audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
    state_changed, is_speech = vad.process_audio_chunk(audio_chunk)

    if state_changed:
        if is_speech:
            on_speech_start()
        else:
            on_speech_end()
            client.generate_response()  # Trigger AI response
```

### VAD Comparison

| Feature | Server VAD | Local VAD |
|---------|------------|-----------|
| Latency | 200-500ms | 10-20ms |
| Provider support | OpenAI only | All providers |
| Customization | Limited | Full control |
| Offline | No | Yes |

---

## Provider Support

### OpenAI (Default)

```python
client = RealtimeAIClient(
    options=RealtimeAIOptions(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-realtime-preview",
    ),
    stream_options=stream_options,
    event_handler=handler,
    provider="openai"
)
```

### Google Gemini

```python
client = RealtimeAIClient(
    options=RealtimeAIOptions(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-2.0-flash-exp",
    ),
    stream_options=stream_options,
    event_handler=handler,
    provider="gemini"
)
```

### xAI Grok

```python
client = RealtimeAIClient(
    options=RealtimeAIOptions(
        api_key=os.environ["XAI_API_KEY"],
        model="grok-2-public",
    ),
    stream_options=stream_options,
    event_handler=handler,
    provider="grok"
)
```

### Provider Feature Matrix

| Feature | OpenAI | Gemini | Grok |
|---------|--------|--------|------|
| Audio I/O | ✅ | ✅ | ✅ |
| Image/Vision input | ✅ | ✅ | ❌ |
| Server VAD | ✅ | ❌ | ❌ |
| Function calling | ✅ | ✅ | ✅ |
| Transcription | ✅ | ✅ | ✅ |
| Multiple voices | ✅ | ✅ | ✅ |

---

## Server-Side Examples

### WebSocket Server (Full Example)

```python
import asyncio
import base64
import json
import websockets
from realtime_ai.aio.realtime_ai_client import RealtimeAIClient
from realtime_ai.aio.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions

class WebSocketBridgeHandler(RealtimeAIEventHandler):
    def __init__(self, client_ws):
        self.client_ws = client_ws
        self.ai_client = None

    def set_ai_client(self, client):
        self.ai_client = client

    async def _send_to_client(self, data: dict):
        await self.client_ws.send(json.dumps(data))

    async def on_response_audio_delta(self, event):
        await self._send_to_client({
            "type": "audio",
            "data": event.delta  # Already base64
        })

    async def on_response_audio_transcript_done(self, event):
        await self._send_to_client({
            "type": "transcript",
            "role": "assistant",
            "text": event.transcript
        })

    async def on_conversation_item_input_audio_transcription_completed(self, event):
        await self._send_to_client({
            "type": "transcript",
            "role": "user",
            "text": event.transcript
        })

    async def on_error(self, event):
        await self._send_to_client({
            "type": "error",
            "message": event.error.message
        })

    # Implement remaining abstract methods...
    async def on_session_created(self, event): pass
    async def on_session_updated(self, event): pass
    async def on_input_audio_buffer_speech_started(self, event):
        # Clear client audio on interruption
        await self._send_to_client({"type": "clear_audio"})
    async def on_input_audio_buffer_speech_stopped(self, event): pass
    async def on_input_audio_buffer_committed(self, event): pass
    async def on_conversation_item_created(self, event): pass
    async def on_response_created(self, event): pass
    async def on_response_content_part_added(self, event): pass
    async def on_response_content_part_done(self, event): pass
    async def on_response_output_item_added(self, event): pass
    async def on_response_output_item_done(self, event): pass
    async def on_response_audio_done(self, event): pass
    async def on_response_audio_transcript_delta(self, event): pass
    async def on_response_done(self, event): pass
    async def on_response_function_call_arguments_delta(self, event): pass
    async def on_response_function_call_arguments_done(self, event): pass
    async def on_rate_limits_updated(self, event): pass


async def handle_client(websocket):
    handler = WebSocketBridgeHandler(websocket)

    options = RealtimeAIOptions(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-realtime-preview",
        modalities=["audio", "text"],
        instructions="You are a helpful assistant.",
        voice="alloy",
        turn_detection={
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": 500,
        },
    )

    stream_options = AudioStreamOptions(
        sample_rate=24000,
        channels=1,
        bytes_per_sample=2
    )

    client = RealtimeAIClient(options, stream_options, handler)
    handler.set_ai_client(client)

    await client.start()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Raw audio from client
                await client.send_audio(message)
            else:
                data = json.loads(message)
                if data["type"] == "text":
                    await client.send_text(data["text"])
                elif data["type"] == "cancel":
                    await client.cancel_response()
    finally:
        await client.stop()


async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # Run forever

asyncio.run(main())
```

### Telephony Integration (Twilio)

```python
class TwilioHandler(RealtimeAIEventHandler):
    def __init__(self, twilio_stream):
        self.twilio = twilio_stream

    async def on_response_audio_delta(self, event):
        # Twilio expects μ-law encoded audio
        pcm_audio = base64.b64decode(event.delta)
        ulaw_audio = convert_pcm_to_ulaw(pcm_audio)
        await self.twilio.send_audio(ulaw_audio)

    # ... implement other methods


async def handle_twilio_call(twilio_ws):
    handler = TwilioHandler(twilio_ws)

    options = RealtimeAIOptions(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-realtime-preview",
        modalities=["audio", "text"],
        input_audio_format="g711_ulaw",
        output_audio_format="g711_ulaw",
        turn_detection=None,  # Use local VAD for telephony
    )

    stream_options = AudioStreamOptions(
        sample_rate=8000,  # Twilio uses 8kHz
        channels=1,
        bytes_per_sample=1
    )

    client = RealtimeAIClient(options, stream_options, handler)
    await client.start()

    async for audio_chunk in twilio_ws.receive_audio():
        await client.send_audio(audio_chunk)
```

### Automated Testing

```python
import wave

async def test_voice_assistant():
    received_audio = []
    received_transcripts = []

    class TestHandler(RealtimeAIEventHandler):
        async def on_response_audio_delta(self, event):
            received_audio.append(base64.b64decode(event.delta))

        async def on_response_audio_transcript_done(self, event):
            received_transcripts.append(event.transcript)

        # ... implement other methods

    # Load test audio
    with wave.open("test_input.wav", "rb") as f:
        test_audio = f.readframes(f.getnframes())

    handler = TestHandler()
    client = RealtimeAIClient(options, stream_options, handler)
    await client.start()

    # Send test audio in chunks
    chunk_size = 4800  # 100ms at 24kHz
    for i in range(0, len(test_audio), chunk_size):
        await client.send_audio(test_audio[i:i+chunk_size])
        await asyncio.sleep(0.1)

    # Wait for response
    await asyncio.sleep(5)
    await client.stop()

    # Assertions
    assert len(received_audio) > 0
    assert len(received_transcripts) > 0
```

---

## Error Handling

### Error Event Structure

```python
async def on_error(self, event: ErrorEvent):
    error = event.error
    print(f"Type: {error.type}")      # e.g., "invalid_request_error"
    print(f"Code: {error.code}")      # e.g., "invalid_api_key"
    print(f"Message: {error.message}")
```

### Common Errors

| Error Code | Cause | Solution |
|------------|-------|----------|
| `invalid_api_key` | Bad API key | Check environment variable |
| `rate_limit_exceeded` | Too many requests | Implement backoff/retry |
| `context_length_exceeded` | Conversation too long | Start new session |
| `server_error` | Provider issue | Retry with backoff |

### Connection Handling

```python
async def run_with_reconnect():
    while True:
        try:
            client = RealtimeAIClient(options, stream_options, handler)
            await client.start()
            # ... handle messages
        except websockets.exceptions.ConnectionClosed:
            print("Connection lost, reconnecting...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(5)
```

---

## Best Practices

### 1. Resource Management

```python
# Always use try/finally for cleanup
client = RealtimeAIClient(options, stream_options, handler)
try:
    await client.start()
    # ... your code
finally:
    await client.stop()
```

### 2. Audio Buffering

```python
# Buffer audio before sending for smoother streaming
BUFFER_SIZE = 4800  # 100ms at 24kHz
audio_buffer = bytearray()

def on_audio_received(chunk: bytes):
    audio_buffer.extend(chunk)
    while len(audio_buffer) >= BUFFER_SIZE:
        await client.send_audio(bytes(audio_buffer[:BUFFER_SIZE]))
        del audio_buffer[:BUFFER_SIZE]
```

### 3. Interruption Handling

```python
class Handler(RealtimeAIEventHandler):
    async def on_input_audio_buffer_speech_started(self, event):
        # Always clear client audio queue on interruption
        await self.send_to_client({"type": "clear_audio"})
        await self.ai_client.cancel_response()
```

### 4. Local VAD for Production

```python
# Use local VAD for lower latency in production
options = RealtimeAIOptions(
    turn_detection=None,  # Disable server VAD
)

# Initialize local VAD
vad = SileroVoiceActivityDetector(
    sample_rate=24000,
    min_silence_duration=0.8,
)
```

### 5. Logging

```python
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logging.getLogger("realtime_ai").setLevel(logging.DEBUG)
```

---

## API Version Compatibility

| Library Version | OpenAI API | Gemini API | Grok API |
|-----------------|------------|------------|----------|
| 0.1.x | Realtime v1 | Live v1 | Realtime v1 |

---

## Support

- **GitHub Issues**: https://github.com/jhakulin/realtime-ai/issues
- **Examples**: See `samples/` directory for complete examples

---

## License

MIT License - See LICENSE file for details.
