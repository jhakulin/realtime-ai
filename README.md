## Overview

This Python project exemplifies a modular approach to interacting with OpenAI's Realtime WebSocket APIs. It enables the capture and processing of real-time audio by streaming it efficiently to the API for analysis or transcription.

---

### API usage

```python
from realtime_ai.realtime_ai_client import RealtimeAIClient
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models.realtime_ai_events import *
from user_functions import user_functions

# Setup your own functions
functions = FunctionTool(functions=user_functions)

class MyAudioCaptureEventHandler(AudioCaptureEventHandler):
   # Implementation of AudioCaptureEventHandler
   # Handles audio callbacks for user's audio capture and sends audio data to RealtimeClient
   # Detects speech start and end for response generation and interruption

class MyRealtimeEventHandler(RealtimeAIEventHandler):
   # Implementation of RealtimeAIEventHandler
   # Handles server events from the OpenAI Realtime service, audio playback data handling, function calling etc.

# Define RealtimeAIOptions for OpenAI Realtime service configuration
options = RealtimeAIOptions(
   api_key=api_key,
   model="gpt-4o-realtime-preview-2024-10-01",
   modalities=["audio", "text"],
   instructions="You are a helpful assistant. Respond concisely.",
   turn_detection=None, # or server vad
   tools=functions.definitions,
   tool_choice="auto",
   temperature=0.8,
   max_output_tokens=None
)

# Define AudioStreamOptions (currently only 16bit PCM 24kHz mono is supported)
stream_options = AudioStreamOptions(
   sample_rate=24000,
   channels=1,
   bytes_per_sample=2
)

# Initialize AudioPlayer to start waiting for audio to play
audio_player = AudioPlayer()

# Initialize RealtimeAIClient with event handler, creates websocket connection to service and ready to handle user's audio
event_handler = MyRealtimeEventHandler(audio_player=audio_player, functions=functions)
client = RealtimeAIClient(options, stream_options, event_handler)
event_handler.set_client(client)
client.start()

# Initialize AudioCapture with the event handler
audio_capture_event_handler = MyAudioCaptureEventHandler(
   client=client,
   event_handler=event_handler
)
audio_capture = AudioCapture(audio_capture_event_handler, ...)
```

### Installation

1. **Installation**:
   - Build realtime-ai wheel using following command: `python setup.py sdist bdist_wheel`
   - Go to generated `dist` folder
   - Install the generated wheel using following command: `pip install --force-reinstall realtime_ai-0.1.0-py3-none-any.whl`

2. **Setup**:
   - Replace placeholders like `"OPENAI_API_KEY"` in the sample script with real information.
   - Check system microphone access and settings to align with the project's audio requirements (e.g., 16bit PCM 24kHz mono).

3. **Execution**:
   - Run the script via command-line or an IDE:
     ```bash
     python samples/main.py
     ```

4. **Handling**:
   - Use the logger outputs to ensure successful connections and audio data transmissions.
   - Dive into provided methods to insert custom logic or explore further improvements.

## Contributions

Contributions in the form of issues or pull requests are welcome! Feel free to enhance functionalities, fix bugs, or improve documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
