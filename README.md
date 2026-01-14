## Overview

This Python library provides a unified interface for interacting with multiple realtime AI providers through WebSocket APIs. It enables real-time audio and text conversations with AI assistants, supporting **OpenAI**, **Grok (xAI)**, and **Gemini (Google)** out of the box.

The library features a clean provider abstraction that allows seamless switching between AI providers while maintaining the same application code. It includes advanced audio processing capabilities such as local voice activity detection and keyword detection using Azure Speech Services.

### Key Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Grok (xAI), and Gemini (Google) using a unified API. Add custom providers by implementing the simple provider interface.

- **Real-time Audio and Text Interaction**: Capture and stream audio data to realtime AI providers, enabling seamless conversations through both speech and text, with the ability to interrupt the assistant for dynamic and interactive dialogue.

- **Local Voice Activity Detection (VAD)**: Built-in voice activity detector identifies when speech starts and ends, efficiently managing audio data to ensure only relevant speech segments are processed, optimizing performance and reducing costs.

- **Keyword Detection**: Integrated with Azure Speech Services, the application supports keyword detection to trigger interactions with the AI assistant. By listening for specific trigger words (e.g., "Computer"), audio is only sent when necessary, enhancing privacy and reducing costs.

- **Provider Abstraction**: Clean architecture with zero OpenAI-specific assumptions in core code. Event normalization layer ensures consistent behavior across all providers.

- **Modular Design**: Structured for easy customization and extension. Define your own functions, event handlers, and even custom providers to tailor the application's behavior to specific needs.

- **Multi-Modal Interaction**: Supports both audio and text modalities across all providers, enabling versatile interaction patterns.

- **Configurable AI Options**: Configure model selection, temperature settings, voice options, tool usage, and more to fine-tune the assistant's responses and behavior.

---

## Multi-Provider Support

The library supports multiple realtime AI providers through a unified interface. Switch providers with a single parameter - all other code remains the same.

### Supported Providers

| Provider | Model | Status | Audio | Text | Function Calling | Voice Options |
|----------|-------|--------|-------|------|------------------|---------------|
| **OpenAI** | gpt-4o-realtime-preview | ✅ Production | ✅ | ✅ | ✅ | alloy, echo, shimmer, sage, ash, coral |
| **Grok** | grok-2-voice | 🚧 Beta | 🚧 | 🚧 | 🚧 | ara, rex, sal, eve, leo |
| **Gemini** | gemini-2.0-flash-exp | ✅ Production | ✅ | ✅ | ✅ | Puck, Charon, Kore, Fenrir, Aoede |

**Status Legend:**
- ✅ **Production**: Fully implemented and tested with real API (WebSocket complete)
- 🚧 **Beta**: Architecture complete, endpoint configuration pending (minor work)

### Quick Start Examples

#### OpenAI (Default)

```python
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.realtime_ai_client import RealtimeAIClient

options = RealtimeAIOptions(
    api_key="sk-...",  # OpenAI API key
    model="gpt-4o-realtime-preview",
    voice="sage",
    modalities=["audio", "text"],
    instructions="You are a helpful assistant."
)

# Provider defaults to "openai" - no need to specify
client = RealtimeAIClient(options, stream_options, event_handler)
client.start()
```

#### Grok (xAI)

```python
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.realtime_ai_client import RealtimeAIClient

options = RealtimeAIOptions(
    api_key="xai-...",  # xAI API key
    model="grok-2-voice",
    voice="ara",  # Grok voice personalities: ara, rex, sal, eve, leo
    modalities=["audio", "text"],
    instructions="You are a helpful assistant."
)

# Specify provider="grok" to use Grok
client = RealtimeAIClient(options, stream_options, event_handler, provider="grok")
client.start()
```

#### Gemini (Google)

```python
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.realtime_ai_client import RealtimeAIClient

options = RealtimeAIOptions(
    api_key="google-api-key",  # Google AI API key
    model="gemini-2.0-flash-exp",
    voice="Puck",  # Gemini voices: Puck, Charon, Kore, Fenrir, Aoede
    modalities=["audio", "text"],
    instructions="You are a helpful assistant."
)

# Specify provider="gemini" to use Gemini
client = RealtimeAIClient(options, stream_options, event_handler, provider="gemini")
client.start()
```

### Migration Guide

Existing code works without changes (defaults to OpenAI). To switch providers:

1. **Update API key** - Use provider-specific key
2. **Update model name** - Use provider-specific model
3. **Update voice (optional)** - Use provider-specific voice
4. **Add provider parameter** - Specify which provider to use

**That's it!** No other code changes needed - event handlers, audio streaming, and all other functionality work identically across providers.

### Provider-Specific Features

#### OpenAI ✅ Production Ready
- Advanced function calling with tool choice
- Response truncation and audio buffer control
- Azure OpenAI endpoint support
- **Status**: Fully functional with real API connections

#### Grok (xAI) 🚧 Beta
- OpenAI-compatible API (easy migration)
- Built-in web search and X (Twitter) search tools
- Five distinct voice personalities
- **Status**: Architecture complete, event handling tested. WebSocket endpoint configuration pending for real API connections.

#### Gemini (Google) ✅ Production Ready
- Event synthesis for OpenAI-compatible events (1:N mapping)
- Native Google AI WebSocket integration
- Advanced conversation capabilities
- Real-time bidirectional audio streaming
- **Status**: Fully functional with real API connections. WebSocket implementation complete, all features tested.

### Implementation Roadmap

**Current Release (v1.1)**:
- ✅ Multi-provider architecture (production ready)
- ✅ OpenAI provider (fully functional)
- ✅ Gemini provider (fully functional)
- 🚧 Grok provider (architecture complete, endpoint config needed)

**Upcoming**:
- Complete Grok WebSocket endpoint configuration (v1.2)
- Add integration tests with real APIs
- Performance optimizations

All providers share the same interface. OpenAI and Gemini are production-ready, and Grok will be fully functional once endpoint configuration is complete (no code changes required).

---

## Example API usage

```python
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.models.realtime_ai_events import *
from realtime_ai.realtime_ai_client import RealtimeAIClient
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler
from user_functions import user_functions

# Setup your own functions
functions = FunctionTool(functions=user_functions)

class MyAudioCaptureEventHandler(AudioCaptureEventHandler):
   # Implementation of AudioCaptureEventHandler
   # Handles audio callbacks for user's audio capture and sends audio data to RealtimeClient after speech has been detected.
   # Handles speech start and end events from the local voice activity detector for response generation and interruption.

class MyRealtimeEventHandler(RealtimeAIEventHandler):
   # Implementation of RealtimeAIEventHandler
   # Handles server events from the OpenAI Realtime service, audio playback data handling, function calling etc.

# Define RealtimeAIOptions for OpenAI Realtime service configuration
options = RealtimeAIOptions(
   api_key=api_key,
   model="gpt-4o-realtime-preview",
   modalities=["audio", "text"],
   instructions="You are a helpful assistant. Respond concisely.",
   turn_detection=None, # or server vad
   tools=functions.definitions,
   tool_choice="auto",
   temperature=0.8,
   max_output_tokens=1000,
   voice="sage",
   enable_auto_reconnect=True,
)

# Define AudioStreamOptions (currently only 16bit PCM 24kHz mono is supported)
stream_options = AudioStreamOptions(
   sample_rate=24000,
   channels=1,
   bytes_per_sample=2
)

# Initialize AudioPlayer to start waiting for audio data to play
audio_player = AudioPlayer()

# Initialize RealtimeAIClient with event handler, creates websocket connection to service and set up to handle user's audio
event_handler = MyRealtimeEventHandler(audio_player=audio_player, functions=functions)
client = RealtimeAIClient(options, stream_options, event_handler)
client.start()

# Initialize AudioCapture with the event handler and starts listening for user's speech
audio_capture_event_handler = MyAudioCaptureEventHandler(
   client=client,
   event_handler=event_handler
)
audio_capture = AudioCapture(audio_capture_event_handler, ...)

audio_player.start()
audio_capture.start()
```

## Installation

1. **Install the realtime AI Python library and dependencies**:

   - Run the following command in your terminal to install all the necessary dependencies as specified in the requirements.txt file.

   ```
   pip install -r requirements.txt
   ```

   - Alternatively if you want to build the wheel yourself, use following command: `python setup.py sdist bdist_wheel`
   - After that go to generated `dist` folder and install the generated wheel using following command: `pip install --force-reinstall realtime_ai-0.1.0-py3-none-any.whl`
   - Or simply install via this git url: `pip install git+https://github.com/jhakulin/realtime-ai `

2. **Setup**:

   Set up environment variables for your chosen provider(s):

   - **OpenAI**

     ```bash
     export OPENAI_API_KEY="sk-..."
     ```

   - **Azure OpenAI** (Optional)

     ```bash
     export AZURE_OPENAI_API_KEY="Your Azure OpenAI Key"
     export AZURE_OPENAI_ENDPOINT="wss://<service-name>.openai.azure.com/openai/realtime"
     export AZURE_OPENAI_API_VERSION="2024-10-01-preview"
     ```

   - **Grok (xAI)** (Optional)

     ```bash
     export XAI_API_KEY="xai-..."
     ```

     Get your API key from [xAI Console](https://console.x.ai/)

   - **Gemini (Google)** (Optional)

     ```bash
     export GOOGLE_API_KEY="your-google-api-key"
     ```

     Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

   - **Audio Configuration**

     - Check system microphone access and settings to align with the project's audio requirements (e.g., 16bit PCM 24kHz mono).

3. **Execution**:

   - Run the script via command-line or an IDE:
     ```bash
     python samples/sample_realtime_ai_with_local_vad.py
     ```

## Audio Configuration

### Audio Configuration on Windows

It is important to have functional Audio Echo Cancellation (AEC) on the device running the samples to ensure clear audio playback and recording.
For example, the Lenovo ThinkPad P16S has been tested and provides a reliable configuration with its **Microphone Array**.

1. **Open Control Panel**:

   - Press `Windows + R` to open the Run dialog.
   - Type `control` and press `Enter` to open the Control Panel.

2. **Navigate to Sound Settings**:

   - In the Control Panel, click on **Hardware and Sound**.
   - Click on **Sound** to open the Sound settings dialog.

3. **Select Recording Device**:

   - In the Sound settings window, navigate to the **Recording** tab.
   - Locate and e.g. select **Microphone Array** from the list of recording devices. This setup is preferred for optimal performance and is known to work well on systems like the Lenovo ThinkPad P16S.
   - Click **Properties** to open the Microphone Properties dialog for the selected device.

4. **Enable Audio Enhancements**:

   - In the Microphone Properties dialog, navigate to the **Advanced** tab.
   - Under the **Signal Enhancements** section, look for the option labeled **Enable audio enhancements**.
   - Check the box next to **Enable audio enhancements** to allow extra signal processing by the audio device.

5. **Apply and Confirm Changes**:

   - Click **Apply** to save the changes.
   - Click **OK** to exit the Microphone Properties dialog.
   - Click **OK** in the Sound settings window to close it.

### Audio Configuration on Mac

1. **Install the PyAudio**:

   If you encounter installation problems in Mac, ensure you have installed portaudio by `brew install portaudio` first.

2. **Install the SSL certificates**:

   If you encounter SSL certification problems when running the samples, install certificates via `/Applications/Python 3.x/Install Certificates.command`

3. **Audio Echo Cancellation**:

   If your Mac do not have integrated audio echo cancellation, using e.g. AirPods is recommended to prevent assistant voice leaking into microphone input.

### Alternative Audio Options

If you encounter issues with audio echo that cannot be resolved through configuration changes, consider using a headset with an integrated microphone and speakers. This setup naturally avoids problems with echo, as the audio output from the speakers is isolated from the microphone input. This can provide a more seamless audio experience without relying on device-based audio echo cancellation.

## Keyword Recognition Configuration

Keyword recognition enables your application to listen for specific trigger word (e.g., "Computer") to initiate interactions with the AI assistant, enhancing privacy and costs by ensuring that audio data is only sent to the assistant when necessary. The sample application `sample_realtime_ai_with_keyword_and_vad.py` implements a design where communication with the AI assistant starts only after a keyword ("Computer") has been detected, and continues without keyword detection until a period of configurable silence timeout and once timout happens, the keyword gets rearmed again.

### Setup

The sample uses Azure CognitiveServices Speech SDK for keyword detection. For context and creating your own customer keywords, read the documentation under Azure for [Creating the Custom Keyword](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/custom-keyword-basics?pivots=programming-language-python).

**NOTE** The audio configuration with Azure Speech Keyword Recognition must be 16kHz, mono, PCM.

1. **Install Azure Speech SDK**

   Install the Azure Speech SDK for Python using `pip`:

   `pip install azure-cognitiveservices-speech`

2. **Run Sample For Quick Testing**:
   The sample code in this repository uses the `.table` file from the [Azure Speech SDK samples](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/quickstart/csharp/uwp/keyword-recognizer/helloworld/Keyword).
   This test model is configured for keyword `Computer`

   - Run the script via command-line or an IDE:
     ```bash
     python samples/sample_realtime_ai_with_keyword_and_vad.py
     ```
   - To start conversation with an assistant, say keyword `Computer`.

## Custom Providers

You can extend the library with custom providers by implementing the `BaseProvider` interface:

```python
from realtime_ai.providers.base_provider import BaseProvider
from realtime_ai.providers.provider_factory import ProviderFactory
from realtime_ai.models.normalized_events import NormalizedEvent
from typing import AsyncIterator, List

class CustomProvider(BaseProvider):
    @property
    def provider_name(self) -> str:
        return "custom"

    async def connect(self) -> None:
        # Connect to your service
        self._is_connected = True

    async def disconnect(self) -> None:
        # Disconnect from your service
        self._is_connected = False

    async def send_audio(self, audio_data: bytes) -> None:
        # Send audio to your service
        pass

    async def send_text(self, text: str, role: str = "user") -> None:
        # Send text to your service
        pass

    async def receive_events(self) -> AsyncIterator[NormalizedEvent]:
        # Receive and yield normalized events
        while self._is_connected:
            # Get events from your service
            # normalized_events = self.normalize_incoming_event(raw_event)
            # for event in normalized_events:
            #     yield event
            pass

    def normalize_incoming_event(self, raw_event: dict) -> List[NormalizedEvent]:
        # Convert your service's events to normalized events
        events = []
        # ... mapping logic ...
        return events

    # Implement other required methods...

# Register your provider
ProviderFactory.register("custom", CustomProvider)

# Use it
client = RealtimeAIClient(options, stream_options, handler, provider="custom")
```

See the [design documentation](design/provider_interface.md) for detailed implementation guidelines.

---

## Contributions

Contributions in the form of issues or pull requests are welcome! Feel free to enhance functionalities, fix bugs, improve documentation, or add new provider implementations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
