## Overview

This Python project exemplifies a modular approach to interacting with OpenAI's Realtime WebSocket APIs. It enables the capture and processing of real-time audio by streaming it efficiently to the API for analysis or transcription. The application samples in this repository are designed to enhance user interaction through audio processing features, including local voice activity detection and keyword detection using Azure Speech Keyword Recognition.

### Key Features

- **Real-time Audio and Text Interaction**: Capture and stream audio data to OpenAI's Realtime WebSocket APIs, enabling seamless real-time conversations with the AI assistant through both speech and text, with the ability to interrupt the assistant for dynamic and interactive dialogue.
  
- **Local Voice Activity Detection (VAD)**: The application incorporates a local voice activity detector to identify when speech starts and ends. This feature allows the system to efficiently manage audio data, ensuring that only relevant speech segments are processed and sent to the AI service. It helps in reducing unnecessary data transmission and processing, thereby optimizing performance and resource usage.

- **Keyword Detection**: Integrated with Azure Speech Services, the application supports keyword detection to trigger interactions with the AI assistant. By listening for specific trigger words (e.g., "Computer"), the system ensures that audio data is only sent to the assistant when necessary, enhancing privacy and reducing costs. This feature is particularly useful in scenarios where continuous listening is not feasible or desired.

- **Modular Design**: The project is structured to allow easy customization and extension. Users can define their own functions and event handlers to tailor the application's behavior to specific needs.

- **Multi-Modal Interaction**: Supports both audio and text modalities, enabling versatile interaction patterns with the AI assistant.

- **Configurable AI Options**: Users can configure various AI options, such as model selection, temperature settings, and tool usage, to fine-tune the assistant's responses and behavior.

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
```

## Installation

1. **Build and install the wheel**:
   - Build realtime-ai wheel using following command: `python setup.py sdist bdist_wheel`
   - Go to generated `dist` folder
   - Install the generated wheel using following command: `pip install --force-reinstall realtime_ai-0.1.0-py3-none-any.whl`

2. **Setup**:
   - Replace placeholders like `"OPENAI_API_KEY"` in the sample script with real information.
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

## Contributions

Contributions in the form of issues or pull requests are welcome! Feel free to enhance functionalities, fix bugs, or improve documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
