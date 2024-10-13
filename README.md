# Realtime AI Client

## Overview

This Python project exemplifies a modular approach to interacting with OpenAI's Realtime WebSocket APIs. It enables the capture and processing of real-time audio by streaming it efficiently to the API for analysis or transcription.

---

#### Key Components

1. **RealtimeAIClient**
   - **Purpose**: Acts as the high-level orchestrator, integrating with service and audio managers for comprehensive functionality.
   - **Main Features**:
     - Coordinates the lifecycle and interactions among different managers.
     - Provides asynchronous and synchronous methods to start and stop the client.
     - Handles audio input and process appropriate events back to the application.

2. **RealtimeAIOptions**
   - **Purpose**: Encapsulates configuration parameters for the OpenAI API, such as API keys, model choices etc.

3. **RealtimeAIServiceManager**
   - **Purpose**: Interfaces with the WebSocketManager to handle event processing and communication logic.
   - **Main Features**:
     - Sends initial setup instructions to the API on connection.
     - Queues incoming events for later processing.
     - Handles message parsing based on received data types.

4. **AudioStreamManager**
   - **Purpose**: Streams real-time audio data to the OpenAI Realtime service via RealtimeServiceManager (which sends the data to websocket)
   - **Main Features**:
     - Uses queues to manage audio data buffering.
     - Encodes audio into an acceptable format and sends it to the API.
     - Offers controls to start and stop audio streaming.

5. **WebSocketManager**
   - **Purpose**: Manages WebSocket connections, providing stability through reconnection strategies.
   - **Main Features**:
     - Establishes and maintains WebSocket connections.
     - Both sends and receives data from the OpenAI API.

6. **Sample Script (`main.py`)**
   - **Purpose**: Demonstrates capturing live audio from a microphone and playback to speaker using OpenAI's realtime API.
   - **Key Activities**:
     - Utilizes `pyaudio` to capture real-time audio input and playback audio output.
     - Sends captured audio to `RealtimeAIClient` for processing.
     - Manages the events received from the RealtimeAIClient for further processing.

---

#### Summary of Features

- **OpenAI's Realtime API Interaction**: Structured to support real-time interactions with OpenAI's services.
- **Audio Handling**: Integrates audio processing through `pyaudio` and NumPy libraries, complying with OpenAI API's audio format requirements.
- **Asynchronous and Synchronous Design**: Takes advantage of `asyncio` and threading to handle WebSocket communications efficiently.
- **Scalability and Modularity**: Each component operates independently, fostering scalability and maintainability for various real-time audio applications.

---

#### Getting Started

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
