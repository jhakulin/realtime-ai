# Realtime AI Client

## Overview

This Python project exemplifies a modular approach to interacting with OpenAI's Realtime REST APIs. It enables the capture and processing of real-time audio through the microphone, streaming it efficiently to the API for analysis or transcription.

---

#### Key Components

1. **RealtimeAIClient**
   - **Purpose**: Acts as the high-level orchestrator, integrating with service and audio managers for comprehensive functionality.
   - **Main Features**:
     - Coordinates the lifecycle and interactions among different managers.
     - Provides synchronous methods to start and stop the client.
     - Handles audio input and output, triggering the appropriate streaming commands.

2. **RealtimeAIOptions**
   - **Purpose**: Encapsulates configuration parameters for the OpenAI API, such as API keys, model choices, and reconnect settings.
   - **Attributes**:
     - `api_key`: The OpenAI API key for authentication.
     - `model`: Specifies the model to be used.
     - `instructions`: Initial instructions or prompts to the AI model.
     - Contains retry settings for controlling connection attempts.

3. **RealtimeAIServiceManager**
   - **Purpose**: Interfaces with the WebSocketManager to handle event processing and communication logic.
   - **Main Features**:
     - Sends initial setup instructions to the API on connection.
     - Queues incoming events for later processing.
     - Handles message parsing based on received data types.

4. **AudioStreamManager**
   - **Purpose**: Streams real-time audio data to the OpenAI Realtime service via RealtimeServiceManager (which sends the data to websocket)
   - **Main Features**:
     - Uses asynchronous queues to manage audio data buffering.
     - Encodes audio into an acceptable format and sends it to the API.
     - Offers controls to start and stop audio streaming.

5. **WebSocketManager**
   - **Purpose**: Manages WebSocket connections, providing stability through reconnection strategies.
   - **Main Features**:
     - Establishes and maintains WebSocket connections using `asyncio`.
     - Implements exponential backoff for reconnection attempts.
     - Both sends and receives data asynchronously from the OpenAI API.

6. **Sample Script (`main.py`)**
   - **Purpose**: Demonstrates capturing live audio from a microphone, sending it to the OpenAI API using the selected model.
   - **Key Activities**:
     - Utilizes `pyaudio` to capture real-time audio input.
     - Sends captured audio to `RealtimeAIClient` for processing.
     - Manages the audio stream and executes control functions, cleanly ending operations on user command.

---

#### Summary of Features

- **OpenAI's Realtime API Interaction**: Structured to support real-time interactions with OpenAI's services.
- **Audio Handling**: Integrates audio processing through `pyaudio` and NumPy libraries, complying with OpenAI API's audio format requirements.
- **Asynchronous and Synchronous Design**: Takes advantage of `asyncio` and threading to handle WebSocket communications efficiently.
- **Scalability and Modularity**: Each component operates independently, fostering scalability and maintainability for various real-time audio applications.

---

#### Getting Started

1. **Installation**:
   - Install dependencies using a package manager:
     ```bash
     pip install pyaudio numpy websockets
     ```

2. **Setup**:
   - Replace placeholders like `"YOUR_API_KEY"` in the sample script with real information.
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
