import logging
import base64
import os, json
from typing import Any, Dict
import threading

from utils.audio_playback import AudioPlayer
from utils.audio_capture import AudioCapture, AudioCaptureEventHandler
from utils.function_tool import FunctionTool
from realtime_ai.realtime_ai_client import RealtimeAIClient
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models.realtime_ai_events import *
from user_functions import user_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Streaming logs to the console
)

# Specific loggers for mentioned packages
logging.getLogger("utils.audio_playback").setLevel(logging.ERROR)
logging.getLogger("utils.audio_capture").setLevel(logging.ERROR)
logging.getLogger("utils.vad").setLevel(logging.ERROR)
logging.getLogger("realtime_ai").setLevel(logging.ERROR)

# Root logger for general logging
logger = logging.getLogger()


class MyAudioCaptureEventHandler(AudioCaptureEventHandler):
    def __init__(self, client: RealtimeAIClient, event_handler: "MyRealtimeEventHandler"):
        """
        Initializes the event handler.
        
        :param client: Instance of RealtimeClient.
        :param event_handler: Instance of MyRealtimeEventHandler.
        """
        self._client = client
        self._event_handler = event_handler

    def send_audio_data(self, audio_data: bytes):
        """
        Sends audio data to the RealtimeClient.

        :param audio_data: Raw audio data in bytes.
        """
        logger.debug("Sending audio data to the client.")
        self._client.send_audio(audio_data)

    def on_speech_start(self):
        """
        Handles actions to perform when speech starts.
        """
        logger.info("Local VAD: User speech started")
        if (self._client.options.turn_detection is None and
            self._event_handler.is_audio_playing()):
            logger.info("User started speaking while assistant is responding; interrupting the assistant's response.")
            self._client.clear_input_audio_buffer()
            self._client.cancel_response()
            self._event_handler.audio_player.drain_and_restart()

    def on_speech_end(self):
        """
        Handles actions to perform when speech ends.
        """
        logger.info("Local VAD: User speech ended")

        if self._client.options.turn_detection is None:
            logger.debug("Using local VAD; requesting the client to generate a response after speech ends.")
            self._client.generate_response()

    def on_keyword_detected(self, result):
        pass


class MyRealtimeEventHandler(RealtimeAIEventHandler):
    def __init__(self, audio_player: AudioPlayer, functions: FunctionTool):
        super().__init__()
        self._audio_player = audio_player
        self._functions = functions
        self._current_item_id = None
        self._current_audio_content_index = None
        self._call_id_to_function_name = {}
        self._lock = threading.Lock()
        self._client = None
        self._function_processing = False

    @property
    def audio_player(self):
        return self._audio_player

    def get_current_conversation_item_id(self):
        return self._current_item_id
    
    def get_current_audio_content_id(self):
        return self._current_audio_content_index
    
    def is_audio_playing(self):
        return self._audio_player.is_audio_playing()
    
    def is_function_processing(self):
        return self._function_processing
    
    def set_client(self, client: RealtimeAIClient):
        self._client = client

    def on_error(self, event: ErrorEvent):
        logger.error(f"Error occurred: {event.error.message}")

    def on_input_audio_buffer_speech_stopped(self, event: InputAudioBufferSpeechStopped):
        logger.info(f"Server VAD: Speech stopped at {event.audio_end_ms}ms, Item ID: {event.item_id}")

    def on_input_audio_buffer_committed(self, event: InputAudioBufferCommitted):
        logger.debug(f"Audio Buffer Committed: {event.item_id}")

    def on_conversation_item_created(self, event: ConversationItemCreated):
        logger.debug(f"New Conversation Item: {event.item}")

    def on_response_created(self, event: ResponseCreated):
        logger.debug(f"Response Created: {event.response}")

    def on_response_content_part_added(self, event: ResponseContentPartAdded):
        logger.debug(f"New Part Added: {event.part}")

    def on_response_audio_delta(self, event: ResponseAudioDelta):
        logger.debug(f"Received audio delta for Response ID {event.response_id}, Item ID {event.item_id}, Content Index {event.content_index}")
        self._current_item_id = event.item_id
        self._current_audio_content_index = event.content_index
        self.handle_audio_delta(event)

    def on_response_audio_transcript_delta(self, event: ResponseAudioTranscriptDelta):
        logger.info(f"Assistant transcription delta: {event.delta}")

    def on_rate_limits_updated(self, event: RateLimitsUpdated):
        for rate in event.rate_limits:
            logger.debug(f"Rate Limit: {rate.name}, Remaining: {rate.remaining}")

    def on_conversation_item_input_audio_transcription_completed(self, event: ConversationItemInputAudioTranscriptionCompleted):
        logger.info(f"User transcription complete: {event.transcript}")

    def on_response_audio_done(self, event: ResponseAudioDone):
        logger.debug(f"Audio done for response ID {event.response_id}, item ID {event.item_id}")

    def on_response_audio_transcript_done(self, event: ResponseAudioTranscriptDone):
        logger.debug(f"Audio transcript done: '{event.transcript}' for response ID {event.response_id}")

    def on_response_content_part_done(self, event: ResponseContentPartDone):
        part_type = event.part.get("type")
        part_text = event.part.get("text", "")
        logger.debug(f"Content part done: '{part_text}' of type '{part_type}' for response ID {event.response_id}")

    def on_response_output_item_done(self, event: ResponseOutputItemDone):
        item_content = event.item.get("content", [])
        if item_content:
            for item in item_content:
                if item.get("type") == "audio":
                    transcript = item.get("transcript")
                    if transcript:
                        logger.info(f"Assistant transcription complete: {transcript}")

    def on_response_done(self, event: ResponseDone):
        logger.debug(f"Assistant's response completed with status '{event.response.get('status')}' and ID '{event.response.get('id')}'")

    def on_session_created(self, event: SessionCreated):
        logger.info(f"Session created: {event.session}")

    def on_session_updated(self, event: SessionUpdated):
        logger.info(f"Session updated: {event.session}")

    def on_input_audio_buffer_speech_started(self, event: InputAudioBufferSpeechStarted):
        logger.info(f"Server VAD: User speech started at {event.audio_start_ms}ms for item ID {event.item_id}")
        if self._client.options.turn_detection is not None:
            self._client.clear_input_audio_buffer()
            self._client.cancel_response()
            self._audio_player.drain_and_restart()

    def on_response_output_item_added(self, event: ResponseOutputItemAdded):
        logger.debug(f"Output item added for response ID {event.response_id} with item: {event.item}")
        if event.item.get("type") == "function_call":
            call_id = event.item.get("call_id")
            function_name = event.item.get("name")
            if call_id and function_name:
                with self._lock:
                    self._call_id_to_function_name[call_id] = function_name
                logger.debug(f"Registered function call. Call ID: {call_id}, Function Name: {function_name}")
            else:
                logger.warning("Function call item missing 'call_id' or 'name' fields.")

    def on_response_function_call_arguments_delta(self, event: ResponseFunctionCallArgumentsDelta):
        logger.debug(f"Function call arguments delta for call ID {event.call_id}: {event.delta}")

    def on_response_function_call_arguments_done(self, event: ResponseFunctionCallArgumentsDone):
        call_id = event.call_id
        arguments_str = event.arguments

        with self._lock:
            function_name = self._call_id_to_function_name.pop(call_id, None)

        if not function_name:
            logger.error(f"No function name found for call ID: {call_id}")
            return

        try:
            self._function_processing = True
            logger.info(f"Executing function '{function_name}' with arguments: {arguments_str} for call ID {call_id}")
            function_output = self._functions.execute(function_name, arguments_str)
            logger.info(f"Function output for call ID {call_id}: {function_output}")
            self._client.generate_response_from_function_call(call_id, function_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse arguments for call ID {call_id}: {e}")
            return
        finally:
            self._function_processing = False

    def on_unhandled_event(self, event_type: str, event_data: Dict[str, Any]):
        logger.warning(f"Unhandled Event: {event_type} - {event_data}")

    def handle_audio_delta(self, event: ResponseAudioDelta):
        delta_audio = event.delta
        if delta_audio:
            try:
                audio_bytes = base64.b64decode(delta_audio)
                self._audio_player.enqueue_audio_data(audio_bytes)
            except base64.binascii.Error as e:
                logger.error(f"Failed to decode audio delta: {e}")
        else:
            logger.warning("Received 'ResponseAudioDelta' event without 'delta' field.")


def get_vad_configuration(use_server_vad=False):
    """
    Configures the VAD settings based on the specified preference.

    :param use_server_vad: Boolean indicating whether to use server-side VAD.
                           Default is False for local VAD.
    :return: Dictionary representing the VAD configuration suitable for RealtimeAIOptions.
    """
    if use_server_vad:
        return {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 200
        }
    else:
        return None  # Local VAD typically requires no special configuration


def main():
    """
    Main function to initialize and run the audio processing and realtime client asynchronously.
    """
    client = None
    audio_player = None
    audio_capture = None

    try:
        # Retrieve OpenAI API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return

        functions = FunctionTool(functions=user_functions)

        # Define RealtimeOptions
        options = RealtimeAIOptions(
            api_key=api_key,
            model="gpt-4o-realtime-preview-2024-10-01",
            modalities=["audio", "text"],
            instructions="You are a helpful assistant. Respond concisely. If user asks to tell story, tell story very shortly.",
            turn_detection=get_vad_configuration(use_server_vad=False),
            tools=functions.definitions,
            tool_choice="auto",
            temperature=0.8,
            max_output_tokens=None,
            voice="ballad",
        )

        # Define AudioStreamOptions
        stream_options = AudioStreamOptions(
            sample_rate=24000,
            channels=1,
            bytes_per_sample=2
        )

        # Initialize AudioPlayer
        audio_player = AudioPlayer(enable_wave_capture=False)

        # Initialize RealtimeAIClient with MyRealtimeEventHandler to handle events
        event_handler = MyRealtimeEventHandler(audio_player=audio_player, functions=functions)
        client = RealtimeAIClient(options, stream_options, event_handler)
        event_handler.set_client(client)
        client.start()

        audio_capture_event_handler = MyAudioCaptureEventHandler(
            client=client,
            event_handler=event_handler
        )
        vad_parameters={
                "sample_rate": 24000,
                "chunk_size": 1024,
                "window_duration": 1.5,
                "silence_ratio": 1.5,
                "min_speech_duration": 0.3,
                "min_silence_duration": 1.0
            }
        if USE_SILERO_VAD_MODEL:
            logger.info("using Silero VAD...")
            vad_parameters["model_path"] = "samples/resources/silero_vad.onnx"
        else:
            logger.info("using VoiceActivityDetector...")

        # Initialize AudioCapture with the event handler
        audio_capture = AudioCapture(
            event_handler=audio_capture_event_handler,
            sample_rate=24000,
            channels=1,
            frames_per_buffer=1024,
            buffer_duration_sec=1.0,
            cross_fade_duration_ms=20,
            vad_parameters=vad_parameters,
            enable_wave_capture=False
        )

        logger.info("Recording... Press Ctrl+C to stop.")
        audio_player.start()
        audio_capture.start()

        # Loop to ensure keyboard interrupt is caught correctly
        stop_event = threading.Event()
        while not stop_event.is_set():
            try:
                stop_event.wait(timeout=0.1)
            except KeyboardInterrupt:
                logger.info("Recording stopped by user.")
                audio_capture.stop()
                audio_player.stop()

                if audio_player:
                    audio_player.close()
                if audio_capture:
                    audio_capture.close()
                stop_event.set()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        if client:
            try:
                logger.info("Stopping client...")
                client.stop()
            except Exception as e:
                logger.error(f"Error during client shutdown: {e}")


if __name__ == "__main__":
    USE_SILERO_VAD_MODEL = True
    main()
