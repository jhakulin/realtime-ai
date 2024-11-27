import logging
import base64
import os, json, sys
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
logging.getLogger("websocket").setLevel(logging.ERROR)

# Root logger for general logging
logger = logging.getLogger()

# Options switch commands
TEXT_MODALITY_COMMAND = "/text"
AUDIO_AND_TEXT_MODALITY_COMMAND = "/audio_text"


class MyRealtimeEventHandler(RealtimeAIEventHandler):

    def __init__(self, audio_player: AudioPlayer, functions: FunctionTool, response_event: threading.Event):
        super().__init__()
        self._audio_player = audio_player
        self._functions = functions
        self._call_id_to_function_name = {}
        self._lock = threading.Lock()
        self._client = None
        self._function_processing = False
        self._is_transcription_for_audio_created = False
        self._response_event = response_event

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
        self.handle_audio_delta(event)

    def on_response_audio_transcript_delta(self, event: ResponseAudioTranscriptDelta):
        logger.debug(f"Assistant transcription delta: {event.delta}")
        self._display_transcript(event.delta)

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
        with self._lock:
            item_content = event.item.get("content", [])
            for item in item_content:
                # If audio content is present, process the text transcription here
                if item.get("type") != "audio":
                    continue
                transcript = item.get("transcript")
                if transcript:
                    logger.debug(f"Assistant transcription complete: {transcript}")
                    self._is_transcription_for_audio_created = True

    def on_response_done(self, event: ResponseDone):
        logger.debug(f"Assistant's response completed with status '{event.response.get('status')}' and ID '{event.response.get('id')}'")

        with self._lock:
            try:
                completed = self._handle_response_done(event)
                if completed:
                    self._is_transcription_for_audio_created = False
                    self._response_event.set()
            except Exception as e:
                error_message = f"Failed to process response: {e}"
                print(error_message)
                logger.error(error_message)
                self._is_transcription_for_audio_created = False

    def _handle_response_done(self, event : ResponseDone) -> bool:
        # Check if the response is failed
        if event.response.get('status') == 'failed':
            self._handle_failed_response(event.response)
            return True
        
        is_function_call_present = self._check_function_call(event.response.get('output', []))

        # if function call is present, do not end the run yet
        if not is_function_call_present:
            if self._is_transcription_for_audio_created is False:
                messages = self._extract_content_messages(event.response.get('output', []))
                logger.debug(f"Assistant transcription complete: {messages}")
                self._display_transcript(messages)
            return True

        return False

    def _handle_failed_response(self, response):
        status_details = response.get('status_details', {})
        error = status_details.get('error', {})
        
        error_type = error.get('type')
        error_code = error.get('code')
        error_message = error.get('message')

        self._display_transcript(f"Error: {error_message}\n")
        logger.debug(f"Failed response: Type: {error_type}, Code: {error_code}, Message: {error_message}")

    def _check_function_call(self, output_list):
        return any(item.get('type') == 'function_call' for item in output_list)

    def _extract_content_messages(self, output_list):
        content_messages = []
        for item in output_list:
            if item.get('type') == 'message':
                content_list = item.get('content', [])
                for content in content_list:
                    if content.get('type') == 'text':
                        content_messages.append(content.get('text'))
        
        if not content_messages:
            return None
        
        return "\n".join(content_messages)

    def on_session_created(self, event: SessionCreated):
        logger.debug(f"Session created: {event.session}")

    def on_session_updated(self, event: SessionUpdated):
        logger.debug(f"Session updated: {event.session}")

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
            logger.debug(f"Executing function '{function_name}' with arguments: {arguments_str} for call ID {call_id}")
            function_output = self._functions.execute(function_name, arguments_str)
            logger.debug(f"Function output for call ID {call_id}: {function_output}")
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

    def _display_transcript(self, transcript: str, end=''):
        print(transcript, end='', flush=True)


def main():
    """
    Main function to initialize and run the audio processing and realtime client asynchronously.
    """
    client = None
    audio_player = None

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
            turn_detection=None,
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
        response_event = threading.Event()
        event_handler = MyRealtimeEventHandler(audio_player=audio_player, functions=functions, response_event=response_event)
        client = RealtimeAIClient(options, stream_options, event_handler)
        event_handler.set_client(client)
        client.start()
        audio_player.start()

        # Start the chat with assistant by asking user input
        stop_event = threading.Event()

        print_instructions()
        
        while not stop_event.is_set():
            try:
                user_input = input("User: ")

                if not user_input.strip():
                    continue  # Skip empty messages

                if user_input.lower() == "exit":
                    stop_event.set()
                    break

                # Check for modality switch command
                if user_input == TEXT_MODALITY_COMMAND or user_input == AUDIO_AND_TEXT_MODALITY_COMMAND:
                    # Update the modality based on the user input
                    new_modality = "text" if user_input == TEXT_MODALITY_COMMAND else "audio_text"
                    options.modalities = ["text"] if new_modality == "text" else ["audio", "text"]
                    client.update_session(options=options)

                    # Inform the user of the modality update
                    print(f"Modality has been updated to {new_modality}.")
                    continue  # Skip the rest and prompt for next input

                print("Assistant: ", end='', flush=True)
                # Send user input to assistant
                client.send_text(user_input)

                # Wait for response to be processed
                response_event.wait()
                response_event.clear()
                print("\n")

            except KeyboardInterrupt:
                stop_event.set()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        if client:
            try:
                logger.debug("Stopping client...")
                client.stop()
            except Exception as e:
                logger.error(f"Error during client shutdown: {e}")

        if audio_player:
            audio_player.close()


def print_instructions():
    print("You can use the following commands to interact with the assistant:")
    print(" - Type your message and press Enter to send it to the assistant.")
    print(" - Type '/text' to switch to text modality.")
    print(" - Type '/audio_text' to switch to audio_text modality. In this sample, only audio output is supported.")
    print(" - Type 'exit' to close the assistant.")
    print()


if __name__ == "__main__":
    main()