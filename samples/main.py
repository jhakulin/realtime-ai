import logging
import base64
import os
import wave
from typing import Any, Dict
import threading

from utils.audio_processing import AudioPlayer, AudioCapture, AudioCaptureEventHandler
from realtime_ai.realtime_ai_client import RealtimeAIClient
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models.realtime_ai_events import *

# Set up logging
logging.getLogger("audio_processing").setLevel(logging.DEBUG)
logging.getLogger("realtime_ai").setLevel(logging.ERROR)

logger = logging.getLogger()
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

ENABLE_WAVE_CAPTURE = False


class MyAudioCaptureEventHandler(AudioCaptureEventHandler):
    def __init__(self, client: RealtimeAIClient, event_handler: "MyRealtimeEventHandler", wave_file):
        self.client = client
        self.event_handler = event_handler
        self.wave_file = wave_file
        self.cancelled = False

    def send_audio_data(self, audio_data: bytes):
        if ENABLE_WAVE_CAPTURE:
            self.wave_file.writeframes(audio_data)

        logger.info("Sending audio data to the client.")
        self.client.send_audio(audio_data)

    def on_speech_start(self, audio_data: bytes):
        logger.info("Speech has started.")
        if self.event_handler.is_audio_playing():
            logger.info(f"User started speaking while audio is playing.")

            logger.info("Clearing input audio buffer.")
            self.client.clear_input_audio_buffer()

            logger.info("Cancelling response.")
            self.client.cancel_response()
            self.cancelled = True

            current_item_id = self.event_handler.get_current_conversation_item_id()
            current_audio_content_index = self.event_handler.get_current_audio_content_id()
            logger.info(f"Truncate the current audio, current item ID: {current_item_id}, current audio content index: {current_audio_content_index}")
            self.client.truncate_response(item_id=current_item_id, content_index=current_audio_content_index, audio_end_ms=1000)
        else:
            logger.info("Assistant is not speaking, cancelling response is not required.")
            self.cancelled = False

    def on_speech_end(self):
        logger.info("Speech has ended")
        logger.info("Requesting the client to generate a response.")
        self.client.generate_response()


class MyRealtimeEventHandler(RealtimeAIEventHandler):
    def __init__(self, audio_player: AudioPlayer):
        super().__init__()
        self.audio_player = audio_player
        self.audio_buffer = []
        self.client = None
        self.current_item_id = None
        self.current_audio_content_index = None
        self._is_audio_playing = False
    
    def get_current_conversation_item_id(self):
        return self.current_item_id
    
    def get_current_audio_content_id(self):
        return self.current_audio_content_index
    
    def is_audio_playing(self):
        return self._is_audio_playing
    
    def set_client(self, client: RealtimeAIClient):
        self.client = client
        logger.info("RealtimeAIClient has been set in MyRealtimeEventHandler.")

    def on_error(self, event: ErrorEvent):
        logger.error(f"Error occurred: {event.error.message}")

    def on_input_audio_buffer_speech_stopped(self, event: InputAudioBufferSpeechStopped):
        logger.info(f"Speech stopped at {event.audio_end_ms}ms, Item ID: {event.item_id}")

    def on_input_audio_buffer_committed(self, event: InputAudioBufferCommitted):
        logger.info(f"Audio Buffer Committed: {event.item_id}")

    def on_conversation_item_created(self, event: ConversationItemCreated):
        logger.info(f"New Conversation Item: {event.item}")

    def on_response_created(self, event: ResponseCreated):
        logger.info(f"Response Created: {event.response}")

    def on_response_content_part_added(self, event: ResponseContentPartAdded):
        logger.info(f"New Part Added: {event.part}")

    def on_response_audio_delta(self, event: ResponseAudioDelta):
        logger.info(f"Received audio delta for Response ID {event.response_id}, Item ID {event.item_id}, Content Index {event.content_index}")
        self.current_item_id = event.item_id
        self.current_audio_content_index = event.content_index
        self.handle_audio_delta(event)

    def on_response_audio_transcript_delta(self, event: ResponseAudioTranscriptDelta):
        logger.info(f"Transcript Delta: {event.delta}")

    def on_rate_limits_updated(self, event: RateLimitsUpdated):
        for rate in event.rate_limits:
            logger.info(f"Rate Limit: {rate.name}, Remaining: {rate.remaining}")

    def on_conversation_item_input_audio_transcription_completed(self, event: ConversationItemInputAudioTranscriptionCompleted):
        logger.info(f"Transcription completed for item {event.item_id}: {event.transcript}")

    def on_response_audio_done(self, event: ResponseAudioDone):
        logger.info(f"Audio done for response ID {event.response_id}, item ID {event.item_id}")
        if self.client:
            self._is_audio_playing = False
        self.audio_player.drain_and_restart()

    def on_response_audio_transcript_done(self, event: ResponseAudioTranscriptDone):
        logger.info(f"Audio transcript done: '{event.transcript}' for response ID {event.response_id}")

    def on_response_content_part_done(self, event: ResponseContentPartDone):
        part_type = event.part.get("type")
        part_text = event.part.get("text", "")
        logger.info(f"Content part done: '{part_text}' of type '{part_type}' for response ID {event.response_id}")

    def on_response_output_item_done(self, event: ResponseOutputItemDone):
        item_content = event.item.get("content", [])
        logger.info(f"Output item done for response ID {event.response_id} with content: {item_content}")

    def on_response_done(self, event: ResponseDone):
        logger.info(f"Response completed with status '{event.response.get('status')}' and ID '{event.response.get('id')}'")

    def on_session_created(self, event: SessionCreated):
        logger.info(f"Session created: {event.session}")

    def on_session_updated(self, event: SessionUpdated):
        logger.info(f"Session updated: {event.session}")

    def on_input_audio_buffer_speech_started(self, event: InputAudioBufferSpeechStarted):
        logger.info(f"Speech started at {event.audio_start_ms}ms for item ID {event.item_id}")

    def on_response_output_item_added(self, event: ResponseOutputItemAdded):
        logger.info(f"Output item added for response ID {event.response_id} with item: {event.item}")

    def on_response_function_call_arguments_delta(self, event: ResponseFunctionCallArgumentsDelta):
        logger.info(f"Function call arguments delta for call ID {event.call_id}: {event.delta}")

    def on_response_function_call_arguments_done(self, event: ResponseFunctionCallArgumentsDone):
        logger.info(f"Function call arguments done for call ID {event.call_id} with arguments: {event.arguments}")

    def on_unhandled_event(self, event_type: str, event_data: Dict[str, Any]):
        logger.warning(f"Unhandled Event: {event_type} - {event_data}")

    def handle_audio_delta(self, event: ResponseAudioDelta):
        delta_audio = event.delta
        if delta_audio:
            try:
                audio_bytes = base64.b64decode(delta_audio)
                self.audio_player.enqueue_audio_data(audio_bytes)
                self._is_audio_playing = True
            except base64.binascii.Error as e:
                logger.error(f"Failed to decode audio delta: {e}")
        else:
            logger.warning("Received 'ResponseAudioDelta' event without 'delta' field.")


def main():
    client = None
    audio_player = None
    audio_capture = None
    wave_file = None

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return

        audio_player = AudioPlayer(min_buffer_fill=3)

        options = RealtimeAIOptions(
            api_key=api_key,
            model="gpt-4o-realtime-preview-2024-10-01",
            modalities=["audio", "text"],
            instructions="You are a helpful assistant. Respond concisely. If user asks to tell story, tell story very shortly.",
            turn_detection=None,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            ],
            tool_choice="auto",
            temperature=0.8,
            max_output_tokens=None
        )

        stream_options = AudioStreamOptions(
            sample_rate=24000,
            channels=1,
            bytes_per_sample=2
        )

        event_handler = MyRealtimeEventHandler(audio_player=audio_player)
        client = RealtimeAIClient(options, stream_options, event_handler)
        event_handler.set_client(client)
        client.start()

        if ENABLE_WAVE_CAPTURE:
            try:
                wave_file = wave.open("microphone_output.wav", "wb")
                wave_file.setnchannels(stream_options.channels)
                wave_file.setsampwidth(stream_options.bytes_per_sample)
                wave_file.setframerate(stream_options.sample_rate)
            except Exception as e:
                logger.error(f"Error opening wave file: {e}")

        audio_capture_event_handler = MyAudioCaptureEventHandler(
            client=client,
            event_handler=event_handler,
            wave_file=wave_file
        )

        audio_capture = AudioCapture(
            event_handler=audio_capture_event_handler,
            sample_rate=24000,
            channels=1,
            frames_per_buffer=1024,
            buffer_duration_sec=1.0,
            cross_fade_duration_ms=20,
            vad_parameters={
                "sample_rate": 24000,
                "chunk_size": 1024,
                "window_duration": 1.0,
                "silence_ratio": 1.5,
                "min_speech_duration": 0.3,
                "min_silence_duration": 0.3
            }
        )

        logger.info("Recording... Press Ctrl+C to stop.")

        # Loop to ensure keyboard interrupt is caught correctly
        stop_event = threading.Event()
        while not stop_event.is_set():
            try:
                stop_event.wait(timeout=0.1)
            except KeyboardInterrupt:
                logger.info("Recording stopped by user.")
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

        if ENABLE_WAVE_CAPTURE and wave_file:
            wave_file.close()
            logger.info("Wave file saved successfully.")

        if audio_player:
            audio_player.close()

        if audio_capture:
            audio_capture.close()

if __name__ == "__main__":
    main()
