"""
Sample: Commit Audio Buffer and Get Transcription

Demonstrates the commit_audio_buffer() method which commits audio
without triggering automatic response generation.

This is useful for:
- Getting user transcription before deciding how to respond
- Push-to-talk scenarios with manual response control
"""

import base64
import logging
import threading
from typing import Optional

from provider_config import get_provider_config, get_provider_env_keys
from utils.audio_capture import AudioCapture, AudioCaptureEventHandler
from utils.audio_playback import AudioPlayer

from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.models.realtime_ai_events import *
from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.realtime_ai_client import RealtimeAIClient
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("realtime_ai").setLevel(logging.INFO)
logging.getLogger("utils").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class SimpleAudioHandler(AudioCaptureEventHandler):
    """Forwards audio to client when recording."""

    def __init__(self, client: RealtimeAIClient):
        self._client = client
        self.is_recording = False

    def send_audio_data(self, audio_data: bytes):
        if self.is_recording:
            self._client.send_audio(audio_data)

    def on_speech_start(self):
        pass

    def on_speech_end(self):
        pass

    def on_keyword_detected(self, result):
        pass


class TranscriptionEventHandler(RealtimeAIEventHandler):
    """Handles events and tracks transcription completion."""

    def __init__(self, audio_player: AudioPlayer):
        super().__init__()
        self._audio_player = audio_player
        self._client: Optional[RealtimeAIClient] = None

        # Synchronization events
        self._transcription_ready = threading.Event()
        self._response_done = threading.Event()

        # Results
        self.last_transcription: Optional[str] = None
        self.last_item_id: Optional[str] = None

    def set_client(self, client: RealtimeAIClient):
        self._client = client

    def wait_for_transcription(self, timeout: float = 30.0) -> Optional[str]:
        """Wait for transcription and return it."""
        if self._transcription_ready.wait(timeout=timeout):
            self._transcription_ready.clear()
            return self.last_transcription
        return None

    def wait_for_response(self, timeout: float = 60.0) -> bool:
        """Wait for response to complete."""
        result = self._response_done.wait(timeout=timeout)
        self._response_done.clear()
        return result

    # Key events for this sample

    def on_input_audio_buffer_committed(self, event: InputAudioBufferCommitted):
        logger.info(f"Audio buffer committed, item_id: {event.item_id}")
        self.last_item_id = event.item_id

    def on_conversation_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompleted
    ):
        logger.info(f"Transcription received: {event.transcript}")
        self.last_transcription = event.transcript
        self._transcription_ready.set()

    def on_response_audio_delta(self, event: ResponseAudioDelta):
        if event.delta:
            audio_bytes = base64.b64decode(event.delta)
            self._audio_player.enqueue_audio_data(audio_bytes)

    def on_response_audio_transcript_delta(self, event: ResponseAudioTranscriptDelta):
        print(event.delta, end="", flush=True)

    def on_response_done(self, event: ResponseDone):
        logger.info("Response complete")
        self._response_done.set()

    def on_error(self, event: ErrorEvent):
        logger.error(f"Error: {event.error.message}")

    def on_session_created(self, event: SessionCreated):
        logger.info(f"Session created: {event.session}")

    # Required but unused handlers
    def on_session_updated(self, event: SessionUpdated):
        logger.info(f"Session updated: {event.session}")

    def on_conversation_item_created(self, event: ConversationItemCreated):
        pass

    def on_response_created(self, event: ResponseCreated):
        logger.info("Response created")

    def on_input_audio_buffer_speech_started(
        self, event: InputAudioBufferSpeechStarted
    ):
        logger.info(f"[DEBUG] Server VAD speech started - VAD should be disabled!")

    def on_input_audio_buffer_speech_stopped(
        self, event: InputAudioBufferSpeechStopped
    ):
        logger.info(f"[DEBUG] Server VAD speech stopped - VAD should be disabled!")

    def on_rate_limits_updated(self, event: RateLimitsUpdated):
        pass

    def on_response_audio_done(self, event: ResponseAudioDone):
        pass

    def on_response_audio_transcript_done(self, event: ResponseAudioTranscriptDone):
        pass

    def on_response_content_part_added(self, event: ResponseContentPartAdded):
        pass

    def on_response_content_part_done(self, event: ResponseContentPartDone):
        pass

    def on_response_function_call_arguments_delta(
        self, event: ResponseFunctionCallArgumentsDelta
    ):
        pass

    def on_response_function_call_arguments_done(
        self, event: ResponseFunctionCallArgumentsDone
    ):
        pass

    def on_response_output_item_added(self, event: ResponseOutputItemAdded):
        pass

    def on_response_output_item_done(self, event: ResponseOutputItemDone):
        pass

    def on_unhandled_event(self, event_type: str, event_data):
        pass


def main():
    config = get_provider_config()
    if not config:
        print(
            f"No API key found. Set one of: {', '.join(get_provider_env_keys().values())}"
        )
        return

    # Disable server VAD for manual control
    options = RealtimeAIOptions(
        api_key=config["api_key"],
        model=config["model"],
        modalities=["audio", "text"],
        instructions="You are a helpful assistant. Respond concisely.",
        turn_detection=None,
        voice=config["voice"],
        input_audio_transcription_enabled=True,
        input_audio_transcription_model="whisper-1",
    )

    stream_options = AudioStreamOptions(
        sample_rate=24000, channels=1, bytes_per_sample=2
    )
    audio_player = AudioPlayer(enable_wave_capture=False)
    event_handler = TranscriptionEventHandler(audio_player)

    client = RealtimeAIClient(
        options, stream_options, event_handler, provider=config["provider"]
    )
    event_handler.set_client(client)
    client.start()
    audio_player.start()

    audio_handler = SimpleAudioHandler(client)
    audio_capture = AudioCapture(
        event_handler=audio_handler,
        sample_rate=24000,
        channels=1,
        frames_per_buffer=1024,
        vad_parameters=None,
        enable_wave_capture=False,
    )
    audio_capture.start()

    print("\n=== COMMIT AUDIO BUFFER DEMO ===")
    print(f"Provider: {config['provider'].upper()}")
    print("\n1. Press ENTER to record")
    print("2. Speak, then press ENTER to stop")
    print("3. See transcription (no auto-response)")
    print("4. Press ENTER to generate response, or 's' to skip")
    print("Type 'quit' to exit\n")

    try:
        while True:
            cmd = input("[ENTER to record, 'quit' to exit] ")
            if cmd.lower() == "quit":
                break

            # Record
            print("Recording... (ENTER to stop)")
            audio_handler.is_recording = True
            input()
            audio_handler.is_recording = False
            print("Stopped.")

            # Commit WITHOUT response
            print("Committing audio...")
            client.commit_audio_buffer()

            # Wait for transcription
            print("Waiting for transcription...")
            transcription = event_handler.wait_for_transcription(timeout=30.0)

            if transcription:
                print(f'\n>>> User said: "{transcription}"\n')

                logger.info("Waiting for user input (ENTER or 's')...")
                choice = input("[ENTER to respond, 's' to skip] ")
                logger.info(f"User chose: '{choice}' (empty=generate, 's'=skip)")

                if choice.lower() != "s":
                    logger.info("Calling generate_response()...")
                    print("Assistant: ", end="", flush=True)
                    client.generate_response(commit_audio_buffer=False)
                    event_handler.wait_for_response()
                    print("\n")
                else:
                    logger.info("Skipped response generation")
            else:
                print("No transcription received.\n")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        audio_capture.stop()
        audio_capture.close()
        audio_player.stop()
        audio_player.close()
        client.stop()
        print("Done.")


if __name__ == "__main__":
    main()
