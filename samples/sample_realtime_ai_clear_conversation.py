"""
Sample: Clear Conversation History

Demonstrates the clear_conversation() method which removes all conversation
items for a fresh start without reconnecting the WebSocket.

This is useful for:
- Starting new independent interactions within the same session
- Product scanning apps where each scan is a new conversation
- Kiosk applications with multiple users
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


class ConversationEventHandler(RealtimeAIEventHandler):
    """Handles events and tracks conversation state."""

    def __init__(self, audio_player: AudioPlayer):
        super().__init__()
        self._audio_player = audio_player
        self._client: Optional[RealtimeAIClient] = None
        self._response_done = threading.Event()

    def set_client(self, client: RealtimeAIClient):
        self._client = client

    def wait_for_response(self, timeout: float = 60.0) -> bool:
        result = self._response_done.wait(timeout=timeout)
        self._response_done.clear()
        return result

    def on_input_audio_buffer_committed(self, event: InputAudioBufferCommitted):
        logger.info(f"Audio buffer committed, item_id: {event.item_id}")

    def on_conversation_item_created(self, event: ConversationItemCreated):
        item_id = event.item.get("id", "unknown")
        item_type = event.item.get("type", "unknown")
        logger.info(f"Conversation item created: {item_id} (type: {item_type})")

    def on_conversation_item_deleted(self, event: ConversationItemDeleted):
        logger.info(f"Conversation item deleted: {event.item_id}")

    def on_conversation_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompleted
    ):
        logger.info(f"User said: {event.transcript}")

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
        logger.info("Session created")

    def on_session_updated(self, event: SessionUpdated):
        logger.info("Session updated")

    # Required abstract methods
    def on_input_audio_buffer_speech_started(self, event: InputAudioBufferSpeechStarted):
        pass

    def on_input_audio_buffer_speech_stopped(self, event: InputAudioBufferSpeechStopped):
        pass

    def on_response_created(self, event: ResponseCreated):
        pass

    def on_response_content_part_added(self, event: ResponseContentPartAdded):
        pass

    def on_rate_limits_updated(self, event: RateLimitsUpdated):
        pass

    def on_response_audio_done(self, event: ResponseAudioDone):
        pass

    def on_response_audio_transcript_done(self, event: ResponseAudioTranscriptDone):
        pass

    def on_response_content_part_done(self, event: ResponseContentPartDone):
        pass

    def on_response_output_item_added(self, event: ResponseOutputItemAdded):
        pass

    def on_response_output_item_done(self, event: ResponseOutputItemDone):
        pass

    def on_response_function_call_arguments_delta(self, event: ResponseFunctionCallArgumentsDelta):
        pass

    def on_response_function_call_arguments_done(self, event: ResponseFunctionCallArgumentsDone):
        pass

    def on_unhandled_event(self, event_type: str, event_data):
        pass


def main():
    config = get_provider_config()
    if not config:
        print(f"No API key found. Set one of: {', '.join(get_provider_env_keys().values())}")
        return

    options = RealtimeAIOptions(
        api_key=config["api_key"],
        model=config["model"],
        modalities=["audio", "text"],
        instructions="You are a helpful assistant. Respond concisely.",
        voice=config["voice"],
        input_audio_transcription_enabled=True,
        input_audio_transcription_model="whisper-1",
        turn_detection=None,  # Manual mode - no auto-commit or auto-response
    )

    stream_options = AudioStreamOptions(sample_rate=24000, channels=1, bytes_per_sample=2)
    audio_player = AudioPlayer(enable_wave_capture=False)
    event_handler = ConversationEventHandler(audio_player)

    client = RealtimeAIClient(options, stream_options, event_handler, provider=config["provider"])
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

    print("\n=== CLEAR CONVERSATION DEMO ===")
    print(f"Provider: {config['provider'].upper()}")
    print("\nCommands:")
    print("  [ENTER] - Record audio message")
    print("  'c'     - Clear conversation history")
    print("  't'     - Send text message")
    print("  'quit'  - Exit")
    print()

    try:
        while True:
            cmd = input("\n> ").strip().lower()

            if cmd == "quit":
                break

            elif cmd == "c":
                print("Clearing conversation history...")
                client.clear_conversation()
                print("Conversation cleared. Start fresh!")

            elif cmd == "t":
                text = input("Enter message: ").strip()
                if text:
                    print("Assistant: ", end="", flush=True)
                    client.send_text(text)
                    event_handler.wait_for_response()
                    print()

            elif cmd == "":
                # Record audio
                print("Recording... (ENTER to stop)")
                audio_handler.is_recording = True
                input()
                audio_handler.is_recording = False
                print("Stopped. Generating response...")
                print("Assistant: ", end="", flush=True)
                client.generate_response()
                event_handler.wait_for_response()
                print()

            else:
                print("Unknown command. Use ENTER, 'c', 't', or 'quit'")

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
