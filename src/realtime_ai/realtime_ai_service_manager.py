import json
import logging
import uuid
import threading
import queue
from typing import Optional, Type

from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.web_socket_manager import WebSocketManager
from realtime_ai.models.realtime_ai_events import (
    EventBase,
    ErrorEvent,
    ErrorDetails,
    InputAudioBufferSpeechStopped,
    InputAudioBufferCommitted,
    ConversationItemCreated,
    ConversationItemInputAudioTranscriptionCompleted,
    ResponseCreated,
    ResponseContentPartAdded,
    ResponseAudioTranscriptDelta,
    RateLimit,
    RateLimitsUpdated,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDone,
    ResponseContentPartDone,
    ResponseOutputItemDone,
    ResponseDone,
    SessionCreated,
    SessionUpdated,
    InputAudioBufferSpeechStarted,
    ResponseOutputItemAdded,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone
)

logger = logging.getLogger(__name__)


class RealtimeAIServiceManager:
    """
    Manages WebSocket connection and communication with OpenAI's Realtime API synchronously.
    """

    def __init__(self, options: RealtimeAIOptions):
        self.options = options
        self.websocket_manager = WebSocketManager(options, self)
        self.event_queue = queue.Queue()
        self.is_connected = False

        self._thread_running_event = threading.Event()

        # Pre-construct session.update event details
        self.session_update_event = {
            "event_id": self._generate_event_id(),
            "type": "session.update",
            "session": {
                "modalities": self.options.modalities,
                "instructions": self.options.instructions,
                "voice": self.options.voice,
                "input_audio_format": self.options.input_audio_format,
                "output_audio_format": self.options.output_audio_format,
                "input_audio_transcription": {
                    "model": self.options.input_audio_transcription_model
                },
                "turn_detection": self.options.turn_detection,
                "tools": self.options.tools,
                "tool_choice": self.options.tool_choice,
                "temperature": self.options.temperature
            }
        }

    def connect(self):
        try:
            self.websocket_manager.connect()
            self.is_connected = True
            logger.info("RealtimeAIServiceManager: Connection started to WebSocket.")
        except Exception as e:
            logger.error(f"RealtimeAIServiceManager: Unexpected error during connect: {e}")

    def disconnect(self):
        try:
            self.event_queue.put(None)  # Signal the event loop to stop
            self.websocket_manager.disconnect()
            self.is_connected = False
            logger.warning("RealtimeAIServiceManager: WebSocket disconnection started.")
        except Exception as e:
            logger.error(f"RealtimeAIServiceManager: Unexpected error during disconnect: {e}")

    def send_event(self, event: dict):
        try:
            self.websocket_manager.send(event)
            logger.debug(f"RealtimeAIServiceManager: Sent event: {event.get('type')}")
        except Exception as e:
            logger.error(f"RealtimeAIServiceManager: Failed to send event {event.get('type')}: {e}")

    def on_connected(self):
        logger.info("RealtimeAIServiceManager: WebSocket connected.")
        self.send_event(self.session_update_event)
        logger.debug("RealtimeAIServiceManager: session.update event sent.")

    def on_disconnected(self, status_code: int, reason: str):
        logger.warning(f"RealtimeAIServiceManager: WebSocket disconnected: {status_code} - {reason}")

    def on_error(self, error: Exception):
        logger.error(f"RealtimeAIServiceManager: WebSocket error: {error}")

    def on_message_received(self, message: str):
        try:
            json_object = json.loads(message)
            event = self.parse_realtime_event(json_object)
            if event:
                self.event_queue.put_nowait(event)
                logger.debug(f"RealtimeAIServiceManager: Event queued: {event.type}")
        except json.JSONDecodeError as e:
            logger.error(f"RealtimeAIServiceManager: JSON parse error: {e}")

    def parse_realtime_event(self, json_object: dict) -> Optional[EventBase]:
        event_type = json_object.get("type")
        event_class = self._get_event_class(event_type)
        if event_class:
            try:
                if event_type == "error" and 'error' in json_object:
                    # Convert error dict to ErrorDetails dataclass
                    error_data = json_object['error']
                    error_details = ErrorDetails(**error_data)
                    return ErrorEvent(event_id=json_object['event_id'], type=event_type, error=error_details)
                elif event_type == "rate_limits.updated" and 'rate_limits' in json_object:
                    rate_limits_data = json_object['rate_limits']
                    rate_limits = [RateLimit(**rate) for rate in rate_limits_data]
                    return RateLimitsUpdated(event_id=json_object['event_id'], type=event_type, rate_limits=rate_limits)
                elif event_type == "response.function_call_arguments.done":
                    # Ensure only relevant fields are passed
                    return ResponseFunctionCallArgumentsDone(
                        event_id=json_object['event_id'], 
                        type=event_type,
                        response_id=json_object.get('response_id'),
                        item_id=json_object.get('item_id'),
                        output_index=json_object.get('output_index'),
                        call_id=json_object.get('call_id'),
                        arguments=json_object.get('arguments')
                    )
                else:
                    return event_class(**json_object)
            except TypeError as e:
                logger.error(f"Error creating event object for {event_type}: {e}")
        else:
            logger.warning(f"RealtimeAIServiceManager: Unknown message type received: {event_type}")
        return None

    def clear_event_queue(self):
        """Clears all events in the event queue."""
        try:
            self.event_queue.queue.clear()
            logger.info("RealtimeAIServiceManager: Event queue cleared.")
        except Exception as e:
            logger.error(f"RealtimeAIServiceManager: Failed to clear event queue: {e}")

    def _get_event_class(self, event_type: str) -> Optional[Type[EventBase]]:
        event_mapping = {
            "error": ErrorEvent,
            "input_audio_buffer.speech_stopped": InputAudioBufferSpeechStopped,
            "input_audio_buffer.committed": InputAudioBufferCommitted,
            "conversation.item.created": ConversationItemCreated,
            "response.created": ResponseCreated,
            "response.content_part.added": ResponseContentPartAdded,
            "response.audio.delta": ResponseAudioDelta,
            "response.audio_transcript.delta": ResponseAudioTranscriptDelta,
            "conversation.item.input_audio_transcription.completed": ConversationItemInputAudioTranscriptionCompleted,
            "rate_limits.updated": RateLimitsUpdated,
            "response.audio.done": ResponseAudioDone,
            "response.audio_transcript.done": ResponseAudioTranscriptDone,
            "response.content_part.done": ResponseContentPartDone,
            "response.output_item.done": ResponseOutputItemDone,
            "response.done": ResponseDone,
            "session.created": SessionCreated,
            "session.updated": SessionUpdated,
            "input_audio_buffer.speech_started": InputAudioBufferSpeechStarted,
            "response.output_item.added": ResponseOutputItemAdded,
            "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDelta,
            "response.function_call_arguments.done": ResponseFunctionCallArgumentsDone,
        }
        return event_mapping.get(event_type)

    def get_next_event(self, timeout=5.0) -> Optional[EventBase]:
        try:
            logger.info("RealtimeAIServiceManager: Waiting for next event...")
            return self.event_queue.get(timeout=timeout)
        except queue.Empty:
            raise

    def _generate_event_id(self) -> str:
        return f"event_{uuid.uuid4()}"
