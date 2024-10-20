import threading, logging
import queue

from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.realtime_ai_service_manager import RealtimeAIServiceManager
from realtime_ai.audio_stream_manager import AudioStreamManager
from realtime_ai.realtime_ai_event_handler import RealtimeAIEventHandler
from realtime_ai.models.realtime_ai_events import EventBase

logger = logging.getLogger(__name__)


class RealtimeAIClient:
    """
    Manages overall interaction with OpenAI's Realtime API.
    """
    def __init__(self, options: RealtimeAIOptions, stream_options: AudioStreamOptions, event_handler: RealtimeAIEventHandler):
        self._options = options
        self.service_manager = RealtimeAIServiceManager(options)
        self.audio_stream_manager = AudioStreamManager(stream_options, self.service_manager)
        self.event_handler = event_handler
        self.is_running = False
        self.event_queue = queue.Queue()
        
        # Thread for consuming events
        self._consume_thread = threading.Thread(target=self._consume_events)

    def start(self):
        """Starts the RealtimeAIClient."""
        self.is_running = True
        try:
            self.service_manager.connect()  # Connect to the service
            logger.info("RealtimeAIClient: Client started.")
            self._consume_thread.start()  # Start event consumption thread
        except Exception as e:
            logger.error(f"RealtimeAIClient: Error during client start: {e}")

    def stop(self):
        """Stops the RealtimeAIClient gracefully."""
        self.is_running = False
        self.audio_stream_manager.stop_stream()
        self.service_manager.disconnect()
        self._consume_thread.join()  # Ensure thread is closed before proceeding
        logger.info("RealtimeAIClient: Services stopped.")

    def send_audio(self, audio_data: bytes):
        """Sends audio data to the audio stream manager for processing."""
        logger.info("RealtimeAIClient: Queuing audio data for streaming.")
        self.audio_stream_manager.write_audio_buffer_sync(audio_data)  # Ensure this is a sync method in the audio_stream_manager

    def generate_response(self):
        """Sends a response.create event to generate a response."""
        logger.info("RealtimeAIClient: Generating response.")
        self._send_event_to_manager({
            "event_id": self.service_manager._generate_event_id(),
            "type": "input_audio_buffer.commit",
        })

        self._send_event_to_manager({
            "event_id": self.service_manager._generate_event_id(),
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]}
        })

    def cancel_response(self):
        """Sends a response.cancel event to interrupt the model when playback is interrupted by user."""
        self._send_event_to_manager({
            "event_id": self.service_manager._generate_event_id(),
            "type": "response.cancel"
        })
        logger.info("Client: Sent response.cancel event to server.")

        # Clear the event queue in the service manager
        self.service_manager.clear_event_queue()
        logger.info("RealtimeAIClient: Event queue cleared after cancellation.")

    def truncate_response(self, item_id: str, content_index: int, audio_end_ms: int):
        """Sends a conversation.item.truncate event to truncate the response."""
        self._send_event_to_manager({
            "event_id": self.service_manager._generate_event_id(),
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms
        })
        logger.info("Client: Sent conversation.item.truncate event to server.")

    def clear_input_audio_buffer(self):
        self._send_event_to_manager({
            "event_id": self.service_manager._generate_event_id(),
            "type": "input_audio_buffer.clear"
        })
        logger.info("Client: Sent input_audio_buffer.clear event to server.")

    def generate_response_from_function_call(self, call_id: str, function_output: str):
        """
        Sends a conversation.item.create message as a function call output and optionally triggers a model response.
        
        :param call_id: The ID of the function call.
        :param name: The name of the function being called.
        :param arguments: The arguments used for the function call, in stringified JSON.
        :param function_output: The output of the function call.
        """

        # Create the function call output event
        item_create_event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "conversation.item.create",
            "item": {
                "id": "1234", # Unique item ID
                "type": "function_call_output",
                "call_id": call_id,
                "output": function_output,
            }
        }

        # Send the function call output event
        self._send_event_to_manager(item_create_event)
        logger.info("Function call output event sent.")

        self._send_event_to_manager({
            "event_id": self.service_manager._generate_event_id(),
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]}
        })

    def _consume_events(self):
        """Consume events from the service manager."""
        while self.is_running:
            try:
                event = self.service_manager.get_next_event()
                if event:
                    self._handle_event(event)
                # Insert a slight delay or make this blocking based on incoming events
                threading.Event().wait(timeout=0.05)
            except Exception as e:
                logger.error(f"RealtimeAIClient: Error in consume_events: {e}")
                break

    def _handle_event(self, event: EventBase):
        """Handles the received event based on its type using the event handler."""
        event_type = event.type
        method_name = f'on_{event_type.replace(".", "_")}'
        handler = getattr(self.event_handler, method_name, None)

        if callable(handler):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in handler {method_name} for event {event_type}: {e}")
        else:
            self.event_handler.on_unhandled_event(event_type, vars(event))

    def _send_event_to_manager(self, event):
        """Helper method to send an event to the manager."""
        self.service_manager.send_event(event)

    @property
    def options(self):
        return self._options