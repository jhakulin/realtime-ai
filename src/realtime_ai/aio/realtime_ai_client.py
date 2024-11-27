import asyncio
import logging
import uuid

from realtime_ai.models.realtime_ai_options import RealtimeAIOptions
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.aio.realtime_ai_service_manager import RealtimeAIServiceManager
from realtime_ai.aio.audio_stream_manager import AudioStreamManager
from realtime_ai.aio.realtime_ai_event_handler import RealtimeAIEventHandler
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
        self._consume_task = None

    async def start(self):
        """Starts the RealtimeAIClient."""
        if not self.is_running:
            self.is_running = True
            try:
                await self.service_manager.connect()  # Connect to the service initially
                logger.info("RealtimeAIClient: Client started.")

                # Schedule the event consumption coroutine as a background task
                self._consume_task = asyncio.create_task(self._consume_events())
            except Exception as e:
                logger.error(f"RealtimeAIClient: Error during client start: {e}")
                self.is_running = False

    async def stop(self):
        """Stops the RealtimeAIClient gracefully."""
        if self.is_running:
            self.is_running = False
            try:
                await self.audio_stream_manager.stop_stream()
                await self.service_manager.disconnect()
                logger.info("RealtimeAIClient: Services stopped.")

                if self._consume_task:
                    # Cancel the consume_events task and wait for it to finish
                    self._consume_task.cancel()
                    try:
                        await self._consume_task
                    except asyncio.CancelledError:
                        logger.info("RealtimeAIClient: consume_events task cancelled.")
            except Exception as e:
                logger.error(f"RealtimeAIClient: Error during client stop: {e}")

    async def send_audio(self, audio_data: bytes):
        """Sends audio data to the audio stream manager for processing."""
        logger.info("RealtimeAIClient: Queuing audio data for streaming.")
        await self.audio_stream_manager.write_audio_buffer(audio_data)

    async def send_text(self, text: str, role: str = "user", generate_response: bool = True):
        """Sends text input to the service manager.
        """
        event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": [
                    {
                        "type": "text" if role == "assistant" else "input_text",
                        "text": text
                    }
                ]
            }
        }
        await self.service_manager.send_event(event)
        logger.info("RealtimeAIClient: Sent text input to server.")

        # Generate a response if required
        if generate_response:
            await self.generate_response(commit_audio_buffer=False)

    async def update_session(self, options: RealtimeAIOptions):
        """Updates the session configuration with the provided options."""
        event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "session.update",
            "session": {
                "modalities": options.modalities,
                "instructions": options.instructions,
                "voice": options.voice,
                "input_audio_format": options.input_audio_format,
                "output_audio_format": options.output_audio_format,
                "input_audio_transcription": {
                    "model": options.input_audio_transcription_model
                },
                "turn_detection": options.turn_detection,
                "tools": options.tools,
                "tool_choice": options.tool_choice,
                "temperature": options.temperature
            }
        }
        await self.service_manager.send_event(event)
        self._options = options
        logger.info("RealtimeAIClient: Sent session update to server.")

    async def generate_response(self, commit_audio_buffer: bool = True):
        """Sends a response.create event to generate a response."""
        logger.info("RealtimeAIClient: Generating response.")
        if commit_audio_buffer:
            commit_event = {
                "event_id": self.service_manager._generate_event_id(),
                "type": "input_audio_buffer.commit"
            }
            await self.service_manager.send_event(commit_event)

        response_create_event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "response.create",
            "response": {"modalities": self.options.modalities}
        }
        await self.service_manager.send_event(response_create_event)

    async def cancel_response(self):
        """Sends a response.cancel event to interrupt the model when playback is interrupted by user."""
        cancel_event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "response.cancel"
        }
        await self.service_manager.send_event(cancel_event)
        logger.info("Client: Sent response.cancel event to server.")

        # Clear the event queue in the service manager
        await self.service_manager.clear_event_queue()
        logger.info("RealtimeAIClient: Event queue cleared after cancellation.")

    async def truncate_response(self, item_id: str, content_index: int, audio_end_ms: int):
        """Sends a conversation.item.truncate event to truncate the response."""
        truncate_event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms
        }
        await self.service_manager.send_event(truncate_event)
        logger.info("Client: Sent conversation.item.truncate event to server.")

    async def clear_input_audio_buffer(self):
        clear_audio_buffers_event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "input_audio_buffer.clear"
        }
        await self.service_manager.send_event(clear_audio_buffers_event)
        logger.info("Client: Sent input_audio_buffer.clear event to server.")

    async def generate_response_from_function_call(self, call_id: str, function_output: str):
        """
        Asynchronously sends a conversation.item.create message as a function call output 
        and optionally triggers a model response.
        
        :param call_id: The ID of the function call.
        :param function_output: The output of the function call.
        """

        # Create the function call output event
        item_create_event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "conversation.item.create",
            "item": {
                "id": str(uuid.uuid4()).replace('-', ''),
                "type": "function_call_output",
                "call_id": call_id,
                "output": function_output,
            }
        }

        # Send the function call output event
        await self.service_manager.send_event(item_create_event)
        logger.info("Function call output event sent.")

        # Create and send the response.create event
        response_event = {
            "event_id": self.service_manager._generate_event_id(),
            "type": "response.create",
            "response": {"modalities": self.options.modalities}
        }
        await self.service_manager.send_event(response_event)

    async def _consume_events(self):
        """Consume events from the service manager asynchronously."""
        logger.info("RealtimeAIClient: Started consuming events.")
        try:
            while self.is_running:
                try:
                    event = await self.service_manager.get_next_event()
                    if event:
                        # Schedule the event handler as an independent task
                        asyncio.create_task(self._handle_event(event))
                    else:
                        await asyncio.sleep(0.05)  # Small delay to prevent tight loop
                except Exception as e:
                    logger.error(f"RealtimeAIClient: Error in consume_events: {e}")
                    await asyncio.sleep(1)  # Optional: Backoff on error to prevent rapid retries
        except asyncio.CancelledError:
            logger.info("RealtimeAIClient: consume_events loop has been cancelled.")
        finally:
            logger.info("RealtimeAIClient: Stopped consuming events.")

    async def _handle_event(self, event: EventBase):
        """Handles the received event based on its type using the event handler."""
        event_type = event.type
        method_name = f'on_{event_type.replace(".", "_")}'
        handler = getattr(self.event_handler, method_name, None)

        if callable(handler):
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in handler {method_name} for event {event_type}: {e}")
        else:
            await self.event_handler.on_unhandled_event(event_type, vars(event))

    @property
    def options(self):
        return self._options
