import asyncio
import json
import logging
import uuid

import websockets

from realtime_ai.models.realtime_ai_options import RealtimeAIOptions

logger = logging.getLogger(__name__)

# Fatal errors that should not trigger reconnection
FATAL_ERROR_PATTERNS = [
    "invalid_api_key",
    "invalid_request_error",
    "authentication_error",
]


class WebSocketManager:
    """
    Manages WebSocket connections using asyncio and the websockets library.
    """

    def __init__(self, options: RealtimeAIOptions, service_manager):
        self._options = options
        self._service_manager = service_manager
        self._websocket = None

        if self._options.azure_openai_endpoint:
            request_id = uuid.uuid4()
            self._url = (
                self._options.azure_openai_endpoint
                + f"?api-version={self._options.azure_openai_api_version}"
                + f"&deployment={self._options.model}"
            )
            self._headers = {
                "x-ms-client-request-id": str(request_id),
                "api-key": self._options.api_key,
            }
        else:
            self._url = f"{self._options.url}?model={self._options.model}"
            self._headers = {
                "Authorization": f"Bearer {self._options.api_key}",
            }
            # Only include openai-beta header for OpenAI endpoints
            if "api.openai.com" in self._options.url:
                self._headers["openai-beta"] = "realtime=v1"

        self._reconnect_delay = (
            5  # Time to wait before attempting to reconnect, in seconds
        )

    async def connect(self, reconnection=False):
        """
        Establishes a WebSocket connection.
        """
        try:
            if self._websocket:
                logger.info("WebSocketManager: Already connected.")
                return

            logger.info(f"WebSocketManager: Connecting to {self._url}")
            self._websocket = await websockets.connect(
                self._url, additional_headers=self._headers
            )
            logger.info("WebSocketManager: WebSocket connection established.")
            await self._service_manager.on_connected(reconnection=reconnection)

            asyncio.create_task(
                self._receive_messages()
            )  # Begin listening as a separate task
        except Exception as e:
            logger.error(f"WebSocketManager: Connection error: {e}")

    async def _receive_messages(self):
        """
        Listens for incoming WebSocket messages and delegates them to the service manager.
        """
        try:
            if not self._websocket:
                logger.warning(
                    "WebSocketManager: Cannot receive messages, websocket is None."
                )
                return

            async for message in self._websocket:
                await self._service_manager.on_message_received(message)
                logger.debug(f"WebSocketManager: Received message: {message}")
                if (
                    "session_expired" in message
                    and "Your session hit the maximum duration" in message
                ):
                    logger.info(
                        "WebSocketManager: Reconnecting due to maximum duration reached."
                    )
                    await asyncio.sleep(self._reconnect_delay)
                    await self.connect(reconnection=True)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(
                f"WebSocketManager: Connection closed during receive: {e.code} - {e.reason}"
            )

            # Check for fatal errors that should not trigger reconnection
            is_fatal = (
                any(error in str(e.reason).lower() for error in FATAL_ERROR_PATTERNS)
                if e.reason
                else False
            )

            if is_fatal:
                logger.error(
                    f"WebSocketManager: Fatal error detected - {e.reason}. Not attempting reconnection."
                )

            await self._service_manager.on_disconnected(e.code, e.reason)
        except asyncio.CancelledError:
            logger.info("WebSocketManager: Receive task was cancelled.")
        except Exception as e:
            logger.error(f"WebSocketManager: Error receiving messages: {e}")

    async def disconnect(self):
        """
        Gracefully disconnects the WebSocket connection.
        """
        if self._websocket:
            try:
                await self._websocket.close()
                logger.info("WebSocketManager: WebSocket closed gracefully.")
            except Exception as e:
                logger.error(f"WebSocketManager: Error closing WebSocket: {e}")
            finally:
                self._websocket = None

    async def send(self, message: dict):
        """
        Sends a message over the WebSocket.
        """
        if not self._websocket:
            logger.debug(
                "WebSocketManager: Cannot send message - WebSocket is not connected."
            )
            return

        # Check if websocket is still open (close_code is set when connection is closed)
        if self._websocket.close_code is not None:
            logger.debug(
                "WebSocketManager: Cannot send message - WebSocket is closed."
            )
            return

        try:
            message_str = json.dumps(message)
            await self._websocket.send(message_str)
            logger.debug(f"WebSocketManager: Sent message: {message_str}")
        except websockets.exceptions.ConnectionClosed as e:
            # Connection was closed, don't flood logs
            logger.debug(f"WebSocketManager: Send failed - connection closed: {e}")
            await self._service_manager.on_disconnected(e.code, e.reason)
        except Exception as e:
            error_str = str(e).lower()
            # Check for fatal errors
            is_fatal = any(error in error_str for error in FATAL_ERROR_PATTERNS)

            if is_fatal:
                logger.error(
                    f"WebSocketManager: Fatal error in send - {e}. Closing connection."
                )
                # Close the websocket to prevent retry loop
                if self._websocket:
                    await self._websocket.close()
                    self._websocket = None
                # Don't call on_error for fatal errors to prevent retry attempts
                return
            else:
                logger.error(f"WebSocketManager: Send failed: {e}")
                await self._service_manager.on_error(e)

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options: RealtimeAIOptions):
        self._options = options
        logger.info(f"WebSocketManager: Options updated: {options}")
