import asyncio
import logging
from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class AudioStreamManager:
    """
    Manages streaming audio data to Realtime AI providers.

    This class provides buffering and streaming capabilities for audio data,
    delegating the actual transmission to the provider implementation.
    """
    def __init__(self, stream_options: AudioStreamOptions, provider: BaseProvider):
        """
        Initialize AudioStreamManager.

        Args:
            stream_options: Audio stream configuration options
            provider: Provider instance to send audio data to
        """
        self._stream_options = stream_options
        self._provider = provider
        self._audio_queue = asyncio.Queue()
        self._is_streaming = False
        self._stream_task = None

    def _start_stream(self):
        if not self._is_streaming:
            self._is_streaming = True
            self._stream_task = asyncio.create_task(self._stream_audio())
            logger.info("Audio streaming started.")

    async def stop_stream(self):
        if self._is_streaming:
            self._is_streaming = False
            if self._stream_task:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    logger.info("Audio streaming task cancelled.")
            logger.info("Audio streaming stopped.")

    async def write_audio_buffer(self, audio_data: bytes):
        if not self._is_streaming:
            self._start_stream()
        logger.info("Enqueuing audio data for streaming.")
        await self._audio_queue.put(audio_data)
        logger.info("Audio data enqueued for streaming.")

    async def _stream_audio(self):
        """Stream audio chunks from queue to provider."""
        logger.info(f"Streaming audio task started, is_streaming: {self._is_streaming}")
        while self._is_streaming:
            try:
                audio_chunk = await self._audio_queue.get()
                processed_audio = self._process_audio(audio_chunk)

                # Send audio data to provider (provider handles encoding and event format)
                await self._provider.send_audio(processed_audio)
                logger.info("Audio data sent to provider.")

            except asyncio.CancelledError:
                logger.info("Streaming audio task cancelled.")
                break
            except Exception as e:
                logger.error(f"Streaming error: {e}")

    def _process_audio(self, audio_data: bytes) -> bytes:
        """
        Process audio data if needed (e.g., resampling, normalization).
        Currently, it returns the audio data as-is.
        """
        return audio_data
