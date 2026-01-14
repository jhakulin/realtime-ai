import asyncio
import logging
import queue
import threading

from realtime_ai.models.audio_stream_options import AudioStreamOptions
from realtime_ai.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class AudioStreamManager:
    """
    Manages streaming audio data to Realtime AI providers in a synchronous manner.

    This class provides buffering and streaming capabilities for audio data,
    delegating the actual transmission to the provider implementation.
    """

    def __init__(
        self,
        stream_options: AudioStreamOptions,
        provider: BaseProvider,
        provider_loop=None,
    ):
        """
        Initialize AudioStreamManager.

        Args:
            stream_options: Audio stream configuration options
            provider: Provider instance to send audio data to
            provider_loop: Event loop for async provider operations (for sync client)
        """
        self._stream_options = stream_options
        self._provider = provider
        self._provider_loop = provider_loop
        self._audio_queue = queue.Queue()
        self._is_streaming = False
        self._stream_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    def _start_stream(self):
        with self._lock:
            if not self._is_streaming:
                self._is_streaming = True
                self._stop_event.clear()
                self._stream_thread = threading.Thread(target=self._stream_audio)
                self._stream_thread.start()
                logger.info("Audio streaming started.")

    def stop_stream(self):
        with self._lock:
            if self._is_streaming:
                self._is_streaming = False
                self._stop_event.set()  # Signal to the thread to stop
                if self._stream_thread:
                    self._stream_thread.join()
                logger.info("Audio streaming stopped.")

    def write_audio_buffer_sync(self, audio_data: bytes):
        with self._lock:
            if not self._is_streaming:
                self._start_stream()
        logger.debug("Enqueuing audio data for streaming.")
        self._audio_queue.put_nowait(audio_data)
        logger.debug("Audio data enqueued for streaming.")

    def _stream_audio(self):
        """Stream audio chunks from queue to provider."""
        logger.info(f"Streaming audio task started, is_streaming: {self._is_streaming}")

        while self._is_streaming and not self._stop_event.is_set():
            try:
                audio_chunk = self._audio_queue.get(
                    timeout=1
                )  # Block for a short moment
                processed_audio = self._process_audio(audio_chunk)

                # Send audio data to provider (provider handles encoding and event format)
                if self._provider_loop:
                    # For sync client with async provider, use event loop
                    future = asyncio.run_coroutine_threadsafe(
                        self._provider.send_audio(processed_audio), self._provider_loop
                    )
                    future.result(timeout=5)
                else:
                    # For async client, this should not be used (use async AudioStreamManager)
                    # But if it is, we can try to run it synchronously
                    asyncio.run(self._provider.send_audio(processed_audio))

                logger.debug("Audio data sent to provider.")

            except queue.Empty:
                # If the queue is empty, just continue looping
                continue
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break

    def _process_audio(self, audio_data: bytes) -> bytes:
        """
        Process audio data if needed (e.g., resampling, normalization).
        Currently, it returns the audio data as-is.
        """
        return audio_data
