import logging
import pyaudio
import numpy as np
import queue
import threading
import time
import wave

# Constants for PyAudio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000  # Default sample rate
FRAMES_PER_BUFFER = 1024

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Handles audio playback for decoded audio data using PyAudio."""

    def __init__(self, min_buffer_fill=3, max_buffer_size=0, enable_wave_capture=False):
        """
        Initializes the AudioPlayer with a pre-fetch buffer threshold.

        :param min_buffer_fill: Minimum number of buffers that should be filled before starting playback initially.
        :param max_buffer_size: Maximum size of the buffer queue.
        """
        self.initial_min_buffer_fill = min_buffer_fill
        self.min_buffer_fill = min_buffer_fill
        self.buffer = queue.Queue(maxsize=max_buffer_size)
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.playback_complete_event = threading.Event()
        self.buffer_lock = threading.Lock()  # Unified lock for synchronization
        self.enable_wave_capture = enable_wave_capture
        self.wave_file = None
        self.buffers_played = 0  # Initialize buffers played counter
        self.play_limit = None
        self._initialize_wave_file()
        self._initialize_stream()
        self._start_thread()

    def _initialize_wave_file(self):
        if self.enable_wave_capture:
            try:
                self.wave_file = wave.open("playback_output.wav", "wb")
                self.wave_file.setnchannels(CHANNELS)
                self.wave_file.setsampwidth(self.pyaudio_instance.get_sample_size(FORMAT))
                self.wave_file.setframerate(RATE)
                logger.info("Wave file for playback capture initialized.")
            except Exception as e:
                logger.error(f"Error opening wave file for playback capture: {e}")

    def _initialize_stream(self):
        """Initializes or reinitializes the PyAudio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )
        logger.info("PyAudio stream initialized.")

    def _start_thread(self):
        """Starts the playback thread."""
        self.thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.thread.start()
        logger.info("Playback thread started.")

    def playback_loop(self):
        self.playback_complete_event.clear()
        self.initial_buffer_fill()

        while not self.stop_event.is_set():
            # Check and reset playback state if necessary
            try:
                if self.reset_event.is_set():
                    logger.debug("Reset event detected; resetting state.")
                    self._reset_playback_state()
                    continue

                # Evaluate playback limits safely
                with self.buffer_lock:
                    if self.play_limit is not None and self.buffers_played >= self.play_limit:
                        logger.info("Reached the buffer play limit for reset.")
                        self.reset_event.set()
                        continue

                try:
                    data = self.buffer.get(timeout=0.1)
                except queue.Empty:
                    logger.debug("Playback queue empty, waiting for data.")
                    time.sleep(0.1)
                    continue

                self._write_data_to_stream(data)

                # Update buffer count outside of lock to prevent recursive locking
                with self.buffer_lock:
                    self.buffers_played += 1

                logger.debug(f"Audio played. Buffers played count: {self.buffers_played}")

            except Exception as e:
                logger.error(f"Unexpected error in playback loop: {e}")

        logger.info("Playback thread terminated.")

    def _reset_playback_state(self):
        logger.debug("Resetting playback state.")
        with self.buffer_lock:
            self.buffer.queue.clear()
            logger.debug("Cleared playback buffer.")
            self.buffers_played = 0
            self.play_limit = None
            self.min_buffer_fill = self.initial_min_buffer_fill
            self.reset_event.clear()
        logger.debug("Playback state has been reset.")

    def _write_data_to_stream(self, data: bytes):
        try:
            if self.enable_wave_capture and self.wave_file:
                self.wave_file.writeframes(data)  # Write audio data to wave file

            for i in range(0, len(data), FRAMES_PER_BUFFER):
                chunk = data[i:i + FRAMES_PER_BUFFER]
                try:
                    self.stream.write(chunk)
                except IOError as e:
                    logger.error(f"I/O error during stream write: {e}")
                    self.stream.stop_stream()
                    self.stream.start_stream()  # Attempt to recover from an I/O error

        except IOError as e:
            logger.error(f"I/O error while writing to audio stream: {e}")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")

    def initial_buffer_fill(self):
        logger.debug("Starting initial buffer fill.")
        while True:
            with self.buffer_lock:
                current_size = self.buffer.qsize()
            if current_size >= self.min_buffer_fill or self.stop_event.is_set():
                break
            #logger.debug(f"Waiting for buffer to fill. Current size: {current_size}")
            time.sleep(0.01)
        logger.debug("Initial buffer fill complete.")

    def enqueue_audio_data(self, audio_data: bytes):
        try:
            with self.buffer_lock:
                self.buffer.put(audio_data, timeout=1)
                logger.debug(f"Enqueued audio data. Queue size: {self.buffer.qsize()}")
        except queue.Full:
            logger.warning("Failed to enqueue audio data: Buffer full.")

    def close(self):
        logger.info("Closing AudioPlayer.")
        self.stop_event.set()
        self.buffer.put(None)
        self.playback_complete_event.wait(timeout=5)
        self.thread.join(timeout=5)
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()
        if self.enable_wave_capture and self.wave_file:
            try:
                self.wave_file.close()
                logger.info("Playback wave file saved successfully.")
            except Exception as e:
                logger.error(f"Error closing wave file for playback: {e}")
        logger.info("AudioPlayer stopped and resources released.")

    def drain_and_restart(self, buffers_to_play_before_reset: int = 0):
        """
        Configures the player to play a specified number of buffers before resetting.

        :param clear_buffer: Boolean flag indicating whether to clear the buffers before resetting.
        :param buffers_to_play_before_reset: Number of buffers to play before resetting if clear_buffer is True.
        """
        with self.buffer_lock:
            self.buffers_played = 0  # Reset buffers played count
            self.play_limit = buffers_to_play_before_reset
            logger.info(f"Configured to reset after playing {buffers_to_play_before_reset} buffers.")

    def is_audio_playing(self):
        """
        Checks if audio is currently playing.

        :return: True if audio is playing, False otherwise.
        """
        with self.buffer_lock:
            buffer_not_empty = not self.buffer.empty()
        is_playing = buffer_not_empty
        logger.debug(f"Checking if audio is playing: Buffer not empty = {buffer_not_empty}, "
                     f"Is playing = {is_playing}")
        return is_playing