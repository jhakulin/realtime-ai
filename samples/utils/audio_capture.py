import logging
import pyaudio
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from .vad import VoiceActivityDetector  # Ensure this is correctly imported based on your project structure
import wave

# Constants for PyAudio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000  # Default sample rate
FRAMES_PER_BUFFER = 1024

logger = logging.getLogger(__name__)


class AudioCaptureEventHandler(ABC):
    """
    Abstract base class defining the interface for handling audio capture events.
    Any event handler must implement these methods.
    """

    @abstractmethod
    def send_audio_data(self, audio_data: bytes):
        """
        Called to send audio data to the client.

        :param audio_data: Raw audio data in bytes.
        """
        pass

    @abstractmethod
    def on_speech_start(self):
        """
        Called when speech starts.
        """
        pass

    @abstractmethod
    def on_speech_end(self):
        """
        Called when speech ends.
        """
        pass


class AudioCapture:
    """
    Handles audio input processing, including Voice Activity Detection (VAD)
    and wave file handling using PyAudio. It communicates with an event handler
    to notify about audio data and speech events.
    """

    def __init__(
        self,
        event_handler: AudioCaptureEventHandler,
        sample_rate: int = RATE,
        channels: int = CHANNELS,
        frames_per_buffer: int = FRAMES_PER_BUFFER,
        buffer_duration_sec: float = 1.0,
        cross_fade_duration_ms: int = 20,
        vad_parameters: Optional[dict] = None,
        enable_wave_capture: bool = False
    ):
        """
        Initializes the AudioCapture instance.

        :param event_handler: An instance of AudioCaptureEventHandler to handle callbacks.
        :param sample_rate: Sampling rate for audio capture.
        :param channels: Number of audio channels.
        :param frames_per_buffer: Number of frames per buffer.
        :param buffer_duration_sec: Duration of the internal audio buffer in seconds.
        :param cross_fade_duration_ms: Duration for cross-fading in milliseconds.
        :param vad_parameters: Parameters for VoiceActivityDetector.
        """
        self.event_handler = event_handler
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.buffer_duration_sec = buffer_duration_sec
        self.buffer_size = int(self.buffer_duration_sec * self.sample_rate)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.int16)
        self.buffer_pointer = 0
        self.cross_fade_duration_ms = cross_fade_duration_ms
        self.cross_fade_samples = int((self.cross_fade_duration_ms / 1000) * self.sample_rate)
        self.speech_started = False
        self.enable_wave_capture = enable_wave_capture

        if self.enable_wave_capture:
            try:
                self.wave_file = wave.open("microphone_output.wav", "wb")
                self.wave_file.setnchannels(channels)
                self.wave_file.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                self.wave_file.setframerate(sample_rate)
            except Exception as e:
                logger.error(f"Error opening wave file: {e}")

        # Initialize VAD
        if vad_parameters is None:
            vad_parameters = {
                "sample_rate": self.sample_rate,
                "chunk_size": self.frames_per_buffer,
                "window_duration": 1.0,
                "silence_ratio": 1.5,
                "min_speech_duration": 0.3,
                "min_silence_duration": 0.3
            }
        try:
            self.vad = VoiceActivityDetector(**vad_parameters)
            logger.info("VoiceActivityDetector initialized with parameters: "
                        f"{vad_parameters}")
        except Exception as e:
            logger.error(f"Failed to initialize VoiceActivityDetector: {e}")
            raise

        # Initialize PyAudio for input
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self.handle_input_audio
            )
            self.stream.start_stream()
            logger.info("AudioCapture initialized and input stream started.")
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio Input Stream: {e}")
            raise

    def handle_input_audio(self, indata: bytes, frame_count: int, time_info, status):
        """
        Combined callback function for PyAudio input stream.
        Processes incoming audio data, performs VAD, and triggers event handler callbacks.

        :param indata: Incoming audio data in bytes.
        :param frame_count: Number of frames.
        :param time_info: Time information.
        :param status: Status flags.
        :return: Tuple containing None and pyaudio.paContinue.
        """
        if status:
            logger.warning(f"Input Stream Status: {status}")

        # Convert bytes to numpy int16 and make sure the array is writable
        audio_data = np.frombuffer(indata, dtype=np.int16).copy()

        # Update internal audio buffer
        self.buffer_pointer = self._update_buffer(audio_data, self.audio_buffer, self.buffer_pointer, self.buffer_size)
        current_buffer = self._get_buffer_content(self.audio_buffer, self.buffer_pointer, self.buffer_size).copy()

        # Process VAD to detect speech
        try:
            speech_detected, is_speech = self.vad.process_audio_chunk(audio_data)
            #logger.debug(f"Speech detected: {speech_detected}, is_speech: {is_speech}")
        except Exception as e:
            logger.error(f"Error processing VAD: {e}")
            speech_detected, is_speech = False, False

        # Synchronously handle audio
        if speech_detected or self.speech_started:
            if is_speech:
                if not self.speech_started:
                    logger.info("Speech started")

                    # Determine fade length for crossfading
                    fade_length = min(self.cross_fade_samples, len(current_buffer), len(audio_data))
                    fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)

                    if fade_length > 0:
                        buffer_fade_section = current_buffer[-fade_length:].astype(np.float32)
                        audio_fade_section = audio_data[:fade_length].astype(np.float32)

                        faded_buffer_section = buffer_fade_section * fade_out
                        faded_audio_section = audio_fade_section * fade_in

                        # Ensure that the slices are writable
                        current_buffer[-fade_length:] = np.round(faded_buffer_section).astype(np.int16)
                        audio_data[:fade_length] = np.round(faded_audio_section).astype(np.int16)

                    # Combine buffered and current audio
                    combined_audio = np.concatenate((current_buffer, audio_data))

                    logger.info("Sending buffered audio to client via event handler...")
                    self.event_handler.on_speech_start()
                    self.event_handler.send_audio_data(combined_audio.tobytes())
                    if self.enable_wave_capture:
                        self.wave_file.writeframes(combined_audio.tobytes())
                else:
                    logger.info("Sending audio to client via event handler...")
                    self.event_handler.send_audio_data(audio_data.tobytes())
                    if self.enable_wave_capture:
                        self.wave_file.writeframes(audio_data.tobytes())
                self.speech_started = True
            else:
                logger.info("Speech ended")
                self.event_handler.on_speech_end()
                #self.vad.reset()  # Reset VAD if necessary
                self.speech_started = False

        return (None, pyaudio.paContinue)

    def _update_buffer(self, new_audio: np.ndarray, buffer: np.ndarray, pointer: int, buffer_size: int) -> int:
        """
        Updates the internal audio buffer with new audio data.

        :param new_audio: New incoming audio data as a NumPy array.
        :param buffer: Internal circular buffer as a NumPy array.
        :param pointer: Current pointer in the buffer.
        :param buffer_size: Total size of the buffer.
        :return: Updated buffer pointer.
        """
        new_length = len(new_audio)
        if new_length >= buffer_size:
            buffer[:] = new_audio[-buffer_size:]  # Keep only last BUFFER_SIZE samples
            pointer = 0
        else:
            end_space = buffer_size - pointer
            if new_length <= end_space:
                buffer[pointer:pointer + new_length] = new_audio
                pointer += new_length
            else:
                buffer[pointer:] = new_audio[:end_space]
                remaining = new_length - end_space
                buffer[:remaining] = new_audio[end_space:]
                pointer = remaining
        return pointer

    def _get_buffer_content(self, buffer: np.ndarray, pointer: int, buffer_size: int) -> np.ndarray:
        """
        Retrieves the current content of the buffer in the correct order.

        :param buffer: Internal circular buffer as a NumPy array.
        :param pointer: Current pointer in the buffer.
        :param buffer_size: Total size of the buffer.
        :return: Ordered audio data as a NumPy array.
        """
        if pointer == 0:
            return buffer.copy()
        return np.concatenate((buffer[pointer:], buffer[:pointer]))

    def close(self):
        """
        Closes the audio capture stream and the wave file, releasing all resources.
        """
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            logger.info("AudioCapture stopped and input stream closed.")

            if self.enable_wave_capture and self.wave_file:
                self.wave_file.close()
                logger.info("Wave file saved successfully.")

        except Exception as e:
            logger.error(f"Error closing AudioCapture: {e}")
