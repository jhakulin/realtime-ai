import azure.cognitiveservices.speech as speechsdk


class AzureKeywordRecognizer:
    """
    A class to recognize specific keywords from PCM audio streams using Azure Cognitive Services.
    """

    def __init__(self, model_file: str, callback, sample_rate: int = 16000, bits_per_sample: int = 16, channels: int = 1):
        """
        Initializes the AzureKeywordRecognizer.

        :param model_file: Path to the keyword recognition model file.
        :type model_file: str
        """

        # Create a push stream to which we'll write PCM audio data
        audio_stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=sample_rate, bits_per_sample=bits_per_sample, channels=channels)
        self.audio_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_stream_format)
        self.audio_config = speechsdk.audio.AudioConfig(stream=self.audio_stream)

        # Initialize the speech recognizer
        self.recognizer = speechsdk.KeywordRecognizer(
            audio_config=self.audio_config
        )

        # Connect callback functions to the recognizer
        self.recognizer.recognized.connect(self._on_recognized)
        self.recognizer.canceled.connect(self._on_canceled)

        # Define the keyword recognition model
        self.keyword_model = speechsdk.KeywordRecognitionModel(filename=model_file)

        if not callable(callback):
            raise ValueError("Callback must be a callable function.")

        self.keyword_detected_callback = callback

    def start_recognition(self):
        """
        Starts the keyword recognition process.

        :param callback: A function to be called when the keyword is detected.
                         It should accept a single argument with the recognition result.
        """
        # Start continuous keyword recognition
        self.recognizer.recognize_once_async(model=self.keyword_model)

    def stop_recognition(self):
        """
        Stops the keyword recognition process.
        """
        self.recognizer.stop_recognition_async()

    def push_audio(self, pcm_data):
        """
        Pushes PCM audio data to the recognizer.

        :param pcm_data: Bytes of PCM audio data.
        """
        self.audio_stream.write(pcm_data)

    def _on_recognized(self, event: speechsdk.SpeechRecognitionEventArgs):
        """
        Internal callback when a keyword is recognized.
        """
        result = event.result
        if result.reason == speechsdk.ResultReason.RecognizedKeyword:
            print(f"Keyword detected")
            if self.keyword_detected_callback:
                self.keyword_detected_callback(result)

    def _on_canceled(self, event: speechsdk.SpeechRecognitionCanceledEventArgs):
        """
        Internal callback when recognition is canceled.
        """
        print(f"Recognition canceled: {event.reason}")
        if event.result.reason == speechsdk.ResultReason.Canceled:
            print(f"Cancellation details: {event.cancellation_details}")