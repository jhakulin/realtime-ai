from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class RealtimeAIOptions:
    """Configuration options for the Realtime API client."""
    api_key: str
    model: str
    modalities: List[str]
    instructions: str
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: Optional[str] = None
    voice: str = "alloy"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription_enabled: bool = True
    input_audio_transcription_model: str = "whisper-1"
    turn_detection: Dict[str, Any] = field(default_factory=lambda: {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 200
    })
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "auto"
    temperature: float = 0.8
    max_output_tokens: Optional[int] = None
    enable_auto_reconnect: bool = False

    def __post_init__(self):
        self.validate_options()

    def validate_options(self):
        """Validates provided configuration settings based on expected constraints."""
        if not self.api_key:
            raise ValueError("API key must be provided.")
        if not self.model:
            raise ValueError("Model must be specified.")