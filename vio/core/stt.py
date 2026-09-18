from pathlib import Path
import tempfile
import wave
import numpy as np
import whisper
import torch
from ..contracts import Transcript
from .audio import convert_to_wav
from .config import Settings


class LocalSTT:
    """MIT-licensed Whisper adapter. Load only when the pipeline needs STT."""
    def __init__(self, config: Settings):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VIO] STT device: {self.device}")
        self.model = whisper.load_model(config.stt_model, device=self.device, download_root=str(config.models_dir / "whisper"))

    def transcribe(self, audio_path: Path) -> Transcript:
        with tempfile.TemporaryDirectory(prefix="vio-stt-") as directory:
            wav = convert_to_wav(audio_path, Path(directory) / "input.wav")
            # Whisper's path-based loader requires a system-wide ffmpeg executable.
            # The file is already mono, 16 kHz PCM thanks to convert_to_wav, so pass
            # the waveform directly and keep the bundled imageio-ffmpeg dependency.
            with wave.open(str(wav), "rb") as stream:
                if stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2:
                    raise RuntimeError("Audio normalization did not produce 16 kHz mono PCM")
                samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(samples, language=self.config.stt_language,
                                           fp16=self.device == "cuda", verbose=False)
        return Transcript(text=result["text"].strip(), language=result.get("language", self.config.stt_language), source=audio_path)
