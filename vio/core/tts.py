import uuid
from pathlib import Path
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from .config import Settings


class LocalTTS:
    """
    Local TTS with Chatterbox Multilingual.
    - Zero-shot voice cloning (use your own voice)
    - Spanish and other languages
    """

    def __init__(self, config: Settings):
        self.config = config
        self.language = config.tts_language
        self.speaker_wav = config.tts_voice

        # Prefer GPU when available.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[WARNING] No GPU detected, will use CPU (may be slow)")

        print(f"[DEBUG] Selected device: {self.device}")

        # Load the multilingual model.
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, t3_model="v3")

    def synthesize(self, text: str, *, speaker_wav: str | Path | None = None, language_id: str | None = None) -> Path:
        """
        Generate audio from text.
        text: text to synthesize
        output_path: output file path
        language_id: language code, ej. "es"
        """
        if not text.strip(): raise ValueError("TTS text cannot be empty")
        lang = language_id or self.language
        voice = Path(speaker_wav or self.speaker_wav) if (speaker_wav or self.speaker_wav) else None

        if voice and voice.is_file():
            audio = self.model.generate(
                text=text,
                audio_prompt_path=str(voice),
                language_id=lang
            )
        else:
            audio = self.model.generate(text=text, language_id=lang)

        output_path = self.config.output_dir / f"{uuid.uuid4().hex}.wav"
        ta.save(str(output_path), audio.cpu(), self.model.sr)
        return output_path
