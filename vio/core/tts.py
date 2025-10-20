import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

class TTSLocal:
    """
    TTS local con Chatterbox Multilingual
    - Zero-shot voice cloning (usar tu propia voz)
    - Español y otros idiomas
    """

    def __init__(self, model_name: str = "chatterbox-0.5b", work_dir: str = "models/chatterbox"):
        self.model_name = model_name
        self.work_dir = Path(work_dir)

        # Forzamos GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[WARNING] No se detectó GPU, se usará CPU (puede ser lento)")

        print(f"[DEBUG] Device seleccionado: {self.device}")

        # Cargamos modelo multilingüe
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

    def create_audio(self, text, speaker_wav=None, output_path="output.wav", language_id="es"):
        """
        text: texto a sintetizar
        speaker_wav: ruta a tu voz (zero-shot)
        language_id: código de idioma, ej. "es"
        """
        if speaker_wav and Path(speaker_wav).exists():
            audio = self.model.generate(
                text=text,
                audio_prompt_path=speaker_wav,
                language_id=language_id
            )
        else:
            audio = self.model.generate(text=text, language_id=language_id)

        ta.save(output_path, audio, self.model.sr)
        return output_path