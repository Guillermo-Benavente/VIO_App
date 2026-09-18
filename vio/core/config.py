import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime configuration; environment variables override safe CPU defaults."""
    base_dir: Path = Path(__file__).resolve().parent.parent
    host: str = os.getenv("VIO_HOST", "127.0.0.1")
    port: int = int(os.getenv("VIO_PORT", "8000"))
    profile: str = os.getenv("VIO_PROFILE", "balanced-cpu")
    stt_model: str = os.getenv("VIO_STT_MODEL", "small")
    stt_language: str = os.getenv("VIO_STT_LANGUAGE", "es")
    llm_model: str = os.getenv("VIO_LLM_MODEL", "Phi-3-mini-4k-instruct-q4.gguf")
    llm_max_tokens: int = int(os.getenv("VIO_LLM_MAX_TOKENS", "180"))
    llm_temperature: float = float(os.getenv("VIO_LLM_TEMPERATURE", "0.65"))
    llm_context: int = int(os.getenv("VIO_LLM_CONTEXT", "4096"))
    # -1 means automatic: offload all layers only when the local backend supports it.
    gpu_layers: int = int(os.getenv("VIO_GPU_LAYERS", "-1"))
    tts_language: str = os.getenv("VIO_TTS_LANGUAGE", "es")
    tts_voice: str | None = os.getenv("VIO_TTS_VOICE") or None
    max_upload_mb: int = int(os.getenv("VIO_MAX_UPLOAD_MB", "30"))

    @property
    def models_dir(self) -> Path: return self.base_dir / "models"
    @property
    def output_dir(self) -> Path:
        path = self.base_dir / "output"; path.mkdir(parents=True, exist_ok=True); return path
    @property
    def upload_dir(self) -> Path:
        path = self.base_dir / "runtime" / "uploads"; path.mkdir(parents=True, exist_ok=True); return path
    @property
    def allowed_audio_suffixes(self) -> set[str]: return {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


settings = Settings()
