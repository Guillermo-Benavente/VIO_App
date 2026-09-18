"""Optional FastAPI adapter. Importing it does not load any ML model."""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .contracts import DialogueRequest
from .core.config import Settings, settings
from .pipeline import VoicePipeline


def create_app(config: Settings = settings, pipeline: VoicePipeline | None = None) -> FastAPI:
    app = FastAPI(title="VIO", version="0.1.0")
    app.state.pipeline = pipeline or VoicePipeline(config)
    app.state.config = config

    def save_upload(audio: UploadFile) -> Path:
        suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"
        if suffix not in config.allowed_audio_suffixes:
            raise HTTPException(415, "Audio format is not allowed")
        path = config.upload_dir / f"{uuid.uuid4().hex}{suffix}"
        received = 0
        with path.open("wb") as target:
            while chunk := audio.file.read(1024 * 1024):
                received += len(chunk)
                if received > config.max_upload_mb * 1024 * 1024:
                    target.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, "Audio upload is too large")
                target.write(chunk)
        return path

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "models_loaded": {"stt": app.state.pipeline._stt is not None, "llm": app.state.pipeline._llm is not None, "tts": app.state.pipeline._tts is not None}}

    @app.post("/stt")
    def stt(audio: UploadFile = File(...)) -> dict:
        path = save_upload(audio)
        result = app.state.pipeline.transcribe(path)
        return {"text": result.text, "language": result.language}

    @app.post("/llm")
    def llm(data: dict) -> dict:
        return {"reply": app.state.pipeline.reply(data)}

    @app.post("/tts")
    def tts(text: str = Form(...), speaker_audio: UploadFile | None = File(None)) -> dict:
        text = text.strip()
        if not text:
            raise HTTPException(422, "The 'text' field is required")
        speaker_path = save_upload(speaker_audio) if speaker_audio else None
        path = app.state.pipeline.synthesize(text, speaker_wav=speaker_path)
        return {"audio_url": f"/audio/{path.name}"}

    @app.post("/full")
    def full(audio: UploadFile = File(...), context_json: str = Form("{}"),
             speaker_audio: UploadFile | None = File(None)) -> dict:
        import json
        try:
            context = json.loads(context_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(422, "context_json must contain valid JSON") from exc
        speaker_path = save_upload(speaker_audio) if speaker_audio else None
        result = app.state.pipeline.process(save_upload(audio), context, speaker_wav=speaker_path)
        return {"text": result.transcript.text, "language": result.transcript.language, "reply": result.reply, "audio_url": f"/audio/{result.audio_path.name}"}

    @app.get("/audio/{filename}")
    def audio(filename: str) -> FileResponse:
        path = (config.output_dir / Path(filename).name).resolve()
        if path.parent != config.output_dir.resolve() or not path.is_file():
            raise HTTPException(404, "Audio not found")
        return FileResponse(path, media_type="audio/wav")

    return app
