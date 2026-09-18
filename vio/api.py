"""Optional FastAPI adapter. Importing it does not load any ML model."""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .core.config import Settings, settings
from .pipeline import VoicePipeline
from .resources import ResourceBusyError


DASHBOARD = """<main><h1>VIO resource monitor</h1><p id='decision'>Loading…</p><section id='meters'></section><p>VIO yields when a metric reaches the hard limit and reduces its requested profile at the soft limit.</p></main><style>body{font:16px system-ui;margin:2rem;max-width:720px}section{display:grid;gap:1rem;grid-template-columns:repeat(2,minmax(0,1fr))}.meter{padding:1rem;background:#f3f4f6;border-radius:.5rem}.bar{height:10px;background:#ddd;border-radius:5px}.fill{height:100%;background:#2563eb;border-radius:5px}@media(prefers-color-scheme:dark){body{background:#111;color:#eee}.meter{background:#222}.bar{background:#444}}</style><script>const names={cpu_percent:'CPU',ram_percent:'RAM',gpu_percent:'GPU',vram_percent:'VRAM'};async function update(){const data=await fetch('/metrics').then(r=>r.json());document.querySelector('#decision').textContent=`Requested: ${data.requested_profile} · Active: ${data.active_profile||'idle'} · Decision: ${data.decision}`;document.querySelector('#meters').innerHTML=Object.entries(names).map(([key,name])=>{const value=data.resources[key];return `<div class=meter><strong>${name}</strong><div>${value==null?'unavailable':value+'%'}</div><div class=bar><div class=fill style='width:${value||0}%'></div></div></div>`}).join('')}update();setInterval(update,1000)</script>"""


def create_app(config: Settings = settings, pipeline: VoicePipeline | None = None) -> FastAPI:
    app = FastAPI(title="VIO", version="0.1.0")
    app.state.pipeline = pipeline or VoicePipeline(config)
    app.state.config = config

    @app.exception_handler(ResourceBusyError)
    def resource_busy(_, exc: ResourceBusyError):
        return HTMLResponse(str(exc), status_code=503, headers={"Retry-After": "5"})

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

    @app.get("/metrics")
    def metrics() -> dict:
        return app.state.pipeline.governor.status(config.profile, app.state.pipeline.active_profile)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD

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
