# VIO

Pipeline local reutilizable para `audio → transcripción → diálogo de personaje → audio`. La librería y la API comparten la misma implementación; no hay lógica de inferencia en las rutas HTTP.

## Instalación y ejecución

Requiere Python 3.10+.

```powershell
pip install -e .[engines,dev]
uvicorn vio.main:app --host 127.0.0.1 --port 8000
```

Coloca el GGUF LLM en `vio/models/` y define `VIO_LLM_MODEL` si su nombre difiere. Para CPU el perfil por defecto usa Whisper `small`, un GGUF Q4 y cero capas GPU. Configura `VIO_STT_MODEL`, `VIO_GPU_LAYERS`, `VIO_LLM_CONTEXT` y `VIO_TTS_VOICE` en el entorno según el equipo.

`GET /health` does not load models. `POST /stt`, `/llm`, `/tts`, and `/full` load them only when needed. `/tts` receives multipart `text` and optional `speaker_audio`; `/full` receives multipart `audio`, `context_json`, and optional `speaker_audio`.

## Uso como librería

```python
from vio import VoicePipeline
from vio.contracts import DialogueRequest

vio = VoicePipeline()
result = vio.process("entrada.wav", DialogueRequest(character_name="Guardia", user_text="Déjame pasar"))
print(result.reply, result.audio_path)
```

Consulta [LICENSES.md](LICENSES.md) antes de elegir pesos para una distribución comercial.
