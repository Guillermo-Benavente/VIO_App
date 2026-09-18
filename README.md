# VIO

Local, open-model speech-to-speech pipeline: `audio → transcription → character dialogue → audio`. The library and API use the same implementation.

## Install and run

Requires Python 3.10+.

```powershell
pip install -e .[engines,dev]
uvicorn vio.main:app --host 127.0.0.1 --port 8000
```

Put GGUF LLM files in `vio/models/`. Chatterbox Multilingual V3 is used for Spanish zero-shot voice cloning. The API detects CUDA for Whisper and Chatterbox; llama.cpp only uses CUDA when its installed wheel supports GPU offload.

`GET /health` does not load models. `POST /stt`, `/llm`, `/tts`, and `/full` load them only when needed. `/tts` receives multipart `text` and optional `speaker_audio`; `/full` receives multipart `audio`, `context_json`, and optional `speaker_audio`.

## Resource-aware profiles

VIO protects the host application first. `GET /metrics` returns CPU, RAM, GPU, and VRAM usage plus the selected profile. Open `/dashboard` for a live browser view.

| Profile | STT | LLM | Intended use |
| --- | --- | --- | --- |
| `eco` | Whisper Base when installed | TinyLlama Q4 | Background work on constrained hardware |
| `balanced` | Whisper Small | Phi-3 Mini Q4 | Default local interaction |
| `quality` | Whisper Large V3 Turbo when installed | Qwen3 4B Q4 when installed | Highest quality on an idle GPU |

Set `VIO_PROFILE`, `VIO_RESOURCE_SOFT_LIMIT` (default `65`), `VIO_RESOURCE_HARD_LIMIT` (default `85`), and `VIO_RESOURCE_WAIT_SECONDS` (default `5`). Under soft pressure VIO selects a lower locally available profile. At the hard limit it yields with HTTP `503` instead of competing with the host. Missing model files never download during a request.

The quality LLM target is `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`; obtain a GGUF derived from the official Apache-2.0 weights and place it in `vio/models/`. See [LICENSES.md](LICENSES.md) before distributing any weights.

## Library use

```python
from vio import VoicePipeline
from vio.contracts import DialogueRequest

vio = VoicePipeline()
result = vio.process("entrada.wav", DialogueRequest(character_name="Guardia", user_text="Déjame pasar"))
print(result.reply, result.audio_path)
```

Review [LICENSES.md](LICENSES.md) before selecting weights for commercial distribution.
