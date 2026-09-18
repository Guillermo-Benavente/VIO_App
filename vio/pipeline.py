"""Reusable orchestration. It deliberately contains no HTTP-specific code."""

from pathlib import Path

from .contracts import DialogueRequest, Transcript, VoiceResult
from .core.config import Settings, settings
from .core.llm import LocalLLM
from .core.stt import LocalSTT
from .core.tts import LocalTTS


class VoicePipeline:
    """A lazily-loaded, local STT → LLM → TTS pipeline."""

    def __init__(self, config: Settings = settings):
        self.config = config
        self._stt: LocalSTT | None = None
        self._llm: LocalLLM | None = None
        self._tts: LocalTTS | None = None

    @property
    def stt(self) -> LocalSTT:
        if self._stt is None:
            self._stt = LocalSTT(self.config)
        return self._stt

    @property
    def llm(self) -> LocalLLM:
        if self._llm is None:
            self._llm = LocalLLM(self.config)
        return self._llm

    @property
    def tts(self) -> LocalTTS:
        if self._tts is None:
            self._tts = LocalTTS(self.config)
        return self._tts

    def transcribe(self, audio_path: str | Path) -> Transcript:
        return self.stt.transcribe(Path(audio_path))

    def reply(self, request: DialogueRequest | dict, user_text: str | None = None) -> str:
        normalized = request if isinstance(request, DialogueRequest) else DialogueRequest.from_mapping(request, user_text)
        return self.llm.respond(normalized)

    def synthesize(self, text: str, *, speaker_wav: str | Path | None = None) -> Path:
        return self.tts.synthesize(text, speaker_wav=speaker_wav)

    def process(self, audio_path: str | Path, context: DialogueRequest | dict,
                *, speaker_wav: str | Path | None = None) -> VoiceResult:
        transcript = self.transcribe(audio_path)
        request = context if isinstance(context, DialogueRequest) else DialogueRequest.from_mapping(context, transcript.text)
        if isinstance(request, DialogueRequest) and not request.user_text:
            request = DialogueRequest(**{**request.__dict__, "user_text": transcript.text})
        reply = self.reply(request)
        return VoiceResult(transcript=transcript, reply=reply,
                           audio_path=self.synthesize(reply, speaker_wav=speaker_wav))
