"""Reusable orchestration. It deliberately contains no HTTP-specific code."""

from pathlib import Path

from .contracts import DialogueRequest, Transcript, VoiceResult
from .core.config import Settings, settings
from .core.llm import LocalLLM
from .core.stt import LocalSTT
from .core.tts import LocalTTS
from .resources import ResourceGovernor


class VoicePipeline:
    """A lazily-loaded, local STT → LLM → TTS pipeline."""

    def __init__(self, config: Settings = settings):
        self.config = config
        self._stt: LocalSTT | None = None
        self._llm: LocalLLM | None = None
        self._tts: LocalTTS | None = None
        self.governor = ResourceGovernor(config)
        self.active_profile: str | None = None

    def _stage_config(self) -> Settings:
        profile = self.governor.select_profile(self.config.profile)
        self.active_profile = profile
        return self.config.for_profile(profile)

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
        stage_config = self._stage_config()
        if self._stt is not None and self._stt.config.stt_model != stage_config.stt_model:
            self._stt = None
        if self._stt is None:
            self._stt = LocalSTT(stage_config)
        return self.stt.transcribe(Path(audio_path))

    def reply(self, request: DialogueRequest | dict, user_text: str | None = None) -> str:
        stage_config = self._stage_config()
        if self._llm is not None and self._llm.config.llm_model != stage_config.llm_model:
            self._llm = None
        if self._llm is None:
            self._llm = LocalLLM(stage_config)
        normalized = request if isinstance(request, DialogueRequest) else DialogueRequest.from_mapping(request, user_text)
        return self.llm.respond(normalized)

    def synthesize(self, text: str, *, speaker_wav: str | Path | None = None) -> Path:
        stage_config = self._stage_config()
        if self._tts is not None and self._tts.config.tts_language != stage_config.tts_language:
            self._tts = None
        if self._tts is None:
            self._tts = LocalTTS(stage_config)
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
