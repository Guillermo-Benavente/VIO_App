"""Stable public data contracts for the VIO library and HTTP API."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    source: Path


@dataclass(frozen=True)
class DialogueRequest:
    """Character and conversation state. All fields have safe defaults."""

    character_name: str = "NPC"
    tone: str = "neutral"
    humor: str = "sin humor"
    emotions: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    location: str = "desconocida"
    summary: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    user_text: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any], user_text: str | None = None) -> "DialogueRequest":
        """Normalize the public English-key NPC JSON contract."""
        npc = data.get("npc", data)
        personality = npc.get("personality", {})
        session = data.get("session", {})
        last_turn = data.get("last_turn", {})
        return cls(
            character_name=npc.get("name", data.get("character_name", "NPC")),
            tone=personality.get("tone", data.get("tone", "neutral")),
            humor=personality.get("humor", data.get("humor", "sin humor")),
            emotions=list(personality.get("base_emotions", data.get("emotions", []))),
            goals=list(npc.get("objectives", data.get("goals", []))),
            location=npc.get("location", data.get("location", "unknown")),
            summary=session.get("summary", data.get("summary", "")),
            history=list(session.get("history", data.get("history", [])))[-8:],
            user_text=user_text if user_text is not None else last_turn.get("text", data.get("user_text", "")),
        )


@dataclass(frozen=True)
class VoiceResult:
    transcript: Transcript
    reply: str
    audio_path: Path
