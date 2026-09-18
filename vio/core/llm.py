"""llama.cpp adapter with bounded generation and a Spanish NPC prompt."""

import torch
from llama_cpp import Llama, llama_supports_gpu_offload
from ..contracts import DialogueRequest
from .config import Settings


class LocalLLM:
    def __init__(self, config: Settings):
        self.config = config
        model_path = config.models_dir / config.llm_model
        if not model_path.is_file():
            raise FileNotFoundError(f"GGUF model was not found: {model_path}")
        backend_supports_gpu = llama_supports_gpu_offload()
        if config.gpu_layers == -1:
            gpu_layers = -1 if torch.cuda.is_available() and backend_supports_gpu else 0
        else:
            gpu_layers = config.gpu_layers
        self.device = "cuda" if gpu_layers != 0 else "cpu"
        print(f"[VIO] LLM device: {self.device} (llama.cpp GPU support: {backend_supports_gpu})")
        self.model = Llama(model_path=str(model_path), n_ctx=config.llm_context,
                           n_gpu_layers=gpu_layers, verbose=False)

    @staticmethod
    def _messages(request: DialogueRequest) -> list[dict[str, str]]:
        history = [{"role": "assistant" if item.get("sender") == "npc" else "user", "content": item.get("text", "")}
                   for item in request.history if item.get("text")]
        system = (
            "You are {name}, a role-playing character. Always reply in Spanish as the character, "
            "with no labels, instructions, lists, or narrator. Maintain tone {tone}, humor {humor}, "
            "emotions {emotions}; you are in {location}. Goals: {goals}. "
            "Scene summary: {summary}. Give a natural response of one to four sentences."
        ).format(name=request.character_name, tone=request.tone, humor=request.humor,
                 emotions=", ".join(request.emotions) or "neutral", location=request.location,
                 goals="; ".join(request.goals) or "converse", summary=request.summary or "no summary")
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": request.user_text}]

    def respond(self, request: DialogueRequest) -> str:
        if not request.user_text.strip():
            raise ValueError("The user message cannot be empty")
        response = self.model.create_chat_completion(
            messages=self._messages(request), max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature, top_p=0.9, repeat_penalty=1.08,
            stop=["\nUsuario:", "\nPlayer:", "\nSistema:"],
        )
        text = response["choices"][0]["message"]["content"].strip()
        if not text:
            raise RuntimeError("The LLM generated an empty response")
        return text
