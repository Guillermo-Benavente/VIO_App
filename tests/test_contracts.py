from vio.contracts import DialogueRequest
from vio.core.llm import LocalLLM


def test_old_npc_json_is_normalized():
    request = DialogueRequest.from_mapping({
        "npc": {"name": "Álvaro", "personality": {"tone": "serio"}, "objectives": ["vigilar"]},
        "session": {"history": [{"sender": "npc", "text": "Alto ahí."}]},
        "last_turn": {"text": "¿Puedo entrar?"},
    })
    assert request.character_name == "Álvaro"
    assert request.user_text == "¿Puedo entrar?"
    assert LocalLLM._messages(request)[-1]["content"] == "¿Puedo entrar?"
