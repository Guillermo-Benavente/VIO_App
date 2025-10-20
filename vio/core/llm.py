from pathlib import Path
from llama_cpp import Llama

class LLMLocal:
    def __init__(self, model_name='tinyllama-1.1b-chat-v1.0.Q4_K_M'):
        base = Path(__file__).resolve().parent.parent
        model_path = base / 'models' / f'{model_name}.gguf'
        self.model = Llama(model_path=str(model_path))
    
    def respond(self, prompt, max_tokens=256):
        response = self.model(prompt, max_tokens=max_tokens)
        return response['choices'][0]['text'].strip()