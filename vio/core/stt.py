import whisper
import librosa
from .audio_proccess import convert_wav

class STTLocal:
    def __init__(self, model_name='small'):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        audio_path = convert_wav(audio_path)

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        result = self.model.transcribe(audio, language='es', fp16=False)
        return {'audio': str(audio_path), 'audio_converter': result['text']}
