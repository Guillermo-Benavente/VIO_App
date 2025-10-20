from flask import Flask, request, jsonify
from pathlib import Path
from core.stt import STTLocal
from core.llm import LLMLocal
from core.tts import TTSLocal
import json
import traceback

app = Flask(__name__)

modelSTT = STTLocal()
modelLLM = LLMLocal()
modelTTS = TTSLocal()

# curl -X POST http://127.0.0.1:5000/stt -H "Content-Type: application/json" -d '{"audio_path": "pruebas/prueba3.m4a"}'
@app.route('/stt', methods=['POST'])
def stt_route():
    audio_rel_path = request.json['audio_path']
    
    audio_path = Path(audio_rel_path).resolve()
    
    if not audio_path.exists():
        return jsonify({'error': 'Archivo no encontrado'}), 404

    result = modelSTT.transcribe(str(audio_path))
    
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
    
# curl -X POST http://127.0.0.1:5000/llm -H "Content-Type: application/json" --data-binary "@pruebas/prueballm.json"
@app.route('/llm', methods=['POST'])
def llm_route():
    text = request.json['text']
    result = modelLLM.respond(f"Usuario dijo: {text}\nResponde de manera amable:")
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

# curl -X POST http://127.0.0.1:5000/tts -H "Content-Type: application/json" --data-binary "@pruebas/pruebatts.json"
@app.route('/tts', methods=['POST'])
def tts_route():
    data = request.get_json(force=True)
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Falta el texto para sintetizar'}), 400

    speaker_wav = data.get('speaker_wav', "pruebas/pruebatts01.wav")
    
    try:
        audio_path = modelTTS.create_audio(text, speaker_wav)
        return jsonify({'audio_path': audio_path}), 200
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500

@app.route('/full', methods=['POST'])
def full_route():
    audio_file = request.files['audio']
    result = modelSTT.transcribe(audio_file.filename)
    result.update(modelLLM.respond(result['audio_converter']))
    result['url'] = modelTTS.create_audio(result['respond'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)