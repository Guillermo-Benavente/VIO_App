from imageio_ffmpeg import get_ffmpeg_exe
import subprocess
from pathlib import Path

def convert_wav(input_path: str) -> Path:
    input_path = Path(input_path).resolve()
    output_path = input_path.with_suffix('.wav')
    ffmpeg_exe = get_ffmpeg_exe()

    if not output_path.exists():
        subprocess.run([
            str(ffmpeg_exe), '-y',
            '-i', str(input_path),
            '-ac', '1', '-ar', '16000',
            str(output_path)
        ], check=True)

    return output_path
