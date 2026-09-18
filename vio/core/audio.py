import subprocess
from pathlib import Path
from imageio_ffmpeg import get_ffmpeg_exe


def convert_to_wav(input_path: str | Path, output_path: str | Path) -> Path:
    """Normalise input without overwriting it or trusting a stale conversion."""
    source, target = Path(input_path).resolve(), Path(output_path).resolve()
    if not source.is_file(): raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(get_ffmpeg_exe()), "-y", "-i", str(source), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)], check=True, capture_output=True)
    return target
