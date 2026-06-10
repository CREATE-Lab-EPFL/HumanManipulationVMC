"""
Extract audio from Canon video recordings and normalise to a shared amplitude scale.

When multiple files are provided, all outputs are normalised to the same global
peak so that RMS intensity is directly comparable across recordings.

Usage:
    python extract_audio.py <video1> [video2] [video3] ...

Output (next to each input file):
    <base>.wav — extracted audio, globally normalised

Dependencies:
    sudo apt install ffmpeg
    pip install soundfile numpy
"""

import subprocess
import sys
import os
import numpy as np
import soundfile as sf


if len(sys.argv) < 2:
    print('Usage: python extract_audio.py <video1> [video2] ...')
    sys.exit(1)

video_paths = sys.argv[1:]
for p in video_paths:
    if not os.path.isfile(p):
        print(f'File not found: {p}')
        sys.exit(1)

# ── Step 1: extract raw audio from each video ────────────────────────────────
results = []   # list of (wav_path, audio_array, sample_rate)
for video_path in video_paths:
    base     = os.path.splitext(video_path)[0]
    wav_path = base + '.wav'

    print(f'\n[{os.path.basename(video_path)}]')
    print('  Extracting audio…')
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', wav_path,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    audio, sr = sf.read(wav_path)
    audio     = audio.astype(np.float64)
    results.append((wav_path, audio, sr))

# ── Step 2: normalise all files to the same global peak ──────────────────────
global_peak = max(np.max(np.abs(audio)) for _, audio, _ in results)
if global_peak > 0:
    print(f'\nGlobal peak: {global_peak:.6f}  — normalising all files to this scale.')
    results = [(path, audio / global_peak, sr) for path, audio, sr in results]

# ── Step 3: save ─────────────────────────────────────────────────────────────
for wav_path, audio, sr in results:
    sf.write(wav_path, audio, sr)
    print(f'  Saved → {wav_path}')
