"""
Extract and band-pass filter audio from a Canon video recording.

Keeps only frequencies within the guitar's audible range (LOW_HZ – HIGH_HZ).
Everything outside that band (rumble, hiss, electronic noise) is suppressed.
No quiet-room profiling required.

Usage:
    python extract_audio.py <video_file>

Output:
    <base>.wav          — raw extracted audio (next to the video file)
    <base>_filtered.wav — band-pass filtered audio

Dependencies:
    sudo apt install ffmpeg
    pip install soundfile scipy numpy
"""

import subprocess
import sys
import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# Guitar frequency range: open low-E (82 Hz) up to ~8 kHz (harmonics + body resonance).
# Anything outside this window is treated as noise and zeroed.
LOW_HZ  = 80
HIGH_HZ = 8000

# Butterworth order — higher = steeper roll-off, but avoid ringing artefacts
FILTER_ORDER = 6


def _bandpass(audio, sr):
    nyq = sr / 2.0
    sos = butter(FILTER_ORDER, [LOW_HZ / nyq, HIGH_HZ / nyq], btype='bandpass', output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    # multi-channel: filter each channel independently
    return np.stack([sosfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)


if len(sys.argv) != 2:
    print('Usage: python extract_audio.py <video_file>')
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.isfile(video_path):
    print(f'File not found: {video_path}')
    sys.exit(1)

base, _ = os.path.splitext(video_path)
raw_path      = base + '.wav'
filtered_path = base + '_filtered.wav'

# ── Step 1: extract audio to WAV ──────────────────────────────────────────────
print('Extracting audio…')
subprocess.run([
    'ffmpeg', '-y', '-i', video_path,
    '-vn', '-acodec', 'pcm_s16le',
    raw_path,
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f'  Raw audio → {raw_path}')

# ── Step 2: bandpass filter ────────────────────────────────────────────────────
print(f'Filtering {LOW_HZ}–{HIGH_HZ} Hz (guitar range)…')
audio, sr = sf.read(raw_path)
filtered  = _bandpass(audio.astype(np.float64), sr)

# Normalise to [-1, 1] to avoid clipping when writing int16
peak = np.max(np.abs(filtered))
if peak > 0:
    filtered /= peak

sf.write(filtered_path, filtered, sr)
print(f'  Filtered  → {filtered_path}')
