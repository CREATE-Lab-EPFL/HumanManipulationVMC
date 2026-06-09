"""
Extract and isolate guitar notes from a Canon video recording.

Two-stage pipeline:
  1. HPSS (Harmonic-Percussive Source Separation): keeps only sustained tonal
     content (string resonance, harmonics) and discards broadband noise,
     transients, and room clutter.
  2. Butterworth bandpass C3–C6 (130.8 – 1046.5 Hz): restricts to the central
     three octaves of the keyboard, removing sub-bass and high-frequency content
     that is not musically relevant for the guitar strumming task.

No quiet-room profiling is required.

Usage:
    python extract_audio.py <video_file>

Output:
    <base>.wav       — raw extracted audio (next to the video file)
    <base>_clean.wav — HPSS + bandpass filtered audio

Dependencies:
    sudo apt install ffmpeg
    pip install librosa soundfile scipy numpy
"""

import subprocess
import sys
import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt

# Central three octaves: C3 to C6
LOW_HZ  = 130.8   # C3
HIGH_HZ = 1046.5  # C6
FILTER_ORDER = 4  # gentle Butterworth roll-off, avoids ringing

# librosa native sample rate for HPSS processing
LIBROSA_SR = 22050


def _bandpass(audio, sr):
    nyq = sr / 2.0
    sos = butter(FILTER_ORDER, [LOW_HZ / nyq, HIGH_HZ / nyq], btype='bandpass', output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    return np.stack([sosfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)


if len(sys.argv) != 2:
    print('Usage: python extract_audio.py <video_file>')
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.isfile(video_path):
    print(f'File not found: {video_path}')
    sys.exit(1)

base, _ = os.path.splitext(video_path)
raw_path   = base + '.wav'
clean_path = base + '_clean.wav'

# ── Step 1: extract audio to WAV ──────────────────────────────────────────────
print('Extracting audio…')
subprocess.run([
    'ffmpeg', '-y', '-i', video_path,
    '-vn', '-acodec', 'pcm_s16le',
    raw_path,
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f'  Raw audio → {raw_path}')

# ── Step 2: HPSS — keep harmonic component only ───────────────────────────────
print('Running HPSS (harmonic-percussive separation)…')
# librosa loads mono at LIBROSA_SR for HPSS; we re-read the original sr for output
y_mono, _ = librosa.load(raw_path, sr=LIBROSA_SR, mono=True)
harmonic, _ = librosa.effects.hpss(y_mono)

# ── Step 3: bandpass to central three octaves (C3–C6) ─────────────────────────
print(f'Bandpass {LOW_HZ:.1f}–{HIGH_HZ:.1f} Hz (C3–C6)…')
clean = _bandpass(harmonic.astype(np.float64), LIBROSA_SR)

# Normalise to avoid clipping
peak = np.max(np.abs(clean))
if peak > 0:
    clean /= peak

sf.write(clean_path, clean, LIBROSA_SR)
print(f'  Clean audio → {clean_path}')
