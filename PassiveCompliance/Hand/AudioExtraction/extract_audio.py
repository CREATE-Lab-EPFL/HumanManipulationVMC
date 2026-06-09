"""
Extract and clean audio from a Canon video recording.

Two-stage denoising:
  1. Butterworth bandpass (LOW_HZ – HIGH_HZ): removes everything outside the
     guitar frequency range (sub-bass rumble, ultrasonic hiss, etc.).
  2. Spectral gating (noisereduce): subtracts the stationary noise floor that
     remains inside the guitar band (HVAC, room hum, electronics).
     Profile is taken from the first NOISE_PROFILE_S seconds — keep the room
     quiet (no playing) for at least that long at the start of each recording.

Usage:
    python extract_audio.py <video_file>

Output:
    <base>.wav       — raw extracted audio (next to the video file)
    <base>_clean.wav — bandpass + spectral-gated audio

Dependencies:
    sudo apt install ffmpeg
    pip install noisereduce soundfile scipy numpy
"""

import subprocess
import sys
import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
import noisereduce as nr

# Guitar range: open low-E (82 Hz) to ~8 kHz (harmonics + body resonance).
LOW_HZ  = 80
HIGH_HZ = 8000
FILTER_ORDER = 4       # gentle roll-off avoids ringing artefacts

# Quiet seconds at the start of the recording used to profile noise.
NOISE_PROFILE_S = 2.0


def _bandpass(audio, sr):
    nyq = sr / 2.0
    sos = butter(FILTER_ORDER, [LOW_HZ / nyq, HIGH_HZ / nyq], btype='bandpass', output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    return np.stack([sosfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)


def _spectral_gate(audio, sr):
    n_profile = int(NOISE_PROFILE_S * sr)
    if audio.ndim == 1:
        noise = audio[:n_profile]
        return nr.reduce_noise(y=audio, sr=sr, y_noise=noise, stationary=True, prop_decrease=0.9)
    channels = []
    for ch in range(audio.shape[1]):
        noise = audio[:n_profile, ch]
        channels.append(nr.reduce_noise(y=audio[:, ch], sr=sr, y_noise=noise,
                                        stationary=True, prop_decrease=0.9))
    return np.stack(channels, axis=1)


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

# ── Step 2: bandpass filter ────────────────────────────────────────────────────
print(f'Bandpass {LOW_HZ}–{HIGH_HZ} Hz…')
audio, sr = sf.read(raw_path)
audio     = audio.astype(np.float64)
filtered  = _bandpass(audio, sr)

# ── Step 3: spectral gating on the bandpass result ────────────────────────────
print(f'Spectral gating (noise profile from first {NOISE_PROFILE_S:.1f} s)…')
clean = _spectral_gate(filtered, sr)

# Normalise to avoid clipping
peak = np.max(np.abs(clean))
if peak > 0:
    clean /= peak

sf.write(clean_path, clean, sr)
print(f'  Clean audio → {clean_path}')
