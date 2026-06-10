"""
Extract and clean audio from Canon video recordings.

Two-stage denoising designed to isolate the guitar against UR5 motor noise:
  1. Butterworth bandpass (LOW_HZ – HIGH_HZ): removes everything outside the
     guitar frequency range (sub-bass rumble, ultrasonic hiss, etc.).
  2. Non-stationary spectral gating (noisereduce, stationary=False): tracks the
     noise floor dynamically throughout the recording. This handles the UR5
     whose noise changes as the arm accelerates and decelerates — guitar strums
     produce large transient bursts well above the tracked floor and are
     preserved; the continuous motor hum is subtracted frame by frame.

When multiple files are provided, all clean outputs are normalised to the same
global peak amplitude so that RMS intensity is directly comparable across files.

Usage:
    python extract_audio.py <video1> [video2] [video3] ...

Output (next to each input file):
    <base>.wav       — raw extracted audio
    <base>_clean.wav — bandpass + spectral-gated, globally normalised

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

def _bandpass(audio, sr):
    nyq = sr / 2.0
    sos = butter(FILTER_ORDER, [LOW_HZ / nyq, HIGH_HZ / nyq], btype='bandpass', output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    return np.stack([sosfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)


def _spectral_gate(audio, sr):
    # stationary=False: noise floor re-estimated each frame → handles UR5 speed changes
    if audio.ndim == 1:
        return nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.95)
    channels = [nr.reduce_noise(y=audio[:, ch], sr=sr, stationary=False, prop_decrease=0.95)
                for ch in range(audio.shape[1])]
    return np.stack(channels, axis=1)


if len(sys.argv) < 2:
    print('Usage: python extract_audio.py <video1> [video2] ...')
    sys.exit(1)

video_paths = sys.argv[1:]
for p in video_paths:
    if not os.path.isfile(p):
        print(f'File not found: {p}')
        sys.exit(1)

# ── Step 1 & 2 & 3: extract, bandpass, spectral-gate each file ───────────────
results = []   # list of (clean_path, audio_array, sample_rate)
for video_path in video_paths:
    base       = os.path.splitext(video_path)[0]
    raw_path   = base + '.wav'
    clean_path = base + '_clean.wav'

    print(f'\n[{os.path.basename(video_path)}]')
    print('  Extracting audio…')
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', raw_path,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f'  Bandpass {LOW_HZ}–{HIGH_HZ} Hz…')
    audio, sr = sf.read(raw_path)
    audio     = audio.astype(np.float64)
    filtered  = _bandpass(audio, sr)

    print('  Spectral gating (non-stationary)…')
    clean = _spectral_gate(filtered, sr)

    results.append((clean_path, clean, sr))

# ── Step 4: normalise all files to the same global peak ──────────────────────
global_peak = max(np.max(np.abs(clean)) for _, clean, _ in results)
if global_peak > 0:
    print(f'\nGlobal peak: {global_peak:.6f}  — normalising all files to this scale.')
    for i in range(len(results)):
        path, clean, sr = results[i]
        results[i] = (path, clean / global_peak, sr)

# ── Step 5: save ──────────────────────────────────────────────────────────────
for clean_path, clean, sr in results:
    sf.write(clean_path, clean, sr)
    print(f'  Saved → {clean_path}')
