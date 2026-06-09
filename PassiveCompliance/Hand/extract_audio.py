"""
Extract and denoise audio from a Canon video recording.

Uses spectral gating (noisereduce) to remove stationary background noise,
profiled from the first NOISE_PROFILE_S seconds of the recording.

Usage:
    python extract_audio.py <video_file>

Output:
    <base>.wav          — raw extracted audio
    <base>_denoised.wav — noise-reduced audio

Dependencies:
    sudo apt install ffmpeg
    pip install noisereduce soundfile
"""

import subprocess
import sys
import os
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr

# Seconds at the start of the recording used to profile background noise.
# Make sure the room is quiet (no strumming) for at least this long before playing.
NOISE_PROFILE_S = 0.5

if len(sys.argv) != 2:
    print('Usage: python extract_audio.py <video_file>')
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.isfile(video_path):
    print(f'File not found: {video_path}')
    sys.exit(1)

base, _ = os.path.splitext(video_path)
raw_path      = base + '.wav'
denoised_path = base + '_denoised.wav'

# ── Step 1: extract audio to WAV ──────────────────────────────────────────────
print('Extracting audio…')
subprocess.run([
    'ffmpeg', '-y', '-i', video_path,
    '-vn', '-acodec', 'pcm_s16le',
    raw_path,
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f'  Raw audio → {raw_path}')

# ── Step 2: denoise ────────────────────────────────────────────────────────────
print('Denoising…')
audio, sr = sf.read(raw_path)

# noisereduce expects mono or (samples, channels); handle both
if audio.ndim == 1:
    noise_profile = audio[:int(NOISE_PROFILE_S * sr)]
    denoised = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
else:
    # process each channel independently
    channels = []
    for ch in range(audio.shape[1]):
        noise_profile = audio[:int(NOISE_PROFILE_S * sr), ch]
        channels.append(nr.reduce_noise(y=audio[:, ch], sr=sr, y_noise=noise_profile))
    denoised = np.stack(channels, axis=1)

sf.write(denoised_path, denoised, sr)
print(f'  Denoised  → {denoised_path}')
