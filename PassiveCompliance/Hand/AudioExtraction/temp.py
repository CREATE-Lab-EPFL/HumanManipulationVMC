"""
Plot the RMS envelope of an audio file over time.

Usage:
    python temp.py <audio_file.wav>
"""

import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# RMS window length in milliseconds — controls smoothing of the envelope
RMS_MS = 50

if len(sys.argv) != 2:
    print('Usage: python temp.py <audio_file.wav>')
    sys.exit(1)

audio, sr = sf.read(sys.argv[1])

if audio.ndim > 1:
    audio = audio.mean(axis=1)

# RMS envelope: compute RMS over non-overlapping windows
win = max(1, int(RMS_MS * sr / 1000))
n_frames = len(audio) // win
rms = np.sqrt(np.mean(audio[:n_frames * win].reshape(n_frames, win) ** 2, axis=1))
t   = (np.arange(n_frames) * win + win / 2) / sr

plt.figure(figsize=(14, 4))
plt.plot(t, rms, lw=0.8, color='steelblue')
plt.xlabel('Time [s]')
plt.ylabel(f'RMS amplitude ({RMS_MS} ms window)')
plt.title(sys.argv[1])
plt.xlim(t[0], t[-1])
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
