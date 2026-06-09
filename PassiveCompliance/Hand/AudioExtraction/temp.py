"""
Plot the RMS envelope of one or more audio files for comparison.

Usage:
    python temp.py <file1.wav> [file2.wav] [file3.wav] ...
"""

import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# RMS window length in milliseconds
RMS_MS = 50

if len(sys.argv) < 2:
    print('Usage: python temp.py <file1.wav> [file2.wav] ...')
    sys.exit(1)

paths = sys.argv[1:]

def rms_envelope(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win      = max(1, int(RMS_MS * sr / 1000))
    n_frames = len(audio) // win
    rms      = np.sqrt(np.mean(audio[:n_frames * win].reshape(n_frames, win) ** 2, axis=1))
    t        = (np.arange(n_frames) * win + win / 2) / sr
    return t, rms

fig, axes = plt.subplots(len(paths), 1, figsize=(14, 3 * len(paths)),
                         sharex=True, squeeze=False)

for ax, path in zip(axes[:, 0], paths):
    t, rms = rms_envelope(path)
    ax.plot(t, rms, lw=0.8, color='steelblue')
    ax.set_ylabel(f'RMS ({RMS_MS} ms)')
    ax.set_title(path)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)

axes[-1, 0].set_xlabel('Time [s]')
fig.tight_layout()
plt.show()
