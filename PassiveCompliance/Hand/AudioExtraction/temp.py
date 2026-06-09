"""
Plot the intensity (waveform) of an audio file over time.

Usage:
    python temp.py <audio_file.wav>
"""

import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Usage: python temp.py <audio_file.wav>')
    sys.exit(1)

audio, sr = sf.read(sys.argv[1])

# collapse to mono for plotting
if audio.ndim > 1:
    audio = audio.mean(axis=1)

t = np.arange(len(audio)) / sr

plt.figure(figsize=(14, 4))
plt.plot(t, audio, lw=0.4, color='steelblue')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title(sys.argv[1])
plt.tight_layout()
plt.show()
