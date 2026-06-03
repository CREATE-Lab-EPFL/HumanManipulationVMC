# HelperGuitar

Microphone interface for the guitar experiment — no ROS2 bridge needed.
The microphone RMS level is the intensity proxy (analogous to MIDI velocity
in the piano experiment).

## Files

| File | Description |
|------|-------------|
| `mic_controller.py` | `MicrophoneController` class — opens an audio input stream, tracks the RMS level continuously, and fires optional callbacks on detected onsets (energy threshold) |
| `guitar_config.py` | Shared experiment constants (UR5 poses, torsional stiffness values, microphone settings, timing) |

## Dependencies

```bash
pip install sounddevice
```

PortAudio must also be available (usually pre-installed on Ubuntu/macOS):

```bash
sudo apt-get install libportaudio2   # Ubuntu
brew install portaudio               # macOS
```

## Check connection

```bash
python3 mic_controller.py --list         # list available audio input devices
python3 mic_controller.py                # print onset events as you pluck
python3 mic_controller.py --device USB   # select device by partial name
python3 mic_controller.py --threshold 0.03   # adjust sensitivity
```

## Using `MicrophoneController` in your own script

```python
from mic_controller import MicrophoneController

# Event-based (callback runs on audio thread)
def on_pluck(intensity, t):
    print(f'pluck  intensity={intensity:.3f}  t={t:.2f}s')

mic = MicrophoneController(on_onset=on_pluck)

# Poll the current RMS level (non-blocking)
level = mic.get_level()   # 0..1

# Access the onset history
for t, intensity in mic.onsets:
    print(t, intensity)

mic.reset()   # clear onset history between runs

mic.close()   # or: use as context manager (with MicrophoneController() as mic: ...)
```

## Onset detection

Onsets are detected with a simple energy-based rule:
- A **rising edge** fires when the block RMS exceeds `onset_threshold`
  *and* at least `refractory` seconds have elapsed since the last onset.
- Hysteresis: the detector resets only when the level drops below
  `0.5 × onset_threshold`.
- The reported **intensity** is the block's peak amplitude (0..1) —
  the loudness proxy.

For the guitar experiment the continuous `mic_level` column (RMS per
control loop tick) is the primary output; the onset callback is available
for other uses.
