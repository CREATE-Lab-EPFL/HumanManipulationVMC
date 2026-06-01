# HelperPianoMIDI

Direct `rtmidi` interface for a physical MIDI keyboard — no ROS2 bridge needed.

## Files

| File | Description |
|------|-------------|
| `midi_controller.py` | `MidiController` class — opens a MIDI port, tracks pressed notes, fires optional callbacks on note-on / note-off |
| `midi_listener.py` | Terminal listener — prints every note and CC event (quick connectivity check) |
| `midi_visualizer.py` | Live piano-roll visualizer — scrolling roll + keyboard, dark/light theme, save figures with `s` |
| `piano_config.py` | Shared experiment constants (UR5 poses, stiffness values, timing) |

## Dependencies

```bash
pip install python-rtmidi
```

If `snd_seq` is not loaded (ALSA sequencer error on first run):
```bash
sudo modprobe snd_seq
```

## Check connection

```bash
python3 midi_listener.py --list          # list available MIDI ports
python3 midi_listener.py                 # print events as you play
python3 midi_listener.py --port Arturia  # select port by partial name
```

## Live visualizer

```bash
python3 midi_visualizer.py                    # dark theme (for video recording)
python3 midi_visualizer.py --light            # light theme (for paper figures)
python3 midi_visualizer.py --persist          # auto-reconnect if device disconnects
python3 midi_visualizer.py --midi-min 36 --midi-max 84   # restrict note range
python3 midi_visualizer.py --save-dir figures            # where to save snapshots
```

While the window is open:
- **`s`** — save the current frame as `midi_snapshot_NNN.png` + `.pdf` (300 DPI)
- **`q`** — quit

## Using `MidiController` in your own script

```python
from midi_controller import MidiController

# Event-based (callbacks run on background thread)
def on_note(note, velocity):
    print(f'pressed {note} vel={velocity}')

ctrl = MidiController(on_note_on=on_note)

# Or poll the current state
pressed = ctrl.get_pressed(midi_min=48, midi_max=83)  # {48: False, 49: True, ...}
held    = ctrl.is_pressed(60)                          # True/False

ctrl.close()  # or: use as context manager (with MidiController() as ctrl: ...)
```
