# HelperPianoMIDI

Routes a physical MIDI keyboard (USB/MIDI) into ROS2 topics.

## Files

| File | Description |
|------|-------------|
| `midi_publisher.py` | Reads live events from the keyboard and publishes on `/midi/note_on`, `/midi/note_off`, `/midi/control` |
| `midi_subscriber.py` | Subscribes to those topics and prints human-readable note/control info |

## Dependencies

```bash
pip install mido python-rtmidi
```

## Usage

**Terminal 1 — publish:**
```bash
python3 midi_publisher.py                      # auto-selects first available port
python3 midi_publisher.py --port "Arturia"     # filter by partial port name
```

**Terminal 2 — monitor:**
```bash
python3 midi_subscriber.py
```

## Topics

| Topic | Type | Format |
|-------|------|--------|
| `/midi/note_on` | `std_msgs/String` | `note=<0-127> velocity=<0-127>` |
| `/midi/note_off` | `std_msgs/String` | `note=<0-127>` |
| `/midi/control` | `std_msgs/String` | `cc=<0-127> value=<0-127>` |

Note-on with `velocity=0` is treated as note-off (standard MIDI spec).
