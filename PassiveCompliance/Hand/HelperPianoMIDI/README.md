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

**Terminal A — must be running before any piano experiment script:**
```bash
python midi_publisher.py                      # auto-selects an available port
python midi_publisher.py --port "Arturia"     # filter by partial port name
```

**Terminal B — optional live monitor:**
```bash
python midi_subscriber.py
```

## Topics

| Topic | Type | Format |
|-------|------|--------|
| `/midi/note_on` | `std_msgs/String` | `note=<midi note> velocity=<midi velocity>` |
| `/midi/note_off` | `std_msgs/String` | `note=<midi note>` |
| `/midi/control` | `std_msgs/String` | `cc=<control index> value=<control value>` |

Note-on with no velocity is treated as note-off (standard MIDI spec).
