# PassiveCompliance — Experimental Area 1

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone, without hardware reconfiguration.
Compliance acts as a bidirectional filter: outward (absorbs noise/impacts before
contact), inward (smoothens deformations, providing adaptability).

## Files

| File | Platform | Description |
|------|----------|-------------|
| `passive_stiffness_sweep.py` | finger | Stiffness sweep — F vs d for varying K (finger space, N·m/rad) |
| `passive_range.py` | finger | Dense biased K sweep, one run per K |
| `passive_stiffness_sweep_linear.py` | finger | Cart stiffness sweep — F vs d for varying K_cart (N/m), vertical direction |
| `passive_range_linear.py` | finger | Dense biased K_cart sweep, vertical direction |
| `directional_stiffness.py` | finger | Cart stiffness in multiple contact directions in the Y-Z plane |
| `pose_sweep.py` | finger | Pose sweep — F vs d from multiple starting Z heights |
| `piano_playing_hand.py` | hand | Three conditions: uniform stiffness sweep, k₁/k₂ per-finger assignment, and damping sweep |
| `plot_passive_stiffness_sweep.ipynb` | finger | Plot passive stiffness sweep experiment |
| `plot_passive_stiffness_sweep_linear.ipynb` | finger | Plot passive stiffness sweep linear experiment |
| `plot_directional_stiffness.ipynb` | finger | Plot directional stiffness experiment |
| `plot_pose_sweep.ipynb` | finger | Plot pose sweep experiment |
| `plot_piano_playing_hand.ipynb` | hand | Plot piano playing hand experiment |

## HelperPianoMIDI

Tools for routing a physical MIDI keyboard into ROS2.

| File | Description |
|------|-------------|
| `HelperPianoMIDI/midi_publisher.py` | Reads live events from a USB/MIDI keyboard and publishes on `/midi/note_on`, `/midi/note_off`, `/midi/control` (`std_msgs/String`) |
| `HelperPianoMIDI/midi_subscriber.py` | Subscribes to the three topics above and prints human-readable note/control info to the terminal |

**Dependencies** — install once:
```bash
pip install mido python-rtmidi
```

**Run the publisher** (terminal 1):
```bash
python3 HelperPianoMIDI/midi_publisher.py           # auto-selects first MIDI port
python3 HelperPianoMIDI/midi_publisher.py --port "Arturia"  # filter by port name
```

**Run the subscriber** (terminal 2):
```bash
python3 HelperPianoMIDI/midi_subscriber.py
```

**Topics published**

| Topic | Message | Content |
|-------|---------|---------|
| `/midi/note_on` | `std_msgs/String` | `note=<0-127> velocity=<0-127>` |
| `/midi/note_off` | `std_msgs/String` | `note=<0-127>` |
| `/midi/control` | `std_msgs/String` | `cc=<0-127> value=<0-127>` |
