# PassiveCompliance — Experimental Area

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone, without hardware reconfiguration.
Compliance acts as a bidirectional filter: outward (absorbs noise and impacts
before contact), inward (smoothens deformations and provides adaptability).

## Structure

```
PassiveCompliance/
├── Finger/     — single finger testbed experiments
└── Hand/       — ADAPT Hand piano-playing experiments
```

---

## Finger/

Each script runs a VMC controller to hold a reference configuration while an
external motion presses the fingertip along a commanded direction. Virtual
stiffness settings are swept and the script logs tip displacement, estimated
force, and joint state to build force-displacement curves. Variants change the
contact direction or starting pose.

Key technical elements:
- Joint-space and task-space virtual springs are used to shape the apparent
	stiffness at the fingertip.
- Directional constraints isolate normal and tangential responses.
- The output is a family of force-displacement curves for comparison across
	directions and poses.

| File | Description |
|------|-------------|
| `passive_stiffness_sweep.py` | Sweep virtual joint stiffness and record force versus displacement |
| `passive_range.py` | Dense sweep with one run per stiffness setting |
| `passive_stiffness_sweep_linear.py` | Sweep task-space stiffness in a pressing direction and record force versus displacement |
| `passive_range_linear.py` | Dense sweep in task space for the pressing direction |
| `directional_stiffness.py` | Task-space stiffness in multiple contact directions in a plane |
| `pose_sweep.py` | Sweep starting poses and record force versus displacement |
| `plot_passive_stiffness_sweep.ipynb` | Plot passive stiffness sweep experiment |
| `plot_passive_stiffness_sweep_linear.ipynb` | Plot passive stiffness sweep linear experiment |
| `plot_directional_stiffness.ipynb` | Plot directional stiffness experiment |
| `plot_pose_sweep.ipynb` | Plot pose sweep experiment |

---

## Hand/

Both scripts require `midi_publisher.py` running in a separate terminal (see `HelperPianoMIDI/`).

The hand maintains a press pose under joint-space VMC while the UR5 drives key
motion. Experiments compare uniform versus mixed fingertip stiffness and log
joint state together with MIDI events to relate compliance to key interaction.

Key technical elements:
- Joint-space VMC stabilizes a press pose while allowing compliance at the tips.
- UR5 motion provides repeatable key interaction trajectories.
- MIDI events provide timing tags for contact and release phases.

| File | Description |
|------|-------------|
| `piano_playing_hand.py` | Hand holds a fixed press pose while the UR5 performs rhythmic press-lift strokes. Runs both uniform and mixed stiffness conditions, saving hand state and MIDI logs per stroke. |
| `piano_glissando.py` | Hand holds a press pose while the UR5 slides along the keyboard and returns. Saves hand state and MIDI logs per run. |
| `plot_piano.ipynb` | Plot piano playing and glissando experiments |
| `HelperPianoMIDI/midi_publisher.py` | Reads physical keyboard and publishes on `/midi/note_on` — must be running alongside experiment scripts |
| `HelperPianoMIDI/midi_subscriber.py` | Subscribe and print MIDI events (debug/monitor) |
| `HelperPianoMIDI/piano_config.py` | Shared constants for piano experiments |
