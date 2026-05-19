# PassiveCompliance — Experimental Area 1

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone, without hardware reconfiguration.
Compliance acts as a bidirectional filter: outward (absorbs noise/impacts before
contact), inward (smoothens deformations, providing adaptability).

## Structure

```
PassiveCompliance/
├── Finger/     — single 2-DOF finger testbed experiments
└── Hand/       — ADAPT Hand piano-playing experiments
```

---

## Finger/

| File | Description |
|------|-------------|
| `passive_stiffness_sweep.py` | Stiffness sweep — F vs d for varying K (finger space, N·m/rad) |
| `passive_range.py` | Dense biased K sweep, one run per K |
| `passive_stiffness_sweep_linear.py` | Cart stiffness sweep — F vs d for varying K_cart (N/m), vertical direction |
| `passive_range_linear.py` | Dense biased K_cart sweep, vertical direction |
| `directional_stiffness.py` | Cart stiffness in multiple contact directions in the Y-Z plane |
| `pose_sweep.py` | Pose sweep — F vs d from multiple starting Z heights |
| `plot_passive_stiffness_sweep.ipynb` | Plot passive stiffness sweep experiment |
| `plot_passive_stiffness_sweep_linear.ipynb` | Plot passive stiffness sweep linear experiment |
| `plot_directional_stiffness.ipynb` | Plot directional stiffness experiment |
| `plot_pose_sweep.ipynb` | Plot pose sweep experiment |

---

## Hand/

Both scripts require `midi_publisher.py` running in a separate terminal (see `HelperPianoMIDI/`).

| File | Description |
|------|-------------|
| `piano_playing_hand.py` | Conditions 1 & 2: hand holds a fixed press pose (index+ring, stiffness K); UR5 performs rhythmic press-lift strokes.  Two conditions: uniform K swept over K_SWEEP, or heterogeneous K_STIFF/K_SOFT.  Saves hand-state and MIDI CSVs per stroke. |
| `piano_glissando.py` | Condition 3: hand holds index+middle at press pose while UR5 slides along the keyboard (Y) for GLISSANDO_DISTANCE and returns.  Saves hand-state and MIDI CSVs per run. |
| `plot_piano.ipynb` | Plot piano playing and glissando experiments |
| `HelperPianoMIDI/midi_publisher.py` | Reads physical keyboard and publishes on `/midi/note_on` — must be running alongside experiment scripts |
| `HelperPianoMIDI/midi_subscriber.py` | Subscribe and print MIDI events (debug/monitor) |
| `HelperPianoMIDI/piano_config.py` | Shared constants for piano experiments |
