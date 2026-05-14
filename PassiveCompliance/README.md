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

| File | Description |
|------|-------------|
| `piano_playing_hand.py` | Conditions 1 & 2: rhythmic index+ring pressing with uniform or heterogeneous cart stiffness |
| `piano_glissando.py` | Condition 3: index+middle glissando slide — UR5 sweeps across keys while fingers stay pressed |
| `plot_piano_playing_hand.ipynb` | Plot piano playing hand experiments |
| `HelperPianoMIDI/midi_publisher.py` | Physical keyboard → ROS2 MIDI topics |
| `HelperPianoMIDI/midi_subscriber.py` | Subscribe and print MIDI events |
| `HelperPianoMIDI/piano_config.py` | UR5 poses and shared constants for piano experiments |
