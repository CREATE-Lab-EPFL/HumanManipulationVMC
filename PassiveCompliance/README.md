# PassiveCompliance — Experimental Area

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip and palm through virtual springs alone, without hardware reconfiguration. Compliance acts as a bidirectional filter: outward (absorbs noise and impacts before contact), inward (smooths deformations and provides adaptability).

```
PassiveCompliance/
├── Finger/   — single finger testbed experiments
└── Hand/     — ADAPT Hand experiments
```

## Finger/

Each script holds a reference configuration under VMC while an external motion presses the fingertip along a commanded direction. Virtual stiffness is swept and tip displacement, estimated force, and joint state are logged to build force-displacement curves. Variants change contact direction or starting pose.

| File | Description |
|------|-------------|
| `passive_stiffness_sweep.py` | Sweep virtual joint stiffness, record force vs displacement |
| `passive_range.py` | Dense sweep with one run per stiffness setting |
| `passive_stiffness_sweep_linear.py` | Sweep task-space stiffness in a pressing direction |
| `passive_range_linear.py` | Dense sweep in task space for the pressing direction |
| `directional_stiffness.py` | Task-space stiffness in multiple contact directions in a plane |
| `pose_sweep.py` | Sweep starting poses, record force vs displacement |
| `plot_passive_stiffness_sweep.ipynb` | Plot passive stiffness sweep experiment |
| `plot_passive_stiffness_sweep_linear.ipynb` | Plot passive stiffness sweep linear experiment |
| `plot_directional_stiffness.ipynb` | Plot directional stiffness experiment |
| `plot_pose_sweep.ipynb` | Plot pose sweep experiment |

## Hand/

All hand experiments require the ADAPT Hand on ROS2 (`dynamixel_node`); shared configuration lives in `hand_config.py`.

**`guitar_playing_hand.py`** — index, middle, and ring fingers are held closed at a fixed flexion pose under a torsional (joint-space) spring while the UR5 performs a single upward strum across the strings (audio captured externally). Thumb and pinky stay at a small neutral pose and are not involved. One run per stiffness condition, comparing [0.05, 0.2, 0.6] N·m/rad — joint stiffness is the only swept variable, isolating its effect on sound. Before each strum the controller briefly re-stabilizes at higher stiffness/damping for identical initial conditions, then softens to the test condition. Set `COLLECTED_DATA = True` to replay the arm protocol without overwriting saved CSVs.

**`weight_compliance_hand.py`** — four fingers hold a fixed pose (`WEIGHT_POSE`) under joint-space springs (thumb stays neutral, unloaded) while the operator adds weights incrementally; each addition is logged over a 5 s convergence window. The hand ramps to the target pose once at the start, then softens to each stiffness condition (`STIFFNESS_CONDITIONS`) in turn — step 0 (no weight) is logged automatically as the baseline. Without a UR5 connected, gravity compensation falls back to identity rotation.

| File | Description |
|------|-------------|
| `guitar_playing_hand.py` | Guitar strumming — one strum per stiffness condition |
| `plot_guitar.ipynb` | Joint angle and torque figures per condition |
| `weight_compliance_hand.py` | Weight-loading — logs joint angles per (stiffness, weight) pair |
| `plot_weight.ipynb` | Deviation-from-target vs weight, per stiffness condition |
| `hand_config.py` | All constants for both experiments (UR5 poses, stiffness values, timing) |
