# PassiveCompliance — Experimental Area

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone, without hardware reconfiguration.
Compliance acts as a bidirectional filter: outward (absorbs noise and impacts
before contact), inward (smoothens deformations and provides adaptability).

## Structure

```
PassiveCompliance/
├── Finger/   — single finger testbed experiments
└── Hand/     — ADAPT Hand experiments
```

---

## Finger/

Each script runs a VMC controller to hold a reference configuration while an
external motion presses the fingertip along a commanded direction. Virtual
stiffness settings are swept and the script logs tip displacement, estimated
force, and joint state to build force-displacement curves. Variants change the
contact direction or starting pose.

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

---

## Hand/

All hand experiments require the ADAPT Hand on ROS2 (`dynamixel_node`).
Shared configuration lives in a single `hand_config.py`.

### Guitar (`guitar_playing_hand.py`)

All fingers except the thumb are held closed at a fixed flexion pose under a
torsional (joint-space) spring. The UR5 drags the closed hand linearly across
the strings at two stiffness conditions; audio is captured externally via the
camera microphone.

Key technical elements:
- Torsional (joint-space) stiffness is the swept variable; task-space springs are
  not used — the experiment isolates the effect of joint stiffness on sound.
- Fingers stay closed throughout all runs within a condition; the arm lifts over
  the top and returns without reopening the hand.
- No friction compensation (FRICTION_TAU_MAX = 0).

| File | Description |
|------|-------------|
| `guitar_playing_hand.py` | Guitar strumming experiment — N_RUNS per stiffness condition |
| `plot_guitar.ipynb` | Joint angle and torque figures per condition |

### Weight compliance (`weight_compliance_hand.py`)

Four fingers are commanded to hold a fixed pose (WEIGHT_POSE) under joint-space
springs. Hanging weights are applied in steps to measure steady-state angular
deflection as a function of stiffness. The hand is fixed (UR5 stationary or absent).

Key technical elements:
- Torsional stiffness (STIFFNESS_CONDITIONS) is the swept variable.
- For each stiffness, the operator steps through WEIGHTS_G and the script logs
  the steady-state joint state after a settle period.
- With UR5 absent, UR5Receiver falls back to identity rotation for gravity
  compensation; connect a stationary UR5 for accurate orientation.

| File | Description |
|------|-------------|
| `weight_compliance_hand.py` | Weight-loading experiment — logs joint angles per (K, weight) pair |
| `plot_weight.ipynb` | Deviation-from-target vs weight, per stiffness condition |

### Configuration

| File | Description |
|------|-------------|
| `hand_config.py` | All constants for both Hand experiments (UR5 poses, stiffness values, timing) |
