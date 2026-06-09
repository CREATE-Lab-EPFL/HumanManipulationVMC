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
camera microphone. The thumb is held at a small neutral pose (2° on all joints)
and is not involved in the strumming motion.

Key technical elements:
- Torsional (joint-space) stiffness is the swept variable; task-space springs are
  not used — the experiment isolates the effect of joint stiffness on sound.
- At the start of every run the script bumps stiffness to K_ROT with higher damping
  (B_SETTLE) for 1 s to guarantee identical initial conditions, then softens to the
  condition K before the arm moves.
- Fingers stay closed throughout all runs within a condition; the arm lifts over
  the top and returns without reopening the hand.
- Set `COLLECTED_DATA = True` in the script to replay the full arm protocol
  without overwriting any saved CSV files.

| File | Description |
|------|-------------|
| `guitar_playing_hand.py` | Guitar strumming experiment — N_RUNS per stiffness condition |
| `plot_guitar.ipynb` | Joint angle and torque figures per condition |

### Weight compliance (`weight_compliance_hand.py`)

Four fingers are commanded to hold a fixed pose (WEIGHT_POSE) under joint-space
springs. The operator adds weights incrementally; the script logs a 5 s convergence
window after each addition. The hand is fixed (UR5 stationary or absent). The thumb
is held at a small neutral pose (2° on all joints) and is not loaded.

Key technical elements:
- Torsional stiffness (STIFFNESS_CONDITIONS) is the swept variable.
- The hand ramps to the target pose once at the start (K_ROT + B_SETTLE for fast
  settling), then softens to the first condition K. Subsequent conditions change K only.
- The operator types ENTER for each weight added, "done" to close the condition.
  Step 0 (no weight) is logged automatically as the baseline.
- With UR5 absent, UR5Receiver falls back to identity rotation for gravity
  compensation; connect a stationary UR5 for accurate orientation.
- Set `COLLECTED_DATA = True` in the script to step through the sequence
  without overwriting any saved CSV files.

| File | Description |
|------|-------------|
| `weight_compliance_hand.py` | Weight-loading experiment — logs joint angles per (K, weight) pair |
| `plot_weight.ipynb` | Deviation-from-target vs weight, per stiffness condition |

### Configuration

| File | Description |
|------|-------------|
| `hand_config.py` | All constants for both Hand experiments (UR5 poses, stiffness values, timing) |
