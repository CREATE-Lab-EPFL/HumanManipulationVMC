# PassiveCompliance — Experimental Area

Validates that VMC generates diverse, predictable stiffness profiles at the fingertip
and palm through virtual springs alone, without hardware reconfiguration.
Compliance acts as a bidirectional filter: outward (absorbs noise and impacts
before contact), inward (smoothens deformations and provides adaptability).

## Structure

```
PassiveCompliance/
├── Finger/     — single finger testbed experiments
└── Hand/       — ADAPT Hand guitar strumming experiment
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

All hand experiments require the ADAPT Hand on ROS2 (`dynamixel_node`) and a UR5
reachable at `UR5_IP`.

### Guitar (`guitar_playing_hand.py`)

All fingers except the thumb are held closed at a fixed flexion pose under a
torsional (joint-space) spring. The UR5 drags the closed hand linearly across
the strings; audio is captured externally via the camera microphone. Two torsional
stiffness values are compared; the camera audio is the intensity proxy.

Key technical elements:
- Torsional (joint-space) stiffness is the swept variable; task-space springs are
  not used — the experiment isolates the effect of joint stiffness on sound.
- The wrist is held (near-)rigid to isolate finger compliance from wrist mobility.
- Fingers stay closed throughout all runs; the arm lifts over the top and returns
  between strums without reopening the hand.

| File | Description |
|------|-------------|
| `guitar_playing_hand.py` | Guitar experiment — linear arm sweep across strings at two torsional stiffnesses, N_RUNS per condition |
| `plot_guitar.ipynb` | Joint angle and torque figures per condition |
| `HelperGuitar/` | Config constants |
