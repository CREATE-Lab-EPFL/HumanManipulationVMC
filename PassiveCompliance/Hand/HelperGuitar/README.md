# HelperGuitar

Shared constants for the guitar strumming experiment.

## Files

| File | Description |
|------|-------------|
| `guitar_config.py` | Experiment constants: UR5 poses, torsional stiffness sweep values, hand pose, timing |

## Key parameters

| Constant | Value | Meaning |
|----------|-------|---------|
| `UR5_POSE_GUITAR` | `[x, y, z, rx, ry, rz]` | Arm reference pose at the guitar strings |
| `SWEEP_VECTOR` | `[0, −0.10, 0]` m | XY displacement per strum |
| `LIFT` | `0.05` m | Z clearance for the return trip |
| `TORSIONAL_SPRINGS` | `[0.1, 0.3]` N·m/rad | Compared joint stiffness conditions |
| `FINGER_CLOSED_POSE` | `[20, 20, 20]` deg | MCP / PIP / DIP target for closed fingers |
| `N_RUNS` | `3` | Strums per stiffness condition |

Audio is recorded externally via the camera microphone and analysed separately.
