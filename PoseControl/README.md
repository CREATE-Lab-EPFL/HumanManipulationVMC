# PoseControl

Position tracking validation for the ADAPT Hand using joint-space VMC.
The hand is commanded through a small set of target poses inspired by hand
synergies (Santello et al.), expressed as joint-space references, and joint
convergence is recorded for each.

Controller details:
- Joint-space virtual springs and dampers drive tracking of the target pose.
- Tracking error, convergence behavior, and steady-state offsets are logged for
	analysis.

All experiments in this folder use the **ADAPT Hand**.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `position_tracker.py` | hand | Commands the pose targets in sequence and logs joint convergence per pose |
| `plot_position_tracker.ipynb` | hand | Plot position tracker experiment |
