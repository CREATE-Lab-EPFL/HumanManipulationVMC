# PoseControl

Position tracking validation for the ADAPT Hand using joint-space VMC.
The hand is commanded through a pair of target poses inspired by hand synergies
(Santello et al.), expressed as joint-space references, and joint convergence is
recorded for each.

All experiments in this folder use the **ADAPT Hand**.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `position_tracker.py` | hand | Commands the pose targets in sequence and logs joint convergence per pose |
| `plot_position_tracker.ipynb` | hand | Plot position tracker experiment |
