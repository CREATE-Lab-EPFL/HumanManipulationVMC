# PoseControl

Position tracking validation for the 15-DOF ADAPT Hand using joint-space VMC.
The hand is commanded through two target poses inspired by hand synergies
(Santello et al. 1998) — PC1 (power grasp, global flexion) and PC2 (precision pinch,
differential) — and joint convergence is recorded for each.

All experiments in this folder use the **ADAPT Hand (15 DOF)**.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `position_tracker.py` | hand | Commands PC1 and PC2 poses in sequence, logs joint convergence per pose |
| `plot_position_tracker.ipynb` | hand | Plot position tracker experiment |
