# ADAPT-StiffControl

End-to-end closed-loop force adaptation on the ADAPT Hand. The operator selects a target force level; the hand closes to a PC1 grasp pose and runs stiffness-descent gradient descent on all five fingertips until analytic tip force converges to the target, then the arm presses down to demonstrate the exerted force.

All experiments use the **ADAPT Hand**.

## How it works

- Force levels: `hard` (0.80 N), `medium` (0.20 N), `soft` (0.02 N), selected per finger at startup.
- A state machine sequences settle → ramp to PC1 (gentle task stiffness `K_TIP_GENTLE`) → gradient-descent adaptation → press → release → home.
- Each tick, stiffness-descent gradient descent updates per-finger `K_task` toward the target `f_des`, with step size scaled to the relative force error (aggressive far from target, gentle near it). Tip force comes from the VMC stiffness model (`tip_stiffness_MixedSpace`) evaluated at the current joint state.
- Convergence: the EMA of mean `|f_meas − f_des|` across all five fingers stays below `F_CONVERGE_THR` for `CONVERGE_HOLD` seconds. The arm then presses down `PRESS_HEIGHT`, holds `HOLD_TIME`, and releases.

**Logged:** joint state (13 motors), per-finger force/stiffness/position, phase label, convergence flag.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Select force level, GD force adaptation, press demonstration |
| `plot_grasp_adaptation.ipynb` | — | Plot GD convergence, stiffness evolution, and hand pose |
| `hand_config.py` | hand | UR5 grasp pose, PC1 joint targets, force levels, GD parameters, and timing |
