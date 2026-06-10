# ADAPT-StiffControl

End-to-end closed-loop force adaptation on the ADAPT Hand. The operator selects
a target force level at startup; the hand closes to a PC1 grasp pose and runs
stiffness-descent gradient descent on all five fingertips until the analytic
tip force converges to the target. The arm then presses down to demonstrate the
exerted force.

All experiments in this folder use the **ADAPT Hand**.

## Adaptation rule

- The operator selects a force level at startup: `hard` (0.80 N), `medium` (0.20 N),
  or `soft` (0.02 N) per finger.
- The arm moves to the grasp pose; the hand ramps from home to the PC1 grasp target
  at a gentle task-space stiffness (`K_TIP_GENTLE`).
- Once the hand has converged, the force direction per finger is estimated from
  the initial contact force and the scalar target magnitude `f_des` is projected
  along that direction.
- Stiffness-descent gradient descent updates `K_task` per finger each tick.
  The step size scales with the relative force error so updates are aggressive far
  from the target and gentle near it.
- GD convergence is declared when the EMA of the mean `|f_meas − f_des|` across
  all five fingers stays below `F_CONVERGE_THR` for `CONVERGE_HOLD` seconds.
- The arm presses down by `PRESS_HEIGHT` to demonstrate the exerted force, holds
  for `HOLD_TIME`, then releases; the hand returns to home.

## Control flow

- A state machine sequences settle, ramp-to-PC1, convergence wait,
  gradient-descent adaptation, press, release, and return-to-home.
- Tip force comes from the VMC stiffness model (`tip_stiffness_MixedSpace`)
  evaluated at the current joint state.
- The GD runs at the full control rate; tip pose comes from forward kinematics.

## Logged signals

- Joint positions and velocities (all 13 motors).
- Per-finger measured force vector and magnitude, task stiffness diagonal.
- Per-finger tip position, reference position, displacement, 1st-order tip force
  and stiffness matrix with eigenvalues.
- Phase label (`adapt_gd`, `press`) and convergence flag.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Select force level, GD force adaptation, press demonstration |
| `plot_grasp_adaptation.ipynb` | — | Plot GD convergence, stiffness evolution, and hand pose |
| `hand_config.py` | hand | UR5 grasp pose, PC1 joint targets, force levels, GD parameters, and timing |
