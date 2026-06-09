# ADAPT-StiffControl

End-to-end closed-loop stiffness adaptation on the ADAPT Hand. The hand
grasps a known object, estimates its compliance from paired contact samples
using FK tip position and analytic VMC tip force, then matches fingertip
stiffness through gradient descent before pressing down to demonstrate the
exerted force.

All experiments in this folder use the **ADAPT Hand**.

## Adaptation rule

- Close with a gentle stiffness and let the hand settle. Freeze the thumb at a
  fixed hold stiffness and record FK tip position and model-based tip force for
  the four probe fingers (index, middle, ring, pinky).
- Ramp probe fingers to a higher probe stiffness, wait for re-convergence, and
  record a follow-up position and force sample.
- Estimate object compliance per finger from the ratio of position change to
  force change; average across the four fingers to obtain $C_O$.
- Map compliance to a target force: $f_{\rm des} = F_{\rm GAIN} / C_O$.
  Harder objects (lower $C_O$) give a higher target force.
- Run stiffness-descent gradient descent independently on each probe finger,
  updating $K_{\rm task}$ per finger each tick until the mean tip-force error
  falls below the convergence threshold.
- Press the arm down by `PRESS_HEIGHT` to demonstrate the exerted force, hold
  for `HOLD_TIME`, then release.

## Control flow

- A state machine sequences approach, contact, compliance sensing, gradient
  descent, force demonstration, and release.
- Tip pose comes from forward kinematics; tip force comes from the VMC
  stiffness model evaluated at the current state.
- The GD runs at the full control rate; convergence is declared when the mean
  $|f_{\rm meas} - f_{\rm des}|$ across probe fingers stays below
  `F_CONVERGE_THR` for `CONVERGE_HOLD` seconds.

## Logged signals

- Joint positions and velocities (all 13 motors).
- Per-finger tip position, reference position, tip force, and tip stiffness.
- Mean GD task stiffness and mean force error across probe fingers.
- Phase labels: `sense`, `probe`, `adapt_gd`, `press`.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `grasp_adaptation.py` | hand | Compliance sensing, GD force adaptation, press demonstration |
| `plot_grasp_adaptation.ipynb` | — | Plot compliance, GD convergence, stiffness, and hand pose |
| `hand_config.py` | hand | UR5 grasp poses, joint targets, object list, stiffness schedule, and timing |
