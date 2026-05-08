# KinematicsHand — ADAPT Hand Kinematics

Forward kinematics, Jacobians, and Hessians for the 15-DOF ADAPT hand. Required by
all VMC hand controllers. Jacobians are pre-compiled symbolically and cached to
avoid recomputation on repeated calls.

See [`KINEMATICS_DOCUMENTATION.md`](KINEMATICS_DOCUMENTATION.md) for the full API reference.

## Files

### `FK_Hand.py`

Forward kinematics for all fingers, thumb, and wrist. Motor-to-joint conversion
handles the differential wrist, thumb per-joint constants, mimic DIP constraint,
and spread motion ratios.

### `JacobiansHand.py`

`HandJacobians` class: pre-compiled symbolic Jacobians for all 15 motors.

- `get_finger_jacobian(finger, q)` — Jacobian for index/middle/ring/pinky fingertip
- `get_thumb_jacobian(q)` — Jacobian for thumb tip
- `get_wrist_palm_jacobian(q)` — Jacobian for palm/wrist point

### `HessiansHand.py`

Second-order derivatives for the geometric stiffness correction (CCT) applied
to the full hand.
