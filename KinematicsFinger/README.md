# KinematicsFinger — Single Finger Kinematics

Forward kinematics, Jacobians, and Hessians for the 2-DOF finger (MCP + PIP, with
mimic DIP = PIP). Required by all VMC finger controllers and the stiffness model.

See [`KINEMATICS_DOCUMENTATION.md`](KINEMATICS_DOCUMENTATION.md) for the full API reference.

## Files

### `FK_Finger.py`

- `motor_to_joint(q_motor)` — converts motor positions to joint angles
- `joint_to_motor(theta)` — inverse conversion
- `FK_MCP(q_motor, r_local)`, `FK_PIP(q_motor, r_local)`, `FK_DIP(q_motor, r_local)` — world-frame position of a local point on the proximal/intermediate/distal phalanx
- `FK(q_motor, link, r_local)` — general FK for an arbitrary link and local point

Transmission: θ_MCP = (r_pulley/r_motor)·q_MCP, θ_PIP = (r_motor/c_param)·q_PIP, θ_DIP = θ_PIP.

### `JacobiansFinger.py`

Jacobian matrices ∂x/∂q and ∂θ/∂q evaluated at the current configuration.
Used in the VMC force and stiffness mapping.

### `HessiansFinger.py`

Second-order derivatives ∂²x/∂q² for the geometric stiffness correction in the
exact (CCT) stiffness formula.
