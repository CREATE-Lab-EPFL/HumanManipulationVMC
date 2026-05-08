# VMCFinger — Single Finger VMC Controllers

ROS2 control stack for the 2-DOF finger. Handles gravity and joint limit
compensation, and provides four VMC controller variants (finger space, task space,
directional Cartesian, and repulsive spring).

See [`FINGER_VMC_DOCUMENTATION.md`](FINGER_VMC_DOCUMENTATION.md) for the full API reference.

## Files

### `FingerController.py`

Main ROS2 control node.
- Subscribes: `/joint_positions` (deg), `/joint_velocities` (deg/s)
- Publishes: `/goal_torque` (N·m)
- Internally stores state in radians (software order).

### `FingerGravFricLim.py`

Compensation layer added on top of any VMC controller:
- Gravity torques via Jacobian transpose
- Friction compensation (Stribeck model)
- Joint limit springs (deadzone)

### `FingerVMCFingerSpace.py`

VMC with virtual springs at MCP, PIP, DIP joints (finger-space coordinates).
Torques computed via τ = K_θ·(θ_ref − θ), then mapped to task-space force via PVW.

### `FingerVMCTaskSpace.py`

VMC with a virtual spring directly at the fingertip in Cartesian coordinates.
Virtual force computed in task space, mapped to motor torques via J^T.

### `FingerVMCDirCart.py`

VMC combining a finger-space spring (baseline stiffness, k_d [N·m/rad]) with a
Cartesian spring constrained to the pressing direction (Z axis, k_cart [N/m]).
Used for directional stiffness shaping experiments.

### `FingerVMCRepulsiveSpring.py`

Same dual-element structure as `FingerVMCDirCart.py` but the Cartesian element is
a Gaussian repulsive spring — adds stiffness only above a displacement threshold,
used for repulsive stiffness shaping.
