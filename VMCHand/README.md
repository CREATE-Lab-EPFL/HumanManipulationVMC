# VMCHand — ADAPT Hand VMC Controllers

ROS2 control stack for the 15-DOF ADAPT hand. Handles motor ordering conversion,
gravity and joint limit compensation across all fingers and the wrist.

See [`HAND_VMC_DOCUMENTATION.md`](HAND_VMC_DOCUMENTATION.md) for the full API reference.

## Files

### `HandController.py`

Main ROS2 control node.
- Subscribes: `/joint_positions` (deg), `/joint_velocities` (deg/s)
- Publishes: `/goal_torque` (N·m)
- Converts between hardware motor ordering and software ordering (see `ModelIDHand/motor_config.py`).

### `HandGravFricLim.py`

Compensation layer for the full hand:
- Gravity compensation per link via Jacobian transpose
- Friction compensation (Stribeck model)
- Deadzone joint-limit springs

### `HandVMCJointSpace.py`

VMC controller in **joint space**: applies virtual springs and dampers independently on
all 22 hand DOFs (wrist pitch/yaw, thumb CMC1/CMC2/MCP/IP, 4 spread joints,
index/middle/ring/pinky MCP/PIP/DIP).

Stiffness and damping are stored as per-joint numpy arrays in `self.stiffness` and
`self.damping` (keyed by joint group). Use `set_stiffness(K)` / `set_damping(B)` to
assign the same value uniformly, or index individual joints for fine-grained control.

Finger targets: `index_target`, `middle_target`, `ring_pinky_target` (ring and pinky
share a reference), plus `wrist`, `thumb`, and `spread` per-finger targets.

### `HandVMCTaskSpace.py`

VMC controller in **task space**: applies virtual springs and dampers to 6 Cartesian
attachment points — the fingertip of each finger (thumb IP link, index/middle/ring/pinky
DIP link) and the palm frame origin.

Stiffness and damping accept scalars (isotropic), 3-vectors (per-axis), or 3×3 matrices.
Any spring/damper class from `VMC_utils.VirtualModels` can be substituted.
Attachment points (`self.attachment_points`) default to the link origin and can be
overridden to place the spring at the true fingertip.
