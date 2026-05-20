# Finger VMC Documentation

Virtual Model Control for a 2-motor, 3-joint underactuated finger (MCP, PIP, DIP).

## Motor Ordering

**Motor order** (all numpy arrays):
```
[0] MCP motor
[1] PIP motor
```

Joint space has 3 DOF (MCP, PIP, DIP) coupled to 2 motors. Mapping handled by the `q2joint()` Jacobian from `KinematicsFinger`.

## Controller

ROS2 node that handles hardware communication.

```python
controller.get_joint_positions()      # motor angles    — see FingerController.py for units
controller.get_joint_velocities()     # motor velocities — see FingerController.py for units
controller.publish_torques(tau)       # motor torques   — see FingerController.py for units
```

Consuming scripts convert to SI (rad / rad·s⁻¹ / N·m) where needed.

**Prerequisite:** `ros2 run dynamixel_interface dynamixel_node`

## GravLim

Gravity and joint limit compensation.

```python
# All compensation (gravity + limits)
tau_comp = comp.compute_compensation_torques(q_motor, q_dot_motor, tau_vmc)

# Individual components
tau_g = comp.gravity_compensation(q_motor)
tau_l = comp.deadzone_spring_torques(q_motor)
```

**Joint limits:** Deadzone springs per joint, zero torque within limits, linear spring outside.

## Coordinate Frame

Finger orientation set via `R_world_to_finger` in the model identification module.

## Quick Reference

| Variable | Dim | Notes |
|----------|-----|-------|
| `q_motor` | 2 | see `FingerController.py` for unit |
| `q_dot_motor` | 2 | see `FingerController.py` for unit |
| `tau_motor` | 2 | published to `/goal_torque` |

Internal VMC computations and the SI convention used elsewhere in the repo
are: motor angles in rad, velocities in rad·s⁻¹, torques in N·m.

See [KINEMATICS_DOCUMENTATION.md](../KinematicsFinger/KINEMATICS_DOCUMENTATION.md) for FK/Jacobians.
