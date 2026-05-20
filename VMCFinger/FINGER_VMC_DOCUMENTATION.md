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
controller.get_joint_positions()      # motor angles, degrees
controller.get_joint_velocities()     # motor velocities, deg/s
controller.publish_torques(tau)       # motor torques, N·m
```

Consumer scripts convert positions / velocities to radians before use.

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

| Variable | Dim | Unit |
|----------|-----|------|
| `q_motor` | 2 | deg (from ROS) |
| `q_dot_motor` | 2 | deg/s (from ROS) |
| `tau_motor` | 2 | N·m |

Consumer scripts convert positions / velocities to radians before passing them
to the VMC, gravity, and Jacobian functions, which expect SI units.

See [KINEMATICS_DOCUMENTATION.md](../KinematicsFinger/KINEMATICS_DOCUMENTATION.md) for FK/Jacobians.
