# Hand VMC Documentation

Virtual Model Control for the 15-DOF ADAPT Hand.

## Motor Ordering

**Software order** (all numpy arrays):
```
[0-1]   Wrist: motor1, motor2
[2-5]   Thumb: CMC1, CMC2, MCP, IP
[6]     Spread motor
[7-8]   Index: MCP, PIP
[9-10]  Middle: MCP, PIP
[11-12] Ring: MCP, PIP
[13-14] Pinky: MCP, PIP
```

**Conversion** handled automatically by `HandController`:
- `get_joint_positions()` returns software order (radians)
- `get_joint_velocities()` returns software order (rad/s)
- `publish_torques(tau)` accepts software order, converts to hardware

Manual conversion (if needed):
```python
from VMCHand.motor_config import hardware_to_software, software_to_hardware
```

## HandController Class

ROS2 node that handles hardware communication and motor ordering.

```python
from VMCHand.HandController import HandController, CONTROL_FREQUENCY

controller = HandController()
q     = controller.get_joint_positions()      # software order, radians
q_dot = controller.get_joint_velocities()     # software order, rad/s
controller.publish_torques(tau)               # software order, N·m
```

## GravFricLim Class

Gravity compensation, friction compensation (Stribeck model), and joint-limit
deadzone springs for the full hand.

```python
from VMCHand.HandGravFricLim import GravFricLim

comp = GravFricLim()

# All compensation at once (gravity + friction + limits)
tau_comp = comp.compute_compensation_torques(
    q_motor, q_dot_motor, tau_vmc, R_tcp)   # R_tcp: 3×3 wrist rotation from UR5

# Individual components (if needed)
tau_g = comp.gravity_compensation(q_motor)
tau_l = comp.joint_limit_torques(q_motor)
```

**Joint limits:** Deadzone springs — zero torque within limits, linear spring outside.

## HandVMCJointSpace Class

Joint-space spring-damper controller across all 22 hand DOFs.

```python
from VMCHand.HandVMCJointSpace import VMC

vmc = VMC()
```

### Stiffness and Damping

Per-joint gains are stored as numpy arrays inside `self.stiffness` and `self.damping`,
keyed by joint group:

| Key | Shape | DOFs |
|-----|-------|------|
| `'wrist'` | (2,) | pitch, yaw |
| `'thumb'` | (4,) | CMC1, CMC2, MCP, IP |
| `'spread_index'` | (1,) | index spread |
| `'spread_middle'` | (1,) | middle spread |
| `'spread_ring'` | (1,) | ring spread |
| `'spread_pinky'` | (1,) | pinky spread |
| `'index'` | (3,) | MCP, PIP, DIP |
| `'middle'` | (3,) | MCP, PIP, DIP |
| `'ring'` | (3,) | MCP, PIP, DIP |
| `'pinky'` | (3,) | MCP, PIP, DIP |

**Uniform assignment** (all joints at once):
```python
vmc.set_stiffness(0.4)    # [N·m/rad]
vmc.set_damping(0.05)     # [N·m·s/rad]
```

**Per-joint fine control** — index directly into the arrays:
```python
vmc.stiffness['wrist'][1]   = 0.6   # wrist yaw only
vmc.stiffness['thumb'][2]   = 1.0   # thumb MCP only
vmc.stiffness['index'][0]   = 0.8   # index MCP only  (0=MCP, 1=PIP, 2=DIP)
vmc.damping['spread_ring']  = np.array([0.02])
```

### Reference Positions

```python
vmc.wrist             # [pitch, yaw]          (rad)
vmc.thumb             # [CMC1, CMC2, MCP, IP] (rad)
vmc.spread            # dict: 'index','middle','ring','pinky' → (1,) array (rad)
vmc.index_target      # [MCP, PIP, DIP]       (rad)
vmc.middle_target     # [MCP, PIP, DIP]       (rad)
vmc.ring_pinky_target # [MCP, PIP, DIP]       (rad) — ring and pinky share one target
```

### Torque Computation

```python
tau = vmc.hand_torques(q_motor, q_dot_motor)   # → [15] N·m
```

Implements per-group Jacobian-transpose spring-dampers:
```
tau = J^T @ ( K*(theta_target - theta) - B*(J @ q_dot) )
```

### Minimal Example

```python
vmc = VMC()
vmc.set_stiffness(0.4)
vmc.set_damping(0.05)
vmc.index_target = np.deg2rad([50.0, 60.0, 60.0])
tau = vmc.hand_torques(q_motor, q_dot_motor)
```

## HandVMCTaskSpace Class

Task-space spring-damper controller for 6 Cartesian attachment points:
thumb fingertip (IP link), index/middle/ring/pinky fingertips (DIP link), and palm origin.

```python
from VMCHand.HandVMCTaskSpace import VMC

vmc = VMC()
```

### Attachment Points

| Key | Link | Default |
|-----|------|---------|
| `'thumb'` | thumb IP | [0,0,0] in link frame |
| `'index'` | index DIP | [0,0,0] in link frame |
| `'middle'` | middle DIP | [0,0,0] in link frame |
| `'ring'` | ring DIP | [0,0,0] in link frame |
| `'pinky'` | pinky DIP | [0,0,0] in link frame |
| `'palm'` | wrist/palm origin | [0,0,0] in palm frame |

Override `self.attachment_points` to shift the spring to the true fingertip:
```python
vmc.attachment_points['index'] = np.array([0.0, 0.0, 0.025])  # 25 mm along finger axis
```

### Stiffness and Damping

Springs and dampers are `LinearSpring` / `LinearDamper` objects stored in `self.springs`
and `self.dampers`. They accept scalar (isotropic), 3-vector (per-axis), or 3×3 matrix gains,
and can be replaced with any class from `VMC_utils.VirtualModels`.

**Uniform assignment** (all points, or a subset):
```python
vmc.set_stiffness(200.0)                        # isotropic, all points
vmc.set_damping(5.0, points=['thumb', 'index']) # subset only
```

**Per-point fine control** — set the `.stiffness` / `.damping` attribute directly:
```python
vmc.springs['index'].stiffness = np.array([200.0, 100.0, 200.0])  # per-axis [x,y,z]
vmc.dampers['thumb'].damping   = 8.0                                # scalar
vmc.springs['palm'].stiffness  = np.diag([100.0, 50.0, 100.0])    # full 3×3
```

**Swap the model class** (e.g. tanh spring):
```python
from VMC_utils.VirtualModels import TanhSpring
vmc.springs['index'] = TanhSpring(K_max=50.0, alpha=10.0)
```

### Target Positions

```python
vmc.targets['index'] = np.array([x, y, z])   # [m] in hand base frame
vmc.targets['palm']  = np.array([x, y, z])
```

### Torque Computation

```python
tau = vmc.hand_torques(q_motor, q_dot_motor)  # → [15] N·m
```

Implements Jacobian-transpose VMC summed over all points:
```
tau = Σ_p  J_p^T @ ( F_spring_p + F_damper_p )
```

### Minimal Example

```python
vmc = VMC()
vmc.set_stiffness(200.0)
vmc.set_damping(5.0)
vmc.targets['index'] = np.array([0.15, 0.05, 0.10])  # index fingertip target [m]
tau = vmc.hand_torques(q_motor, q_dot_motor)
```

## Coordinate Frame

- **Origin**: Wrist base (hand base_link)
- **x**: Toward pinky side
- **y**: Out from palm (forward)
- **z**: Upward (gravity = −z)

## Quick Reference

| Variable | Dim | Unit |
|----------|-----|------|
| `q_motor` | 15 | rad |
| `q_dot_motor` | 15 | rad/s |
| `tau_motor` | 15 | N·m |
| joint-space stiffness | per-joint | N·m/rad |
| joint-space damping | per-joint | N·m·s/rad |
| task-space stiffness | per-point | N/m |
| task-space damping | per-point | N·s/m |

See [KINEMATICS_DOCUMENTATION.md](../KinematicsHand/KINEMATICS_DOCUMENTATION.md) for FK/Jacobians.
