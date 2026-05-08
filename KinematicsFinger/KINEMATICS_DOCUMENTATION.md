# Single Finger Kinematics - Quick Reference

## Overview

The single finger has **2 motors** controlling **3 joints**:
- 1 MCP motor (proximal phalanx flexion)
- 1 PIP motor (middle phalanx flexion, DIP mimics PIP)

**Units**: Lengths in meters [m], Angles in radians [rad]

---

## Quick Start

```python
import numpy as np
from FK_Finger import FK, motor_to_joint, joint_to_motor
from JacobiansFinger import q2joint, q2MCP, q2PIP, q2DIP
from HessiansFinger import qq2joint, qq2MCP, qq2PIP, qq2DIP

# Motor angles (2 motors: [q_MCP, q_PIP])
q_motor = np.array([0.5, 0.3])
r_local = np.array([0.0, 0.0175, 0.0])  # Fingertip in distal link frame

# Forward Kinematics
pos_tip = FK(q_motor, "DIP", r_local)

# Motor ↔ Joint conversion
theta = motor_to_joint(q_motor)    # → [theta_MCP, theta_PIP, theta_DIP]
q     = joint_to_motor(theta)      # → [q_MCP, q_PIP]

# Jacobians (build once, call many times)
J_DIP_func = q2DIP()
J = J_DIP_func(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])  # 3x2

# Hessians (build once, call many times)
H_DIP_func = qq2DIP()
H = H_DIP_func(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])  # 3x2x2
```

---

## Motor Order Convention

```python
q_motor[0] = q_MCP    # MCP motor angle
q_motor[1] = q_PIP    # PIP motor angle
```

---

## Kinematic Chain

```
Base → [MCP rotation] → Proximal phalanx (length a) →
       [PIP rotation] → Middle phalanx   (length b) →
       [DIP rotation] → Distal phalanx   (length c) → Fingertip
```

All rotations are about the **x-axis** (single-plane flexion):

```python
R(theta) = [[1,  0,           0         ],
            [0,  cos(theta), -sin(theta) ],
            [0,  sin(theta),  cos(theta) ]]
```

---

## API Reference

### FK_Finger.py - Forward Kinematics

**Motor ↔ Joint Conversions:**
```python
motor_to_joint(q_motor)        → [theta_MCP, theta_PIP, theta_DIP]
joint_to_motor(theta)          → [q_MCP, q_PIP]
```

**Forward Kinematics:**
```python
FK_MCP(q_motor, r_local)      → pos  (point on proximal phalanx)
FK_PIP(q_motor, r_local)      → pos  (point on middle phalanx)
FK_DIP(q_motor, r_local)      → pos  (point on distal phalanx)
FK(q_motor, link, r_local)    → pos  (general, link = 'MCP'|'PIP'|'DIP')
```

### JacobiansFinger.py - Jacobian Computation

Each function **returns a builder** (sympy → lambdify). Call the builder once, then evaluate many times:

```python
q2joint()   → func(q_MCP, q_PIP)               → 3x2  (joint angles w.r.t. motor angles)
q2MCP()     → func(q_MCP, q_PIP, x, y, z)      → 3x2  (MCP position w.r.t. motor angles)
q2PIP()     → func(q_MCP, q_PIP, x, y, z)      → 3x2  (PIP position w.r.t. motor angles)
q2DIP()     → func(q_MCP, q_PIP, x, y, z)      → 3x2  (DIP position w.r.t. motor angles)
```

### HessiansFinger.py - Hessian Computation

Same pattern — builder functions returning evaluable Hessians `H[i,j,k] = d²f_i / dq_j dq_k`:

```python
qq2joint()  → func(q_MCP, q_PIP)               → 3x2x2  (all zeros — linear mapping)
qq2MCP()    → func(q_MCP, q_PIP, x, y, z)      → 3x2x2
qq2PIP()    → func(q_MCP, q_PIP, x, y, z)      → 3x2x2
qq2DIP()    → func(q_MCP, q_PIP, x, y, z)      → 3x2x2
```

---

## Testing

Run built-in tests:
```bash
python3 FK_Finger.py
python3 JacobiansFinger.py
python3 HessiansFinger.py
```

---

## Important Notes

1. **All functions take `q_motor` as a 2-element array** `[q_MCP, q_PIP]`
2. **Jacobians return (3x2) matrices** — derivatives w.r.t. both motors
3. **Hessians return (3x2x2) tensors** — symmetric in last two indices
4. **Jacobian/Hessian functions are builders**: call once to compile, then evaluate many times
5. **DIP joint mimics PIP**: Mechanically coupled, same angle
6. **All rotations are planar** (x-axis): the finger moves in the y-z plane

---

## References

- **Parameters**: `ModelIDFinger/finger_params.py`
- **System ID**: `ModelIDFinger/finger_fitting.ipynb`
- **Hand kinematics**: See `KinematicsHand/KINEMATICS_DOCUMENTATION.md` for the full hand
