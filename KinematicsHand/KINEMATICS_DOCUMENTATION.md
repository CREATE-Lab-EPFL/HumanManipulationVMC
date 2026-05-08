# ADAPT Hand Kinematics - Quick Reference

## Overview

The ADAPT Hand has **15 motors**:
- 2 wrist motors (differential: pitch/yaw)
- 4 thumb motors (CMC1, CMC2, MCP, IP)
- 8 finger motors (2 per finger: MCP, PIP/DIP for Index, Middle, Ring, Pinky)
- 1 spread motor (controls Index, Ring, Pinky abduction)

**Units**: Lengths in meters [m], Angles in radians [rad]

---

## Quick Start

```python
import numpy as np
from FK_Hand import FK_motor2thumbPos, FK_motor2fingerPos, FK_motor2palm
from JacobiansHand import HandJacobians

# Motor angles (15 motors: [wrist1, wrist2, thumb(4), spread, index(2), middle(2), ring(2), pinky(2)])
q_motor = np.zeros(15)
r_local = np.array([0.0, 0.0, 0.02])  # Local offset in link frame

# Forward Kinematics
pos_thumb_IP = FK_motor2thumbPos(q_motor, "IP", r_local)
pos_index_DIP = FK_motor2fingerPos(q_motor, "index", "DIP", r_local)
R_palm, pos_palm = FK_motor2palm(q_motor, r_local)

# Jacobians (initialize once, reuse many times)
jac = HandJacobians()  # Takes ~5-10 seconds to build all Jacobians
J_thumb = jac.get_thumb_jacobian("IP", q_motor, r_local)      # 3x15
J_index = jac.get_finger_jacobian("index", "DIP", q_motor, r_local)  # 3x15
J_palm = jac.get_wrist_palm_jacobian(q_motor, r_local)        # 3x15
```

---

## Motor Order Convention

```python
q_motor[0]  = wrist_motor1     # Pinky side
q_motor[1]  = wrist_motor2     # Thumb side
q_motor[2]  = thumb_CMC1
q_motor[3]  = thumb_CMC2
q_motor[4]  = thumb_MCP
q_motor[5]  = thumb_IP
q_motor[6]  = spread           # Shared across fingers
q_motor[7]  = index_MCP
q_motor[8]  = index_PIP
q_motor[9]  = middle_MCP
q_motor[10] = middle_PIP
q_motor[11] = ring_MCP
q_motor[12] = ring_PIP
q_motor[13] = pinky_MCP
q_motor[14] = pinky_PIP
```

---

## API Reference

### FK_Hand.py - Forward Kinematics

**Motor to Joint Conversions:**
```python
FK_motor2wrist(q_motor)                    → [pitch, yaw]
FK_motor2thumb(q_motor)                    → [CMC1, CMC2, MCP, IP]
FK_motor2finger(q_motor, finger_name)      → [MCP, PIP, DIP]
FK_motor2spread(q_motor, finger_name)      → spread_angle
```

**Forward Kinematics:**
```python
FK_motor2palm(q_motor, r_local)            → (R_palm, pos)
FK_motor2thumbPos(q_motor, link, r_local)  → pos
    # link: 'CMC1', 'CMC2', 'MCP', 'IP'
FK_motor2fingerPos(q_motor, finger, link, r_local) → pos
    # finger: 'index', 'middle', 'ring', 'pinky'
    # link: 'MCP', 'PIP', 'DIP'
```

### JacobiansHand.py - Jacobian Computation

**Individual Jacobians (build once, call many times):**
```python
Jacobian_motor2wrist()                    → func(q_motor) → 2x15
Jacobian_motor2palm()                     → func(q_motor, r_local) → 3x15
Jacobian_motor2thumbPos(link)             → func(q_motor, r_local) → 3x15
FK_motor2fingerPos(finger, link)          → func(q_motor, r_local) → 3x15
Jacobian_motor2spread(finger)             → func(q_motor) → 1x15
```

**HandJacobians Class (recommended):**
```python
jac = HandJacobians()  # Pre-builds all Jacobians
J = jac.get_thumb_jacobian(link, q_motor, r_local)
J = jac.get_finger_jacobian(finger, link, q_motor, r_local)
J = jac.get_wrist_palm_jacobian(q_motor, r_local)
```

---

## Testing

Run built-in tests:
```bash
python3 FK_Hand.py
python3 JacobiansHand.py
```

---

## Important Notes

1. **All functions take `q_motor` as a single 15-element array**
2. **Jacobians return full (Nx15) matrices** - derivatives w.r.t. all 15 motors
3. **Use `HandJacobians` class** to avoid re-computing symbolic expressions
4. **Thread-safe**: No global state modification (bug fixed with `.copy()`)
5. **DIP joint mimics PIP**: Mechanically coupled, same angle

---

## References

- **URDF**: `ADAPT_Hand.urdf`
- **Config**: `hand.config.json`
- **Full documentation**: See original version for detailed derivations
