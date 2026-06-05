# ADAPT Hand Kinematics - Quick Reference

## Overview

The ADAPT Hand has **13 software motors** (the 2 wrist motors are held rigidly
by the hardware in position-control mode and are **not visible to kinematics**):

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

# Motor angles (13 motors: [thumb(4), spread, index(2), middle(2), ring(2), pinky(2)])
q_motor = np.zeros(13)
r_local = np.array([0.0, 0.0, 0.02])  # Local offset in link frame

# Forward Kinematics
pos_thumb_IP = FK_motor2thumbPos(q_motor, "IP", r_local)
pos_index_DIP = FK_motor2fingerPos(q_motor, "index", "DIP", r_local)
R_palm, pos_palm = FK_motor2palm(r_local)   # palm position is constant (no q_motor arg)

# Jacobians (initialize once, reuse many times)
jac = HandJacobians()  # Takes ~5-10 seconds to build all Jacobians
J_thumb = jac.get_thumb_jacobian("IP", q_motor, r_local)      # 3x13
J_index = jac.get_finger_jacobian("index", "DIP", q_motor, r_local)  # 3x13
J_palm = jac.get_wrist_palm_jacobian(q_motor, r_local)        # 3x13
```

---

## Motor Order Convention

```python
q_motor[0]  = thumb_CMC1
q_motor[1]  = thumb_CMC2
q_motor[2]  = thumb_MCP
q_motor[3]  = thumb_IP
q_motor[4]  = spread           # Shared across fingers
q_motor[5]  = index_MCP
q_motor[6]  = index_PIP
q_motor[7]  = middle_MCP
q_motor[8]  = middle_PIP
q_motor[9]  = ring_MCP
q_motor[10] = ring_PIP
q_motor[11] = pinky_MCP
q_motor[12] = pinky_PIP
```

---

## API Reference

### FK_Hand.py - Forward Kinematics

**Motor to Joint Conversions:**
```python
FK_motor2thumb(q_motor)                    → [CMC1, CMC2, MCP, IP]
FK_motor2finger(q_motor, finger_name)      → [MCP, PIP, DIP]
FK_motor2spread(q_motor, finger_name)      → spread_angle
```

**Forward Kinematics:**
```python
FK_motor2palm(r_local)                     → (R_palm, pos)   # palm is constant, no q_motor
FK_motor2thumbPos(q_motor, link, r_local)  → pos
    # link: 'CMC1', 'CMC2', 'MCP', 'IP'
FK_motor2fingerPos(q_motor, finger, link, r_local) → pos
    # finger: 'index', 'middle', 'ring', 'pinky'
    # link: 'MCP', 'PIP', 'DIP'
```

### JacobiansHand.py - Jacobian Computation

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

1. **All kinematics functions take `q_motor` as a single 13-element array**
2. **Jacobians return full (Nx13) matrices** — derivatives w.r.t. all 13 software motors
3. **Use `HandJacobians` class** to avoid re-computing symbolic expressions
4. **Thread-safe**: No global state modification (bug fixed with `.copy()`)
5. **DIP joint mimics PIP**: Mechanically coupled, same angle
6. **Wrist is rigid**: Hardware motors 13 & 14 are position-controlled; `FK_motor2palm` takes no `q_motor` arg

---

## References

- **URDF**: `ADAPT_Hand.urdf`
- **Config**: `hand.config.json`
- **Full documentation**: See original version for detailed derivations
