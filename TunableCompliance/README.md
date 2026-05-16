# TunableCompliance — Experimental Area 2

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid
spectrum, adapting to task phases (approach, hold, release), object properties,
and contact events — all without hardware changes and while preserving passivity.
Includes single-finger characterisation and full-hand in-hand manipulation and
dynamic-grasp studies.

## Structure

```
TunableCompliance/
├── Finger/     — single 2-DOF finger testbed experiments
└── Hand/       — ADAPT Hand experiments
```

---

## Finger/

| File | Description |
|------|-------------|
| `stiffening_contact.py` | K_d increased online upon contact detection |
| `repulsive_stiffness_shaping.py` | Effective stiffness shaped above the mechanical baseline |
| `plot_stiffening_contact.ipynb` | Plot stiffening contact experiment |
| `plot_repulsive_stiffness_shaping.ipynb` | Plot repulsive stiffness shaping experiment |

---

## Hand/

| File | Description |
|------|-------------|
| `inhand_manipulation.py` | In-hand object reorientation via asymmetric tip stiffness across hand sides (UNIFORM / ASYM_A / ASYM_B) |
| `dynamic_grasp.py` | Dynamic grasping of a bottle while UR5 transports it; three conditions: `soft`, `stiff`, `adaptive` (soft → stiff after a delay) |
| `plot_emergent_grasps.ipynb` | Plot in-hand manipulation and dynamic grasp experiments |
| `hand_config.py` | UR5 poses, PC1 / home joint targets, asymmetric-side groupings, condition list |
