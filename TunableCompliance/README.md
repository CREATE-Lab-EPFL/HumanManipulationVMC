# TunableCompliance — Experimental Area 2

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid
spectrum, adapting to task phases (approach, hold, release), object properties,
and contact events — all without hardware changes and while preserving passivity.
Includes single-finger characterisation and a full-hand emergence-of-grasp study.

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
| `emergent_grasps.py` | Finger postures emerging from controller configuration (virtual elements, stiffening on/off) across objects |
| `plot_emergent_grasps.ipynb` | Plot emergent grasps experiment |
