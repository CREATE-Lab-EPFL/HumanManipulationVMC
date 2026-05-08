# TunableCompliance — Experimental Area 2

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid
spectrum, adapting to task phases (approach, hold, release), object properties,
and contact events — all without hardware changes and while preserving passivity.
Includes single-finger characterisation and a full-hand emergence-of-grasp study.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `stiffening_contact.py` | finger | K_d increased online upon contact detection |
| `repulsive_stiffness_shaping.py` | finger | Effective stiffness shaped above the mechanical baseline |
| `emergent_grasps.py` | hand | Finger postures emerging from controller configuration (virtual elements, stiffening on/off) across objects |
| `plot_stiffening_contact.ipynb` | finger | Plot stiffening contact experiment |
| `plot_repulsive_stiffness_shaping.ipynb` | finger | Plot repulsive stiffness shaping experiment |
| `plot_emergent_grasps.ipynb` | hand | Plot emergent grasps experiment |
