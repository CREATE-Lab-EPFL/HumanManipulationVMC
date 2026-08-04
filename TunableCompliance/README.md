# TunableCompliance — Experimental Area

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid spectrum, adapting to task phase (approach, hold, release), object properties, and contact events — without hardware changes and while preserving passivity. Includes single-finger characterisation and full-hand in-hand manipulation and dynamic-grasp studies.

```
TunableCompliance/
├── Finger/     — single finger testbed experiments
└── Hand/       — ADAPT Hand experiments
```

## Finger/

A contact detector based on deformation and force-estimate trends triggers virtual-stiffness switches between softer and stiffer schedules. The repulsive-shaping variant adds a nonlinear Cartesian element that activates only above a displacement threshold, raising apparent stiffness beyond baseline while staying passive. Scripts log displacement, force estimates, and controller state.

| File | Description |
|------|-------------|
| `stiffening_contact.py` | Increase virtual stiffness after contact is detected |
| `repulsive_stiffness_shaping.py` | Add a repulsive stiffness component above the baseline |
| `plot_stiffening_contact.ipynb` | Plot stiffening contact experiment |
| `plot_repulsive_stiffness_shaping.ipynb` | Plot repulsive stiffness shaping experiment |

## Hand/

A state machine assigns different fingertip-stiffness patterns across sides of the hand to bias object motion (in-hand reorientation), and adjusts stiffness schedules during UR5 transport based on task phase (dynamic grasping). Scripts log joint state and task events; analysis compares trajectory outcomes and grasp stability across conditions.

| File | Description |
|------|-------------|
| `inhand_manipulation.py` | In-hand object reorientation via asymmetric fingertip stiffness across hand sides |
| `dynamic_grasp.py` | Dynamic grasping during UR5 transport under soft, stiff, and adaptive stiffness schedules |
| `plot_emergent_grasps.ipynb` | Plot in-hand manipulation and dynamic grasp experiments |
| `hand_config.py` | UR5 poses, home and synergy pose targets, asymmetric-side groupings, condition list |
