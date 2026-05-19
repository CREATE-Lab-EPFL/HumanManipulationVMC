# TunableCompliance — Experimental Area

Shows that VMC enables real-time stiffness modulation across the full soft-to-rigid
spectrum, adapting to task phases (approach, hold, release), object properties,
and contact events — all without hardware changes and while preserving passivity.
Includes single-finger characterisation and full-hand in-hand manipulation and
dynamic-grasp studies.

## Structure

```
TunableCompliance/
├── Finger/     — single finger testbed experiments
└── Hand/       — ADAPT Hand experiments
```

---

## Finger/

Finger experiments run a contact detector based on deformation and force
estimates, then adjust virtual stiffness on contact. The repulsive shaping
variant adds a nonlinear repulsive element in task space to raise apparent
stiffness beyond the baseline while maintaining passivity. Scripts log
displacement, force estimates, and controller state for comparison.

Key technical elements:
- Contact detection uses changes in deformation and force estimate trends.
- Stiffness schedules switch between softer and stiffer phases based on state.
- Repulsive shaping adds a nonlinear virtual element that activates beyond a
	displacement threshold to increase apparent stiffness without violating
	passive behavior.

| File | Description |
|------|-------------|
| `stiffening_contact.py` | Increase virtual stiffness after contact is detected |
| `repulsive_stiffness_shaping.py` | Add a repulsive stiffness component above the baseline |
| `plot_stiffening_contact.ipynb` | Plot stiffening contact experiment |
| `plot_repulsive_stiffness_shaping.ipynb` | Plot repulsive stiffness shaping experiment |

---

## Hand/

Hand experiments use a simple state machine that assigns different fingertip
stiffness patterns across hand sides to bias object motion. Dynamic grasping
updates stiffness schedules during UR5 transport based on contact and task
phase. Scripts log joint state and task events for analysis.

Key technical elements:
- Asymmetric fingertip stiffness creates differential contact forces to drive
	in-hand reorientation.
- A schedule-based controller adjusts stiffness during approach, transport, and
	release phases.
- Analysis compares trajectory outcomes and grasp stability across conditions.

| File | Description |
|------|-------------|
| `inhand_manipulation.py` | In-hand object reorientation via asymmetric fingertip stiffness across hand sides |
| `dynamic_grasp.py` | Dynamic grasping during UR5 transport under soft, stiff, and adaptive stiffness schedules |
| `plot_emergent_grasps.ipynb` | Plot in-hand manipulation and dynamic grasp experiments |
| `hand_config.py` | UR5 poses, home and synergy pose targets, asymmetric-side groupings, condition list |
