# ProprioceptiveSensing — Experimental Area

Contact force and object stiffness estimated from kinematics and virtual stiffness alone — no external force sensors. The deformation that absorbs impacts is the same signal that encodes contact force: tip force comes from virtual-spring deflection and the stiffness model, tip pose from forward kinematics.

```
ProprioceptiveSensing/
├── Finger/     — single finger testbed experiments
└── Hand/       — ADAPT Hand experiments
```

## Finger/

Calibrates motor efficiency by relating commanded torque to observed motion and force, then uses that calibration with virtual stiffness and kinematics to infer contact force from deformation. Logs motor commands, joint state, estimated tip force, and reference trajectories used for validation.

| File | Description |
|------|-------------|
| `finger_eta.py` | Motor efficiency identification and force estimation validation |
| `plot_finger_eta.ipynb` | Plot finger eta experiment |

## Hand/

Squeezes an object at a low and a high virtual compliance setting; an additive compliance model uses the paired measurements to solve for object compliance without external sensing. Logs joint state, tip pose, inferred force, and the applied compliance schedule.

| File | Description |
|------|-------------|
| `object_stiffness_hand.py` | Object stiffness estimation by squeezing using an additive compliance model |
| `plot_object_stiffness_hand.ipynb` | Plot object stiffness hand experiment |
