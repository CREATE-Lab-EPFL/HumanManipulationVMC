# ProprioceptiveSensing — Experimental Area

Contact force and object stiffness estimated from kinematics and virtual stiffness
alone — no external force sensors required. The deformation that absorbs impacts
is the same that encodes contact force. Sensing sensitivity is maximised when the
virtual compliance C_A matches the object compliance C_O.

## Structure

```
ProprioceptiveSensing/
├── Finger/     — single finger testbed experiments
└── Hand/       — ADAPT Hand experiments
```

---

## Finger/

The finger experiment calibrates motor efficiency by relating commanded torque
to observed motion and force. That calibration is then used with virtual
stiffness and kinematics to infer contact force from deformation.

| File | Description |
|------|-------------|
| `finger_eta.py` | Motor efficiency identification and force estimation validation |
| `plot_finger_eta.ipynb` | Plot finger eta experiment |

---

## Hand/

The hand experiment squeezes an object using a low and a high virtual
compliance setting. An additive compliance model uses the paired measurements
to solve for object compliance without external sensing.

| File | Description |
|------|-------------|
| `object_stiffness_hand.py` | Object stiffness estimation by squeezing using an additive compliance model |
| `plot_object_stiffness_hand.ipynb` | Plot object stiffness hand experiment |
