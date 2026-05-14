# ProprioceptiveSensing — Experimental Area 3

Contact force and object stiffness estimated from kinematics and virtual stiffness
alone — no external force sensors required. The deformation that absorbs impacts
is the same that encodes contact force. Sensing sensitivity is maximised when the
virtual compliance C_A matches the object compliance C_O.

## Structure

```
ProprioceptiveSensing/
├── Finger/     — single 2-DOF finger testbed experiments
└── Hand/       — ADAPT Hand experiments
```

---

## Finger/

| File | Description |
|------|-------------|
| `finger_eta.py` | Motor efficiency identification and force estimation validation |
| `plot_finger_eta.ipynb` | Plot finger eta experiment |

---

## Hand/

| File | Description |
|------|-------------|
| `object_stiffness_hand.py` | Object stiffness estimation by squeezing — C_O from (C_A + C_O) |
| `plot_object_stiffness_hand.ipynb` | Plot object stiffness hand experiment |
