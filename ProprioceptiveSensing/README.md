# ProprioceptiveSensing — Experimental Area 3

Contact force and object stiffness estimated from kinematics and virtual stiffness
alone — no external force sensors required. The deformation that absorbs impacts
is the same that encodes contact force. Sensing sensitivity is maximised when the
virtual compliance C_A matches the object compliance C_O.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `finger_eta.py` | finger | Motor efficiency identification and force estimation validation |
| `object_stiffness_hand.py` | hand | Object stiffness estimation by squeezing — C_O from (C_A + C_O) |
| `plot_finger_eta.ipynb` | finger | Plot finger eta experiment |
| `plot_object_stiffness_hand.ipynb` | hand | Plot object stiffness hand experiment |
