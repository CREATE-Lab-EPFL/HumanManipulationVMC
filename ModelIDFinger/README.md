# ModelIDFinger — Single Finger System Identification

Contains all parameters and identification routines for the 2-DOF finger (MCP + PIP,
with mimic DIP). `finger_params.py` is the **single source of truth** — all other
finger modules import from here.

## Files

| File | Description |
|------|-------------|
| `finger_params.py` | Transmissions, link geometry, masses, COGs, joint limits, friction model — single source of truth |
| `finger_node.py` | ROS2 node for running system identification experiments on the finger |
| `friction_ablation.py` | Step and ramp response with/without friction compensation |
| `finger_fitting.ipynb` | Fits linear transmission model q = k·θ from joint data; saves to finger_params.py |
| `friction_study.ipynb` | Identifies Stribeck friction parameters from ablation data |
