"""
UR5 poses and hand configuration for ADAPT-StiffControl experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── UR5 poses ─────────────────────────────────────────────────────────────────
# grasp_adaptation.py — grasp contact pose for each known object.
UR5_POSE_GRASP_OBJ = {
    'object_1': np.array([-0.1405, 0.6833, 0.2594, -0.1861, -0.027, -2.7656]),
    'object_2': np.array([-0.1405, 0.5500, 0.2594, -0.1861, -0.027, -2.7656]),
}

# ── PC1 grasp pose (Santello et al. 1998 — first principal component) ─────────
PC1_WRIST  = np.deg2rad([0.0,  0.0])
PC1_THUMB  = np.deg2rad([70.0, 0.0, 80.0, 80.0])
PC1_SPREAD = {
    'index':  np.deg2rad(-2.0),
    'middle': 0.0,
    'ring':   np.deg2rad(2.0),
    'pinky':  np.deg2rad(2.0),
}
PC1_INDEX  = np.deg2rad([80.0, 85.0, 85.0])
PC1_MIDDLE = np.deg2rad([80.0, 85.0, 85.0])
PC1_RING   = np.deg2rad([80.0, 85.0, 85.0])
PC1_PINKY  = np.deg2rad([80.0, 85.0, 85.0])

# ── Home pose (all joints at zero) ────────────────────────────────────────────
HOME_WRIST  = np.zeros(2)
HOME_THUMB  = np.zeros(4)
HOME_SPREAD = {f: np.zeros(1) for f in ['index', 'middle', 'ring', 'pinky']}
HOME_FINGER = np.zeros(3)

# ── Experiment lists ───────────────────────────────────────────────────────────
FINGERTIPS = ['thumb', 'index', 'middle', 'ring', 'pinky']
OBJECTS    = ['object_1', 'object_2']
