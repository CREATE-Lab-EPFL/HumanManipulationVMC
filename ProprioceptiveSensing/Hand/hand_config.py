"""
UR5 poses and hand configuration for ProprioceptiveSensing/Hand experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── UR5 poses ─────────────────────────────────────────────────────────────────
# Hand horizontal, fingers pointing down, centred over the squeezing fixture.
UR5_POSE_SQUEEZING = np.array([0, 0.66, 0.29, 0.0, 0.0, -1.96])

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
OBJECTS    = ['object_1', 'object_2', 'object_3']

# ── Tip stiffness levels ──────────────────────────────────────────────────────
K_TIP_GENTLE = 10.0            # [N/m]   gentle-grasp stiffness (baseline)
K_TIP_SWEEP  = [50, 100, 150]  # [N/m]   stiffness levels for the sweep

# ── Background joint regulation ───────────────────────────────────────────────
K_ROT       = 0.1       # [N·m/rad]
B_ROT       = 0.0001    # [N·m·s/rad]
B_TIP       = 0.001     # [N·s/m]
K_RETURN    = 0.2       # [N·m/rad]    joint stiffness for ramp back to HOME
B_FLEX_DAMP = B_ROT     # [N·m·s/rad]  flexion damping when task spring takes over

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME      = 3.0   # [s]
RAMP_DURATION    = 5.0   # [s]
CONVERGE_VEL_THR = 0.02  # [rad/s]
CONVERGE_HOLD    = 1.0   # [s]
CONVERGE_TIMEOUT = 12.0  # [s]
RECORD_DURATION  = 5.0   # [s]
