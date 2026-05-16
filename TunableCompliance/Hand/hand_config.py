"""
UR5 poses and hand configuration for TunableCompliance/Hand experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── UR5 poses ─────────────────────────────────────────────────────────────────
# inhand_manipulation.py — same squeezing fixture as ProprioceptiveSensing.
UR5_POSE_SQUEEZING = np.array([-0.1405, 0.6833, 0.2594, -0.1861, -0.027, -2.7656])

# dynamic_grasp.py — hand open, aligned with the bottle; UR5 slides along +X.
UR5_POSE_BOTTLE_START = np.array([-0.30, 0.60, 0.25, -0.1861, -0.027, -2.7656])

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
FINGERTIPS  = ['thumb', 'index', 'middle', 'ring', 'pinky']

# inhand_manipulation.py — asymmetric stiffness sides.
SIDE_A_SOFT = ['pinky', 'ring']
SIDE_B_SOFT = ['thumb', 'index', 'middle']

# dynamic_grasp.py — available compliance conditions.
CONDITIONS  = ['soft', 'stiff', 'adaptive']

# ── Shared joint regulation (both experiments) ────────────────────────────────
K_ROT       = 0.1       # [N·m/rad]    background joint stiffness
B_ROT       = 0.0001    # [N·m·s/rad]  background joint damping
B_TIP       = 0.001     # [N·s/m]      task-space damping
B_FLEX_DAMP = B_ROT     # [N·m·s/rad]  flexion damping when task spring takes over

# ── inhand_manipulation.py — stiffness levels ────────────────────────────────
K_UNIFORM = 30.0    # [N/m]  baseline uniform tip stiffness
K_HIGH    = 150.0   # [N/m]  stiff-side stiffness
K_LOW     = 5.0     # [N/m]  compliant-side stiffness
K_RETURN  = 0.2     # [N·m/rad]  joint stiffness for ramp back to HOME

# ── inhand_manipulation.py — timing ──────────────────────────────────────────
SETTLE_TIME      = 3.0   # [s]
RAMP_DURATION    = 5.0   # [s]   joint-target / K ramp duration
CONVERGE_VEL_THR = 0.02  # [rad/s]
CONVERGE_HOLD    = 1.0   # [s]
CONVERGE_TIMEOUT = 12.0  # [s]
RECORD_DURATION  = 5.0   # [s]

# ── dynamic_grasp.py — stiffness levels ──────────────────────────────────────
K_SOFT        = 5.0    # [N/m]  very compliant tip spring
K_STIFF       = 150.0  # [N/m]  very stiff tip spring
SOFT_DURATION = 2.0    # [s]    (adaptive) time at K_SOFT before switching

# ── dynamic_grasp.py — UR5 motion ────────────────────────────────────────────
APPROACH_SPEED  = 0.05   # [m/s]
TOTAL_DISTANCE  = 0.40   # [m]   total X displacement
CLOSE_DISTANCE  = 0.15   # [m]   X at which the hand closes to PC1
