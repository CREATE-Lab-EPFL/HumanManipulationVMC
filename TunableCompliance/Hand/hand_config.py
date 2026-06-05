"""
UR5 poses and hand configuration for TunableCompliance/Hand experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── UR5 poses ─────────────────────────────────────────────────────────────────
# inhand_manipulation.py — hand horizontal, fingers down over the in-hand
# reorientation fixture. Independent from ProprioceptiveSensing's
# UR5_POSE_SQUEEZING (object-stiffness fixture) — edit them separately.
UR5_POSE_INHAND = np.array([-0.25, 0.59, 0.26, -1.72, -0.57, 0.83])

# dynamic_grasp.py — hand open, aligned with the bottle; UR5 slides along +X.
UR5_POSE_BOTTLE_START = np.array([-0.30, 0.52, 0.12, -0.71, -1.52, -1.66])

# ── PC1 grasp pose for in-hand manipulation ───────────────────────────────────
INHAND_PC1_WRIST  = np.deg2rad([0.0,  0.0])
INHAND_PC1_THUMB  = np.deg2rad([40.0, 0.0, 90.0, 90.0])
INHAND_PC1_SPREAD = {
    'index':  np.deg2rad(-2.0),
    'middle': 0.0,
    'ring':   np.deg2rad(2.0),
    'pinky':  np.deg2rad(2.0),
}
INHAND_PC1_INDEX  = np.deg2rad([55.0, 65.0, 65.0])
INHAND_PC1_MIDDLE = np.deg2rad([55.0, 65.0, 65.0])
INHAND_PC1_RING   = np.deg2rad([55.0, 65.0, 65.0])
INHAND_PC1_PINKY  = np.deg2rad([55.0, 65.0, 65.0])

# ── PC1 grasp pose for dynamic grasping ───────────────────────────────────────
GRASP_PC1_WRIST  = np.deg2rad([0.0,  0.0])
GRASP_PC1_THUMB  = np.deg2rad([30.0, 0.0, 70.0, 70.0])
GRASP_PC1_SPREAD = {
    'index':  np.deg2rad(-2.0),
    'middle': 0.0,
    'ring':   np.deg2rad(2.0),
    'pinky':  np.deg2rad(2.0),
}
GRASP_PC1_INDEX  = np.deg2rad([55.0, 65.0, 65.0])
GRASP_PC1_MIDDLE = np.deg2rad([55.0, 65.0, 65.0])
GRASP_PC1_RING   = np.deg2rad([55.0, 65.0, 65.0])
GRASP_PC1_PINKY  = np.deg2rad([55.0, 65.0, 65.0])

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
K_ROT       = 0.01      # [N·m/rad]    background joint stiffness
K_ROT_FLEX  = 0.01      # [N·m/rad]    flex joint stiffness when task spring is active
B_ROT       = 0.0001    # [N·m·s/rad]  background joint damping
B_TIP       = 0.001     # [N·s/m]      task-space damping
B_FLEX_DAMP = B_ROT     # [N·m·s/rad]  flexion damping when task spring takes over

# ── inhand_manipulation.py — stiffness levels ────────────────────────────────
K_UNIFORM = 0.0     # [N/m]  baseline uniform tip stiffness
K_HIGH    = 100.0   # [N/m]  stiff-side stiffness
K_LOW     = 0.0     # [N/m]  compliant-side stiffness
K_RETURN  = 0.1     # [N·m/rad]  joint stiffness for ramp back to HOME

# ── inhand_manipulation.py — timing ──────────────────────────────────────────
SETTLE_TIME      = 1.0   # [s]
RAMP_DURATION    = 3.0   # [s]   joint-target / K ramp duration
CONVERGE_VEL_THR = 0.02  # [rad/s]
CONVERGE_HOLD    = 2.0   # [s]
CONVERGE_TIMEOUT = 8.0  # [s]
RECORD_DURATION  = 5.0   # [s]

# ── dynamic_grasp.py — stiffness levels ──────────────────────────────────────
K_SOFT        = 1.0    # [N/m]  very compliant tip spring
K_STIFF       = 100.0  # [N/m]  very stiff tip spring
SOFT_DURATION    = 2.0    # [s]    (adaptive) time at K_SOFT before ramp starts
K_RAMP_DURATION  = 1.0    # [s]    (adaptive) stiffness ramp duration
HOME_DURATION    = 5.0    # [s]    time to hold home targets before shutdown

# ── dynamic_grasp.py — UR5 motion ────────────────────────────────────────────
APPROACH_SPEED  = 0.025  # [m/s]
TOTAL_DISTANCE  = 0.40   # [m]   total X displacement
CLOSE_DISTANCE  = 0.15   # [m]   X at which the hand closes to PC1
K_HOME          = 0.15   # [N·m/rad]  finger/thumb stiffness for hand return to HOME
K_HOME_WRIST    = 0.05   # [N·m/rad]  wrist stiffness for return to HOME (softer — wrist oscillates most)
