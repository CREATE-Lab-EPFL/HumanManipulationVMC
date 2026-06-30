"""
UR5 poses and hand configuration for TunableCompliance/Hand experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── Shared ────────────────────────────────────────────────────────────────────
HOME_THUMB  = np.zeros(4)
HOME_SPREAD = {f: np.zeros(1) for f in ['index', 'middle', 'ring', 'pinky']}
HOME_FINGER = np.zeros(3)
FINGERTIPS  = ['thumb', 'index', 'middle', 'ring', 'pinky']

# ── Shared joint regulation (both experiments) ────────────────────────────────
K_ROT          = 0.2      # [N·m/rad]    background joint stiffness (spread, thumb CMC)
B_ROT          = 0.01     # [N·m·s/rad]  background joint damping
B_TIP          = 0.01     # [N·s/m]      task-space damping
FRICTION_TAU_MAX = 0.06   # [N·m]        max stiction compensation

# =============================================================================
# inhand_manipulation.py
# =============================================================================

# ── UR5 pose ──────────────────────────────────────────────────────────────────
UR5_POSE_INHAND = np.array([-0.30, 0.70, 0.34, -0.26, 1.05, 2.80])

# ── PC1 grasp pose ────────────────────────────────────────────────────────────
INHAND_PC1_THUMB  = np.deg2rad([60.0, -180, 90.0, 90.0])
INHAND_PC1_SPREAD = {
    'index':  np.deg2rad(-10.0),
    'middle': 0.0,
    'ring':   np.deg2rad(10.0),
    'pinky':  np.deg2rad(10.0),
}
INHAND_PC1_INDEX  = np.deg2rad([120.0, 70.0, 70.0])
INHAND_PC1_MIDDLE = np.deg2rad([120.0, 70.0, 70.0])
INHAND_PC1_RING   = np.deg2rad([120.0, 70.0, 70.0])
INHAND_PC1_PINKY  = np.deg2rad([120.0, 70.0, 70.0])

# ── Asymmetric stiffness sides ────────────────────────────────────────────────
SIDE_A_SOFT = ['pinky', 'ring']   # K_LOW in ASYM_A, K_HIGH in ASYM_B
SIDE_B_SOFT = ['index', 'middle'] # K_LOW in ASYM_B, K_HIGH in ASYM_A

# ── Stiffness levels ──────────────────────────────────────────────────────────
K_UNIFORM        = 10.0   # [N/m]        baseline uniform tip stiffness
K_HIGH           = 200.0  # [N/m]        stiff-side stiffness
K_LOW            = 0.1    # [N/m]        compliant-side stiffness
THUMB_ASYM_RATIO = 4      # [-]          thumb stiffness divisor in asymmetric phases (K_UNIFORM/RATIO)
K_RETURN         = 0.1    # [N·m/rad]    joint stiffness for ramp back to HOME
K_BACKGROUND_IH  = 0.1    # [N·m/rad]    background stiffness on passive joints (MCP/IP, flex)

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME      = 1.0    # [s]
RAMP_DURATION    = 1.0    # [s]   joint-target / K ramp duration
CONVERGE_VEL_THR = 0.02   # [rad/s]
CONVERGE_HOLD    = 2.0    # [s]
CONVERGE_TIMEOUT = 8.0    # [s]
RECORD_DURATION  = 5.0    # [s]

# =============================================================================
# dynamic_grasp.py
# =============================================================================

# ── UR5 pose ──────────────────────────────────────────────────────────────────
UR5_POSE_BOTTLE_START = np.array([0.01, 0.67, 0.055, -1.37, -1.09, -1.79])

# ── PC1 grasp pose ────────────────────────────────────────────────────────────
GRASP_PC1_THUMB  = np.deg2rad([40.0, -90.0, 60.0, 60.0])
GRASP_PC1_SPREAD = {
    'index':  np.deg2rad(-2.0),
    'middle': 0.0,
    'ring':   np.deg2rad(2.0),
    'pinky':  np.deg2rad(2.0),
}
GRASP_PC1_INDEX  = np.deg2rad([120.0, 120.0, 120.0])
GRASP_PC1_MIDDLE = np.deg2rad([120.0, 120.0, 120.0])
GRASP_PC1_RING   = np.deg2rad([120.0, 120.0, 120.0])
GRASP_PC1_PINKY  = np.deg2rad([120.0, 120.0, 120.0])

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = ['soft', 'stiff', 'adaptive']

# ── Stiffness levels ──────────────────────────────────────────────────────────
K_BACKGROUND_DG  = 0.05   # [N·m/rad]    uniform background stiffness on all joints
K_SOFT           = 0.1    # [N/m]        very compliant tip spring
K_STIFF          = 150.0  # [N/m]        very stiff tip spring
SOFT_DURATION    = 0.5    # [s]          (adaptive) time at K_SOFT before ramp starts
K_RAMP_DURATION  = 1.0    # [s]          (adaptive) stiffness ramp duration
HOME_DURATION    = 5.0    # [s]          time to hold home targets before shutdown

# ── Thumb pre-grip ────────────────────────────────────────────────────────────
# At startup (before UR5 moves), thumb CMC1 is driven to its final PC1 target;
# CMC2 is pre-bent to this fraction of its PC1 target.  MCP/IP stay at HOME
# until _close_hand() fires.  Regulated with K_ROT / B_ROT.
CMC2_PREGRIP_FRAC    = 0.9   # [0–1]  fraction of thumb PC1_CMC2 to pre-set at start
FINGER_PREGRIP_FRAC  = 0.15  # [0–1]  fraction of finger PC1 flexion to pre-set at start

# ── UR5 motion ────────────────────────────────────────────────────────────────
APPROACH_DIRECTION    = np.array([-1.0, 0.0, 0.0])  # unit vector for sliding direction
APPROACH_SPEED        = 0.08   # [m/s]
APPROACH_ACCELERATION = 0.20   # [m/s²]
TOTAL_DISTANCE        = 0.50   # [m]   total displacement along APPROACH_DIRECTION
CLOSE_DISTANCE        = 0.20   # [m]   displacement at which the hand closes to PC1
K_HOME                = 0.15   # [N·m/rad]  stiffness for hand return to HOME
