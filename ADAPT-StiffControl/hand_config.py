"""
UR5 poses and hand configuration for ADAPT-StiffControl experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── UR5 pose ───────────────────────────────────────────────────────────────────
# grasp contact pose for the apple.
UR5_POSE_GRASP = np.array([-0.52,  0.31,  0.36, -1.54, -1.64, 0.90])

# ── PC1 grasp pose (Santello et al. 1998 — first principal component) ─────────
PC1_THUMB  = np.deg2rad([70.0, -60.0, 100.0, 100.0])
PC1_SPREAD = {
    'index':  np.deg2rad(-5.0),
    'middle': np.deg2rad(0.0),
    'ring':   np.deg2rad(5.0),
    'pinky':  np.deg2rad(5.0),
}
PC1_INDEX  = np.deg2rad([90.0, 110.0, 110.0])
PC1_MIDDLE = np.deg2rad([90.0, 110.0, 110.0])
PC1_RING   = np.deg2rad([90.0, 110.0, 110.0])
PC1_PINKY  = np.deg2rad([90.0, 110.0, 110.0])

# ── Home pose (all joints at zero) ────────────────────────────────────────────
HOME_THUMB  = np.zeros(4)
HOME_SPREAD = {f: np.zeros(1) for f in ['index', 'middle', 'ring', 'pinky']}
HOME_FINGER = np.zeros(3)

# ── Fingertip list ─────────────────────────────────────────────────────────────
FINGERTIPS = ['thumb', 'index', 'middle', 'ring', 'pinky']

# ── Force levels ───────────────────────────────────────────────────────────────
# User selects one at startup; GD drives all 5 fingertip forces toward f_des.
FORCE_LEVELS = ['hard', 'medium', 'soft']
F_DES = {
    'hard':   0.8,   # [N]  target tip force per finger
    'medium': 0.3,   # [N]
    'soft':   0.1,   # [N]
}

# ── Stiffness schedule ─────────────────────────────────────────────────────────
K_TIP_GENTLE = 20.0   # [N/m]  starting task-space stiffness for all fingertips

# ── Gradient-descent adaptation ────────────────────────────────────────────────
# GD drives K_task per finger until mean |f_meas − f_des| < F_CONVERGE_THR.
# No timeout: waits until fully converged.
GD_LR          = 1e-4   # [-]   learning rate (matched to stiffness_descent default)
F_CONVERGE_THR = 0.05   # [N]   mean |f_meas − f_des| threshold across all fingers

# ── Background joint regulation ───────────────────────────────────────────────
K_ROT          = 0.1       # [N·m/rad]    background joint stiffness
B_ROT          = 0.001     # [N·m·s/rad]  background joint damping
B_TIP          = 0.01      # [N·s/m]      task-space damping
K_RETURN       = 0.2       # [N·m/rad]    joint stiffness during ramp back to HOME
B_FLEX_DAMP    = B_ROT     # [N·m·s/rad]  flexion damping when task spring takes over
FRICTION_TAU_MAX = 0.04    # [N·m]        max stiction compensation

# ── UR5 motion geometry ───────────────────────────────────────────────────────
PRESS_HEIGHT = 0.15   # [m]  Z press below grasp pose (elastic band shows force)

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME      = 3.0   # [s]      wait after UR5 reaches pose
RAMP_DURATION    = 1.0   # [s]      joint-target ramp duration
CONVERGE_VEL_THR = 0.02  # [rad/s]  velocity threshold for "converged"
CONVERGE_HOLD    = 1.0   # [s]      time below threshold to declare convergence
CONVERGE_TIMEOUT = 5.0   # [s]      max wait for initial closing convergence only
HOLD_TIME        = 3.0   # [s]      hold at PRESS_POSE before releasing
