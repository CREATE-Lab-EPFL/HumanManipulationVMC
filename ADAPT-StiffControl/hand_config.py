"""
UR5 poses and hand configuration for ADAPT-StiffControl experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── UR5 poses ─────────────────────────────────────────────────────────────────
# grasp_adaptation.py — grasp contact pose for each known object.
UR5_POSE_GRASP_OBJ = {
    'hard_obj': np.array([-0.1405, 0.6833, 0.2594, -0.1861, -0.027, -2.7656]),
    'soft_obj': np.array([-0.1405, 0.5500, 0.2594, -0.1861, -0.027, -2.7656]),
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
PC1_INDEX  = np.deg2rad([55.0, 65.0, 65.0])
PC1_MIDDLE = np.deg2rad([55.0, 65.0, 65.0])
PC1_RING   = np.deg2rad([55.0, 65.0, 65.0])
PC1_PINKY  = np.deg2rad([55.0, 65.0, 65.0])

# ── Home pose (all joints at zero) ────────────────────────────────────────────
HOME_WRIST  = np.zeros(2)
HOME_THUMB  = np.zeros(4)
HOME_SPREAD = {f: np.zeros(1) for f in ['index', 'middle', 'ring', 'pinky']}
HOME_FINGER = np.zeros(3)

# ── Experiment lists ───────────────────────────────────────────────────────────
FINGERTIPS = ['thumb', 'index', 'middle', 'ring', 'pinky']
OBJECTS    = ['hard_obj', 'soft_obj']

# ── Grasp adaptation rule (grasp_adaptation.py) ───────────────────────────────
# Two-point compliance estimation: settle at K_TIP_GENTLE, then push at
# K_TIP_PROBE.  Per finger:  C_O = ||Δpos|| / ||Δforce||.  Average across
# fingers, then  k_applied = clip(K_GAIN / C_O_mean, K_MIN, K_MAX).
K_TIP_GENTLE = 10.0     # [N/m]   first sensing stiffness
K_TIP_PROBE  = 50.0     # [N/m]   second sensing stiffness (must differ from gentle)
K_GAIN       = 5.0      # [-]     proportionality of k_applied to 1/C_O
K_MIN        = 5.0      # [N/m]   lower bound on adapted stiffness
K_MAX        = 150.0    # [N/m]   upper bound on adapted stiffness

# ── Background joint regulation ───────────────────────────────────────────────
K_ROT       = 0.1       # [N·m/rad]    background joint stiffness
B_ROT       = 0.0001    # [N·m·s/rad]  background joint damping
B_TIP       = 0.001     # [N·s/m]      task-space damping
K_RETURN    = 0.2       # [N·m/rad]    joint stiffness during ramp back to HOME
B_FLEX_DAMP = B_ROT     # [N·m·s/rad]  flexion damping when task spring takes over

# ── UR5 motion geometry ───────────────────────────────────────────────────────
APPROACH_HEIGHT = 0.10   # [m] Z offset above grasp pose for safe approach
LIFT_HEIGHT     = 0.08   # [m] Z lift after adaptation

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME      = 2.0   # [s]      wait after UR5 reaches pose
RAMP_DURATION    = 5.0   # [s]      joint-target / K ramp duration
CONVERGE_VEL_THR = 0.02  # [rad/s]  velocity threshold for "converged"
CONVERGE_HOLD    = 1.0   # [s]      time below threshold to declare convergence
CONVERGE_TIMEOUT = 12.0  # [s]      max wait before forcing transition
SENSE_DURATION   = 5.0   # [s]      sensing window (δ averaged over this)
HOLD_TIME        = 3.0   # [s]      hold at lifted pose
