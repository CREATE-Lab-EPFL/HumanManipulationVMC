"""
UR5 poses and hand configuration for ADAPT-StiffControl experiments.
PLACEHOLDER — tune UR5 poses to the actual setup before running.
"""

import numpy as np

# ── UR5 poses ─────────────────────────────────────────────────────────────────
# grasp_adaptation.py — grasp contact pose for each known object.
UR5_POSE_GRASP_OBJ = {
    'hard_obj': np.array([-0.52,  0.31,  0.37,  -1.54, -1.64, 0.90]),
    'soft_obj': np.array([-0.52,  0.31,  0.25, -1.54, -1.64, 0.90]),
}

# ── PC1 grasp pose (Santello et al. 1998 — first principal component) ─────────
PC1_THUMB  = np.deg2rad([90.0, -90.0, 90.0, 90.0])
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

# ── Experiment lists ───────────────────────────────────────────────────────────
FINGERTIPS = ['thumb', 'index', 'middle', 'ring', 'pinky']
OBJECTS    = ['hard_obj', 'soft_obj']

# ── Grasp adaptation rule (grasp_adaptation.py) ───────────────────────────────
# Two-point compliance sensing: 4 fingers clamped at K_TIP_HOLD (frozen
# position), thumb probes from K_TIP_GENTLE → K_TIP_PROBE.
# C_O = ||Δpos_thumb|| / ||ΔF_thumb||; then gradient descent drives
# tip force toward f_des = F_GAIN / C_O (MODE='stiffness': updates K;
# MODE='ref': updates virtual equilibrium positions).
K_TIP_GENTLE = 20.0    # [N/m]   gentle baseline stiffness (thumb)
K_TIP_PROBE  = 200.0   # [N/m]   probe stiffness (thumb)
K_TIP_HOLD   = 100.0   # [N/m]   constant hold stiffness for the 4 clamping fingers

# ── Gradient-descent adaptation ───────────────────────────────────────────────
F_GAIN          = 0.025  # [N·m]   tune to set the desired force range
GD_LR           = 1e-6   # [-]     gradient-descent learning rate (stiffness & ref modes)
F_CONVERGE_THR  = 0.05   # [N]     |f_meas - f_des| threshold to declare GD converged

# ── Background joint regulation ───────────────────────────────────────────────
K_ROT          = 0.1       # [N·m/rad]    background joint stiffness
B_ROT          = 0.001     # [N·m·s/rad]  background joint damping
B_TIP          = 0.01      # [N·s/m]      task-space damping
K_RETURN       = 0.2       # [N·m/rad]    joint stiffness during ramp back to HOME
B_FLEX_DAMP    = B_ROT     # [N·m·s/rad]  flexion damping when task spring takes over
FRICTION_TAU_MAX = 0.04    # [N·m]        max stiction compensation (overrides hand_params default)

# ── UR5 motion geometry ───────────────────────────────────────────────────────
PRESS_HEIGHT = 0.15   # [m] Z press below grasp pose (force demonstration)

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME      = 3.0   # [s]      wait after UR5 reaches pose
RAMP_DURATION    = 1.0   # [s]      joint-target / K ramp duration
CONVERGE_VEL_THR = 0.02  # [rad/s]  velocity threshold for "converged"
CONVERGE_HOLD    = 1.0   # [s]      time below threshold to declare convergence
CONVERGE_TIMEOUT = 5.0   # [s]      max wait before forcing transition
SENSE_DURATION   = 2.0   # [s]      sensing window (δ averaged over this)
HOLD_TIME        = 3.0   # [s]      hold at lifted pose
