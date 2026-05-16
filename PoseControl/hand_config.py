"""
Hand configuration for PoseControl experiments.
"""

import numpy as np

# ── VMC parameters ─────────────────────────────────────────────────────────────
STIFFNESS = 0.6    # [N·m/rad]
DAMPING   = 0.001  # [N·m·s/rad]

# ── Timing ────────────────────────────────────────────────────────────────────
CONVERGE_VEL_THR = 0.02   # [rad/s]
CONVERGE_HOLD    = 1.0    # [s]
CONVERGE_TIMEOUT = 10.0   # [s]
LOG_DURATION     = 8.0    # [s]
RAMP_DURATION    = 5.0    # [s]

# ── Target poses (Santello et al. 1998) ───────────────────────────────────────
POSES = [
    {   # PC1: power grasp (~50% variance) — global flexion, thumb opposition
        "label":      "PC1",
        "wrist":      np.deg2rad([-5.0,  0.0]),
        "thumb":      np.deg2rad([ 65.0,  0.0,  60.0, 45.0]),
        "spread":     {"index":  np.array([np.deg2rad( -2.0)]),
                       "middle": np.array([0.0]),
                       "ring":   np.array([np.deg2rad(  2.0)]),
                       "pinky":  np.array([np.deg2rad(  2.0)])},
        "middle":     np.deg2rad([55.0, 65.0, 65.0]),
        "ring_pinky": np.deg2rad([55.0, 65.0, 65.0]),
        "index":      np.deg2rad([55.0, 65.0, 65.0]),
    },
    {   # PC2: precision pinch (~30% variance) — index+thumb opposed
        "label":      "PC2",
        "wrist":      np.deg2rad([ -1.0,  2.0]),
        "thumb":      np.deg2rad([ 50.0, 12.0,  35.0, 25.0]),
        "spread":     {"index":  np.array([np.deg2rad(-1.0)]),
                       "middle": np.array([0.0]),
                       "ring":   np.array([np.deg2rad( 1.0)]),
                       "pinky":  np.array([np.deg2rad( 1.0)])},
        "middle":     np.deg2rad([25.0, 35.0, 35.0]),
        "ring_pinky": np.deg2rad([55.0, 65.0, 65.0]),
        "index":      np.deg2rad([25.0, 35.0, 40.0]),
    },
]

# ── Home pose ──────────────────────────────────────────────────────────────────
HOME_POSE = {
    "label":      "HOME",
    "wrist":      np.deg2rad([0.0, 0.0]),
    "thumb":      np.deg2rad([0.0, 0.0, 0.0, 0.0]),
    "spread":     {"index":  np.array([0.0]),
                   "middle": np.array([0.0]),
                   "ring":   np.array([0.0]),
                   "pinky":  np.array([0.0])},
    "middle":     np.deg2rad([0.0, 0.0, 0.0]),
    "ring_pinky": np.deg2rad([0.0, 0.0, 0.0]),
    "index":      np.deg2rad([0.0, 0.0, 0.0]),
}
