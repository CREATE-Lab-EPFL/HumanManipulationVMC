"""
UR5 poses and hand configuration for ProprioceptiveSensing/Hand experiments.
"""

import numpy as np

# ── UR5 poses ─────────────────────────────────────────────────────────────────
# Hand horizontal, fingers pointing down, centred over the squeezing fixture.
UR5_POSE_SQUEEZING = np.array([-0.381, 0.55, 0.29, -1.86, -0.12, 0.59])

# ── PC1 grasp pose (Santello et al. 1998 — first principal component) ─────────
PC1_THUMB  = np.deg2rad([100.0, -70.0, 120.0, 120.0])
PC1_SPREAD = {
    'index':  np.deg2rad(0.0),
    'middle': np.deg2rad(0.0),
    'ring':   np.deg2rad(0.0),
    'pinky':  np.deg2rad(0.0),
}
PC1_INDEX  = np.deg2rad([70.0, 80.0, 80.0])
PC1_MIDDLE = np.deg2rad([70.0, 80.0, 80.0])
PC1_RING   = np.deg2rad([70.0, 80.0, 80.0])
PC1_PINKY  = np.deg2rad([70.0, 80.0, 80.0])

# ── Home pose (all joints at zero) ────────────────────────────────────────────
HOME_THUMB  = np.zeros(4)
HOME_SPREAD = {f: np.zeros(1) for f in ['index', 'middle', 'ring', 'pinky']}
HOME_FINGER = np.zeros(3)

# ── Experiment lists ───────────────────────────────────────────────────────────
FINGERTIPS = ['thumb', 'index', 'middle', 'ring', 'pinky']
OBJECTS    = ['hard', 'medium', 'soft']

# ── Tip stiffness levels ──────────────────────────────────────────────────────
K_TIP_GENTLE = 20.0              # [N/m]  gentle-grasp stiffness (baseline)
K_TIP_SWEEP  = [50, 100, 200]    # [N/m]  stiffness levels for the thumb sweep
K_TIP_HOLD   = 100.0             # [N/m]  constant hold stiffness for the 4 clamping fingers
N_RUNS       = 5                 # independent grasp trials per object (data collection only; 1 for video)

# ── Background joint regulation ───────────────────────────────────────────────
K_ROT          = 0.1       # [N·m/rad]
B_ROT          = 0.001     # [N·m·s/rad]
B_TIP          = 0.01      # [N·s/m]      task-space damping
K_RETURN       = 0.2       # [N·m/rad]    joint stiffness for ramp back to HOME
B_FLEX_DAMP    = B_ROT     # [N·m·s/rad]  flexion damping when task spring takes over
FRICTION_TAU_MAX = 0.04    # [N·m]        max stiction compensation (overrides hand_params default)

# ── object_stiffness_dynamic.py — dynamic ramp parameters ────────────────────
K_TIP_PROBE        = 100.0              # [N/m]  ramp target for the thumb probe
RAMP_DURATIONS     = list(np.round(np.logspace(np.log10(0.01), np.log10(RAMP_DURATION), 8), 4))  # [s]
POST_RAMP_DURATION = 5.0               # [s]  record window after reaching K_TIP_PROBE
BASELINE_DURATION  = 2.0              # [s]  baseline window before each ramp

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_TIME      = 3.0   # [s]
RAMP_DURATION    = 1.0   # [s]
CONVERGE_VEL_THR = 0.02  # [rad/s]
CONVERGE_HOLD    = 1.0   # [s]
CONVERGE_TIMEOUT = 5.0   # [s]
RECORD_DURATION  = 5.0   # [s]
