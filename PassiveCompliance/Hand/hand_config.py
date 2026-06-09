"""Shared configuration for PassiveCompliance/Hand experiments.

Covers:
  guitar_playing_hand.py     — torsional-spring guitar strumming (UR5 + hand)
  weight_compliance_hand.py  — static weight loading at different stiffnesses (hand fixed)
"""

import numpy as np

# ── Shared hardware ────────────────────────────────────────────────────────────
UR5_IP         = "192.168.1.10"
UR5_INIT_SPEED = 0.05    # [m/s]
UR5_INIT_ACCEL = 0.05    # [m/s²]

# ── Shared thumb background regulation ────────────────────────────────────────
# Thumb is not involved in either experiment — held at a small neutral pose.
# Uses lower stiffness (matching ProprioceptiveSensing / ADAPT-StiffControl)
# to avoid oscillation with the high K_ROT used for finger ramps.
K_THUMB = 0.1    # [N·m/rad]
B_THUMB = 0.001  # [N·m·s/rad]

# =============================================================================
# Guitar experiment
# =============================================================================
UR5_POSE_GUITAR = np.array([-0.3964,  0.5314,  0.0639,  0.8173, -0.9609, -1.521 ])  # [x,y,z,rx,ry,rz]
SWEEP_VECTOR    = np.array([0.10, 0.0, 0.0])   # [m] XYZ displacement per strum
LIFT            = 0.05    # [m]  Z clearance for the return trip

SWEEP_SPEED    = 0.05    # [m/s]  strum speed (contact phase)
SWEEP_ACCEL    = 0.5     # [m/s²]
RETURN_SPEED   = 0.2     # [m/s]  return (no contact)
RETURN_ACCEL   = 1.0     # [m/s²]

GUITAR_FINGER_CLOSED_POSE = np.deg2rad([30.0, 30.0, 30.0])   # [MCP, PIP, DIP]
GUITAR_CLOSED_FINGERS     = ['index', 'middle', 'ring', 'pinky']
GUITAR_SPREAD_ANGLE_DEG   = 0.0

TORSIONAL_SPRINGS       = [0.01, 0.08, 0.3]    # [N·m/rad]  compared conditions
GUITAR_K_ROT            = 0.6           # [N·m/rad]  approach stiffness (used only to close fingers; replaced by condition K after ramp)
GUITAR_B_ROT            = 0.01          # [N·m·s/rad]  normal strum damping
GUITAR_B_SETTLE         = 0.05          # [N·m·s/rad]  higher damping for per-run stabilisation
GUITAR_K_RETURN         = 0.15          # [N·m/rad]  return-to-home stiffness (matches dynamic_grasp K_HOME)
GUITAR_RAMP_DURATION    = 3.0           # [s]  initial close ramp
GUITAR_RESTAB_DURATION  = 1.0           # [s]  per-run re-stabilisation ramp duration
GUITAR_RESTAB_TIME      = 1.0           # [s]  settle at K_ROT during per-run stabilisation
GUITAR_SETTLE_TIME      = 2.0           # [s]  settle after initial close
GUITAR_N_RUNS           = 3             # strums per stiffness condition
GUITAR_FRICTION_TAU_MAX = 0.03          # [N·m]

# =============================================================================
# Weight compliance experiment
# =============================================================================
WEIGHT_POSE             = np.deg2rad([30.0, 30.0, 30.0])   # [MCP, PIP, DIP] target
WEIGHT_FINGERS          = ['index', 'middle', 'ring', 'pinky']
WEIGHT_SPREAD_ANGLE_DEG = 0.0

STIFFNESS_CONDITIONS    = [0.05, 0.1, 0.2, 0.4]   # [N·m/rad]  swept stiffness values
WEIGHT_LABELS           = ['soft', 'medium', 'hard']  # operator-named weight steps (magnitudes not fixed)

WEIGHT_K_ROT            = 0.4    # [N·m/rad]  approach stiffness (used only to reach pose; replaced by condition K after ramp)
WEIGHT_B_ROT            = 0.01   # [N·m·s/rad]  normal damping
WEIGHT_B_SETTLE         = 0.05   # [N·m·s/rad]  higher damping for initial ramp-to-pose
WEIGHT_K_RETURN         = 0.15   # [N·m/rad]  return-to-home stiffness (matches dynamic_grasp K_HOME)
WEIGHT_RAMP_DURATION    = 3.0    # [s]
WEIGHT_SETTLE_TIME      = 3.0    # [s]  settle after hanging weight before logging
WEIGHT_LOG_DURATION     = 5.0    # [s]  convergence window per weight step
WEIGHT_FRICTION_TAU_MAX = 0.04   # [N·m]
