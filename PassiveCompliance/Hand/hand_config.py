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

# =============================================================================
# Guitar experiment
# =============================================================================
UR5_POSE_GUITAR = np.array([-0.3964, 0.5314, 0.2458, 1.3352, -1.4089, -1.2771])  # [x,y,z,rx,ry,rz]
SWEEP_VECTOR    = np.array([0.10, 0.0, 0.0])   # [m] XYZ displacement per strum
LIFT            = 0.05    # [m]  Z clearance for the return trip

SWEEP_SPEED    = 0.05    # [m/s]  strum speed (contact phase)
SWEEP_ACCEL    = 0.5     # [m/s²]
RETURN_SPEED   = 0.2     # [m/s]  return (no contact)
RETURN_ACCEL   = 1.0     # [m/s²]

GUITAR_FINGER_CLOSED_POSE = np.deg2rad([20.0, 30.0, 30.0])   # [MCP, PIP, DIP]
GUITAR_CLOSED_FINGERS     = ['index', 'middle', 'ring', 'pinky']
GUITAR_SPREAD_ANGLE_DEG   = 0.0

TORSIONAL_SPRINGS       = [0.1, 0.3]    # [N·m/rad]  compared conditions
GUITAR_K_ROT            = 0.4           # [N·m/rad]  ramp / background stiffness
GUITAR_B_ROT            = 0.01          # [N·m·s/rad]
GUITAR_K_RETURN         = 0.8           # [N·m/rad]  return-to-home stiffness
GUITAR_RAMP_DURATION    = 3.0           # [s]
GUITAR_SETTLE_TIME      = 2.0           # [s]  settle after initial close
GUITAR_N_RUNS           = 3             # strums per stiffness condition
GUITAR_FRICTION_TAU_MAX = 0.05          # [N·m]  no friction compensation

# =============================================================================
# Weight compliance experiment
# =============================================================================
WEIGHT_POSE             = np.deg2rad([30.0, 30.0, 30.0])   # [MCP, PIP, DIP] target
WEIGHT_FINGERS          = ['index', 'middle', 'ring', 'pinky']
WEIGHT_SPREAD_ANGLE_DEG = 0.0

STIFFNESS_CONDITIONS    = [0.05, 0.1, 0.2, 0.4]   # [N·m/rad]  swept stiffness values
WEIGHTS_G               = [0, 50, 100, 200, 300]   # [g]  total hanging weight per step

WEIGHT_K_ROT            = 0.4    # [N·m/rad]  ramp / background stiffness
WEIGHT_B_ROT            = 0.01   # [N·m·s/rad]
WEIGHT_K_RETURN         = 0.4    # [N·m/rad]  return-to-home stiffness
WEIGHT_RAMP_DURATION    = 3.0    # [s]
WEIGHT_SETTLE_TIME      = 3.0    # [s]  settle after hanging weight before logging
WEIGHT_LOG_DURATION     = 5.0    # [s]  logging window per weight step
WEIGHT_FRICTION_TAU_MAX = 0.04   # [N·m]
