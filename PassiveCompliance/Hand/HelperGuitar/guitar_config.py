"""Shared constants for the guitar experiment (ADAPT Hand + UR5).

All fingers except the thumb are held closed at FINGER_CLOSED_POSE by a torsional
(joint-space) spring. The experiment compares TORSIONAL_SPRINGS values: for each,
the UR5 strums SWEEP_VECTOR on the XY plane N_RUNS times. After each strum the arm
lifts by LIFT, returns to the start position at height, then descends — fingers
stay closed throughout. Stiffness is changed online between conditions.
Audio is captured externally via the camera microphone. No friction compensation.
"""

import numpy as np

# ── UR5 reference pose ──────────────────────────────────────────────────────────
UR5_POSE_GUITAR = np.array([0.0, 0.61, 0.22, -1.66, -0.61, -0.54])  # [x,y,z,rx,ry,rz]
SWEEP_VECTOR    = np.array([0.0, -0.10, 0.0])   # XYZ displacement per strum [m]
LIFT            = 0.05   # [m]  Z clearance added for the return trip

SWEEP_SPEED   = 0.05    # [m/s]  strum speed (no contact after this speed)
SWEEP_ACCEL   = 0.5     # [m/s²]
RETURN_SPEED  = 0.2     # [m/s]  return speed (no contact)
RETURN_ACCEL  = 1.0     # [m/s²]

UR5_IP         = "192.168.1.10"
UR5_INIT_SPEED = 0.05    # [m/s]
UR5_INIT_ACCEL = 0.05    # [m/s²]

# ── Hand pose / compliance ──────────────────────────────────────────────────────
FINGER_CLOSED_POSE = np.deg2rad([20.0, 20.0, 20.0])   # [MCP, PIP, DIP]
CLOSED_FINGERS     = ['index', 'middle', 'ring', 'pinky']
SPREAD_ANGLE_DEG   = 0.0

TORSIONAL_SPRINGS  = [0.1, 0.3]   # [N·m/rad]  compared conditions
B_ROT              = 0.01   # [N·m·s/rad]  uniform joint damping
K_ROT              = 0.4    # [N·m/rad]  ramp / background stiffness

FRICTION_TAU_MAX   = 0.0    # no friction compensation

RAMP_DURATION  = 3.0    # [s]  stiffness / pose ramp duration
SETTLE_TIME    = 2.0    # [s]  settle after initial close, before first run
K_RETURN       = 0.8    # [N·m/rad]  stiffness used during return-to-home ramp
N_RUNS         = 3      # strums per stiffness condition
