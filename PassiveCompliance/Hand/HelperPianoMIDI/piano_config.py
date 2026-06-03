"""Shared constants for piano experiments."""

import numpy as np

# ── Piano playing ──────────────────────────────────────────────────────────────
UR5_POSE_PIANO = np.array([0.12, 0.51, 0.07, -1.45, -0.42, -0.64])
PRESS_DEPTH    = 0.06    # [m]  UR5 descends along base-frame Z to press keys
PRESS_SPEED    = 3.0     # [m/s]  UR5 moveL caps TCP speed at 3 m/s (must be in [0, 3])
PRESS_ACCEL    = 1.5     # [m/s²]  press-stroke acceleration — the real lever for strike speed
                         #         (speed is capped at 3, but over 0.08 m the move is
                         #         acceleration-limited; raise this for a faster strike)

# ── Glissando ─────────────────────────────────────────────────────────────────
UR5_POSE_GLISSANDO_START = np.array([0.13, 0.51, 0.07, -1.45, -0.42, -0.64])
GLISSANDO_DIRECTION      = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
GLISSANDO_DISTANCE       = 0.23    # [m]
GLISSANDO_SPEED          = 0.05    # [m/s]

# ── Shared ─────────────────────────────────────────────────────────────────────
UR5_IP         = "192.168.1.10"
UR5_INIT_SPEED = 0.05    # [m/s]
UR5_INIT_ACCEL = 0.05    # [m/s²]

PRESS_POSE              = np.deg2rad([80.0, 80.0, 80.0])  # [MCP, PIP, DIP] flexion [rad]
SPREAD_ANGLE_DEG        = 20.0           # [deg]  finger abduction/adduction
PIANO_FINGERS_PLAYING   = ['index', 'ring']
PIANO_FINGERS_GLISSANDO = ['index', 'middle']

K_ROT       = 0.4    # [N·m/rad]  background (non-playing fingers)
K_ROT_PRESS = 0.05   # [N·m/rad]  playing fingers, PIP/DIP (task spring takes over)
K_MCP_PRESS = 0.20   # [N·m/rad]  playing fingers, MCP — fixed, NOT a function of K_SWEEP
B_ROT       = 0.01   # [N·m·s/rad]
B_ROT_HOLD  = 0.05   # [N·m·s/rad]  damping for held (non-playing) DOFs — suppresses oscillation

WRIST_PITCH_DEG = 20.0   # [deg]  hand wrist pitch reference (rotation about wrist -X axis;
                         #         old default was -20. Verify up/down direction on the rig).

# Friction-compensation max torque [N·m] for this task
FRICTION_TAU_MAX = 0.0

# ── piano_playing_hand.py ─────────────────────────────────────────────────────
K_SWEEP             = [5.0, 100.0]           # [N/m]
K_SOFT              = K_SWEEP[0]             # [N/m]  heterogeneous: ring = sweep min
K_STIFF             = K_SWEEP[1]             # [N/m]  heterogeneous: index = sweep max
B_CART_PLAYING      = 0.5                    # [N·s/m]
N_CYCLES            = 3                      # strokes per stiffness value
PLAYING_SETTLE_TIME = 3.0                    # [s]
RAMP_DURATION       = 3.0                    # [s]  gradual approach to / return from press pose

# ── piano_glissando.py ────────────────────────────────────────────────────────
B_CART_GLISSANDO      = 1.0     # [N·s/m]
N_RUNS                = 5
GLISSANDO_SETTLE_TIME = 3.0     # [s]
