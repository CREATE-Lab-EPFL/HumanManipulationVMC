"""Shared constants for piano experiments."""

import numpy as np

# ── Piano playing ──────────────────────────────────────────────────────────────
# PLACEHOLDER — tune UR5_POSE_PIANO and PRESS_DEPTH to the actual keyboard.
UR5_POSE_PIANO = np.array([-0.15, 0.68, 0.14, -0.85, -1.19, 0.29])
PRESS_DEPTH    = 0.025   # [m]  UR5 descends along base-frame Z to press keys
PRESS_SPEED    = 0.05    # [m/s]

# ── Glissando ─────────────────────────────────────────────────────────────────
# PLACEHOLDER — tune UR5_POSE_GLISSANDO_START to the actual keyboard.
UR5_POSE_GLISSANDO_START = np.array([-0.15, 0.68, 0.14, -0.85, -1.19, 0.29])
GLISSANDO_DIRECTION      = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
GLISSANDO_DISTANCE       = 0.23    # [m]
GLISSANDO_SPEED          = 0.05    # [m/s]

# ── Shared ─────────────────────────────────────────────────────────────────────
UR5_IP         = "192.168.1.10"
UR5_INIT_SPEED = 0.05    # [m/s]
UR5_INIT_ACCEL = 0.05    # [m/s²]

PRESS_ANGLE_DEG         = 40.0           # [deg]  MCP+PIP flexion
SPREAD_ANGLE_DEG        = 0.0            # [deg]  finger abduction/adduction
PIANO_FINGERS_PLAYING   = ['index', 'ring']
PIANO_FINGERS_GLISSANDO = ['index', 'ring']

K_ROT       = 0.4    # [N·m/rad]  background (non-playing fingers)
K_ROT_PRESS = 0.05   # [N·m/rad]  playing fingers (task spring takes over)
B_ROT       = 0.001  # [N·m·s/rad]

# ── piano_playing_hand.py ─────────────────────────────────────────────────────
K_SWEEP             = [5.0, 20.0]           # [N/m]
K_STIFF             = 100.0                  # [N/m]
K_SOFT              = 10.0                   # [N/m]
B_CART_PLAYING      = 0.5                    # [N·s/m]
N_CYCLES            = 3                      # strokes per stiffness value
PLAYING_SETTLE_TIME = 3.0                    # [s]
RAMP_DURATION       = 3.0                    # [s]  gradual approach to / return from press pose

# ── piano_glissando.py ────────────────────────────────────────────────────────
B_CART_GLISSANDO      = 1.0     # [N·s/m]
N_RUNS                = 5
GLISSANDO_SETTLE_TIME = 3.0     # [s]
