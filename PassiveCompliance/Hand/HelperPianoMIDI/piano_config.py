"""Shared constants for piano experiments."""

import numpy as np

# ── Piano playing ──────────────────────────────────────────────────────────────
# PLACEHOLDER — tune UR5_POSE_PIANO and PRESS_DEPTH to the actual keyboard.
UR5_POSE_PIANO = np.array([0.13, 0.51, 0.07, -1.45, -0.42, -0.64])
PRESS_DEPTH    = 0.05    # [m]  UR5 descends along base-frame Z to press keys
PRESS_SPEED    = 0.05    # [m/s]

# ── Glissando ─────────────────────────────────────────────────────────────────
# PLACEHOLDER — tune UR5_POSE_GLISSANDO_START to the actual keyboard.
UR5_POSE_GLISSANDO_START = np.array([0.13, 0.51, 0.07, -1.45, -0.42, -0.64])
GLISSANDO_DIRECTION      = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
GLISSANDO_DISTANCE       = 0.23    # [m]
GLISSANDO_SPEED          = 0.05    # [m/s]

# ── Shared ─────────────────────────────────────────────────────────────────────
UR5_IP         = "192.168.1.10"
UR5_INIT_SPEED = 0.05    # [m/s]
UR5_INIT_ACCEL = 0.05    # [m/s²]

PRESS_POSE              = np.deg2rad([80.0, 50.0, 50.0])  # [MCP, PIP, DIP] flexion [rad]
SPREAD_ANGLE_DEG        = 0.0            # [deg]  finger abduction/adduction
PIANO_FINGERS_PLAYING   = ['index', 'ring']
PIANO_FINGERS_GLISSANDO = ['index', 'middle']

K_ROT       = 0.4    # [N·m/rad]  background (non-playing fingers)
K_ROT_PRESS = 0.05   # [N·m/rad]  playing fingers (task spring takes over)
B_ROT       = 0.01   # [N·m·s/rad]
B_ROT_HOLD  = 0.05   # [N·m·s/rad]  damping for held (non-playing) DOFs — suppresses oscillation

WRIST_PITCH_DEG = 20.0   # [deg]  hand wrist pitch reference (rotation about wrist -X axis;
                         #         old default was -20. Verify up/down direction on the rig).

# ── piano_playing_hand.py ─────────────────────────────────────────────────────
K_SWEEP             = [20.0, 100.0]           # [N/m]
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
