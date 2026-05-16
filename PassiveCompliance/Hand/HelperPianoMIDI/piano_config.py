"""
UR5 poses and shared constants for piano experiments.

All poses: [x, y, z, rx, ry, rz] — position in metres, rotation as axis-angle vector.

piano_playing_hand  — hand horizontal above keys, index and ring aligned with two keys
piano_glissando     — hand tilted ~15 deg, index and middle side-by-side, positioned
                      at the left end of the glissando range so the UR5 slides rightward
"""

import numpy as np

# ── Piano playing (conditions 1 & 2) ──────────────────────────────────────────
# Hand horizontal, palm facing down, index over one key, ring over another key
# ~3 cm above the keyboard surface.  PLACEHOLDER — tune to actual keyboard height.
UR5_POSE_PIANO = np.array([0.0, 0.50, 0.120, 0.04, -2.18, -2.18])

# ── Glissando ─────────────────────────────────────────────────────────────────
# Hand slightly tilted forward (~15 deg pitch), index and middle side-by-side,
# positioned at the left end of the glissando range.  PLACEHOLDER — tune in place.
UR5_POSE_GLISSANDO_START = np.array([0.0, 0.42, 0.110, -0.30, -2.18, -2.18])

# Direction of the UR5 slide (unit vector in base-frame, in pose-vector space).
# [dx, dy, dz, 0, 0, 0] — pure translation along the keyboard (Y axis here).
GLISSANDO_DIRECTION = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

# Total slide distance [m] (~10 white keys at ~23 mm spacing ≈ 0.23 m).
GLISSANDO_DISTANCE = 0.23   # [m]

# Slide speed [m/s] — quasistatic, similar to UR5_DESCENT_SPEED in finger experiments.
GLISSANDO_SPEED = 0.05      # [m/s]

# ── Shared movement parameters ────────────────────────────────────────────────
UR5_IP           = "192.168.1.10"
UR5_INIT_SPEED   = 0.05     # [m/s]
UR5_INIT_ACCEL   = 0.05     # [m/s²]

# ── Press pose ────────────────────────────────────────────────────────────────
# MCP + PIP flexion angle [deg] used to define the fingertip press target via FK.
PRESS_ANGLE_DEG = 30.0

# Fingers active in each experiment (used to build Q_PRESS and VMC targets).
PIANO_FINGERS_PLAYING   = ['index', 'ring']
PIANO_FINGERS_GLISSANDO = ['index', 'middle']

# ── Shared joint regulation (both piano experiments) ──────────────────────────
K_ROT = 0.1     # [N·m/rad]    background joint stiffness
B_ROT = 0.001   # [N·m·s/rad]  background joint damping

# ── piano_playing_hand.py — stiffness levels and timing ──────────────────────
K_SWEEP           = [10.0, 20.0, 100.0]   # [N/m]  cart stiffness sweep (uniform condition)
K_STIFF           = 100.0                 # [N/m]  stiff finger (heterogeneous — index)
K_SOFT            = 10.0                  # [N/m]  soft finger  (heterogeneous — ring)
B_CART_PLAYING    = 0.5                   # [N·s/m] task-space damping
PRESS_FREQUENCY   = 1.0                   # [Hz]   one full press-lift cycle per second
N_CYCLES          = 10                    # cycles recorded per stiffness value
PLAYING_SETTLE_TIME = 3.0                 # [s]
REF_RAMP_DURATION = 0.1                   # [s]    linear ramp duration for REST↔PRESS target

# ── piano_glissando.py — stiffness and timing ────────────────────────────────
K_CART              = 50.0    # [N/m]    fingertip stiffness during glissando
B_CART_GLISSANDO    = 1.0     # [N·s/m]  fingertip damping
N_RUNS              = 5
GLISSANDO_SETTLE_TIME = 3.0   # [s]
