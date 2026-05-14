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
