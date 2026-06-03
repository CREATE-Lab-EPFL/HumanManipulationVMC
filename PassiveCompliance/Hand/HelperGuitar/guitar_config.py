"""Shared constants for the guitar experiment (ADAPT Hand + UR5 + microphone).

All fingers except the thumb are held closed at FINGER_CLOSED_POSE by a torsional
(joint-space) spring. Three spring stiffnesses (TORSIONAL_SPRINGS) are compared:
for each, the UR5 moves linearly from UR5_POSE_GUITAR_START to UR5_POSE_GUITAR_END
(dragging the closed hand across the strings) N_RUNS times while a microphone
records the sound. The recorded output is the microphone INTENSITY (audio level).
No friction compensation, as in the piano experiment.
"""

import numpy as np

# ── UR5 sweep: linear move from START to END ────────────────────────────────────
# PLACEHOLDER — tune to the actual guitar mounting.
UR5_POSE_GUITAR_START = np.array([0.0, 0.61, 0.22, -1.66, -0.61, -0.54])
UR5_POSE_GUITAR_END   = np.array([0.0, 0.52, 0.22, -1.66, -0.61, -0.54])
SWEEP_SPEED           = 0.05    # [m/s]  linear sweep speed (must be in [0, 3])
SWEEP_ACCEL           = 0.5     # [m/s²]

UR5_IP         = "192.168.1.10"
UR5_INIT_SPEED = 0.05    # [m/s]
UR5_INIT_ACCEL = 0.05    # [m/s²]

# ── Hand pose / compliance ──────────────────────────────────────────────────────
# All fingers EXCEPT the thumb are held closed at this flexion (customizable):
FINGER_CLOSED_POSE     = np.deg2rad([30.0, 30.0, 30.0])   # [MCP, PIP, DIP]
CLOSED_FINGERS         = ['index', 'middle', 'ring', 'pinky']
SPREAD_ANGLE_DEG       = 0.0         # [deg]  finger abduction/adduction

# Torsional (joint-space) spring stiffnesses to compare:
TORSIONAL_SPRINGS      = [0.1, 0.5]   # [N·m/rad]
B_ROT                  = 0.02   # [N·m·s/rad]  joint damping
K_ROT                  = 0.4    # [N·m/rad]  background stiffness (thumb, wrist, spreads)
B_ROT_HOLD             = 0.05   # [N·m·s/rad]  damping for K=0 (non-playing) DOFs
WRIST_PITCH_DEG        = 0.0    # [deg]  wrist pitch reference
WRIST_K_FIX            = 8.0    # [N·m/rad]  wrist held (near-)rigid
WRIST_B_FIX            = 0.3    # [N·m·s/rad]

FRICTION_TAU_MAX       = 0.0    # [N·m]  no friction compensation (as in the piano)

RAMP_DURATION          = 3.0    # [s]  gradual stiffness / pose transitions
SETTLE_TIME            = 2.0    # [s]  after stiffness change, before first run
N_RUNS                 = 3      # sweeps per torsional-spring condition

# ── Microphone / audio ──────────────────────────────────────────────────────────
SAMPLE_RATE      = 44100  # [Hz]
AUDIO_CHANNELS   = 1      # mono
AUDIO_BLOCKSIZE  = 1024   # frames per callback block (~23 ms at 44.1 kHz)
MIC_DEVICE       = None   # None = default input device, or a name substring
ONSET_THRESHOLD  = 0.05   # RMS level (0..1) for onset detection in mic_controller
ONSET_REFRACTORY = 0.10   # [s]  minimum gap between detected onsets
