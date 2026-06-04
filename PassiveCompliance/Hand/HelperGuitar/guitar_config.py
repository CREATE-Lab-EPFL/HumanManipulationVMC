"""Shared constants for the guitar experiment (ADAPT Hand + UR5 + microphone).

All fingers except the thumb are held closed at FINGER_CLOSED_POSE by a torsional
(joint-space) spring. The experiment compares TORSIONAL_SPRINGS values: for each,
the UR5 strums SWEEP_VECTOR on the XY plane N_RUNS times. After each strum the arm
lifts by LIFT, returns to the start position at height, then descends — fingers
stay closed throughout. Stiffness is changed online between conditions.
No friction compensation, as in the piano experiment.
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
FINGER_CLOSED_POSE = np.deg2rad([30.0, 30.0, 30.0])   # [MCP, PIP, DIP]
CLOSED_FINGERS     = ['index', 'middle', 'ring', 'pinky']
SPREAD_ANGLE_DEG   = 0.0

TORSIONAL_SPRINGS  = [0.1, 0.3]   # [N·m/rad]  compared conditions
B_ROT              = 0.01   # [N·m·s/rad]  uniform joint damping (as in piano)
K_ROT              = 0.4    # [N·m/rad]  ramp / background stiffness
WRIST_PITCH_DEG    = 0.0
WRIST_K_FIX        = 8.0    # [N·m/rad]  wrist held near-rigid
WRIST_B_FIX        = 0.5    # [N·m·s/rad]  matched damping (as in piano)

FRICTION_TAU_MAX   = 0.0    # no friction compensation (as in piano)

RAMP_DURATION  = 3.0    # [s]  stiffness / pose ramp duration
SETTLE_TIME    = 2.0    # [s]  settle after initial close, before first run
N_RUNS         = 3      # strums per stiffness condition

# ── Microphone / audio ──────────────────────────────────────────────────────────
SAMPLE_RATE      = 44100  # [Hz]
AUDIO_CHANNELS   = 1      # mono
AUDIO_BLOCKSIZE  = 1024   # frames per callback block (~23 ms at 44.1 kHz)
MIC_DEVICE       = None   # None = default input device, or a name substring
ONSET_THRESHOLD  = 0.05   # RMS level (0..1) for onset detection in mic_controller
ONSET_REFRACTORY = 0.10   # [s]  minimum gap between detected onsets
