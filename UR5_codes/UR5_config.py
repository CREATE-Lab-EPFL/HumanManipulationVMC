"""
UR5 Initial Configuration
=========================
This file contains the initial pose configurations for the UR5 robot.
Import this file in scripts that need the UR5 starting position.

Target pose format: [x, y, z, rx, ry, rz]
- x, y, z: position in meters
- rx, ry, rz: rotation vector in radians (axis-angle representation)
"""

import numpy as np

# Normal experiment configuration (vertical press — UR5 descends in Z)
UR5_POSE = np.array([0.0, 0.70, 0.085, 0.04, -2.18, -2.18])  # [x, y, z, rx, ry, rz]

# Finger target pose (for VMC bending control)
FINGER_TARGET = np.array([np.deg2rad(30.0), np.deg2rad(30.0), np.deg2rad(30.0)])

# Default pose (for backward compatibility)
UR5_INIT_POSE = UR5_POSE

# UR5 connection settings
UR5_IP = "192.168.1.10"

# Movement parameters
UR5_INIT_SPEED = 0.05           # m/s
UR5_INIT_ACCELERATION = 0.05    # m/s²

# Start experiment again
FINGER_STRAIGHT = np.array([np.deg2rad(-1.0), np.deg2rad(-1.0), np.deg2rad(-1.0)])

# Bending damping for VMC
BENDING_DAMPING   = 0.003       # [N·m·s/rad] — damping during bending/return phases

# Distances / speeds
UR5_DESCENT       = 0.02        # [m]   — vertical press depth
UR5_DESCENT_SPEED = 0.002       # [m/s] — quasistatic descent speed (both directions)

# Experiment repetitions (shared across all finger experiments)
N_RUNS = 5                       # repetitions per condition#
