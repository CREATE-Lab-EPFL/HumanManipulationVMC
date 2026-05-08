## Hand Model Parameters
## ======================
## Single source of truth for all physical and identified parameters
## of the robotic hand. Import from here instead of hardcoding.
##
## Units: SI - lengths [m], angles [rad], masses [kg], torques [N*m]

import numpy as np

# =============================================================================
# Motor transmission (identified from ModelID experiments)
# =============================================================================
FINGER_TRANSMISSIONS = {
    "index": {
        "r_motor":  0.005,    # motor pulley radius [m]
        "r_pulley": 0.00223,  # finger pulley radius [m]
        "c_param":  0.00810,  # PIP transmission constant [m]
    },
    "middle": {
        "r_motor":  0.005,    # motor pulley radius [m]
        "r_pulley": 0.00223,  # finger pulley radius [m]
        "c_param":  0.00813,  # PIP transmission constant [m]
    },
    "ring": {
        "r_motor":  0.005,    # motor pulley radius [m]
        "r_pulley": 0.00223,  # finger pulley radius [m]
        "c_param":  0.00813,  # PIP transmission constant [m]
    },
    "pinky": {
        "r_motor":  0.005,    # motor pulley radius [m]
        "r_pulley": 0.00223,  # finger pulley radius [m]
        "c_param":  0.00813,  # PIP transmission constant [m]
    },
}

THUMB_TRANSMISSION = {
    "r_motor":    0.005,     # motor pulley radius [m]
    "CMC1_pulley": 0.01478,  # CMC1 pulley radius [m]
    "CMC2_pulley": 0.01475,  # CMC2 pulley radius [m]
    "MCP_c":      0.00903,   # MCP transmission constant [m]
    "IP_c":       0.00983,   # IP transmission constant [m]
}

SPREAD_MOTION_RATIO = {
    "index":  1.0,
    "middle": 0.0,
    "ring":  -1.0,
    "pinky": -1.5,
}
SPREAD_ANGLE_CORRECTION = -1.0 / 30.0

# =============================================================================
# URDF parameters (from ADAPT_Hand.urdf)
# =============================================================================

WRIST = {
    "yaw_origin":   np.array([-0.002826,  0.018021, -0.027987]),
    "yaw_axis":     np.array([ 0.0,        1.0,       0.000749]),
    "pitch_origin": np.array([-0.015,     -0.015,    -1.1e-05]),
    "pitch_axis":   np.array([-1.0,        0.0,       0.0]),
    "spur_ratio":   0.8,
    "bevel_ratio":  0.85714286,
}

FINGER_BASE_ORIGINS = {
    "thumb":  np.array([-0.001126,  0.011348,  0.056215]),
    "index":  np.array([-0.001654,  0.040836,  0.093101]),
    "middle": np.array([ 0.02173,   0.039146,  0.092417]),
    "ring":   np.array([ 0.032774,  0.039367,  0.090557]),
    "pinky":  np.array([ 0.061943,  0.039534,  0.075302]),
}

THUMB = {
    "CMC1_origin": np.array([-0.014257,  0.010829, -0.007467]),
    "CMC1_axis":   np.array([ 0.182125, -0.177737, -0.967078]),
    "CMC2_origin": np.array([-0.008631,  0.00194,  -0.008807]),
    "CMC2_axis":   np.array([ 0.0,       0.983527, -0.180761]),
    "MCP_origin":  np.array([-0.04412,  -0.000957, -0.00886]),
    "MCP_axis":    np.array([ 0.168863,  0.20364,  -0.964373]),
    "IP_origin":   np.array([-0.031464, -0.001054, -0.005732]),
    "IP_axis":     np.array([ 0.168863,  0.20364,  -0.964373]),
}

INDEX = {
    "Spread_origin": np.array([-0.006284,  0.000712,  0.005798]),
    "Spread_axis":   np.array([ 0.140517,  0.943569, -0.299886]),
    "MCP_origin":    np.array([-0.001682, -0.00044,   0.009499]),
    "MCP_axis":      np.array([-0.981214,  0.092278, -0.169421]),
    "PIP_origin":    np.array([-0.005287,  0.012722,  0.037552]),
    "PIP_axis":      np.array([-0.981214,  0.092278, -0.169421]),
    "DIP_origin":    np.array([-0.003966,  0.009542,  0.028164]),
    "DIP_axis":      np.array([-0.981214,  0.092278, -0.169421]),
}

MIDDLE = {
    "Spread_origin": np.array([-0.005011,  0.000697,  0.006929]),
    "Spread_axis":   np.array([ 0.036359,  0.936295, -0.349327]),
    "MCP_origin":    np.array([ 0.000289, -0.000149,  0.009651]),
    "MCP_axis":      np.array([-0.998268,  0.050205,  0.030661]),
    "PIP_origin":    np.array([ 0.002081,  0.015643,  0.042143]),
    "PIP_axis":      np.array([-0.998268,  0.050205,  0.030661]),
    "DIP_origin":    np.array([ 0.001387,  0.010428,  0.028095]),
    "DIP_axis":      np.array([-0.998268,  0.050205,  0.030661]),
}

RING = {
    "Spread_origin": np.array([ 0.006841,  0.000228,  0.005174]),
    "Spread_axis":   np.array([-0.055584,  0.937022, -0.34482]),
    "MCP_origin":    np.array([ 0.002501, -0.000155,  0.009326]),
    "MCP_axis":      np.array([-0.964998,  0.038236,  0.259457]),
    "PIP_origin":    np.array([ 0.010252,  0.013887,  0.036084]),
    "PIP_axis":      np.array([-0.964998,  0.038236,  0.259457]),
    "DIP_origin":    np.array([ 0.007689,  0.010415,  0.027062]),
    "DIP_axis":      np.array([-0.964998,  0.038236,  0.259457]),
}

PINKY = {
    "Spread_origin": np.array([-0.001073, -0.000153,  0.008511]),
    "Spread_axis":   np.array([-0.211896,  0.947676, -0.238768]),
    "MCP_origin":    np.array([ 0.004931, -0.000503,  0.008288]),
    "MCP_axis":      np.array([-0.859324, -0.064308,  0.507372]),
    "PIP_origin":    np.array([ 0.014895,  0.010006,  0.026496]),
    "PIP_axis":      np.array([-0.859324, -0.064308,  0.507372]),
    "DIP_origin":    np.array([ 0.009309,  0.006254,  0.016559]),
    "DIP_axis":      np.array([-0.859324, -0.064308,  0.507372]),
}

FINGERS = {"thumb": THUMB, "index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}

SOFTWARE_MOTOR_ORDER = [
    'wrist_motor1',  # [0]
    'wrist_motor2',  # [1]
    'thumb_CMC1',    # [2]
    'thumb_CMC2',    # [3]
    'thumb_MCP',     # [4]
    'thumb_IP',      # [5]
    'spread',        # [6]
    'index_MCP',     # [7]
    'index_PIP',     # [8]
    'middle_MCP',    # [9]
    'middle_PIP',    # [10]
    'ring_MCP',      # [11]
    'ring_PIP',      # [12]
    'pinky_MCP',     # [13]
    'pinky_PIP',     # [14]
]

# =============================================================================
# Link masses (from URDF / physical measurement)  [kg]
# =============================================================================
MASSES = {
    'wrist_joint':       0.0104,
    'palm':              0.0808,
    'thumb_base':        0.0035,
    'thumb_2dof_joint':  0.0070,
    'thumb_proximal':    0.0068,
    'thumb_middle':      0.0043,
    'thumb_distal':      0.0035,
    'index_base':        0.0014,
    'index_2dof_joint':  0.0011,
    'index_proximal':    0.0057,
    'index_middle':      0.0040,
    'index_distal':      0.0025,
    'middle_base':       0.0014,
    'middle_2dof_joint': 0.0011,
    'middle_proximal':   0.0066,
    'middle_middle':     0.0040,
    'middle_distal':     0.0025,
    'ring_base':         0.0014,
    'ring_2dof_joint':   0.0011,
    'ring_proximal':     0.0057,
    'ring_middle':       0.0040,
    'ring_distal':       0.0025,
    'pinky_base':        0.0014,
    'pinky_2dof_joint':  0.0011,
    'pinky_proximal':    0.0043,
    'pinky_middle':      0.0022,
    'pinky_distal':      0.0025,
}

# =============================================================================
# Centers of gravity in link frame (from URDF)  [m]
# =============================================================================
COG = {
    'wrist_joint':       np.array([-2.553e-07, -0.0150,  -1.074e-05]),
    'palm':              np.array([ 0.0193,     0.0162,   0.0492]),
    'thumb_base':        np.array([-0.0059,     0.0034,  -0.0113]),
    'thumb_2dof_joint':  np.array([-0.0027,    -0.0041,  -0.0065]),
    'thumb_proximal':    np.array([-0.0210,    -0.0001,  -0.0038]),
    'thumb_middle':      np.array([-0.0156,     0.0007,  -0.0034]),
    'thumb_distal':      np.array([-0.0130,     0.0005,  -0.0016]),
    'index_base':        np.array([-0.0058,    -0.0049,   1.46e-05]),
    'index_2dof_joint':  np.array([-0.0021,    -0.0022,   0.0035]),
    'index_proximal':    np.array([-0.0018,     0.0075,   0.0185]),
    'index_middle':      np.array([-0.0027,     0.0059,   0.0135]),
    'index_distal':      np.array([-0.0005,     0.0041,   0.0096]),
    'middle_base':       np.array([-0.0054,    -0.0051,   0.0013]),
    'middle_2dof_joint': np.array([-0.0012,    -0.0022,   0.0039]),
    'middle_proximal':   np.array([ 0.0016,     0.0090,   0.0205]),
    'middle_middle':     np.array([-0.0001,     0.0063,   0.0136]),
    'middle_distal':     np.array([ 0.0012,     0.0045,   0.0094]),
    'ring_base':         np.array([ 0.0051,    -0.0055,  -0.0001]),
    'ring_2dof_joint':   np.array([-0.0003,    -0.0022,   0.0041]),
    'ring_proximal':     np.array([ 0.0056,     0.0081,   0.0174]),
    'ring_middle':       np.array([ 0.0028,     0.0063,   0.0133]),
    'ring_distal':       np.array([ 0.0033,     0.0045,   0.0089]),
    'pinky_base':        np.array([-0.0037,    -0.0057,   0.0033]),
    'pinky_2dof_joint':  np.array([ 0.0009,    -0.0024,   0.0038]),
    'pinky_proximal':    np.array([ 0.0079,     0.0061,   0.0125]),
    'pinky_middle':      np.array([ 0.0030,     0.0037,   0.0089]),
    'pinky_distal':      np.array([ 0.0052,     0.0042,   0.0081]),
}

# =============================================================================
# Fingertip offsets in DIP/IP local frame [m]  (z-axis = phalanx axis)
# =============================================================================
FINGER_TIP_OFFSETS = {
    'thumb':  np.array([0.0, 0.0, 0.0175]),
    'index':  np.array([0.0, 0.0, 0.0175]),
    'middle': np.array([0.0, 0.0, 0.0175]),
    'ring':   np.array([0.0, 0.0, 0.0175]),
    'pinky':  np.array([0.0, 0.0, 0.0175]),
}

# =============================================================================
# Joint limits  [rad]
# =============================================================================
JOINT_LIMITS = {
    'wrist_pitch':  (-0.5,  0.5),
    'wrist_yaw':    (-0.5,  0.5),
    'thumb_CMC1':   ( 0.0,  1.5),
    'thumb_CMC2':   (-0.5,  0.5),
    'thumb_MCP':    ( 0.0,  1.5),
    'thumb_IP':     ( 0.0,  1.2),
    'index_spread': (-0.3,  0.0),
    'index_MCP':    ( 0.0,  1.5),
    'index_PIP':    ( 0.0,  1.2),
    'index_DIP':    ( 0.0,  1.2),
    'middle_spread': (0.0,  0.0),
    'middle_MCP':   ( 0.0,  1.5),
    'middle_PIP':   ( 0.0,  1.2),
    'middle_DIP':   ( 0.0,  1.2),
    'ring_spread':  ( 0.0,  0.3),
    'ring_MCP':     ( 0.0,  1.5),
    'ring_PIP':     ( 0.0,  1.2),
    'ring_DIP':     ( 0.0,  1.2),
    'pinky_spread': ( 0.0,  0.3),
    'pinky_MCP':    ( 0.0,  1.5),
    'pinky_PIP':    ( 0.0,  1.2),
    'pinky_DIP':    ( 0.0,  1.2),
}

# =============================================================================
# Motor efficiency
# One η per motor axis, ordered as SOFTWARE_MOTOR_ORDER
# =============================================================================
ETA_WRIST1    = 1.0   # wrist_motor1  [0]
ETA_WRIST2    = 1.0   # wrist_motor2  [1]
ETA_THUMB_CMC1 = 1.0  # thumb_CMC1    [2]
ETA_THUMB_CMC2 = 1.0  # thumb_CMC2    [3]
ETA_THUMB_MCP  = 1.0  # thumb_MCP     [4]
ETA_THUMB_IP   = 1.0  # thumb_IP      [5]
ETA_SPREAD     = 1.0  # spread        [6]
ETA_INDEX_MCP  = 1.0  # index_MCP     [7]
ETA_INDEX_PIP  = 1.0  # index_PIP     [8]
ETA_MIDDLE_MCP = 1.0  # middle_MCP    [9]
ETA_MIDDLE_PIP = 1.0  # middle_PIP    [10]
ETA_RING_MCP   = 1.0  # ring_MCP      [11]
ETA_RING_PIP   = 1.0  # ring_PIP      [12]
ETA_PINKY_MCP  = 1.0  # pinky_MCP     [13]
ETA_PINKY_PIP  = 1.0  # pinky_PIP     [14]

eta = np.array([
    ETA_WRIST1,     # [0]  wrist_motor1
    ETA_WRIST2,     # [1]  wrist_motor2
    ETA_THUMB_CMC1, # [2]  thumb_CMC1
    ETA_THUMB_CMC2, # [3]  thumb_CMC2
    ETA_THUMB_MCP,  # [4]  thumb_MCP
    ETA_THUMB_IP,   # [5]  thumb_IP
    ETA_SPREAD,     # [6]  spread
    ETA_INDEX_MCP,  # [7]  index_MCP
    ETA_INDEX_PIP,  # [8]  index_PIP
    ETA_MIDDLE_MCP, # [9]  middle_MCP
    ETA_MIDDLE_PIP, # [10] middle_PIP
    ETA_RING_MCP,   # [11] ring_MCP
    ETA_RING_PIP,   # [12] ring_PIP
    ETA_PINKY_MCP,  # [13] pinky_MCP
    ETA_PINKY_PIP,  # [14] pinky_PIP
])

# =============================================================================
# Joint limit spring
# =============================================================================
limit_stiffness = 0.6  # [N·m/rad]

# =============================================================================
# Maximum torques
# =============================================================================
goal_limit_torque_wrist = 0.05  # [N·m]
goal_limit_torque = 0.8          # [N·m]

# =============================================================================
# Friction compensation (Stribeck model, fitted from ModelIDHand experiments)
# τ_friction = friction_max * exp(-(v/friction_vlim)²) * sign(τ_commanded)
# =============================================================================
friction_max  = 0.10  # max static friction torque [N·m]
friction_vlim = 0.03  # Stribeck velocity          [rad/s]

# =============================================================================
# Hand properties for UR5-based gravity compensation
# =============================================================================
hand_mass = 1.270     # [kg]
Cx = 8.00             # [mm]
Cy = -2.00            # [mm]
Cz = 77.00            # [mm]

# =============================================================================
# Rotation of the hand around the UR5 flange z-axis (the finger direction).
# This is the in-plane rotation between the hand base frame and the flange
# screw pattern — measure it physically and set it here.
# =============================================================================
HAND_MOUNTING_ANGLE = np.deg2rad(45)  # [rad]
