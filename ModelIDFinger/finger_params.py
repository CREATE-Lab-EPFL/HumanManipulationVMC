## Finger Model Parameters
## ========================
## Single source of truth for all physical and identified parameters
## of the robotic finger. Import from here instead of hardcoding.
##
## Units: SI — lengths [m], angles [rad], masses [kg], torques [N·m]

import numpy as np

# =============================================================================
# Motor transmission (identified from ModelID experiments)
# =============================================================================
r_motor  = 0.005    # motor pulley radius [m]
r_pulley = 0.00223  # finger pulley radius [m]
c_param  = 0.00813  # PIP cable-driven transmission constant [m]

# =============================================================================
# Link lengths (from URDF / physical measurement)
# =============================================================================
a = 0.040   # proximal phalanx length (MCP → PIP) [m]
b = 0.030   # middle phalanx length   (PIP → DIP) [m]
c = 0.0175  # distal phalanx length   (DIP → tip) [m]

# =============================================================================
# Link masses (from URDF / physical measurement)
# =============================================================================
mass_MCP = 0.0057  # proximal phalanx [kg]
mass_PIP = 0.0040  # middle phalanx   [kg]
mass_DIP = 0.025   # distal phalanx   [kg]

# =============================================================================
# Centers of gravity in link frame (from URDF)
# =============================================================================
cog_MCP = np.array([-0.000639,  0.020,  0.001276])
cog_PIP = np.array([ 0.000925,  0.015,  0.001138])
cog_DIP = np.array([-0.000716,  0.0105, 0.000955])

# =============================================================================
# World-to-finger rotation
# =============================================================================
R_world_to_finger = np.array([[1,  0,  0],
                               [0, -1,  0],
                               [0,  0, -1]])

# =============================================================================
# Joint limits (joint angles, not motor angles)
# =============================================================================
joint_limits = {
    'MCP': (0.0, 1.570796),  # [0, π/2] rad
    'PIP': (0.0, 1.22173),
    'DIP': (0.0, 1.22173),
}

# =============================================================================
# Joint limit spring
# =============================================================================
limit_stiffness = 1.0  # [N·m/rad]

# =============================================================================
# Torque limits
# =============================================================================
goal_torque_limit = 0.8  # [N·m]

# =============================================================================
# Friction compensation (Stribeck model, fitted from ModelIDFinger experiments)
# τ_friction = friction_max * exp(-(v/friction_vlim)²) * sign(τ_commanded)
# =============================================================================
friction_max  = 0.08  # max static friction torque [N·m]
friction_vlim = 0.03  # Stribeck velocity          [rad/s]

# =============================================================================
# Motor efficiency (fitted from ProprioceptiveSensing/plot_finger_eta.ipynb)
# Two-η model: one efficiency per motor axis
# =============================================================================
ETA_MCP = 0.8373   # MCP motor efficiency [-]
ETA_PIP = 0.3594   # PIP motor efficiency [-]
eta = np.array([ETA_MCP, ETA_PIP])
