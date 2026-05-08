import numpy as np
from ModelIDFinger.finger_params import r_motor, r_pulley, c_param, a, b, c

# Units of measure:
# - LENGTHS: [m]
# - ANGLES: [rad]

def motor_to_joint(q_motor):
    """
    Convert motor angles to joint angles.

    Args:
        q_motor: [q_motor_MCP, q_motor_PIP] (rad)

    Returns:
        theta: [theta_MCP, theta_PIP, theta_DIP] (rad)
    """

    q_MCP, q_PIP = q_motor

    theta_MCP = (r_pulley/r_motor) * q_MCP
    theta_PIP = (r_motor/c_param) * q_PIP
    theta_DIP = theta_PIP  # Mimic constraint

    return np.array([theta_MCP, theta_PIP, theta_DIP])

def joint_to_motor(theta):
    """
    Convert joint angles to motor angles.

    Args:
        theta: [theta_MCP, theta_PIP] (rad)
               Note: theta_DIP is not needed as it's a mimic of theta_PIP

    Returns:
        q_motor: [q_motor_MCP, q_motor_PIP] (rad)
    """

    theta_MCP, theta_PIP = theta[:2]

    q_MCP = (r_motor/r_pulley) * theta_MCP
    q_PIP = (c_param/r_motor) * theta_PIP

    return np.array([q_MCP, q_PIP])

def FK_MCP(q_motor, r_local):
    """
    Forward kinematics for a point on the Proximal phalanx.

    Args:
        q_motor: [q_motor_MCP, q_motor_PIP] (rad)
        r_local: [x, y, z] in link frame (m)

    Returns:
        pos: [x, y, z] in world frame (m)
    """

    q_MCP = q_motor[0]
    theta_MCP = (r_pulley/r_motor) * q_MCP

    R_MCP = np.array([
        [1,  0,                0],
        [0,  np.cos(theta_MCP), -np.sin(theta_MCP)],
        [0,  np.sin(theta_MCP),  np.cos(theta_MCP)]
    ])

    return R_MCP @ r_local

def FK_PIP(q_motor, r_local):
    """
    Forward kinematics for a point on the Middle phalanx.

    Args:
        q_motor: [q_motor_MCP, q_motor_PIP] (rad)
        r_local: [x, y, z] in link frame (m)

    Returns:
        pos: [x, y, z] in world frame (m)
    """

    q_MCP, q_PIP = q_motor

    theta_MCP = (r_pulley/r_motor) * q_MCP
    theta_PIP = (r_motor/c_param) * q_PIP

    R_MCP = np.array([
        [1,  0,                0],
        [0,  np.cos(theta_MCP), -np.sin(theta_MCP)],
        [0,  np.sin(theta_MCP),  np.cos(theta_MCP)]
    ])

    p_PIP = R_MCP @ np.array([0, a, 0])

    theta_total = theta_MCP + theta_PIP
    R_PIP = np.array([
        [1,  0,                  0],
        [0,  np.cos(theta_total), -np.sin(theta_total)],
        [0,  np.sin(theta_total),  np.cos(theta_total)]
    ])

    return p_PIP + R_PIP @ r_local

def FK_DIP(q_motor, r_local):
    """
    Forward kinematics for a point on the Distal phalanx.

    Args:
        q_motor: [q_motor_MCP, q_motor_PIP] (rad)
        r_local: [x, y, z] in link frame (m)

    Returns:
        pos: [x, y, z] in world frame (m)
    """

    q_MCP, q_PIP = q_motor

    theta_MCP = (r_pulley/r_motor) * q_MCP
    theta_PIP = (r_motor/c_param) * q_PIP
    theta_DIP = theta_PIP  # Mimic constraint

    R_MCP = np.array([
        [1,  0,                0],
        [0,  np.cos(theta_MCP), -np.sin(theta_MCP)],
        [0,  np.sin(theta_MCP),  np.cos(theta_MCP)]
    ])

    p_PIP = R_MCP @ np.array([0, a, 0])

    theta_MCP_PIP = theta_MCP + theta_PIP
    R_PIP = np.array([
        [1,  0,                     0],
        [0,  np.cos(theta_MCP_PIP), -np.sin(theta_MCP_PIP)],
        [0,  np.sin(theta_MCP_PIP),  np.cos(theta_MCP_PIP)]
    ])

    p_DIP = p_PIP + R_PIP @ np.array([0, b, 0])

    theta_total = theta_MCP + theta_PIP + theta_DIP
    R_DIP = np.array([
        [1,  0,                   0],
        [0,  np.cos(theta_total), -np.sin(theta_total)],
        [0,  np.sin(theta_total),  np.cos(theta_total)]
    ])

    return p_DIP + R_DIP @ r_local

def FK(q_motor, link, r_local):
    """
    General forward kinematics function.

    Args:
        q_motor: [q_motor_MCP, q_motor_PIP] (rad)
        link: "MCP", "PIP", or "DIP"
        r_local: [x, y, z] in link frame (m)

    Returns:
        pos: [x, y, z] in world frame (m)
    """

    if link == "MCP":
        return FK_MCP(q_motor, r_local)
    elif link == "PIP":
        return FK_PIP(q_motor, r_local)
    elif link == "DIP":
        return FK_DIP(q_motor, r_local)
    else:
        raise ValueError(f"Unknown link: {link}. Must be 'MCP', 'PIP', or 'DIP'")

# Example usage
if __name__ == "__main__":
    q_motor = np.array([-0.5, -0.3])  # rad
    fingertip = np.array([0, c, 0])   # tip in distal link frame

    pos = FK_DIP(q_motor, fingertip)
    print(f"Fingertip position: {pos} m")

    pos = FK(q_motor, "DIP", fingertip)
    print(f"Fingertip position: {pos} m")
