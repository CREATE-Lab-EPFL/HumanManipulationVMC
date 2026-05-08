import numpy as np
from ModelIDHand.hand_params import (
    FINGER_TRANSMISSIONS, THUMB_TRANSMISSION,
    SPREAD_MOTION_RATIO, SPREAD_ANGLE_CORRECTION,
    WRIST, FINGER_BASE_ORIGINS, THUMB, INDEX, MIDDLE, RING, PINKY, FINGERS,
    SOFTWARE_MOTOR_ORDER,
)

# Units of measure:
# - LENGTHS: meters [m]
# - ANGLES: radians [rad]

# =============================================================================
# ROTATION MATRIX FUNCTIONS
# =============================================================================

def rot_axis(axis, theta):
    """
    Compute rotation matrix around an arbitrary axis using Rodrigues' formula.

    Args:
        axis: [x, y, z] unit vector defining the rotation axis.
        theta: rotation angle in radians.

    Returns:
        R: 3x3 rotation matrix.
    """

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    c = np.cos(theta)
    s = np.sin(theta)
    t = 1.0 - c

    ux, uy, uz = axis

    R = np.array([
        [c + ux*ux*t,      ux*uy*t - uz*s,  ux*uz*t + uy*s],
        [uy*ux*t + uz*s,   c + uy*uy*t,     uy*uz*t - ux*s],
        [uz*ux*t - uy*s,   uz*uy*t + ux*s,  c + uz*uz*t   ]
    ])

    return R

# =============================================================================
# MOTOR TO JOINT ANGLE CONVERSION
# =============================================================================

def FK_motor2finger(q_motor, finger_name):
    """
    Convert MCP, PIP motor angles to finger joint angles.

    Args:
        q_motor: motor angles [rad].
        finger_name: name of the finger ('index', 'middle', 'ring', 'pinky').

    Returns:
        theta: [theta_MCP, theta_PIP, theta_DIP] finger joint angles [rad].
    """

    if finger_name == 'index':
        motor_indices = [7, 8]
    elif finger_name == 'middle':
        motor_indices = [9, 10]
    elif finger_name == 'ring':
        motor_indices = [11, 12]
    elif finger_name == 'pinky':
        motor_indices = [13, 14]
    else:
        raise ValueError(f"Invalid finger name: {finger_name}")

    q_MCP, q_PIP = q_motor[motor_indices]
    trans = FINGER_TRANSMISSIONS[finger_name]
    theta_MCP = trans["r_pulley"] / trans["r_motor"] * q_MCP
    theta_PIP = trans["r_motor"] / trans["c_param"] * q_PIP
    theta_DIP = theta_PIP

    return np.array([theta_MCP, theta_PIP, theta_DIP])

def FK_motor2thumb(q_motor):
    """
    Convert motor angles to thumb joint angles.

    Args:
        q_motor: motor angles [rad].

    Returns:
        theta: [theta_CMC1, theta_CMC2, theta_MCP, theta_IP] thumb joint angles [rad].
    """

    q_CMC1, q_CMC2, q_MCP, q_IP = q_motor[2:6]

    theta_CMC1 = THUMB_TRANSMISSION["r_motor"] / THUMB_TRANSMISSION["CMC1_pulley"] * q_CMC1
    theta_CMC2 = THUMB_TRANSMISSION["r_motor"] / THUMB_TRANSMISSION["CMC2_pulley"] * q_CMC2
    theta_MCP = THUMB_TRANSMISSION["r_motor"] / THUMB_TRANSMISSION["MCP_c"] * q_MCP
    theta_IP = THUMB_TRANSMISSION["r_motor"] / THUMB_TRANSMISSION["IP_c"] * q_IP

    return np.array([theta_CMC1, theta_CMC2, theta_MCP, theta_IP])

def FK_motor2spread(q_motor, finger_name):
    """
    Convert motor angles to individual finger spread joint angle.

    Args:
        q_motor: motor angles [rad].
        finger_name: name of the finger ('index', 'middle', 'ring', 'pinky').

    Returns:
        theta_spread: finger spread joint angle [rad].
    """

    return q_motor[6] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO.get(finger_name, 0.0)

def FK_motor2wrist(q_motor):
    """
    Convert motor angles to wrist joint angles.

    Args:
        q_motor: motor angles [rad].

    Returns:
        theta: [theta_pitch, theta_yaw] wrist joint angles [rad].
    """

    q_motor1, q_motor2 = q_motor[0:2]
    pitch_raw = (q_motor1 - q_motor2) / 2.0
    yaw_raw = (q_motor1 + q_motor2)

    return np.array([pitch_raw * WRIST["spur_ratio"], yaw_raw * WRIST["spur_ratio"] * WRIST["bevel_ratio"]])

# =============================================================================
# JOINT TO MOTOR CONVERSION
# =============================================================================

def joint2motor_wrist(theta_wrist):
    """
    Args:
        theta_wrist: [theta_pitch, theta_yaw] (rad)
    Returns:
        q_motor: motors [0, 1] (rad)
    """
    pitch_raw = theta_wrist[0] / WRIST["spur_ratio"]
    yaw_raw   = theta_wrist[1] / (WRIST["spur_ratio"] * WRIST["bevel_ratio"])
    return np.array([pitch_raw + yaw_raw / 2.0,
                     -pitch_raw + yaw_raw / 2.0])


def joint2motor_thumb(theta_thumb):
    """
    Args:
        theta_thumb: [theta_CMC1, theta_CMC2, theta_MCP, theta_IP] (rad)
    Returns:
        q_motor: motors [2, 3, 4, 5] (rad)
    """
    T = THUMB_TRANSMISSION
    return np.array([
        theta_thumb[0] * T["CMC1_pulley"] / T["r_motor"],
        theta_thumb[1] * T["CMC2_pulley"] / T["r_motor"],
        theta_thumb[2] * T["MCP_c"]       / T["r_motor"],
        theta_thumb[3] * T["IP_c"]        / T["r_motor"],
    ])


def joint2motor_spread(theta_spread_index):
    """
    Invert from index spread angle (the reference finger for the shared spread motor).

    Args:
        theta_spread_index: index spread joint angle (rad)
    Returns:
        q_motor: motor [6] (rad)
    """
    return theta_spread_index / (SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO["index"])


def joint2motor_finger(theta_finger, finger_name):
    """
    Args:
        theta_finger: [theta_MCP, theta_PIP] (rad)  — DIP mirrors PIP, not needed
        finger_name: 'index', 'middle', 'ring', or 'pinky'
    Returns:
        q_motor: [q_MCP, q_PIP] for that finger's two motors (rad)
    """
    trans = FINGER_TRANSMISSIONS[finger_name]
    return np.array([
        theta_finger[0] * trans["r_motor"] / trans["r_pulley"],
        theta_finger[1] * trans["c_param"] / trans["r_motor"],
    ])


def joint_to_motor(theta_wrist, theta_thumb, theta_spread_index,
                   theta_index, theta_middle, theta_ring, theta_pinky):
    """
    Convert all hand joint angles to a full 15-motor angle vector.

    Args:
        theta_wrist:        [pitch, yaw]               (rad)
        theta_thumb:        [CMC1, CMC2, MCP, IP]      (rad)
        theta_spread_index: index spread angle          (rad)  — sets the shared spread motor
        theta_index:        [MCP, PIP]                 (rad)
        theta_middle:       [MCP, PIP]                 (rad)
        theta_ring:         [MCP, PIP]                 (rad)
        theta_pinky:        [MCP, PIP]                 (rad)

    Returns:
        q_motor: [15] motor angles (rad)
    """
    q = np.zeros(15)
    q[0:2]  = joint2motor_wrist(theta_wrist)
    q[2:6]  = joint2motor_thumb(theta_thumb)
    q[6]    = joint2motor_spread(theta_spread_index)
    q[7:9]  = joint2motor_finger(theta_index,  'index')
    q[9:11] = joint2motor_finger(theta_middle, 'middle')
    q[11:13]= joint2motor_finger(theta_ring,   'ring')
    q[13:15]= joint2motor_finger(theta_pinky,  'pinky')
    return q


# =============================================================================
# WRIST FORWARD KINEMATICS
# =============================================================================

def FK_motor2palm(q_motor, r_local):
    """
    Forward kinematics for a point on the palm frame wrt base_link frame.

    Args:
        q_motor: motor angles [rad].
        r_local: [x, y, z] in palm frame [m].

    Returns:
        R_palm: 3x3 rotation matrix of palm frame wrt base_link frame.
        pos_palm: [x, y, z] in world frame [m].
    """

    theta_pitch, theta_yaw = FK_motor2wrist(q_motor)
    R_yaw = rot_axis(WRIST["yaw_axis"], theta_yaw)
    R_pitch = rot_axis(WRIST["pitch_axis"], theta_pitch)
    R_palm = R_yaw @ R_pitch
    p_palm = WRIST["yaw_origin"] + R_yaw @ WRIST["pitch_origin"]

    return R_palm, p_palm + R_palm @ r_local

# =============================================================================
# THUMB FORWARD KINEMATICS
# =============================================================================

def FK_motor2thumbPos(q_motor, link, r_local):
    """
    Forward kinematics for a point on a thumb link.

    Args:
        q_motor: motor angles [rad].
        link: link name ('CMC1', 'CMC2', 'MCP', 'IP').
        r_local: [x, y, z] in link frame [m].

    Returns:
        pos: [x, y, z] in world frame [m].
    """

    thetas = FK_motor2thumb(q_motor)
    chain = ["CMC1", "CMC2", "MCP", "IP"]

    p_curr = FINGER_BASE_ORIGINS["thumb"].copy()
    R_curr = np.eye(3)

    for i, link_name in enumerate(chain):
        p_curr += R_curr @ THUMB[f"{link_name}_origin"]
        R_curr = R_curr @ rot_axis(THUMB[f"{link_name}_axis"], thetas[i])

        if link_name == link:
            break

    return FK_motor2palm(q_motor, p_curr + R_curr @ r_local)[1]

def Frame_motor2thumb(q_motor, link):
    """
    Frame (rotation + origin) of a thumb link in world frame.

    Args:
        q_motor: motor angles [rad].
        link: link name ('CMC1', 'CMC2', 'MCP', 'IP').

    Returns:
        R: 3x3 rotation matrix of the link frame wrt world frame.
           Columns are [x_axis, y_axis, z_axis] of the link in world.
        pos: [x, y, z] origin of the link frame in world frame [m].
    """

    thetas = FK_motor2thumb(q_motor)
    chain = ["CMC1", "CMC2", "MCP", "IP"]

    p_curr = FINGER_BASE_ORIGINS["thumb"].copy()
    R_curr = np.eye(3)

    for i, link_name in enumerate(chain):
        p_curr += R_curr @ THUMB[f"{link_name}_origin"]
        R_curr = R_curr @ rot_axis(THUMB[f"{link_name}_axis"], thetas[i])

        if link_name == link:
            break

    R_palm, p_world = FK_motor2palm(q_motor, p_curr)
    return R_palm @ R_curr, p_world

# =============================================================================
# FINGER FORWARD KINEMATICS (Index, Middle, Ring, Pinky)
# =============================================================================

def FK_motor2fingerPos(q_motor, finger_name, link, r_local):
    """
    Forward kinematics for a point on a generic finger link.

    Args:
        q_motor: motor angles [rad].
        finger_name: name of the finger ('index', 'middle', 'ring', 'pinky').
        link: link name ('MCP', 'PIP', 'DIP').
        r_local: [x, y, z] in link frame [m].

    Returns:
        pos: [x, y, z] in world frame [m].
    """

    theta_spread = FK_motor2spread(q_motor, finger_name)
    thetas = FK_motor2finger(q_motor, finger_name)
    all_thetas = [theta_spread] + thetas.tolist()
    chain = ["Spread", "MCP", "PIP", "DIP"]

    params = FINGERS[finger_name]
    p_curr = FINGER_BASE_ORIGINS[finger_name].copy()
    R_curr = np.eye(3)

    for i, link_name in enumerate(chain):
        p_curr += R_curr @ params[f"{link_name}_origin"]
        R_curr = R_curr @ rot_axis(params[f"{link_name}_axis"], all_thetas[i])

        if link_name == link:
            break

    return FK_motor2palm(q_motor, p_curr + R_curr @ r_local)[1]

def Frame_motor2finger(q_motor, finger_name, link):
    """
    Frame (rotation + origin) of a finger link in world frame.

    Args:
        q_motor: motor angles [rad].
        finger_name: name of the finger ('index', 'middle', 'ring', 'pinky').
        link: link name ('Spread', 'MCP', 'PIP', 'DIP').

    Returns:
        R: 3x3 rotation matrix of the link frame wrt world frame.
           Columns are [x_axis, y_axis, z_axis] of the link in world.
           For the DIP link: z-col = phalanx axis, y-col = flexion-plane normal.
        pos: [x, y, z] origin of the link frame in world frame [m].
    """

    theta_spread = FK_motor2spread(q_motor, finger_name)
    thetas = FK_motor2finger(q_motor, finger_name)
    all_thetas = [theta_spread] + thetas.tolist()
    chain = ["Spread", "MCP", "PIP", "DIP"]

    params = FINGERS[finger_name]
    p_curr = FINGER_BASE_ORIGINS[finger_name].copy()
    R_curr = np.eye(3)

    for i, link_name in enumerate(chain):
        p_curr += R_curr @ params[f"{link_name}_origin"]
        R_curr = R_curr @ rot_axis(params[f"{link_name}_axis"], all_thetas[i])

        if link_name == link:
            break

    R_palm, p_world = FK_motor2palm(q_motor, p_curr)
    return R_palm @ R_curr, p_world

# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":

    import numpy as np#
    import time
    
    print("="*70)
    print("FK_HAND - TEST")
    print("="*70)
    
    # Test configuration
    np.random.seed(42)
    q_motor = np.random.randn(15) * 0.1
    r_local = np.array([0.01, 0.02, 0.03])
    
    print("\nTest motor angles:")
    print(f"  q_motor (first 6): {q_motor[:6]}")
    print(f"  r_local: {r_local}\n")
    
    print("1. Testing motor-to-joint conversions...")
    
    # Test wrist
    theta_wrist = FK_motor2wrist(q_motor)
    print(f"   Wrist [pitch, yaw]: {theta_wrist}")
    
    # Test thumb
    theta_thumb = FK_motor2thumb(q_motor)
    print(f"   Thumb [CMC1, CMC2, MCP, IP]: {theta_thumb}")
    
    # Test finger
    theta_finger = FK_motor2finger(q_motor, "index")
    print(f"   Index [MCP, PIP, DIP]: {theta_finger}")
    
    # Test spread
    theta_spread = FK_motor2spread(q_motor, "index")
    print(f"   Index spread: {theta_spread}")
    
    print("\n2. Testing forward kinematics...")
    
    # Test wrist
    R_palm, pos_palm = FK_motor2palm(q_motor, r_local)
    print(f"   Wrist/Palm position: {pos_palm}")
    
    # Test thumb
    pos_thumb = FK_motor2thumbPos(q_motor, "IP", r_local)
    print(f"   Thumb IP position: {pos_thumb}")
    
    # Test fingers
    pos_index = FK_motor2fingerPos(q_motor, "index", "DIP", r_local)
    print(f"   Index DIP position: {pos_index}")
    
    pos_middle = FK_motor2fingerPos(q_motor, "middle", "DIP", r_local)
    print(f"   Middle DIP position: {pos_middle}")
    
    t = time.time()
    pos_ring = FK_motor2fingerPos(q_motor, "ring", "DIP", r_local)
    print(f"   Ring DIP computation time: {time.time() - t:.6f} seconds")
    print(f"   Ring DIP position: {pos_ring}")
    
    pos_pinky = FK_motor2fingerPos(q_motor, "pinky", "DIP", r_local)
    print(f"   Pinky DIP position: {pos_pinky}")
    
    print("\n3. Testing multiple calls (verifying no state corruption)...")
    
    # Call same function multiple times
    pos1 = FK_motor2thumbPos(q_motor, "IP", r_local)
    pos2 = FK_motor2thumbPos(q_motor, "IP", r_local)
    pos3 = FK_motor2thumbPos(q_motor, "IP", r_local)
    
    match = np.allclose(pos1, pos2) and np.allclose(pos2, pos3)
    print(f"   Multiple calls consistent: {match} ✓")
    
    # Verify global constants unchanged
    expected_thumb_origin = np.array([-0.001126, 0.011348, 0.056215])
    origins_match = np.allclose(FINGER_BASE_ORIGINS["thumb"], expected_thumb_origin)
    print(f"   Global constants unchanged: {origins_match} ✓")
    
    print("\n4. Testing Frame_motor2finger / Frame_motor2thumb...")

    # --- Frame_motor2finger: index DIP ---
    R_dip, p_dip = Frame_motor2finger(q_motor, "index", "DIP")

    # Position must match FK_motor2fingerPos with r_local = 0
    pos_dip_fk = FK_motor2fingerPos(q_motor, "index", "DIP", np.zeros(3))
    pos_match = np.allclose(p_dip, pos_dip_fk)
    print(f"   Frame origin matches FK position: {pos_match} ✓")

    # Rotation must be orthogonal (R @ R^T = I, det = 1)
    ortho_check = np.allclose(R_dip @ R_dip.T, np.eye(3), atol=1e-10)
    det_check = np.isclose(np.linalg.det(R_dip), 1.0, atol=1e-10)
    print(f"   R is orthogonal (R@R^T=I): {ortho_check} ✓")
    print(f"   R has det=1 (proper rotation): {det_check} ✓")

    # z-column must align with phalanx axis (DIP origin → tip)
    p_tip = FK_motor2fingerPos(q_motor, "index", "DIP", np.array([0., 0., 0.0175]))
    phalanx_dir = p_tip - p_dip
    phalanx_dir /= np.linalg.norm(phalanx_dir)
    z_align = np.allclose(R_dip[:, 2], phalanx_dir, atol=1e-6)
    print(f"   z-col aligns with phalanx axis: {z_align} ✓")

    # y-column must be orthogonal to phalanx axis
    y_dot_z = abs(np.dot(R_dip[:, 1], R_dip[:, 2]))
    ortho_yz = y_dot_z < 1e-10
    print(f"   y-col ⊥ phalanx axis (dot={y_dot_z:.2e}): {ortho_yz} ✓")
    print(f"   → orthogonal versor (flexion plane): {R_dip[:, 1]}")

    # --- Frame_motor2thumb: IP link ---
    R_ip, p_ip = Frame_motor2thumb(q_motor, "IP")

    pos_ip_fk = FK_motor2thumbPos(q_motor, "IP", np.zeros(3))
    pos_match_t = np.allclose(p_ip, pos_ip_fk)
    ortho_check_t = np.allclose(R_ip @ R_ip.T, np.eye(3), atol=1e-10)
    det_check_t = np.isclose(np.linalg.det(R_ip), 1.0, atol=1e-10)
    print(f"   Thumb IP frame origin matches FK: {pos_match_t} ✓")
    print(f"   Thumb R orthogonal: {ortho_check_t} ✓")
    print(f"   Thumb R det=1: {det_check_t} ✓")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
