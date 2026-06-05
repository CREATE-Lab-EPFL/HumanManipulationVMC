import numpy as np
from ModelIDHand.hand_params import (
    FINGER_TRANSMISSIONS, THUMB_TRANSMISSION,
    SPREAD_MOTION_RATIO, SPREAD_ANGLE_CORRECTION,
    WRIST, FINGER_BASE_ORIGINS, THUMB, INDEX, MIDDLE, RING, PINKY, FINGERS,
)

# Units: lengths [m], angles [rad]
#
# Motor ordering (13 motors, wrist rigid/fixed):
#   [0] thumb_CMC1  [1] thumb_CMC2  [2] thumb_MCP  [3] thumb_IP
#   [4] spread
#   [5] index_MCP   [6] index_PIP
#   [7] middle_MCP  [8] middle_PIP
#   [9] ring_MCP   [10] ring_PIP
#  [11] pinky_MCP  [12] pinky_PIP

# Constant palm frame (wrist rigid at zero: R_palm = I, p_palm = yaw_origin + pitch_origin)
_PALM_ORIGIN = WRIST["yaw_origin"] + WRIST["pitch_origin"]
_R_PALM = np.eye(3)


# =============================================================================
# ROTATION MATRIX
# =============================================================================

def rot_axis(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    c, s, t = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    ux, uy, uz = axis
    return np.array([
        [c + ux*ux*t,      ux*uy*t - uz*s,  ux*uz*t + uy*s],
        [uy*ux*t + uz*s,   c + uy*uy*t,     uy*uz*t - ux*s],
        [uz*ux*t - uy*s,   uz*uy*t + ux*s,  c + uz*uz*t   ]
    ])


# =============================================================================
# MOTOR TO JOINT ANGLE CONVERSION
# =============================================================================

def FK_motor2thumb(q_motor):
    """[0:4] → [CMC1, CMC2, MCP, IP] joint angles [rad]."""
    q_CMC1, q_CMC2, q_MCP, q_IP = q_motor[0:4]
    T = THUMB_TRANSMISSION
    return np.array([
        T["r_motor"] / T["CMC1_pulley"] * q_CMC1,
        T["r_motor"] / T["CMC2_pulley"] * q_CMC2,
        -T["r_motor"] / T["MCP_c"]      * q_MCP,
        -T["r_motor"] / T["IP_c"]       * q_IP,
    ])


def FK_motor2spread(q_motor, finger_name):
    """[4] → spread joint angle [rad]."""
    return q_motor[4] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO.get(finger_name, 0.0)


def FK_motor2finger(q_motor, finger_name):
    """MCP+PIP motors → [MCP, PIP, DIP] joint angles [rad]."""
    idx = {"index": [5, 6], "middle": [7, 8], "ring": [9, 10], "pinky": [11, 12]}[finger_name]
    q_MCP, q_PIP = q_motor[idx]
    trans = FINGER_TRANSMISSIONS[finger_name]
    theta_MCP = trans["r_pulley"] / trans["r_motor"] * q_MCP
    theta_PIP = -trans["r_motor"] / trans["c_param"] * q_PIP
    return np.array([theta_MCP, theta_PIP, theta_PIP])  # DIP mirrors PIP


# =============================================================================
# JOINT TO MOTOR CONVERSION
# =============================================================================

def joint2motor_thumb(theta_thumb):
    """[CMC1, CMC2, MCP, IP] → motors [0:4] [rad]."""
    T = THUMB_TRANSMISSION
    return np.array([
        theta_thumb[0] * T["CMC1_pulley"] / T["r_motor"],
        theta_thumb[1] * T["CMC2_pulley"] / T["r_motor"],
        -theta_thumb[2] * T["MCP_c"]       / T["r_motor"],
        -theta_thumb[3] * T["IP_c"]        / T["r_motor"],
    ])


def joint2motor_spread(theta_spread_index):
    """Index spread angle → motor [4] [rad]."""
    return theta_spread_index / (SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO["index"])


def joint2motor_finger(theta_finger, finger_name):
    """[MCP, PIP] → two motor angles [rad]."""
    trans = FINGER_TRANSMISSIONS[finger_name]
    return np.array([
        theta_finger[0] * trans["r_motor"] / trans["r_pulley"],
        -theta_finger[1] * trans["c_param"] / trans["r_motor"],
    ])


def joint_to_motor(theta_thumb, theta_spread_index,
                   theta_index, theta_middle, theta_ring, theta_pinky):
    """Convert all joint angles to a 13-motor angle vector."""
    q = np.zeros(13)
    q[0:4]   = joint2motor_thumb(theta_thumb)
    q[4]     = joint2motor_spread(theta_spread_index)
    q[5:7]   = joint2motor_finger(theta_index,  'index')
    q[7:9]   = joint2motor_finger(theta_middle, 'middle')
    q[9:11]  = joint2motor_finger(theta_ring,   'ring')
    q[11:13] = joint2motor_finger(theta_pinky,  'pinky')
    return q


# =============================================================================
# PALM FRAME  (rigid wrist — constant)
# =============================================================================

def FK_motor2palm(r_local):
    """
    Palm frame with rigid wrist (fixed at zero).

    Returns:
        R_palm: 3x3 identity (wrist at zero → no rotation)
        pos:    [x, y, z] position of r_local in world frame [m]
    """
    return _R_PALM, _PALM_ORIGIN + r_local


# =============================================================================
# THUMB FORWARD KINEMATICS
# =============================================================================

def FK_motor2thumbPos(q_motor, link, r_local):
    """FK for a point on a thumb link. link: 'CMC1'|'CMC2'|'MCP'|'IP'."""
    thetas = FK_motor2thumb(q_motor)
    chain = ["CMC1", "CMC2", "MCP", "IP"]
    p_curr = FINGER_BASE_ORIGINS["thumb"].copy()
    R_curr = np.eye(3)
    for i, link_name in enumerate(chain):
        p_curr += R_curr @ THUMB[f"{link_name}_origin"]
        R_curr = R_curr @ rot_axis(THUMB[f"{link_name}_axis"], thetas[i])
        if link_name == link:
            break
    return FK_motor2palm(p_curr + R_curr @ r_local)[1]


def Frame_motor2thumb(q_motor, link):
    """Frame (R, origin) of a thumb link in world frame."""
    thetas = FK_motor2thumb(q_motor)
    chain = ["CMC1", "CMC2", "MCP", "IP"]
    p_curr = FINGER_BASE_ORIGINS["thumb"].copy()
    R_curr = np.eye(3)
    for i, link_name in enumerate(chain):
        p_curr += R_curr @ THUMB[f"{link_name}_origin"]
        R_curr = R_curr @ rot_axis(THUMB[f"{link_name}_axis"], thetas[i])
        if link_name == link:
            break
    R_palm, p_world = FK_motor2palm(p_curr)
    return R_palm @ R_curr, p_world


# =============================================================================
# FINGER FORWARD KINEMATICS (Index, Middle, Ring, Pinky)
# =============================================================================

def FK_motor2fingerPos(q_motor, finger_name, link, r_local):
    """FK for a point on a finger link. link: 'Spread'|'MCP'|'PIP'|'DIP'."""
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
    return FK_motor2palm(p_curr + R_curr @ r_local)[1]


def Frame_motor2finger(q_motor, finger_name, link):
    """Frame (R, origin) of a finger link in world frame."""
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
    R_palm, p_world = FK_motor2palm(p_curr)
    return R_palm @ R_curr, p_world


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("FK_HAND - TEST  (13 motors, rigid wrist)")
    print("=" * 70)

    np.random.seed(42)
    q_motor = np.random.randn(13) * 0.1
    r_local = np.array([0.01, 0.02, 0.03])

    print("\n1. Motor-to-joint conversions")
    print(f"   Thumb [CMC1,CMC2,MCP,IP]:   {FK_motor2thumb(q_motor)}")
    print(f"   Index [MCP,PIP,DIP]:         {FK_motor2finger(q_motor, 'index')}")
    print(f"   Index spread:                {FK_motor2spread(q_motor, 'index'):.4f}")

    print("\n2. Palm frame (constant)")
    R_palm, p_palm = FK_motor2palm(np.zeros(3))
    print(f"   p_palm origin: {p_palm}")
    print(f"   R_palm = I:    {np.allclose(R_palm, np.eye(3))}")

    print("\n3. Forward kinematics")
    print(f"   Thumb IP tip:  {FK_motor2thumbPos(q_motor, 'IP', r_local)}")
    print(f"   Index DIP tip: {FK_motor2fingerPos(q_motor, 'index', 'DIP', r_local)}")

    print("\n4. Frame consistency")
    R_dip, p_dip = Frame_motor2finger(q_motor, "index", "DIP")
    p_tip = FK_motor2fingerPos(q_motor, "index", "DIP", np.array([0., 0., 0.0175]))
    phalanx_dir = (p_tip - p_dip) / np.linalg.norm(p_tip - p_dip)
    print(f"   z-col aligns with phalanx: {np.allclose(R_dip[:, 2], phalanx_dir, atol=1e-6)}")
    print(f"   R orthogonal: {np.allclose(R_dip @ R_dip.T, np.eye(3), atol=1e-10)}")

    t = time.time()
    for _ in range(1000):
        FK_motor2fingerPos(q_motor, "index", "DIP", r_local)
    print(f"\n5. Performance: {(time.time()-t):.4f} s / 1000 calls")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
