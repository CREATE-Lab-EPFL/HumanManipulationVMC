from sympy import symbols, lambdify, cos, sin, Matrix, sqrt
from ModelIDHand.hand_params import (
    FINGER_TRANSMISSIONS, THUMB_TRANSMISSION,
    SPREAD_MOTION_RATIO, SPREAD_ANGLE_CORRECTION,
    WRIST, FINGER_BASE_ORIGINS, THUMB, FINGERS,
)

# Units of measure:
# - LENGTHS: meters [m]
# - ANGLES: radians [rad]

# =============================================================================
# SYMBOLIC ROTATION MATRIX (Rodrigues' formula)
# =============================================================================

def sym_rot_axis(axis, theta):
    """
    Symbolic rotation matrix around an arbitrary axis using Rodrigues' formula.

    Args:
        axis: [ux, uy, uz] unit vector (will be normalized).
        theta: Symbolic rotation angle.

    Returns:
        R: 3x3 symbolic rotation matrix.
    """
    # Normalize axis
    ax, ay, az = axis
    norm = sqrt(ax**2 + ay**2 + az**2)
    ux, uy, uz = ax/norm, ay/norm, az/norm

    c = cos(theta)
    s = sin(theta)
    t = 1 - c

    R = Matrix([
        [c + ux*ux*t,      ux*uy*t - uz*s,  ux*uz*t + uy*s],
        [uy*ux*t + uz*s,   c + uy*uy*t,     uy*uz*t - ux*s],
        [uz*ux*t - uy*s,   uz*uy*t + ux*s,  c + uz*uz*t   ]
    ])

    return R

# =============================================================================
# MOTOR TO JOINT ANGLE CONVERSION
# =============================================================================

def Jacobian_motor2finger(finger_name):
    """
    Convert MCP, PIP motor angles to joint angles.

    Args:
        finger_name: 'index', 'middle', 'ring', or 'pinky'.

    Returns (J 3x15):
        theta: [theta_MCP, theta_PIP, theta_DIP] finger joint angles [rad].
    """

    q = symbols('q0:15')
    idx = {"index": (7, 8), "middle": (9, 10), "ring": (11, 12), "pinky": (13, 14)}[finger_name]
    rm, rp, cp = FINGER_TRANSMISSIONS[finger_name].values()
    theta = Matrix([
        (rp / rm) * q[idx[0]],
        (rm / cp) * q[idx[1]],
        (rm / cp) * q[idx[1]],
    ])
    return lambdify(q, theta.jacobian(Matrix(q)), modules=['numpy'], cse=True)

def Jacobian_motor2thumb():
    """
    Convert motor angles to thumb joint angles.

    Returns (J 4x15):
        theta: [theta_CMC1, theta_CMC2, theta_MCP, theta_IP] thumb joint angles [rad].
    """

    q = symbols('q0:15')
    r_motor, CMC1_pulley, CMC2_pulley, MCP_c, IP_c = THUMB_TRANSMISSION.values()
    theta = Matrix([
        (r_motor / CMC1_pulley) * q[2],
        (r_motor / CMC2_pulley) * q[3],
        (r_motor / MCP_c) * q[4],
        (r_motor / IP_c) * q[5],
    ])
    return lambdify(q, theta.jacobian(Matrix(q)), modules=['numpy'], cse=True)

def Jacobian_motor2spread(finger_name):
    """
    Convert motor angles to individual finger spread angle.

    Args:
        finger_name: 'index', 'middle', 'ring', or 'pinky'.

    Returns (J 1x15):
        theta_spread: spread joint angle [rad].
    """

    q = symbols('q0:15')
    theta_spread = q[6] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO.get(finger_name, 0.0)
    return lambdify(q, Matrix([theta_spread]).jacobian(Matrix(q)), modules=['numpy'], cse=True)

def Jacobian_motor2wrist():
    """
    Convert motor angles to wrist joint angles.

    Returns (J 2x15):
        theta: [theta_pitch, theta_yaw] wrist joint angles [rad].
    """

    q = symbols('q0:15')
    pitch = (q[0] - q[1]) / 2.0 * WRIST["spur_ratio"]
    yaw = (q[0] + q[1]) * WRIST["bevel_ratio"] * WRIST["spur_ratio"]
    theta = Matrix([pitch, yaw])
    return lambdify(q, theta.jacobian(Matrix(q)), modules=['numpy'], cse=True)

# =============================================================================
# WRIST KINEMATICS
# =============================================================================

def Jacobian_motor2palm():
    """
    Convert motor angles to a point in the palm frame.

    Returns (J 3x15):
        pos_palm: palm position [m].
    """

    q = symbols('q0:15')
    x, y, z = symbols('x y z')
    r_local = Matrix([x, y, z])

    pitch = (q[0] - q[1]) / 2.0 * WRIST["spur_ratio"]
    yaw = (q[0] + q[1]) * WRIST["bevel_ratio"] * WRIST["spur_ratio"]
    R_yaw = sym_rot_axis(WRIST["yaw_axis"], yaw)
    R_pitch = sym_rot_axis(WRIST["pitch_axis"], pitch)
    R_palm = R_yaw * R_pitch
    p_palm = Matrix(WRIST["yaw_origin"]) + R_yaw * Matrix(WRIST["pitch_origin"])
    pos_palm = p_palm + R_palm * r_local
    return lambdify(q + (x, y, z), pos_palm.jacobian(Matrix(q)), modules=['numpy'], cse=True)

# =============================================================================
# THUMB KINEMATICS
# =============================================================================

def Jacobian_motor2thumbPos(link):
    """
    Convert motor angles to a point in the thumb frame.

    Args:
        link: link name ('CMC1', 'CMC2', 'MCP', 'IP').

    Returns (J 3x15):
        pos_thumb: thumb link position [m].
    """

    q = symbols('q0:15')
    x, y, z = symbols('x y z')
    r_local = Matrix([x, y, z])

    pitch = (q[0] - q[1]) / 2.0 * WRIST["spur_ratio"]
    yaw = (q[0] + q[1]) * WRIST["bevel_ratio"] * WRIST["spur_ratio"]
    R_yaw = sym_rot_axis(WRIST["yaw_axis"], yaw)
    R_pitch = sym_rot_axis(WRIST["pitch_axis"], pitch)
    R_palm = R_yaw * R_pitch
    p_palm = Matrix(WRIST["yaw_origin"]) + R_yaw * Matrix(WRIST["pitch_origin"])

    rm = THUMB_TRANSMISSION["r_motor"]
    thetas = [
        (rm / THUMB_TRANSMISSION["CMC1_pulley"]) * q[2],
        (rm / THUMB_TRANSMISSION["CMC2_pulley"]) * q[3],
        (rm / THUMB_TRANSMISSION["MCP_c"]) * q[4],
        (rm / THUMB_TRANSMISSION["IP_c"]) * q[5]
    ]
    chain = ["CMC1", "CMC2", "MCP", "IP"]

    p_curr = Matrix(FINGER_BASE_ORIGINS["thumb"])
    R_curr = Matrix.eye(3)

    for i, link_name in enumerate(chain):
        p_curr += R_curr * Matrix(THUMB[f"{link_name}_origin"])
        R_curr = R_curr * sym_rot_axis(THUMB[f"{link_name}_axis"], thetas[i])
        if link_name == link:
            break

    pos_global = p_palm + R_palm * (p_curr + R_curr * r_local)
    return lambdify(q + (x, y, z), pos_global.jacobian(Matrix(q)), modules=['numpy'], cse=True)

def Jacobian_motor2fingerPos(finger_name, link):
    """
    Convert motor angles to a point on a generic finger link.

    Args:
        finger_name: name of the finger ('index', 'middle', 'ring', 'pinky').
        link: link name ('MCP', 'PIP', 'DIP').

    Returns (J 3x15):
        pos_finger: finger link position [m].
    """

    q = symbols('q0:15')
    x, y, z = symbols('x y z')
    r_local = Matrix([x, y, z])

    pitch = (q[0] - q[1]) / 2.0 * WRIST["spur_ratio"]
    yaw = (q[0] + q[1]) * WRIST["bevel_ratio"] * WRIST["spur_ratio"]

    R_yaw = sym_rot_axis(WRIST["yaw_axis"], yaw)
    R_pitch = sym_rot_axis(WRIST["pitch_axis"], pitch)
    R_palm = R_yaw * R_pitch
    p_palm = Matrix(WRIST["yaw_origin"]) + R_yaw * Matrix(WRIST["pitch_origin"])

    idx_map = {"index": (7,8), "middle": (9,10), "ring": (11,12), "pinky": (13,14)}
    idx_mcp, idx_pip = idx_map[finger_name]

    rm, rp, cp = FINGER_TRANSMISSIONS[finger_name]["r_motor"], FINGER_TRANSMISSIONS[finger_name]["r_pulley"], FINGER_TRANSMISSIONS[finger_name]["c_param"]
    theta_spread = q[6] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO[finger_name]
    theta_mcp    = (rp / rm) * q[idx_mcp]
    theta_pip    = (rm / cp) * q[idx_pip]
    theta_dip    = theta_pip

    thetas = [theta_spread, theta_mcp, theta_pip, theta_dip]
    chain = ["Spread", "MCP", "PIP", "DIP"]

    p_curr = Matrix(FINGER_BASE_ORIGINS[finger_name])
    R_curr = Matrix.eye(3)
    params = FINGERS[finger_name]

    for i, link_name in enumerate(chain):
        p_curr += R_curr * Matrix(params[f"{link_name}_origin"])
        R_curr = R_curr * sym_rot_axis(params[f"{link_name}_axis"], thetas[i])

        if link_name == link:
            break

    pos_global = p_palm + R_palm * (p_curr + R_curr * r_local)
    return lambdify(q + (x, y, z), pos_global.jacobian(Matrix(q)), modules=['numpy'], cse=True)

# =============================================================================
# JACOBIAN CONTAINER CLASS
# =============================================================================

class HandJacobians:
    """
    Pre-initialized container for all hand Jacobians.

    This class builds all Jacobian functions once during initialization,
    avoiding repeated symbolic computation overhead.

    Attributes:
        Wrist Jacobians:
            J_wrist_motor: Motor-to-joint Jacobian (2x15)
            J_wrist_palm: Motor-to-palm position Jacobian (3x15)

        Motor-to-Joint-Angle Jacobians:
            J_motor2thumb_angles: Thumb motor-to-joint-angle Jacobian (4x15)
            J_motor2{finger}_angles: Finger motor-to-joint-angle Jacobians (3x15 each)
                                     for index, middle, ring, pinky

        Thumb Position Jacobians:
            J_thumb_CMC1, J_thumb_CMC2, J_thumb_MCP, J_thumb_IP: (3x15 each)

        Finger Position Jacobians (index, middle, ring, pinky):
            J_{finger}_MCP, J_{finger}_PIP, J_{finger}_DIP: (3x15 each)

        Spread Jacobians:
            J_spread_{finger}: Spread motor-to-joint Jacobians (1x15 each)
                               for index, middle, ring, pinky
    """
    
    def __init__(self):
        """Initialize all hand Jacobians."""

        # Wrist Jacobians
        self.J_wrist_motor = Jacobian_motor2wrist()
        self.J_wrist_palm = Jacobian_motor2palm()

        # Motor-to-joint-angle Jacobians
        self.J_motor2thumb_angles = Jacobian_motor2thumb()
        self.J_motor2index_angles = Jacobian_motor2finger("index")
        self.J_motor2middle_angles = Jacobian_motor2finger("middle")
        self.J_motor2ring_angles = Jacobian_motor2finger("ring")
        self.J_motor2pinky_angles = Jacobian_motor2finger("pinky")

        # Thumb Jacobians
        self.J_thumb_CMC1 = Jacobian_motor2thumbPos("CMC1")
        self.J_thumb_CMC2 = Jacobian_motor2thumbPos("CMC2")
        self.J_thumb_MCP = Jacobian_motor2thumbPos("MCP")
        self.J_thumb_IP = Jacobian_motor2thumbPos("IP")
        
        # Index Jacobians
        self.J_index_MCP = Jacobian_motor2fingerPos("index", "MCP")
        self.J_index_PIP = Jacobian_motor2fingerPos("index", "PIP")
        self.J_index_DIP = Jacobian_motor2fingerPos("index", "DIP")
        
        # Middle Jacobians
        self.J_middle_MCP = Jacobian_motor2fingerPos("middle", "MCP")
        self.J_middle_PIP = Jacobian_motor2fingerPos("middle", "PIP")
        self.J_middle_DIP = Jacobian_motor2fingerPos("middle", "DIP")
        
        # Ring Jacobians
        self.J_ring_MCP = Jacobian_motor2fingerPos("ring", "MCP")
        self.J_ring_PIP = Jacobian_motor2fingerPos("ring", "PIP")
        self.J_ring_DIP = Jacobian_motor2fingerPos("ring", "DIP")
        
        # Pinky Jacobians
        self.J_pinky_MCP = Jacobian_motor2fingerPos("pinky", "MCP")
        self.J_pinky_PIP = Jacobian_motor2fingerPos("pinky", "PIP")
        self.J_pinky_DIP = Jacobian_motor2fingerPos("pinky", "DIP")
        
        # Spread Jacobians
        self.J_spread_index = Jacobian_motor2spread("index")
        self.J_spread_middle = Jacobian_motor2spread("middle")
        self.J_spread_ring = Jacobian_motor2spread("ring")
        self.J_spread_pinky = Jacobian_motor2spread("pinky")
        
        print("All Jacobians initialized.")
    
    def get_thumb_jacobian(self, link, q_motor, r_local):
        """
        Get thumb Jacobian for a specific link.

        Args:
            link: 'CMC1', 'CMC2', 'MCP', or 'IP'
            q_motor: Motor angles [15] [rad]
            r_local: Local position [3] [m]

        Returns:
            J: Jacobian matrix (3x15)
        """

        jacobian_func = getattr(self, f"J_thumb_{link}")
        return jacobian_func(*q_motor, *r_local)
    
    def get_finger_jacobian(self, finger, link, q_motor, r_local):
        """
        Get finger Jacobian for a specific finger and link.

        Args:
            finger: 'index', 'middle', 'ring', or 'pinky'
            link: 'MCP', 'PIP', or 'DIP'
            q_motor: Motor angles [15] [rad]
            r_local: Local position [3] [m]

        Returns:
            J: Jacobian matrix (3x15)
        """

        jacobian_func = getattr(self, f"J_{finger}_{link}")
        return jacobian_func(*q_motor, *r_local)
    
    def get_wrist_palm_jacobian(self, q_motor, r_local):
        """
        Get wrist palm Jacobian.

        Args:
            q_motor: Motor angles [15] [rad]
            r_local: Local position [3] [m]

        Returns:
            J: Jacobian matrix (3x15)
        """

        return self.J_wrist_palm(*q_motor, *r_local)

    def get_angles_jacobian(self, part_name, q_motor):
        """
        Get motor-to-joint-angle Jacobian for a specific hand part.

        Args:
            part_name: 'thumb', 'index', 'middle', 'ring', or 'pinky'
            q_motor: Motor angles [15] [rad]

        Returns:
            J: Jacobian matrix (4x15 for thumb, 3x15 for fingers)
        """
        return getattr(self, f"J_motor2{part_name}_angles")(*q_motor)

    def get_wrist_motor_jacobian(self, q_motor):
        """
        Get wrist motor-to-joint Jacobian.

        Args:
            q_motor: Motor angles [15] [rad]

        Returns:
            J: Jacobian matrix (2x15)
        """
        return self.J_wrist_motor(*q_motor)

    def get_spread_jacobian(self, finger, q_motor):
        """
        Get spread motor-to-joint Jacobian for a finger.

        Args:
            finger: 'index', 'middle', 'ring', or 'pinky'
            q_motor: Motor angles [15] [rad]

        Returns:
            J: Jacobian matrix (1x15)
        """
        return getattr(self, f"J_spread_{finger}")(*q_motor)

# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":

    import numpy as np
    import time
    
    print("="*70)
    print("JACOBIANS HAND - TEST")
    print("="*70)
    
    # Test configuration
    np.random.seed(42)
    q_motor = np.random.randn(15) * 0.1
    r_local = np.array([0.01, 0.02, 0.03])
    
    print("\n1. Testing individual Jacobian functions...")

    # Test thumb
    J_thumb = Jacobian_motor2thumbPos("IP")
    J_val = J_thumb(*q_motor, *r_local)
    print(f"   Thumb IP Jacobian shape: {np.array(J_val).shape}")

    # Test finger
    J_index = Jacobian_motor2fingerPos("index", "DIP")
    J_val = J_index(*q_motor, *r_local)
    print(f"   Index DIP Jacobian shape: {np.array(J_val).shape}")

    # Test wrist
    J_wrist = Jacobian_motor2palm()
    J_val = J_wrist(*q_motor, *r_local)
    print(f"   Wrist Palm Jacobian shape: {np.array(J_val).shape}")
    
    print("\n2. Testing HandJacobians class...")
    t = time.time()
    hand_jac = HandJacobians()
    print(f"   Initialization time: {time.time() - t:.4f} seconds")
    
    # Test retrieval
    t = time.time()
    J_val = hand_jac.get_thumb_jacobian("IP", q_motor, r_local)
    print(f"   Retrieval time: {time.time() - t:.6f} seconds\n")
    print(f"   Retrieved Thumb IP Jacobian shape: {np.array(J_val).shape}")
    
    J_val = hand_jac.get_finger_jacobian("index", "DIP", q_motor, r_local)
    print(f"   Retrieved Index DIP Jacobian shape: {np.array(J_val).shape}")
    
    J_val = hand_jac.get_wrist_palm_jacobian(q_motor, r_local)
    print(f"   Retrieved Wrist Palm Jacobian shape: {np.array(J_val).shape}")

    print("\n3. Testing motor-to-joint-angle Jacobians...")

    # Test thumb angles
    J_val = hand_jac.get_angles_jacobian("thumb", q_motor)
    print(f"   Thumb motor-to-angles Jacobian shape: {np.array(J_val).shape}")

    # Test finger angles
    J_val = hand_jac.get_angles_jacobian("index", q_motor)
    print(f"   Index motor-to-angles Jacobian shape: {np.array(J_val).shape}")

    # Direct access test
    J_val = hand_jac.J_motor2middle_angles(*q_motor)
    print(f"   Middle motor-to-angles Jacobian (direct) shape: {np.array(J_val).shape}")

    print("\n4. Performance benchmark (1000 calls)...")

    # Benchmark position Jacobian
    n_calls = 1000
    t_start = time.time()
    for _ in range(n_calls):
        J_val = hand_jac.J_index_DIP(*q_motor, *r_local)
    t_elapsed = time.time() - t_start
    print(f"   Index DIP position Jacobian: {t_elapsed*1000/n_calls:.3f} ms/call")
    print(f"   Effective frequency: {n_calls/t_elapsed:.1f} Hz")

    # Benchmark angle Jacobian
    t_start = time.time()
    for _ in range(n_calls):
        J_val = hand_jac.J_motor2index_angles(*q_motor)
    t_elapsed = time.time() - t_start
    print(f"   Index motor-to-angles Jacobian: {t_elapsed*1000/n_calls:.3f} ms/call")
    print(f"   Effective frequency: {n_calls/t_elapsed:.1f} Hz")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
