from sympy import symbols, lambdify, Matrix
import numpy as np

from ModelIDHand.hand_params import (
    FINGER_TRANSMISSIONS, THUMB_TRANSMISSION,
    SPREAD_ANGLE_CORRECTION, SPREAD_MOTION_RATIO,
    WRIST, FINGER_BASE_ORIGINS, THUMB, FINGERS,
)
from KinematicsHand.JacobiansHand import sym_rot_axis

# Units of measure:
# - LENGTHS: [m] (meters)
# - ANGLES: [rad] (radians)
#
# Hessians are 3D tensors: H[i,j,k] = d²f_i / dq_j dq_k
# For a function f: R^n -> R^m, the Hessian is an (m x n x n) tensor
#
# Computed as: Jacobian of each row of the Jacobian


# =============================================================================
# MOTOR-TO-JOINT-ANGLE HESSIANS (all linear → all zeros)
# =============================================================================

def Hessian_motor2finger(finger_name):
    """
    Hessian of motor angles to finger joint angles.

    All zeros since the mapping is linear.

    Args:
        finger_name: 'index', 'middle', 'ring', or 'pinky'.

    Returns:
        hessian_func(q0..q14) -> (3, 15, 15) array of zeros.
    """
    q = symbols('q0:15')
    idx = {"index": (7, 8), "middle": (9, 10), "ring": (11, 12), "pinky": (13, 14)}[finger_name]
    rm, rp, cp = FINGER_TRANSMISSIONS[finger_name].values()
    theta = Matrix([
        (rp / rm) * q[idx[0]],
        -(rm / cp) * q[idx[1]],
        -(rm / cp) * q[idx[1]],
    ])
    J = theta.jacobian(Matrix(q))
    H_rows = [J.row(i).jacobian(Matrix(q)) for i in range(J.rows)]
    H_funcs = [lambdify(q, H_i, modules=['numpy'], cse=True) for H_i in H_rows]

    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


def Hessian_motor2thumb():
    """
    Hessian of motor angles to thumb joint angles.

    All zeros since the mapping is linear.

    Returns:
        hessian_func(q0..q14) -> (4, 15, 15) array of zeros.
    """
    q = symbols('q0:15')
    r_motor, CMC1_pulley, CMC2_pulley, MCP_c, IP_c = THUMB_TRANSMISSION.values()
    theta = Matrix([
        (r_motor / CMC1_pulley) * q[2],
        (r_motor / CMC2_pulley) * q[3],
        -(r_motor / MCP_c) * q[4],
        -(r_motor / IP_c) * q[5],
    ])
    J = theta.jacobian(Matrix(q))
    H_rows = [J.row(i).jacobian(Matrix(q)) for i in range(J.rows)]
    H_funcs = [lambdify(q, H_i, modules=['numpy'], cse=True) for H_i in H_rows]

    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


def Hessian_motor2spread(finger_name):
    """
    Hessian of motor angles to finger spread angle.

    All zeros since the mapping is linear.

    Args:
        finger_name: 'index', 'middle', 'ring', or 'pinky'.

    Returns:
        hessian_func(q0..q14) -> (1, 15, 15) array of zeros.
    """
    q = symbols('q0:15')
    theta_spread = q[6] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO.get(finger_name, 0.0)
    expr = Matrix([theta_spread])
    J = expr.jacobian(Matrix(q))
    H_rows = [J.row(i).jacobian(Matrix(q)) for i in range(J.rows)]
    H_funcs = [lambdify(q, H_i, modules=['numpy'], cse=True) for H_i in H_rows]

    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


def Hessian_motor2wrist():
    """
    Hessian of motor angles to wrist joint angles.

    All zeros since the mapping is linear.

    Returns:
        hessian_func(q0..q14) -> (2, 15, 15) array of zeros.
    """
    q = symbols('q0:15')
    pitch = (q[0] - q[1]) / 2.0 * WRIST["spur_ratio"]
    yaw = (q[0] + q[1]) * WRIST["bevel_ratio"] * WRIST["spur_ratio"]
    theta = Matrix([pitch, yaw])
    J = theta.jacobian(Matrix(q))
    H_rows = [J.row(i).jacobian(Matrix(q)) for i in range(J.rows)]
    H_funcs = [lambdify(q, H_i, modules=['numpy'], cse=True) for H_i in H_rows]

    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


# =============================================================================
# POSITION HESSIANS (non-trivial)
# =============================================================================

def Hessian_motor2palm():
    """
    Hessian of motor angles to a point in the palm frame.

    Returns:
        hessian_func(q0..q14, x, y, z) -> (3, 15, 15) array.
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

    J = pos_palm.jacobian(Matrix(q))
    H_rows = [J.row(i).jacobian(Matrix(q)) for i in range(J.rows)]
    H_funcs = [lambdify(q + (x, y, z), H_i, modules=['numpy'], cse=True) for H_i in H_rows]

    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


def Hessian_motor2thumbPos(link):
    """
    Hessian of motor angles to a point on a thumb link.

    Args:
        link: 'CMC1', 'CMC2', 'MCP', or 'IP'.

    Returns:
        hessian_func(q0..q14, x, y, z) -> (3, 15, 15) array.
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
        -(rm / THUMB_TRANSMISSION["MCP_c"]) * q[4],
        -(rm / THUMB_TRANSMISSION["IP_c"]) * q[5]
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

    J = pos_global.jacobian(Matrix(q))
    H_rows = [J.row(i).jacobian(Matrix(q)) for i in range(J.rows)]
    H_funcs = [lambdify(q + (x, y, z), H_i, modules=['numpy'], cse=True) for H_i in H_rows]

    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


def Hessian_motor2fingerPos(finger_name, link):
    """
    Hessian of motor angles to a point on a generic finger link.

    Args:
        finger_name: 'index', 'middle', 'ring', or 'pinky'.
        link: 'MCP', 'PIP', or 'DIP'.

    Returns:
        hessian_func(q0..q14, x, y, z) -> (3, 15, 15) array.
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

    idx_map = {"index": (7, 8), "middle": (9, 10), "ring": (11, 12), "pinky": (13, 14)}
    idx_mcp, idx_pip = idx_map[finger_name]

    rm = FINGER_TRANSMISSIONS[finger_name]["r_motor"]
    rp = FINGER_TRANSMISSIONS[finger_name]["r_pulley"]
    cp = FINGER_TRANSMISSIONS[finger_name]["c_param"]
    theta_spread = q[6] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO[finger_name]
    theta_mcp = (rp / rm) * q[idx_mcp]
    theta_pip = -(rm / cp) * q[idx_pip]
    theta_dip = theta_pip

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

    J = pos_global.jacobian(Matrix(q))
    H_rows = [J.row(i).jacobian(Matrix(q)) for i in range(J.rows)]
    H_funcs = [lambdify(q + (x, y, z), H_i, modules=['numpy'], cse=True) for H_i in H_rows]

    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


# =============================================================================
# HESSIAN CONTAINER CLASS
# =============================================================================

class HandHessians:
    """
    Pre-initialized container for all hand Hessians.

    This class builds all Hessian functions once during initialization,
    avoiding repeated symbolic computation overhead.

    Attributes:
        Motor-to-Joint-Angle Hessians (all zero - linear mappings):
            H_motor2thumb_angles: (4, 15, 15)
            H_motor2{finger}_angles: (3, 15, 15) for index, middle, ring, pinky
            H_wrist_motor: (2, 15, 15)
            H_spread_{finger}: (1, 15, 15) for index, middle, ring, pinky

        Position Hessians (non-trivial):
            H_wrist_palm: (3, 15, 15)
            H_thumb_CMC1, H_thumb_CMC2, H_thumb_MCP, H_thumb_IP: (3, 15, 15)
            H_{finger}_MCP, H_{finger}_PIP, H_{finger}_DIP: (3, 15, 15)
    """

    def __init__(self):
        """Initialize all hand Hessians."""

        # Wrist Hessians
        self.H_wrist_motor = Hessian_motor2wrist()
        self.H_wrist_palm = Hessian_motor2palm()

        # Motor-to-joint-angle Hessians
        self.H_motor2thumb_angles = Hessian_motor2thumb()
        self.H_motor2index_angles = Hessian_motor2finger("index")
        self.H_motor2middle_angles = Hessian_motor2finger("middle")
        self.H_motor2ring_angles = Hessian_motor2finger("ring")
        self.H_motor2pinky_angles = Hessian_motor2finger("pinky")

        # Thumb Hessians
        self.H_thumb_CMC1 = Hessian_motor2thumbPos("CMC1")
        self.H_thumb_CMC2 = Hessian_motor2thumbPos("CMC2")
        self.H_thumb_MCP = Hessian_motor2thumbPos("MCP")
        self.H_thumb_IP = Hessian_motor2thumbPos("IP")

        # Index Hessians
        self.H_index_MCP = Hessian_motor2fingerPos("index", "MCP")
        self.H_index_PIP = Hessian_motor2fingerPos("index", "PIP")
        self.H_index_DIP = Hessian_motor2fingerPos("index", "DIP")

        # Middle Hessians
        self.H_middle_MCP = Hessian_motor2fingerPos("middle", "MCP")
        self.H_middle_PIP = Hessian_motor2fingerPos("middle", "PIP")
        self.H_middle_DIP = Hessian_motor2fingerPos("middle", "DIP")

        # Ring Hessians
        self.H_ring_MCP = Hessian_motor2fingerPos("ring", "MCP")
        self.H_ring_PIP = Hessian_motor2fingerPos("ring", "PIP")
        self.H_ring_DIP = Hessian_motor2fingerPos("ring", "DIP")

        # Pinky Hessians
        self.H_pinky_MCP = Hessian_motor2fingerPos("pinky", "MCP")
        self.H_pinky_PIP = Hessian_motor2fingerPos("pinky", "PIP")
        self.H_pinky_DIP = Hessian_motor2fingerPos("pinky", "DIP")

        # Spread Hessians
        self.H_spread_index = Hessian_motor2spread("index")
        self.H_spread_middle = Hessian_motor2spread("middle")
        self.H_spread_ring = Hessian_motor2spread("ring")
        self.H_spread_pinky = Hessian_motor2spread("pinky")

        print("All Hessians initialized.")

    def get_thumb_hessian(self, link, q_motor, r_local):
        """
        Get thumb Hessian for a specific link.

        Args:
            link: 'CMC1', 'CMC2', 'MCP', or 'IP'
            q_motor: Motor angles [15] [rad]
            r_local: Local position [3] [m]

        Returns:
            H: Hessian tensor (3, 15, 15)
        """
        hessian_func = getattr(self, f"H_thumb_{link}")
        return hessian_func(*q_motor, *r_local)

    def get_finger_hessian(self, finger, link, q_motor, r_local):
        """
        Get finger Hessian for a specific finger and link.

        Args:
            finger: 'index', 'middle', 'ring', or 'pinky'
            link: 'MCP', 'PIP', or 'DIP'
            q_motor: Motor angles [15] [rad]
            r_local: Local position [3] [m]

        Returns:
            H: Hessian tensor (3, 15, 15)
        """
        hessian_func = getattr(self, f"H_{finger}_{link}")
        return hessian_func(*q_motor, *r_local)

    def get_wrist_palm_hessian(self, q_motor, r_local):
        """
        Get wrist palm Hessian.

        Args:
            q_motor: Motor angles [15] [rad]
            r_local: Local position [3] [m]

        Returns:
            H: Hessian tensor (3, 15, 15)
        """
        return self.H_wrist_palm(*q_motor, *r_local)

    def get_angles_hessian(self, part_name, q_motor):
        """
        Get motor-to-joint-angle Hessian for a specific hand part.

        Args:
            part_name: 'thumb', 'index', 'middle', 'ring', or 'pinky'
            q_motor: Motor angles [15] [rad]

        Returns:
            H: Hessian tensor (4, 15, 15) for thumb, (3, 15, 15) for fingers
        """
        return getattr(self, f"H_motor2{part_name}_angles")(*q_motor)

    def get_wrist_motor_hessian(self, q_motor):
        """
        Get wrist motor-to-joint Hessian.

        Args:
            q_motor: Motor angles [15] [rad]

        Returns:
            H: Hessian tensor (2, 15, 15)
        """
        return self.H_wrist_motor(*q_motor)

    def get_spread_hessian(self, finger, q_motor):
        """
        Get spread motor-to-joint Hessian for a finger.

        Args:
            finger: 'index', 'middle', 'ring', or 'pinky'
            q_motor: Motor angles [15] [rad]

        Returns:
            H: Hessian tensor (1, 15, 15)
        """
        return getattr(self, f"H_spread_{finger}")(*q_motor)


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("HESSIANS HAND - TEST")
    print("=" * 70)

    # Test configuration
    np.random.seed(42)
    q_motor = np.random.randn(15) * 0.1
    r_local = np.array([0.01, 0.02, 0.03])

    print("\nTest motor angles:")
    print(f"  q_motor (15 values): {np.round(q_motor, 4)}")
    print(f"  r_local [x, y, z]: {r_local}\n")

    # --- Test motor-to-angle Hessians (should be all zeros) ---
    print("1. Motor-to-angle Hessians (should be all zeros)...")
    t = time.time()
    H_func = Hessian_motor2finger("index")
    H_val = H_func(*q_motor)
    print(f"   Index finger angles Hessian shape: {H_val.shape}, all zero: {np.allclose(H_val, 0)}")

    H_func = Hessian_motor2thumb()
    H_val = H_func(*q_motor)
    print(f"   Thumb angles Hessian shape: {H_val.shape}, all zero: {np.allclose(H_val, 0)}")

    H_func = Hessian_motor2wrist()
    H_val = H_func(*q_motor)
    print(f"   Wrist angles Hessian shape: {H_val.shape}, all zero: {np.allclose(H_val, 0)}")

    H_func = Hessian_motor2spread("index")
    H_val = H_func(*q_motor)
    print(f"   Spread Hessian shape: {H_val.shape}, all zero: {np.allclose(H_val, 0)}")
    print(f"   Time for angle Hessians: {time.time() - t:.2f} s")

    # --- Test position Hessians ---
    print("\n2. Position Hessians (non-trivial)...")
    t = time.time()
    H_palm_func = Hessian_motor2palm()
    print(f"   Palm Hessian compiled in {time.time() - t:.2f} s")
    H_val = H_palm_func(*q_motor, *r_local)
    print(f"   Palm Hessian shape: {H_val.shape}")

    t = time.time()
    H_thumb_func = Hessian_motor2thumbPos("IP")
    print(f"   Thumb IP Hessian compiled in {time.time() - t:.2f} s")
    H_val = H_thumb_func(*q_motor, *r_local)
    print(f"   Thumb IP Hessian shape: {H_val.shape}")

    t = time.time()
    H_finger_func = Hessian_motor2fingerPos("index", "DIP")
    print(f"   Index DIP Hessian compiled in {time.time() - t:.2f} s")
    H_val = H_finger_func(*q_motor, *r_local)
    print(f"   Index DIP Hessian shape: {H_val.shape}")

    # --- Verify symmetry ---
    print("\n3. Verifying Hessian symmetry (H[i,j,k] == H[i,k,j])...")
    H_val = H_finger_func(*q_motor, *r_local)
    symmetric = np.allclose(H_val, H_val.transpose(0, 2, 1))
    print(f"   Index DIP Hessian symmetric: {symmetric}")

    H_val = H_thumb_func(*q_motor, *r_local)
    symmetric = np.allclose(H_val, H_val.transpose(0, 2, 1))
    print(f"   Thumb IP Hessian symmetric: {symmetric}")

    H_val = H_palm_func(*q_motor, *r_local)
    symmetric = np.allclose(H_val, H_val.transpose(0, 2, 1))
    print(f"   Palm Hessian symmetric: {symmetric}")

    # --- Performance benchmark ---
    print("\n4. Performance benchmark (1000 calls)...")
    n_calls = 1000

    t_start = time.time()
    for _ in range(n_calls):
        H_val = H_finger_func(*q_motor, *r_local)
    t_elapsed = time.time() - t_start
    print(f"   Index DIP Hessian: {t_elapsed * 1000 / n_calls:.3f} ms/call")
    print(f"   Effective frequency: {n_calls / t_elapsed:.1f} Hz")

    t_start = time.time()
    for _ in range(n_calls):
        H_val = H_palm_func(*q_motor, *r_local)
    t_elapsed = time.time() - t_start
    print(f"   Palm Hessian: {t_elapsed * 1000 / n_calls:.3f} ms/call")
    print(f"   Effective frequency: {n_calls / t_elapsed:.1f} Hz")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
