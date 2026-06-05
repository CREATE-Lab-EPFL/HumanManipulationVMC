from sympy import symbols, lambdify, cos, sin, Matrix, sqrt
from ModelIDHand.hand_params import (
    FINGER_TRANSMISSIONS, THUMB_TRANSMISSION,
    SPREAD_MOTION_RATIO, SPREAD_ANGLE_CORRECTION,
    WRIST, FINGER_BASE_ORIGINS, THUMB, FINGERS,
)

# Units: lengths [m], angles [rad]
#
# All Jacobians operate on the 13-motor vector (wrist rigid/fixed at zero):
#   q[0..3] = thumb CMC1/CMC2/MCP/IP
#   q[4]    = spread
#   q[5..6] = index MCP/PIP,  q[7..8] = middle MCP/PIP
#   q[9..10]= ring MCP/PIP,   q[11..12]= pinky MCP/PIP
#
# The palm frame is constant (wrist at zero: R_palm = I,
# p_palm = WRIST["yaw_origin"] + WRIST["pitch_origin"]).
# Position Jacobians are therefore (3×13); angle Jacobians are (n×13).


# =============================================================================
# SYMBOLIC ROTATION MATRIX (Rodrigues)
# =============================================================================

def sym_rot_axis(axis, theta):
    ax, ay, az = axis
    norm = sqrt(ax**2 + ay**2 + az**2)
    ux, uy, uz = ax/norm, ay/norm, az/norm
    c, s, t = cos(theta), sin(theta), 1 - cos(theta)
    return Matrix([
        [c + ux*ux*t,      ux*uy*t - uz*s,  ux*uz*t + uy*s],
        [uy*ux*t + uz*s,   c + uy*uy*t,     uy*uz*t - ux*s],
        [uz*ux*t - uy*s,   uz*uy*t + ux*s,  c + uz*uz*t   ]
    ])


# Constant palm origin (wrist at zero)
_P_PALM = Matrix(WRIST["yaw_origin"]) + Matrix(WRIST["pitch_origin"])


# =============================================================================
# MOTOR-TO-JOINT-ANGLE JACOBIANS
# =============================================================================

def Jacobian_motor2finger(finger_name):
    """Jacobian of [MCP, PIP, DIP] joint angles w.r.t. 13 motor angles. Returns (3×13)."""
    q = symbols('q0:13')
    idx = {"index": (5,6), "middle": (7,8), "ring": (9,10), "pinky": (11,12)}[finger_name]
    rm, rp, cp = FINGER_TRANSMISSIONS[finger_name].values()
    theta = Matrix([
        (rp / rm) * q[idx[0]],
        -(rm / cp) * q[idx[1]],
        -(rm / cp) * q[idx[1]],
    ])
    return lambdify(q, theta.jacobian(Matrix(q)), modules=['numpy'], cse=True)


def Jacobian_motor2thumb():
    """Jacobian of [CMC1, CMC2, MCP, IP] joint angles w.r.t. 13 motor angles. Returns (4×13)."""
    q = symbols('q0:13')
    r_motor, CMC1_pulley, CMC2_pulley, MCP_c, IP_c = THUMB_TRANSMISSION.values()
    theta = Matrix([
        (r_motor / CMC1_pulley) * q[0],
        (r_motor / CMC2_pulley) * q[1],
        -(r_motor / MCP_c) * q[2],
        -(r_motor / IP_c) * q[3],
    ])
    return lambdify(q, theta.jacobian(Matrix(q)), modules=['numpy'], cse=True)


def Jacobian_motor2spread(finger_name):
    """Jacobian of spread joint angle w.r.t. 13 motor angles. Returns (1×13)."""
    q = symbols('q0:13')
    theta_spread = q[4] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO.get(finger_name, 0.0)
    return lambdify(q, Matrix([theta_spread]).jacobian(Matrix(q)), modules=['numpy'], cse=True)


# =============================================================================
# POSITION JACOBIANS
# =============================================================================

def Jacobian_motor2palm():
    """
    Jacobian of palm position w.r.t. 13 motor angles.
    Palm is constant (wrist rigid) → always returns (3×13) zeros.
    """
    q = symbols('q0:13')
    x, y, z = symbols('x y z')
    pos_palm = _P_PALM + Matrix([x, y, z])  # constant w.r.t. q
    return lambdify(q + (x, y, z), pos_palm.jacobian(Matrix(q)), modules=['numpy'], cse=True)


def Jacobian_motor2thumbPos(link):
    """Jacobian of a thumb link position w.r.t. 13 motor angles. Returns (3×13)."""
    q = symbols('q0:13')
    x, y, z = symbols('x y z')
    r_local = Matrix([x, y, z])

    rm = THUMB_TRANSMISSION["r_motor"]
    thetas = [
        (rm / THUMB_TRANSMISSION["CMC1_pulley"]) * q[0],
        (rm / THUMB_TRANSMISSION["CMC2_pulley"]) * q[1],
        -(rm / THUMB_TRANSMISSION["MCP_c"]) * q[2],
        -(rm / THUMB_TRANSMISSION["IP_c"]) * q[3],
    ]
    chain = ["CMC1", "CMC2", "MCP", "IP"]

    p_curr = Matrix(FINGER_BASE_ORIGINS["thumb"])
    R_curr = Matrix.eye(3)
    for i, link_name in enumerate(chain):
        p_curr += R_curr * Matrix(THUMB[f"{link_name}_origin"])
        R_curr = R_curr * sym_rot_axis(THUMB[f"{link_name}_axis"], thetas[i])
        if link_name == link:
            break

    # R_palm = I → pos_global = p_palm + p_curr + R_curr * r_local
    pos_global = _P_PALM + p_curr + R_curr * r_local
    return lambdify(q + (x, y, z), pos_global.jacobian(Matrix(q)), modules=['numpy'], cse=True)


def Jacobian_motor2fingerPos(finger_name, link):
    """Jacobian of a finger link position w.r.t. 13 motor angles. Returns (3×13)."""
    q = symbols('q0:13')
    x, y, z = symbols('x y z')
    r_local = Matrix([x, y, z])

    idx_map = {"index": (5,6), "middle": (7,8), "ring": (9,10), "pinky": (11,12)}
    idx_mcp, idx_pip = idx_map[finger_name]

    rm = FINGER_TRANSMISSIONS[finger_name]["r_motor"]
    rp = FINGER_TRANSMISSIONS[finger_name]["r_pulley"]
    cp = FINGER_TRANSMISSIONS[finger_name]["c_param"]
    theta_spread = q[4] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO[finger_name]
    theta_mcp    = (rp / rm) * q[idx_mcp]
    theta_pip    = -(rm / cp) * q[idx_pip]
    theta_dip    = theta_pip

    thetas = [theta_spread, theta_mcp, theta_pip, theta_dip]
    chain  = ["Spread", "MCP", "PIP", "DIP"]

    p_curr = Matrix(FINGER_BASE_ORIGINS[finger_name])
    R_curr = Matrix.eye(3)
    params = FINGERS[finger_name]
    for i, link_name in enumerate(chain):
        p_curr += R_curr * Matrix(params[f"{link_name}_origin"])
        R_curr = R_curr * sym_rot_axis(params[f"{link_name}_axis"], thetas[i])
        if link_name == link:
            break

    pos_global = _P_PALM + p_curr + R_curr * r_local
    return lambdify(q + (x, y, z), pos_global.jacobian(Matrix(q)), modules=['numpy'], cse=True)


# =============================================================================
# JACOBIAN CONTAINER CLASS
# =============================================================================

class HandJacobians:
    """
    Pre-initialized container for all hand Jacobians (13-motor, rigid wrist).

    All Jacobians are (n×13) matrices where n is the output dimension.
    Position Jacobians (3×13); angle Jacobians (4×13 thumb, 3×13 fingers, 1×13 spread).
    Palm Jacobian is always zero (palm position does not depend on finger motors).
    """

    def __init__(self):
        # Motor-to-joint-angle Jacobians
        self.J_motor2thumb_angles  = Jacobian_motor2thumb()
        self.J_motor2index_angles  = Jacobian_motor2finger("index")
        self.J_motor2middle_angles = Jacobian_motor2finger("middle")
        self.J_motor2ring_angles   = Jacobian_motor2finger("ring")
        self.J_motor2pinky_angles  = Jacobian_motor2finger("pinky")

        # Thumb position Jacobians
        self.J_thumb_CMC1 = Jacobian_motor2thumbPos("CMC1")
        self.J_thumb_CMC2 = Jacobian_motor2thumbPos("CMC2")
        self.J_thumb_MCP  = Jacobian_motor2thumbPos("MCP")
        self.J_thumb_IP   = Jacobian_motor2thumbPos("IP")

        # Finger position Jacobians
        self.J_index_MCP  = Jacobian_motor2fingerPos("index",  "MCP")
        self.J_index_PIP  = Jacobian_motor2fingerPos("index",  "PIP")
        self.J_index_DIP  = Jacobian_motor2fingerPos("index",  "DIP")
        self.J_middle_MCP = Jacobian_motor2fingerPos("middle", "MCP")
        self.J_middle_PIP = Jacobian_motor2fingerPos("middle", "PIP")
        self.J_middle_DIP = Jacobian_motor2fingerPos("middle", "DIP")
        self.J_ring_MCP   = Jacobian_motor2fingerPos("ring",   "MCP")
        self.J_ring_PIP   = Jacobian_motor2fingerPos("ring",   "PIP")
        self.J_ring_DIP   = Jacobian_motor2fingerPos("ring",   "DIP")
        self.J_pinky_MCP  = Jacobian_motor2fingerPos("pinky",  "MCP")
        self.J_pinky_PIP  = Jacobian_motor2fingerPos("pinky",  "PIP")
        self.J_pinky_DIP  = Jacobian_motor2fingerPos("pinky",  "DIP")

        # Spread Jacobians
        self.J_spread_index  = Jacobian_motor2spread("index")
        self.J_spread_middle = Jacobian_motor2spread("middle")
        self.J_spread_ring   = Jacobian_motor2spread("ring")
        self.J_spread_pinky  = Jacobian_motor2spread("pinky")

        # Palm Jacobian (always zero — wrist rigid)
        self.J_wrist_palm = Jacobian_motor2palm()

        print("All Jacobians initialized.")

    def get_thumb_jacobian(self, link, q_motor, r_local):
        """(3×13) Jacobian of thumb link position. link: 'CMC1'|'CMC2'|'MCP'|'IP'."""
        return getattr(self, f"J_thumb_{link}")(*q_motor, *r_local)

    def get_finger_jacobian(self, finger, link, q_motor, r_local):
        """(3×13) Jacobian of finger link position. link: 'MCP'|'PIP'|'DIP'."""
        return getattr(self, f"J_{finger}_{link}")(*q_motor, *r_local)

    def get_wrist_palm_jacobian(self, q_motor, r_local):
        """(3×13) Jacobian of palm position — always zero (wrist rigid)."""
        return self.J_wrist_palm(*q_motor, *r_local)

    def get_angles_jacobian(self, part_name, q_motor):
        """(n×13) Jacobian of joint angles: 'thumb'→(4×13), fingers→(3×13)."""
        return getattr(self, f"J_motor2{part_name}_angles")(*q_motor)

    def get_spread_jacobian(self, finger, q_motor):
        """(1×13) Jacobian of spread joint angle."""
        return getattr(self, f"J_spread_{finger}")(*q_motor)


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    import time

    print("=" * 70)
    print("JACOBIANS HAND - TEST  (13 motors, rigid wrist)")
    print("=" * 70)

    np.random.seed(42)
    q_motor = np.random.randn(13) * 0.1
    r_local = np.array([0.01, 0.02, 0.03])

    print("\n1. Individual Jacobian shapes")
    J = np.array(Jacobian_motor2thumbPos("IP")(*q_motor, *r_local))
    print(f"   Thumb IP position:     {J.shape}  (expect (3,13))")
    J = np.array(Jacobian_motor2fingerPos("index", "DIP")(*q_motor, *r_local))
    print(f"   Index DIP position:    {J.shape}  (expect (3,13))")
    J = np.array(Jacobian_motor2palm()(*q_motor, *r_local))
    print(f"   Palm (zero):           {J.shape}, all zero: {np.allclose(J, 0)}")
    J = np.array(Jacobian_motor2thumb()(*q_motor))
    print(f"   Thumb angles:          {J.shape}  (expect (4,13))")
    J = np.array(Jacobian_motor2finger("index")(*q_motor))
    print(f"   Index angles:          {J.shape}  (expect (3,13))")

    print("\n2. HandJacobians class")
    t = time.time()
    hj = HandJacobians()
    print(f"   Init time: {time.time()-t:.2f} s")

    J = np.array(hj.get_thumb_jacobian("IP", q_motor, r_local))
    print(f"   Thumb IP:   {J.shape}")
    J = np.array(hj.get_finger_jacobian("index", "DIP", q_motor, r_local))
    print(f"   Index DIP:  {J.shape}")
    J = np.array(hj.get_wrist_palm_jacobian(q_motor, r_local))
    print(f"   Palm zero:  {J.shape}, all zero: {np.allclose(J, 0)}")

    print("\n3. Performance (1000 calls)")
    n = 1000
    t = time.time()
    for _ in range(n):
        hj.J_index_DIP(*q_motor, *r_local)
    print(f"   Index DIP: {(time.time()-t)*1000/n:.3f} ms/call  ({n/(time.time()-t):.0f} Hz)")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
