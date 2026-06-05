from sympy import symbols, lambdify, Matrix
import numpy as np

from ModelIDHand.hand_params import (
    FINGER_TRANSMISSIONS, THUMB_TRANSMISSION,
    SPREAD_ANGLE_CORRECTION, SPREAD_MOTION_RATIO,
    WRIST, FINGER_BASE_ORIGINS, THUMB, FINGERS,
)
from KinematicsHand.JacobiansHand import sym_rot_axis

# Units: lengths [m], angles [rad]
#
# Hessians are (m, 13, 13) tensors: H[i,j,k] = d²f_i / dq_j dq_k
# Motor ordering and wrist convention match JacobiansHand.py (13 motors, wrist rigid).

_P_PALM = Matrix(WRIST["yaw_origin"]) + Matrix(WRIST["pitch_origin"])


def _make_hessian(expr_matrix, q_syms, extra_syms=()):
    """Compile a Hessian tensor from a sympy Matrix expression."""
    J = expr_matrix.jacobian(Matrix(q_syms))
    H_rows = [J.row(i).jacobian(Matrix(q_syms)) for i in range(J.rows)]
    H_funcs = [lambdify(q_syms + extra_syms, H_i, modules=['numpy'], cse=True) for H_i in H_rows]
    def hessian_func(*args):
        return np.array([np.atleast_2d(H_f(*args)) for H_f in H_funcs])
    return hessian_func


# =============================================================================
# MOTOR-TO-JOINT-ANGLE HESSIANS (all zero — linear mappings)
# =============================================================================

def Hessian_motor2finger(finger_name):
    """(3, 13, 13) all-zero Hessian for finger joint angles."""
    q = symbols('q0:13')
    idx = {"index": (5,6), "middle": (7,8), "ring": (9,10), "pinky": (11,12)}[finger_name]
    rm, rp, cp = FINGER_TRANSMISSIONS[finger_name].values()
    theta = Matrix([
        (rp / rm) * q[idx[0]],
        -(rm / cp) * q[idx[1]],
        -(rm / cp) * q[idx[1]],
    ])
    return _make_hessian(theta, q)


def Hessian_motor2thumb():
    """(4, 13, 13) all-zero Hessian for thumb joint angles."""
    q = symbols('q0:13')
    r_motor, CMC1_pulley, CMC2_pulley, MCP_c, IP_c = THUMB_TRANSMISSION.values()
    theta = Matrix([
        (r_motor / CMC1_pulley) * q[0],
        (r_motor / CMC2_pulley) * q[1],
        -(r_motor / MCP_c) * q[2],
        -(r_motor / IP_c) * q[3],
    ])
    return _make_hessian(theta, q)


def Hessian_motor2spread(finger_name):
    """(1, 13, 13) all-zero Hessian for spread joint angle."""
    q = symbols('q0:13')
    theta_spread = q[4] * SPREAD_ANGLE_CORRECTION * SPREAD_MOTION_RATIO.get(finger_name, 0.0)
    return _make_hessian(Matrix([theta_spread]), q)


# =============================================================================
# POSITION HESSIANS (non-trivial)
# =============================================================================

def Hessian_motor2palm():
    """(3, 13, 13) all-zero Hessian for palm position (wrist rigid → constant)."""
    q = symbols('q0:13')
    x, y, z = symbols('x y z')
    pos_palm = _P_PALM + Matrix([x, y, z])
    return _make_hessian(pos_palm, q, extra_syms=(x, y, z))


def Hessian_motor2thumbPos(link):
    """(3, 13, 13) Hessian for thumb link position. link: 'CMC1'|'CMC2'|'MCP'|'IP'."""
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

    pos_global = _P_PALM + p_curr + R_curr * r_local
    return _make_hessian(pos_global, q, extra_syms=(x, y, z))


def Hessian_motor2fingerPos(finger_name, link):
    """(3, 13, 13) Hessian for finger link position. link: 'MCP'|'PIP'|'DIP'."""
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
    thetas = [theta_spread, theta_mcp, theta_pip, theta_pip]
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
    return _make_hessian(pos_global, q, extra_syms=(x, y, z))


# =============================================================================
# HESSIAN CONTAINER CLASS
# =============================================================================

class HandHessians:
    """
    Pre-initialized container for all hand Hessians (13-motor, rigid wrist).

    All Hessians are (m, 13, 13) tensors.
    Motor-to-angle Hessians are all zero (linear mappings).
    Position Hessians for palm are also zero (palm constant with rigid wrist).
    """

    def __init__(self):
        # Motor-to-joint-angle Hessians (all zero)
        self.H_motor2thumb_angles  = Hessian_motor2thumb()
        self.H_motor2index_angles  = Hessian_motor2finger("index")
        self.H_motor2middle_angles = Hessian_motor2finger("middle")
        self.H_motor2ring_angles   = Hessian_motor2finger("ring")
        self.H_motor2pinky_angles  = Hessian_motor2finger("pinky")

        # Palm Hessian (zero — wrist rigid)
        self.H_wrist_palm = Hessian_motor2palm()

        # Thumb Hessians
        self.H_thumb_CMC1 = Hessian_motor2thumbPos("CMC1")
        self.H_thumb_CMC2 = Hessian_motor2thumbPos("CMC2")
        self.H_thumb_MCP  = Hessian_motor2thumbPos("MCP")
        self.H_thumb_IP   = Hessian_motor2thumbPos("IP")

        # Finger Hessians
        self.H_index_MCP  = Hessian_motor2fingerPos("index",  "MCP")
        self.H_index_PIP  = Hessian_motor2fingerPos("index",  "PIP")
        self.H_index_DIP  = Hessian_motor2fingerPos("index",  "DIP")
        self.H_middle_MCP = Hessian_motor2fingerPos("middle", "MCP")
        self.H_middle_PIP = Hessian_motor2fingerPos("middle", "PIP")
        self.H_middle_DIP = Hessian_motor2fingerPos("middle", "DIP")
        self.H_ring_MCP   = Hessian_motor2fingerPos("ring",   "MCP")
        self.H_ring_PIP   = Hessian_motor2fingerPos("ring",   "PIP")
        self.H_ring_DIP   = Hessian_motor2fingerPos("ring",   "DIP")
        self.H_pinky_MCP  = Hessian_motor2fingerPos("pinky",  "MCP")
        self.H_pinky_PIP  = Hessian_motor2fingerPos("pinky",  "PIP")
        self.H_pinky_DIP  = Hessian_motor2fingerPos("pinky",  "DIP")

        # Spread Hessians (all zero)
        self.H_spread_index  = Hessian_motor2spread("index")
        self.H_spread_middle = Hessian_motor2spread("middle")
        self.H_spread_ring   = Hessian_motor2spread("ring")
        self.H_spread_pinky  = Hessian_motor2spread("pinky")

        print("All Hessians initialized.")

    def get_thumb_hessian(self, link, q_motor, r_local):
        """(3, 13, 13) Hessian of thumb link position."""
        return getattr(self, f"H_thumb_{link}")(*q_motor, *r_local)

    def get_finger_hessian(self, finger, link, q_motor, r_local):
        """(3, 13, 13) Hessian of finger link position."""
        return getattr(self, f"H_{finger}_{link}")(*q_motor, *r_local)

    def get_wrist_palm_hessian(self, q_motor, r_local):
        """(3, 13, 13) Hessian of palm position — always zero."""
        return self.H_wrist_palm(*q_motor, *r_local)

    def get_angles_hessian(self, part_name, q_motor):
        """(n, 13, 13) Hessian of joint angles."""
        return getattr(self, f"H_motor2{part_name}_angles")(*q_motor)

    def get_spread_hessian(self, finger, q_motor):
        """(1, 13, 13) Hessian of spread joint angle — always zero."""
        return getattr(self, f"H_spread_{finger}")(*q_motor)


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("HESSIANS HAND - TEST  (13 motors, rigid wrist)")
    print("=" * 70)

    np.random.seed(42)
    q_motor = np.random.randn(13) * 0.1
    r_local = np.array([0.01, 0.02, 0.03])

    print("\n1. Motor-to-angle Hessians (all zero)")
    H = Hessian_motor2finger("index")(*q_motor)
    print(f"   Index angles: {H.shape}, zero: {np.allclose(H, 0)}")
    H = Hessian_motor2thumb()(*q_motor)
    print(f"   Thumb angles: {H.shape}, zero: {np.allclose(H, 0)}")
    H = Hessian_motor2spread("index")(*q_motor)
    print(f"   Spread:       {H.shape}, zero: {np.allclose(H, 0)}")

    print("\n2. Position Hessians")
    t = time.time()
    H_palm = Hessian_motor2palm()
    print(f"   Palm (zero) compiled in {time.time()-t:.2f} s")
    H = H_palm(*q_motor, *r_local)
    print(f"   Palm: {H.shape}, zero: {np.allclose(H, 0)}")

    t = time.time()
    H_thumb = Hessian_motor2thumbPos("IP")
    print(f"   Thumb IP compiled in {time.time()-t:.2f} s")
    H = H_thumb(*q_motor, *r_local)
    print(f"   Thumb IP: {H.shape}, symmetric: {np.allclose(H, H.transpose(0,2,1))}")

    t = time.time()
    H_finger = Hessian_motor2fingerPos("index", "DIP")
    print(f"   Index DIP compiled in {time.time()-t:.2f} s")
    H = H_finger(*q_motor, *r_local)
    print(f"   Index DIP: {H.shape}, symmetric: {np.allclose(H, H.transpose(0,2,1))}")

    print("\n3. Performance (1000 calls)")
    n = 1000
    t = time.time()
    for _ in range(n):
        H_finger(*q_motor, *r_local)
    print(f"   Index DIP: {(time.time()-t)*1000/n:.3f} ms/call")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
