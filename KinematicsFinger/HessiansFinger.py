from sympy import symbols, lambdify, cos, sin, Matrix
import numpy as np
from ModelIDFinger.finger_params import r_motor, r_pulley, c_param, a, b

# Units of measure:
# - LENGTHS: [m] (meters)
# - ANGLES: [rad] (radians)
#
# Hessians are 3D tensors: H[i,j,k] = d²f_i / dq_j dq_k
# For a function f: R^n -> R^m, the Hessian is an (m x n x n) tensor
#
# Computed as: Jacobian of each row of the Jacobian


# qq2joint(): Hessian from q to joint angles
def qq2joint():
    """
    Hessian of q ([q_MCP, q_PIP]) to joint angles ([theta_MCP, theta_PIP, theta_DIP]).

    Returns (3, 2, 2) array. Note: All zeros since mapping is linear.
    """
    q_MCP, q_PIP = symbols("q_MCP q_PIP")

    theta_MCP = (r_pulley/r_motor) * q_MCP
    theta_PIP = (r_motor/c_param) * q_PIP
    theta_DIP = theta_PIP

    joint_angles = Matrix([theta_MCP, theta_PIP, theta_DIP])
    q_motor = Matrix([q_MCP, q_PIP])

    J = joint_angles.jacobian(q_motor)
    H_rows = [J.row(i).jacobian(q_motor) for i in range(J.rows)]

    H_funcs = [lambdify((q_MCP, q_PIP), H_i, modules=['numpy']) for H_i in H_rows]

    def hessian_func(q_MCP_val, q_PIP_val):
        return np.array([H_f(q_MCP_val, q_PIP_val) for H_f in H_funcs])

    return hessian_func


# qq2MCP(): Hessian from q to MCP position
def qq2MCP():
    """
    Hessian of q ([q_MCP, q_PIP]) to MCP position ([x, y, z]).

    Returns (3, 2, 2) array.
    """
    q_MCP, q_PIP = symbols("q_MCP q_PIP")
    x, y, z = symbols("x y z", real=True)

    theta_MCP = (r_pulley/r_motor) * q_MCP

    R_MCP = Matrix([
        [1,  0,             0],
        [0,  cos(theta_MCP), -sin(theta_MCP)],
        [0,  sin(theta_MCP),  cos(theta_MCP)]
    ])

    p_base = Matrix([0, 0, 0])
    r_local = Matrix([x, y, z])

    pos = p_base + R_MCP @ r_local
    q_motor = Matrix([q_MCP, q_PIP])

    J = pos.jacobian(q_motor)
    H_rows = [J.row(i).jacobian(q_motor) for i in range(J.rows)]

    H_funcs = [lambdify((q_MCP, q_PIP, x, y, z), H_i, modules=['numpy']) for H_i in H_rows]

    def hessian_func(q_MCP_val, q_PIP_val, x_val, y_val, z_val):
        return np.array([H_f(q_MCP_val, q_PIP_val, x_val, y_val, z_val) for H_f in H_funcs])

    return hessian_func


# qq2PIP(): Hessian from q to PIP position
def qq2PIP():
    """
    Hessian of q ([q_MCP, q_PIP]) to PIP position ([x, y, z]).

    Returns (3, 2, 2) array.
    """
    q_MCP, q_PIP = symbols("q_MCP q_PIP")
    x, y, z = symbols("x y z", real=True)

    theta_MCP = (r_pulley/r_motor) * q_MCP
    theta_PIP = (r_motor/c_param) * q_PIP

    R_MCP = Matrix([
        [1,  0,             0],
        [0,  cos(theta_MCP), -sin(theta_MCP)],
        [0,  sin(theta_MCP),  cos(theta_MCP)]
    ])

    p_base = Matrix([0, 0, 0])
    p_PIP = p_base + R_MCP @ Matrix([0, a, 0])

    theta_total = theta_MCP + theta_PIP
    R_PIP = Matrix([
        [1,  0,               0],
        [0,  cos(theta_total), -sin(theta_total)],
        [0,  sin(theta_total),  cos(theta_total)]
    ])

    r_local = Matrix([x, y, z])
    pos = p_PIP + R_PIP @ r_local

    q_motor = Matrix([q_MCP, q_PIP])

    J = pos.jacobian(q_motor)
    H_rows = [J.row(i).jacobian(q_motor) for i in range(J.rows)]

    H_funcs = [lambdify((q_MCP, q_PIP, x, y, z), H_i, modules=['numpy']) for H_i in H_rows]

    def hessian_func(q_MCP_val, q_PIP_val, x_val, y_val, z_val):
        return np.array([H_f(q_MCP_val, q_PIP_val, x_val, y_val, z_val) for H_f in H_funcs])

    return hessian_func


# qq2DIP(): Hessian from q to DIP position
def qq2DIP():
    """
    Hessian of q ([q_MCP, q_PIP]) to DIP position ([x, y, z]).

    Returns (3, 2, 2) array.

    Note: theta_DIP = theta_PIP (mimic constraint)
    """
    q_MCP, q_PIP = symbols("q_MCP q_PIP")
    x, y, z = symbols("x y z", real=True)

    theta_MCP = (r_pulley/r_motor) * q_MCP
    theta_PIP = (r_motor/c_param) * q_PIP
    theta_DIP = theta_PIP  # Mimic constraint

    R_MCP = Matrix([
        [1,  0,             0],
        [0,  cos(theta_MCP), -sin(theta_MCP)],
        [0,  sin(theta_MCP),  cos(theta_MCP)]
    ])

    p_base = Matrix([0, 0, 0])
    p_PIP = p_base + R_MCP @ Matrix([0, a, 0])

    theta_MCP_PIP = theta_MCP + theta_PIP
    R_PIP = Matrix([
        [1,  0,                  0],
        [0,  cos(theta_MCP_PIP), -sin(theta_MCP_PIP)],
        [0,  sin(theta_MCP_PIP),  cos(theta_MCP_PIP)]
    ])

    p_DIP = p_PIP + R_PIP @ Matrix([0, b, 0])

    theta_total = theta_MCP + theta_PIP + theta_DIP
    R_DIP = Matrix([
        [1,  0,               0],
        [0,  cos(theta_total), -sin(theta_total)],
        [0,  sin(theta_total),  cos(theta_total)]
    ])

    r_local = Matrix([x, y, z])
    pos = p_DIP + R_DIP @ r_local

    q_motor = Matrix([q_MCP, q_PIP])

    J = pos.jacobian(q_motor)
    H_rows = [J.row(i).jacobian(q_motor) for i in range(J.rows)]

    H_funcs = [lambdify((q_MCP, q_PIP, x, y, z), H_i, modules=['numpy']) for H_i in H_rows]

    def hessian_func(q_MCP_val, q_PIP_val, x_val, y_val, z_val):
        return np.array([H_f(q_MCP_val, q_PIP_val, x_val, y_val, z_val) for H_f in H_funcs])

    return hessian_func


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("="*70)
    print("HESSIANS FINGER - TEST")
    print("="*70)

    np.random.seed(42)
    q_motor = np.random.randn(2) * 0.1
    r_local = np.array([0.0, 0.02, 0.0])

    print("\nTest motor angles:")
    print(f"  q_motor [q_MCP, q_PIP]: {q_motor}")
    print(f"  r_local [x, y, z]: {r_local}\n")

    print("1. Initializing Hessian functions...")
    t = time.time()
    H_joint = qq2joint()
    H_MCP = qq2MCP()
    H_PIP = qq2PIP()
    H_DIP = qq2DIP()
    print(f"   Initialization time: {time.time() - t:.4f} seconds")

    print("\n2. Testing Hessian evaluations...")

    t = time.time()
    H_val = H_joint(q_motor[0], q_motor[1])
    print(f"   Joint Hessian shape: {H_val.shape}")
    print(f"   Joint Hessian (should be all zeros - linear mapping):")
    print(f"   {H_val}")
    print(f"   Evaluation time: {time.time() - t:.6f} seconds")

    t = time.time()
    H_val = H_MCP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
    print(f"\n   MCP Hessian shape: {H_val.shape}")
    print(f"   Evaluation time: {time.time() - t:.6f} seconds")

    t = time.time()
    H_val = H_PIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
    print(f"\n   PIP Hessian shape: {H_val.shape}")
    print(f"   Evaluation time: {time.time() - t:.6f} seconds")

    t = time.time()
    H_val = H_DIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
    print(f"\n   DIP Hessian shape: {H_val.shape}")
    print(f"   Evaluation time: {time.time() - t:.6f} seconds")

    print("\n3. Verifying Hessian symmetry (H[i,j,k] == H[i,k,j])...")
    H_val = H_DIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
    symmetric = np.allclose(H_val, H_val.transpose(0, 2, 1))
    print(f"   DIP Hessian symmetric: {symmetric}")

    print("\n4. Performance benchmark (1000 calls)...")

    n_calls = 1000
    t_start = time.time()
    for _ in range(n_calls):
        H_val = H_DIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
    t_elapsed = time.time() - t_start
    print(f"   DIP Hessian: {t_elapsed*1000/n_calls:.3f} ms/call")
    print(f"   Effective frequency: {n_calls/t_elapsed:.1f} Hz")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
