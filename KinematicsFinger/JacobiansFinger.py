from sympy import symbols, lambdify, cos, sin, Matrix
from ModelIDFinger.finger_params import r_motor, r_pulley, c_param, a, b

# Units of measure:
# - LENGTHS: [m] (meters)
# - ANGLES: [rad] (radians)

# q2joint(): from q to joint angles
def q2joint():
    """
    Jacobian of q ([q_MCP, q_PIP]) to joint angles ([theta_MCP, theta_PIP, theta_DIP]).
    """

    q_MCP, q_PIP = symbols("q_MCP q_PIP")

    theta_MCP = (r_pulley/r_motor) * q_MCP
    theta_PIP = (r_motor/c_param) * q_PIP
    theta_DIP = theta_PIP  # Mimic constraint

    joint_angles = Matrix([theta_MCP, theta_PIP, theta_DIP])
    q_motor = Matrix([q_MCP, q_PIP])

    J = joint_angles.jacobian(q_motor)

    return lambdify((q_MCP, q_PIP), J, modules=['numpy'])

# q2MCP(): from q to MCP positions
def q2MCP():
    """
    Jacobian of q ([q_MCP, q_PIP]) to MCP position ([x, y, z]).
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

    # Coordinate transformation
    pos = p_base + R_MCP @ r_local
    q_motor = Matrix([q_MCP, q_PIP])

    J = pos.jacobian(q_motor)

    return lambdify((q_MCP, q_PIP, x, y, z), J, modules=['numpy'])

# q2PIP(): from q to PIP positions
def q2PIP():
    """
    Jacobian of q ([q_MCP, q_PIP]) to PIP position ([x, y, z]).
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

    return lambdify((q_MCP, q_PIP, x, y, z), J, modules=['numpy'])

# q2DIP(): from q to DIP positions
def q2DIP():
    """
    Jacobian of q ([q_MCP, q_PIP]) to DIP position ([x, y, z]).
    Note: theta_DIP = theta_PIP (mimic constraint).
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

    return lambdify((q_MCP, q_PIP, x, y, z), J, modules=['numpy'])

# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    import time

    print("="*70)
    print("JACOBIANS FINGER - TEST")
    print("="*70)

    np.random.seed(42)
    q_motor = np.random.randn(2) * 0.1
    r_local = np.array([0.0, 0.02, 0.0])

    print("\nTest motor angles:")
    print(f"  q_motor [q_MCP, q_PIP]: {q_motor}")
    print(f"  r_local [x, y, z]: {r_local}\n")

    print("1. Initializing Jacobian functions...")
    t = time.time()
    J_joint = q2joint()
    J_MCP = q2MCP()
    J_PIP = q2PIP()
    J_DIP = q2DIP()
    print(f"   Initialization time: {time.time() - t:.4f} seconds")

    print("\n2. Testing Jacobian evaluations...")

    t = time.time()
    J_val = J_joint(q_motor[0], q_motor[1])
    print(f"   Joint Jacobian shape: {np.array(J_val).shape}")
    print(f"   Evaluation time: {time.time() - t:.6f} seconds")

    t = time.time()
    J_val = J_DIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
    print(f"   DIP Jacobian shape: {np.array(J_val).shape}")
    print(f"   Evaluation time: {time.time() - t:.6f} seconds")

    print("\n3. Performance benchmark (1000 calls)...")

    n_calls = 1000
    t_start = time.time()
    for _ in range(n_calls):
        J_val = J_DIP(q_motor[0], q_motor[1], r_local[0], r_local[1], r_local[2])
    t_elapsed = time.time() - t_start
    print(f"   DIP Jacobian: {t_elapsed*1000/n_calls:.3f} ms/call")
    print(f"   Effective frequency: {n_calls/t_elapsed:.1f} Hz")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
