"""
Compute task space stiffness from VMC finger space stiffness and Jacobian.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KinematicsFinger.FK_Finger import *
from KinematicsFinger.JacobiansFinger import *
from KinematicsFinger.HessiansFinger import *
from ModelIDFinger.finger_params import c


class tip_stiffness_FingerSpace():
    """
    Compute tip stiffness matrix for VMC stiffness model.

    Uses constant efficiency model:
        η = diag([η_MCP, η_PIP])
    """

    def __init__(self, n=np.array([0, 0, 1]), rtip=np.array([0, c, 0]),
                 eta=None):
        """
        Initialize Jacobians and Hessians.

        Args:
            n: 3x1 vector orthogonal to the cart plane (must be normalized).
               E.g. n = [1, 0, 0] means stiffness acts along x only.
            rtip: 3x1 vector of tip position in FingerSpace [m]
            eta: 2x1 efficiency for each motor (default [1, 1])
        """
        self.J_theta = q2joint()
        self.J_tip = q2DIP()
        self.H_theta = qq2joint()
        self.H_tip = qq2DIP()
        if n is not None:
            n = np.asarray(n).reshape(-1, 1)
            n = n / np.linalg.norm(n)
            self.P = n @ n.T
        else:
            self.P = np.eye(3)
        self.rtip = rtip

        # Constant efficiency parameters
        eta_vec = np.asarray(eta) if eta is not None else np.array([1.0, 1.0])
        self.eta = np.diag(eta_vec)

    def motor_stiffness(self, q, K, theta_ref_deg=None, f_ext=None):
        """
        Compute motor stiffness matrix for VMC stiffness model.

        Args:
            q:             (2,) motor angles [deg]
            K:             (3,3) stiffness matrix in FingerSpace [N/rad]
            theta_ref_deg: (3,) joint-space reference angles [deg]
            f_ext:         (3,) external force at the tip, if 2nd order term [N]

        Returns:
            K_motor: 2x2 motor stiffness matrix [N/m]
        """
        # Convert to radians
        q_rad = np.radians(q)
        J_theta = self.J_theta(q_rad[0], q_rad[1])

        # Compute motor space stiffness (1st order)
        K_motor = self.eta @ J_theta.T @ K @ J_theta

        # Add Hessian correction (2nd order, geometric stiffness)
        if f_ext is not None:
            H_theta = self.H_theta(q_rad[0], q_rad[1])  # (3, 2, 2)
            H_tip_P = np.tensordot(self.P, self.H_tip(q_rad[0], q_rad[1], self.rtip[0], self.rtip[1], self.rtip[2]), axes=([1], [0]))

            # Compute joint deflection
            theta_ref_ = np.radians(theta_ref_deg) if theta_ref_deg is not None else np.zeros(3)
            delta_theta = theta_ref_ - motor_to_joint(q_rad)

            # Compute geometric stiffness
            K_g_1 = np.tensordot(f_ext, H_tip_P, axes=([0], [0]))  # (2, 2)

            # Exact virtual torque from spring deflection (CCT formulation)
            tau_theta = K @ delta_theta
            K_g_2 = - self.eta @ np.tensordot(tau_theta, H_theta, axes=([0], [0]))  # (2, 2)

            K_g = K_g_1 + K_g_2
            K_motor += K_g

        return K_motor

    def tip_stiffness(self, q, K, theta_ref_deg=None, f_ext=None):
        """
        Compute tip stiffness matrix for VMC stiffness model.

        Args:
            q:             (2,) motor angles [deg]
            K:             (3,3) stiffness matrix in FingerSpace [N/rad]
            theta_ref_deg: (3,) joint-space reference angles [deg]
            f_ext:         (3,) external force at the tip, if 2nd order term [N]

        Returns:
            K_tip: 3x3 tip stiffness matrix [N/m]
        """
        # Convert q to radians and compute Jacobian
        q_rad = np.radians(q)
        J_tip_P = self.P @ self.J_tip(q_rad[0], q_rad[1], self.rtip[0], self.rtip[1], self.rtip[2])
        J_pinv_tip_P = np.linalg.pinv(J_tip_P)

        # Compute tip stiffness using motor stiffness and Jacobian
        K_motor = self.motor_stiffness(q, K, theta_ref_deg, f_ext)
        K_tip = J_pinv_tip_P.T @ K_motor @ J_pinv_tip_P
        return K_tip

    def tip_force(self, q, theta_ref_deg, K):
        """
        Compute the force the fingertip exerts on the environment.

        This is the contact force when the finger is in static equilibrium,
        pushing against an obstacle due to spring deflection.

        Args:
            q:             (2,) current motor angles [deg]
            theta_ref_deg: (3,) joint-space reference angles [deg]
            K:             (3,3) stiffness matrix in FingerSpace [N/rad]

        Returns:
            f: 3x1 vector of force exerted by fingertip on environment [N]
        """
        # Convert to radians
        q_rad = np.radians(q)
        theta_ref = np.radians(theta_ref_deg)
        J_theta = self.J_theta(q_rad[0], q_rad[1])
        J_tip_P = self.P @ self.J_tip(q_rad[0], q_rad[1], self.rtip[0], self.rtip[1], self.rtip[2])
        J_pinv_tip_P = np.linalg.pinv(J_tip_P)

        # Compute joint deflection and motor torques
        delta_theta = theta_ref - motor_to_joint(q_rad)
        tau = J_theta.T @ K @ delta_theta

        # Compute force exerted by fingertip on environment
        # f_x = ((P · J_x)*)^T · η · J_θ^T · K · (θ_ref − θ)
        f = J_pinv_tip_P.T @ self.eta @ tau

        return f
    
    def stiffness_inversion(self, q, K_des):
        """
        Minimum-norm K_ff such that tip_stiffness(q, K_ff) best approximates K_des.

        Vectorises the forward model  K_x = L @ K @ R  as
            vec(K_x) = (R^T ⊗ L) · vec(K),
        where  L = J_pinv^T · η · J_θ^T,  R = J_θ · J_pinv,
        and returns the minimum-norm least-squares inverse K_ff = M^+ · vec(K_des).

        Args:
            q:     (2,) current motor angles [deg]
            K_des: (3,3) desired tip stiffness [N/m]

        Returns:
            K_ff: (3,3) finger-space stiffness that best produces K_des
        """
        q_rad   = np.radians(q)
        J_th    = self.J_theta(q_rad[0], q_rad[1])
        J_tip_P = self.P @ self.J_tip(q_rad[0], q_rad[1],
                                      self.rtip[0], self.rtip[1], self.rtip[2])
        J_pinv  = np.linalg.pinv(J_tip_P)
        L = J_pinv.T @ self.eta @ J_th.T   # (3,3)
        R = J_th @ J_pinv                   # (3,3)
        M = np.kron(R.T, L)                 # (9,9)
        return (np.linalg.pinv(M) @ K_des.flatten('F')).reshape(3, 3, order='F')
    
    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def stiffness_descent(self, q, theta_ref_deg, K, f_meas, f_des, lr=1e-4):
        """
        Perform one step of gradient descent on K to reduce the force error.

        Minimises  L = ½ ||f_meas - f_des||²  w.r.t. K.
        Gradient:  ∂L/∂vec(K) = Cᵀ · (f_meas - f_des)

        Args:
            q:             (2,) current motor angles [deg]
            theta_ref_deg: (3,) joint-space reference angles [deg]
            K:             (3,3) stiffness matrix in FingerSpace [N/rad]
            f_meas:        (3,) measured (or predicted) tip force [N]
            f_des:         (3,) desired tip force [N]
            lr:            learning rate

        Returns:
            K_new: (3, 3) updated stiffness matrix
        """
        q_rad = np.radians(q)
        theta_ref = np.radians(theta_ref_deg)

        J_theta = self.J_theta(q_rad[0], q_rad[1])
        J_tip_P = self.P @ self.J_tip(q_rad[0], q_rad[1], self.rtip[0], self.rtip[1], self.rtip[2])
        J_pinv_tip_P = np.linalg.pinv(J_tip_P)

        delta_theta = theta_ref - motor_to_joint(q_rad)
        A = J_pinv_tip_P.T @ self.eta @ J_theta.T

        error = f_meas - f_des
        grad_K = A.T @ error.reshape(-1, 1) @ delta_theta.reshape(1, -1)

        return K - lr * grad_K

    def ref_descent(self, q, K, theta_ref_deg, f_meas, f_des, lr=4.84e-03):
        """
        One gradient descent step on theta_ref_deg to reduce force error.

        Args:
            q:             (2,) current motor angles [deg]
            K:             (3,3) stiffness matrix in FingerSpace [N/rad]
            theta_ref_deg: (3,) current VMC target [deg]
            f_meas:        (3,) measured (or predicted) tip force [N]
            f_des:         (3,) desired tip force [N]
            lr:            learning rate [deg/N · (180/π)⁻¹] — same as the [rad/N] value

        Returns:
            theta_ref_deg_new: (3,) updated VMC target [deg]
        """
        q_rad = np.radians(q)

        J_theta = self.J_theta(q_rad[0], q_rad[1])
        J_tip_P = self.P @ self.J_tip(q_rad[0], q_rad[1], self.rtip[0], self.rtip[1], self.rtip[2])
        J_pinv_tip_P = np.linalg.pinv(J_tip_P)

        S = J_pinv_tip_P.T @ self.eta @ J_theta.T @ K
        error = f_meas - f_des
        return theta_ref_deg - lr * np.degrees(S.T @ error) # stay in degrees


if __name__ == "__main__":
    q_base        = np.array([30.0, 20.0])
    q_ref         = np.array([50.0, 50.0])
    theta_ref_deg = np.degrees(motor_to_joint(np.radians(q_ref)))
    K   = np.diag([0.5, 0.3, 0.2])
    eta = np.array([0.8, 0.7])

    model = tip_stiffness_FingerSpace(n=None, eta=eta)

    # Ground truth: actual force at q_base
    f_true = model.tip_force(q_base, theta_ref_deg, K)
    print(f"True force at q_base: {f_true}")

    # --- FD stiffness validation: K_tip · Δx ≈ -Δf ---
    print("\n--- FD stiffness validation (finger space) ---")
    np.random.seed(1)
    f0 = model.tip_force(q_base, theta_ref_deg, K)
    K1 = model.tip_stiffness(q_base, K, theta_ref_deg)
    K2 = model.tip_stiffness(q_base, K, theta_ref_deg, f_ext=f0)
    x0 = FK_DIP(np.radians(q_base), model.rtip)
    for eps in [0.05, 0.005, 0.0005]:
        errs_1, errs_2 = [], []
        for _ in range(6):
            dq = np.random.randn(2)
            dq = dq * (eps / np.linalg.norm(dq))
            dx = FK_DIP(np.radians(q_base + dq), model.rtip) - x0
            df = model.tip_force(q_base + dq, theta_ref_deg, K) - f0
            scale = np.linalg.norm(df) + 1e-15
            errs_1.append(np.linalg.norm(K1 @ dx + df) / scale)
            errs_2.append(np.linalg.norm(K2 @ dx + df) / scale)
        print(f"  eps={eps:.4f}°  1st-order rel={np.mean(errs_1):.2e}"
              f"  2nd-order rel={np.mean(errs_2):.2e}"
              f"  (ratio {np.mean(errs_1) / np.mean(errs_2):.1f}x)")

    # --- stiffness_descent convergence test ---
    print("\n--- stiffness_descent convergence test (finger space) ---")
    f_des = np.array([0.0, 0.05, 0.03])
    K_opt = np.diag([0.5, 0.3, 0.2])
    MAX_ITERS = 10000
    for i in range(MAX_ITERS):
        f_meas = model.tip_force(q_base, theta_ref_deg, K_opt)
        err = np.linalg.norm(f_meas - f_des)
        if i % 200 == 0:
            print(f"  iter {i:4d}  ||error|| = {err:.6f}")
        K_opt = model.stiffness_descent(
            q_base, theta_ref_deg, K_opt, f_meas, f_des, lr=5e-4)
    print(f"  Final ||error|| = {np.linalg.norm(model.tip_force(q_base, theta_ref_deg, K_opt) - f_des):.6f}")

    # --- ref_descent convergence test ---
    print("\n--- ref_descent convergence test (finger space) ---")
    f_des = np.array([0.0, 0.05, 0.03])
    theta_ref_opt_deg = np.degrees(motor_to_joint(np.radians(q_ref)))
    err = float('inf')
    for i in range(MAX_ITERS):
        f_meas = model.tip_force(q_base, theta_ref_opt_deg, K)
        err = np.linalg.norm(f_meas - f_des)
        if i % 200 == 0:
            print(f"  iter {i:4d}  ||error|| = {err:.6f}")
        theta_ref_opt_deg = model.ref_descent(
            q_base, K, theta_ref_opt_deg, f_meas, f_des, lr=1e-3)
    print(f"  Final ||error|| = {err:.6f}")

    # --- stiffness_inversion check ---
    print("\n--- stiffness_inversion check (finger space) ---")
    K_target = np.diag([0.8, 0.4, 0.25])
    K_des    = model.tip_stiffness(q_base, K_target)  # achievable target
    K_ff     = model.stiffness_inversion(q_base, K_des)
    K_recon  = model.tip_stiffness(q_base, K_ff)
    recon_err = np.linalg.norm(K_recon - K_des, 'fro')
    print(f"  ||K_recon - K_des||_F = {recon_err:.6f}")
