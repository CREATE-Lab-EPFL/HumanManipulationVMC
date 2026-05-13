"""
Compute task space stiffness from VMC task space stiffness and Jacobian.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KinematicsFinger.FK_Finger import *
from KinematicsFinger.JacobiansFinger import *
from KinematicsFinger.HessiansFinger import *
from ModelIDFinger.finger_params import a, c

class tip_stiffness_TaskSpace():
    """
    Compute tip stiffness matrix for VMC stiffness model.
    Can be:
        - Linear: without Hessian correction (1st order, Salisbury)
        - Nonlinear: with Hessian correction (2nd order, CCT geometric stiffness)
    """

    def __init__(self, n=np.array([0, 0, 1]), rtip=np.array([0, c, 0]),
                 rbase=np.array([0, a, 0]), eta=None):
        """
        Initialize Jacobians and Hessians.

        Args:
            n: 3x1 vector orthogonal to the cart plane (must be normalized).
               E.g. n = [1, 0, 0] means stiffness acts along x only.
            rtip: Position of fingertip in its local frame (m)
            rbase: Position of base attachment point in its local frame (m)
            eta: 2x2 tendon transmission efficiency matrix (diagonal, eta_i in (0, 1]).
                 If None, defaults to identity (no losses).
        """
        self.J_base = q2MCP()
        self.J_tip  = q2DIP()
        self.H_base = qq2MCP()
        self.H_tip  = qq2DIP()
        if n is not None:
            n = np.asarray(n).reshape(-1, 1)
            n = n / np.linalg.norm(n)
            self.P = n @ n.T
        else:
            self.P = np.eye(3)
        self.rtip  = rtip
        self.rbase = rbase
        self.eta   = eta if eta is not None else np.eye(2)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _jacs(self, q_rad):
        """
        Evaluate J_tip, J_base, and J_pinv_tip_P at q_rad (radians).
        """
        J_ti = self.J_tip( q_rad[0], q_rad[1], self.rtip[0],  self.rtip[1],  self.rtip[2])
        J_ba = self.J_base(q_rad[0], q_rad[1], self.rbase[0], self.rbase[1], self.rbase[2])
        return J_ti, J_ba, np.linalg.pinv(self.P @ J_ti)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def motor_stiffness(self, q, K_tip, K_base, d_ref=None, f_ext=None):
        """
        Compute motor stiffness matrix for VMC stiffness model.

        Args:
            q: (2,) motor angles [deg]
            K_tip: (3,3) stiffness matrix at fingertip [N/m]
            K_base: (3,3) stiffness matrix at base [N/m]
            d_ref: dict with 'tip' (3,) and 'base' (3,) reference positions [m]
                   (required when f_ext is given)
            f_ext: (3,) external force at the tip for 2nd-order correction [N]

        Returns:
            K_motor: (2,2) motor stiffness matrix [N/m]
        """
        q_rad = np.radians(q)
        J_ti, J_ba, _ = self._jacs(q_rad)

        # 1st order (Salisbury)
        K_motor = self.eta @ (J_ba.T @ K_base @ J_ba + J_ti.T @ K_tip @ J_ti)

        if f_ext is not None:
            H_tip   = self.H_tip( q_rad[0], q_rad[1], self.rtip[0],  self.rtip[1],  self.rtip[2])
            H_base  = self.H_base(q_rad[0], q_rad[1], self.rbase[0], self.rbase[1], self.rbase[2])
            H_tip_P = np.tensordot(self.P, H_tip, axes=([1], [0]))

            K_g_out = np.tensordot(f_ext, H_tip_P, axes=([0], [0]))

            # Exact virtual torques from spring deflection (CCT formulation)
            tau_tip  = K_tip  @ (d_ref['tip']  - FK_DIP(q_rad, self.rtip))
            tau_base = K_base @ (d_ref['base'] - FK_MCP(q_rad, self.rbase))
            K_g_in   = self.eta @ np.tensordot(
                np.concatenate([tau_tip, tau_base]),
                np.vstack([H_tip, H_base]),
                axes=([0], [0]))

            K_motor += K_g_out - K_g_in

        return K_motor

    def tip_stiffness(self, q, K_tip, K_base, d_ref=None, f_ext=None):
        """
        Compute tip stiffness matrix for VMC stiffness model.

        Args:
            q: (2,) motor angles [deg]
            K_tip: (3,3) stiffness matrix at fingertip [N/m]
            K_base: (3,3) stiffness matrix at base [N/m]
            d_ref: dict with 'tip' (3,) and 'base' (3,) reference positions [m]
                   (required when f_ext is given)
            f_ext: (3,) external force at the tip for 2nd-order correction [N]

        Returns:
            K_x: (3,3) tip stiffness matrix [N/m]
        """
        _, _, J_pinv = self._jacs(np.radians(q))
        return J_pinv.T @ self.motor_stiffness(q, K_tip, K_base, d_ref, f_ext) @ J_pinv

    def tip_force(self, q, d_ref, K_tip, K_base):
        """
        Compute the force the fingertip exerts on the environment.

        f_x = ((P·J_x)*)^T · η · (J_tip^T·K_tip·Δx_tip + J_base^T·K_base·Δx_base)

        Args:
            q: (2,) current motor angles [deg]
            d_ref: dict with 'tip' (3,) and 'base' (3,) reference positions [m]
            K_tip: (3,3) stiffness matrix at fingertip [N/m]
            K_base: (3,3) stiffness matrix at base [N/m]

        Returns:
            f: (3,) force exerted by fingertip on environment [N]
        """
        q_rad = np.radians(q)
        J_ti, J_ba, J_pinv = self._jacs(q_rad)
        return J_pinv.T @ self.eta @ (
            J_ti.T @ K_tip  @ (d_ref['tip']  - FK_DIP(q_rad, self.rtip)) +
            J_ba.T @ K_base @ (d_ref['base'] - FK_MCP(q_rad, self.rbase)))

    def stiffness_inversion(self, q, K_des):
        """
        Joint minimum-norm inversion for K_tip and K_base given K_des.

        vec(K_des) ≈ [M_tip  M_base] · [vec(K_tip); vec(K_base)]
        where M_i = (R_i^T ⊗ L_i),  L_i = J_pinv^T·η·J_i^T,  R_i = J_i·J_pinv.

        Args:
            q:     (2,) current motor angles [deg]
            K_des: (3,3) desired tip stiffness [N/m]

        Returns:
            K_tip_ff:  (3,3)
            K_base_ff: (3,3)
        """
        J_ti, J_ba, J_pinv = self._jacs(np.radians(q))
        L_tip  = J_pinv.T @ self.eta @ J_ti.T
        R_tip  = J_ti @ J_pinv
        L_base = J_pinv.T @ self.eta @ J_ba.T
        R_base = J_ba @ J_pinv
        M_tip  = np.kron(R_tip.T, L_tip)
        M_base = np.kron(R_base.T, L_base)
        M_all  = np.hstack([M_tip, M_base])
        vec_all = np.linalg.pinv(M_all) @ K_des.flatten('F')
        K_tip_ff  = vec_all[:9].reshape(3, 3, order='F')
        K_base_ff = vec_all[9:].reshape(3, 3, order='F')
        return K_tip_ff, K_base_ff

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def stiffness_descent(self, q, d_ref, K_tip, K_base, f_meas, f_des, lr=1e-3):
        """
        Perform one step of gradient descent on K_tip and K_base simultaneously.

        Minimises  L = ½ ||f_meas - f_des||²  w.r.t. K_tip and K_base.
        Gradients (decoupled, f linear in each):
            ∂L/∂vec(K_tip)  = C_tip^T  · (f_meas - f_des)
            ∂L/∂vec(K_base) = C_base^T · (f_meas - f_des)

        Args:
            q:     (2,) motor angles [deg]
            d_ref: dict with 'tip' (3,) and 'base' (3,) reference positions [m]
            K_tip: (3,3) stiffness matrix at fingertip [N/m]
            K_base: (3,3) stiffness matrix at base [N/m]
            f_meas, f_des: (3,) measured and desired tip forces [N]
            lr: learning rate

        Returns:
            K_tip_new: (3, 3) updated tip stiffness
            K_base_new: (3, 3) updated base stiffness
        """
        q_rad = np.radians(q)
        J_ti, J_ba, J_pinv = self._jacs(q_rad)

        delta_x_tip  = d_ref['tip']  - FK_DIP(q_rad, self.rtip)
        delta_x_base = d_ref['base'] - FK_MCP(q_rad, self.rbase)

        A_tip  = J_pinv.T @ self.eta @ J_ti.T
        A_base = J_pinv.T @ self.eta @ J_ba.T

        error = f_meas - f_des
        grad_K_tip  = A_tip.T  @ error.reshape(-1, 1) @ delta_x_tip.reshape(1, -1)
        grad_K_base = A_base.T @ error.reshape(-1, 1) @ delta_x_base.reshape(1, -1)

        return (K_tip  - lr * grad_K_tip,
            K_base - lr * grad_K_base)

    def ref_descent(self, q, d_ref, K_tip, K_base, f_meas, f_des, lr=7e-04):
        """
        One gradient descent step on d_ref to reduce force error.

        Specular counterpart of stiffness_descent:
          stiffness_descent : K_new      = K     - lr * C.T @ error   (C = tip_sensitivity_K_*)
          ref_descent       : d_ref_new  = d_ref - lr * S.T @ error   (S = ref_sensitivity_K_tip + ref_sensitivity_K_base)

        Args:
            q:      (2,) current motor angles [deg]
            d_ref:  dict with 'tip' and/or 'base' keys, each a (3,) task-space position [m]
            K_tip:  (3,3) stiffness at fingertip [N/m]
            K_base: (3,3) stiffness at base [N/m]
            f_meas, f_des: (3,) measured and desired tip forces [N]
        Returns:
            d_ref_new: dict with updated 'tip' and/or 'base' positions [m]
        """
        J_ti, J_ba, J_pinv = self._jacs(np.radians(q))
        S_tip  = J_pinv.T @ self.eta @ J_ti.T @ K_tip
        S_base = J_pinv.T @ self.eta @ J_ba.T @ K_base
        error  = f_meas - f_des
        d_ref_new = dict(d_ref)
        if 'tip'  in d_ref: d_ref_new['tip']  = d_ref['tip']  - lr * (S_tip.T  @ error)
        if 'base' in d_ref: d_ref_new['base'] = d_ref['base'] - lr * (S_base.T @ error)
        return d_ref_new


if __name__ == "__main__":
    q_base = np.array([30.0, 20.0])
    q_ref  = np.array([50.0, 50.0])
    K_tip  = np.diag([1.0, 1.0, 1.0])
    K_base = np.diag([0.5, 0.5, 0.5])
    eta    = np.diag([0.8, 0.7])

    model = tip_stiffness_TaskSpace(n=None, eta=eta)

    # Compute d_ref: task-space Lagrangian reference positions [m]
    d_ref = {
        'tip':  FK_DIP(np.radians(q_ref), model.rtip),
        'base': FK_MCP(np.radians(q_ref), model.rbase),
    }

    # Ground truth: actual force at q_base
    f_true = model.tip_force(q_base, d_ref, K_tip, K_base)
    print(f"True force at q_base: {f_true}")

    # --- FD stiffness validation: K_tip · Δx ≈ -Δf ---
    print("\n--- FD stiffness validation (task space) ---")
    np.random.seed(1)
    f0 = model.tip_force(q_base, d_ref, K_tip, K_base)
    K1 = model.tip_stiffness(q_base, K_tip, K_base)
    K2 = model.tip_stiffness(q_base, K_tip, K_base, d_ref=d_ref, f_ext=f0)
    x0 = FK_DIP(np.radians(q_base), model.rtip)
    for eps in [0.05, 0.005, 0.0005]:
        errs_1, errs_2 = [], []
        for _ in range(6):
            dq = np.random.randn(2)
            dq = dq * (eps / np.linalg.norm(dq))
            dx = FK_DIP(np.radians(q_base + dq), model.rtip) - x0
            df = model.tip_force(q_base + dq, d_ref, K_tip, K_base) - f0
            scale = np.linalg.norm(df) + 1e-15
            errs_1.append(np.linalg.norm(K1 @ dx + df) / scale)
            errs_2.append(np.linalg.norm(K2 @ dx + df) / scale)
        print(f"  eps={eps:.4f}°  1st-order rel={np.mean(errs_1):.2e}"
              f"  2nd-order rel={np.mean(errs_2):.2e}"
              f"  (ratio {np.mean(errs_1) / np.mean(errs_2):.1f}x)")

    # --- stiffness_descent convergence test ---
    print("\n--- stiffness_descent convergence test (task space) ---")
    f_des = np.array([0.0, 0.05, 0.03])
    K_tip_opt  = np.diag([1.0, 1.0, 1.0])
    K_base_opt = np.diag([0.5, 0.5, 0.5])
    MAX_ITERS = 10000
    for i in range(MAX_ITERS):
        f_meas = model.tip_force(q_base, d_ref, K_tip_opt, K_base_opt)
        err = np.linalg.norm(f_meas - f_des)
        if i % 200 == 0:
            print(f"  iter {i:3d}  ||error|| = {err:.6f}")
        K_tip_opt, K_base_opt = model.stiffness_descent(
            q_base, d_ref, K_tip_opt, K_base_opt, f_meas, f_des, lr=50)
    print(f"  Final ||error|| = {np.linalg.norm(model.tip_force(q_base, d_ref, K_tip_opt, K_base_opt) - f_des):.6f}")

    # --- ref_descent convergence test ---
    print("\n--- ref_descent convergence test (task space) ---")
    f_des = np.array([0.0, 0.023, 0.037])
    q_ref_seed = np.array([50.0, 50.0])
    d_ref_opt = {
        'tip':  FK_DIP(np.radians(q_ref_seed), model.rtip),
        'base': FK_MCP(np.radians(q_ref_seed), model.rbase),
    }
    _q_rad  = np.radians(q_base)
    _Jt     = model.J_tip( _q_rad[0], _q_rad[1], model.rtip[0],  model.rtip[1],  model.rtip[2])
    _Jb     = model.J_base(_q_rad[0], _q_rad[1], model.rbase[0], model.rbase[1], model.rbase[2])
    _JpT    = np.linalg.pinv(model.P @ _Jt).T
    _x_tip  = FK_DIP(_q_rad, model.rtip)
    _x_base = FK_MCP(_q_rad, model.rbase)

    def _force(dr):
        return _JpT @ model.eta @ (
            _Jt.T @ K_tip  @ (dr['tip']  - _x_tip) +
            _Jb.T @ K_base @ (dr['base'] - _x_base))

    for i in range(MAX_ITERS):
        f_meas = _force(d_ref_opt)
        err = np.linalg.norm(f_meas - f_des)
        if i % 200 == 0:
            print(f"  iter {i:3d}  ||error|| = {err:.6f}")
        d_ref_opt = model.ref_descent(
            q_base, d_ref_opt, K_tip, K_base, f_meas, f_des, lr=0.5)
    print(f"  Final ||error|| = {np.linalg.norm(_force(d_ref_opt) - f_des):.6f}")

    # --- stiffness_inversion check (task space) ---
    print("\n--- stiffness_inversion check (task space) ---")
    K_target_tip  = np.diag([1.5, 1.5, 1.5])
    K_target_base = np.diag([0.8, 0.8, 0.8])
    K_des_stiff   = model.tip_stiffness(q_base, K_target_tip, K_target_base)
    K_tip_ff, K_base_ff = model.stiffness_inversion(q_base, K_des_stiff)
    K_recon = model.tip_stiffness(q_base, K_tip_ff, K_base_ff)
    recon_err = np.linalg.norm(K_recon - K_des_stiff, 'fro')
    print(f"  ||K_recon - K_des||_F = {recon_err:.6f}")
