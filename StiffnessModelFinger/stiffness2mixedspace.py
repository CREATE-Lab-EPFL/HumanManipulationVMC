"""
Compute task-space stiffness and force from VMC virtual elements in mixed
(finger-space + task-space) coordinates.

General formulation (mixed d = finger-space + task-space coordinates):

    d = [θ_MCP, θ_PIP, θ_DIP, x_tip, x_base]^T

    f_x = ((P·J_x)*)^T · η · Σ_i  J_di^T · K_di · (d_i,ref - d_i)

    K_x ≈ ((P·J_x)*)^T · η · (Σ_i  J_di^T · K_di · J_di) · (P·J_x)*

Virtual contributions from any coordinate space compose additively through PVW.
This class combines tip_stiffness_FingerSpace and tip_stiffness_TaskSpace
into a single model that can use all three simultaneously.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KinematicsFinger.FK_Finger import *
from KinematicsFinger.JacobiansFinger import *
from KinematicsFinger.HessiansFinger import *
from ModelIDFinger.finger_params import a, c


class tip_stiffness_MixedSpace():
    """
    Compute tip stiffness and force for VMC with virtual elements in both
    finger space (joint springs) and task space (Cartesian springs).

    Virtual coordinate vector:
        d = [θ_MCP, θ_PIP, θ_DIP,   ← finger-space contribution (K_theta)
             x_tip,                   ← task-space at fingertip  (K_tip)
             x_base]                  ← task-space at MCP base   (K_base)

    Any subset can be disabled by setting the corresponding K to zero.

    Uses constant efficiency model:
        η = diag([η_MCP, η_PIP])
    """

    def __init__(self, n=np.array([0, 0, 1]),
                 rtip=np.array([0, c, 0]),
                 rbase=np.array([0, a, 0]),
                 eta=None):
        """
        Initialize Jacobians and Hessians for all virtual element types.

        Args:
            n: 3x1 contact normal (must be normalized). None → P = I (free space).
            rtip: Position of fingertip in its local frame [m].
            rbase: Position of base attachment point in its local frame [m].
            eta: 2x2 tendon transmission efficiency matrix (diagonal, η_i ∈ (0,1]).
                 If None, defaults to identity (no losses).
        """
        # Finger-space Jacobian: ∂θ/∂q  (3×2)
        self.J_theta = q2joint()
        # Task-space Jacobians: ∂x/∂q  (3×2)
        self.J_tip   = q2DIP()
        self.J_base  = q2MCP()

        # Finger-space Hessian: ∂²θ/∂q²  (3×2×2)
        self.H_theta = qq2joint()
        # Task-space Hessians: ∂²x/∂q²  (3×2×2)
        self.H_tip   = qq2DIP()
        self.H_base  = qq2MCP()

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _jacs(self, q_rad):
        """Evaluate all Jacobians at q_rad (radians)."""
        J_th = self.J_theta(q_rad[0], q_rad[1])
        J_ti = self.J_tip( q_rad[0], q_rad[1],
                           self.rtip[0],  self.rtip[1],  self.rtip[2])
        J_ba = self.J_base(q_rad[0], q_rad[1],
                           self.rbase[0], self.rbase[1], self.rbase[2])
        return J_th, J_ti, J_ba

    def _deflections(self, q_rad, theta_ref, d_ref):
        """Virtual element deflections (d_ref - d) for all three contributions.

        Args:
            q_rad:     (2,) current motor angles [rad]
            theta_ref: (3,) joint-space reference [rad]
            d_ref:     dict with 'tip' (3,) and 'base' (3,) reference positions [m]
        """
        delta_theta  = theta_ref - motor_to_joint(q_rad)
        delta_x_tip  = d_ref['tip']  - FK_DIP(q_rad, self.rtip)
        delta_x_base = d_ref['base'] - FK_MCP(q_rad, self.rbase)
        return delta_theta, delta_x_tip, delta_x_base

    def _J_tip_P_pinv(self, q_rad):
        J_tip_P = self.P @ self.J_tip(q_rad[0], q_rad[1],
                                      self.rtip[0], self.rtip[1], self.rtip[2])
        return np.linalg.pinv(J_tip_P)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def motor_stiffness(self, q, K_theta, K_tip, K_base,
                        theta_ref_deg=None, d_ref=None, f_ext=None):
        """
        Combined motor-space stiffness from all three virtual element types.

        1st order (Salisbury):
            K_motor = η · (J_θ^T·K_θ·J_θ + J_tip^T·K_tip·J_tip + J_base^T·K_base·J_base)

        2nd order (CCT): adds geometric stiffness from Hessians of all elements,
        weighted by the external force f_ext and the corresponding virtual torques.

        Args:
            q:             (2,) current motor angles [deg]
            K_theta:       (3,3) finger-space stiffness [N/rad]
            K_tip:         (3,3) task-space stiffness at fingertip [N/m]
            K_base:        (3,3) task-space stiffness at MCP base [N/m]
            theta_ref_deg: (3,) joint-space reference [deg] (required when f_ext is given)
            d_ref:         dict with 'tip' (3,) and 'base' (3,) reference positions [m]
                           (required when f_ext is given)
            f_ext:         (3,) external contact force for 2nd-order correction [N].
                           If None, returns the 1st-order result.

        Returns:
            K_motor: (2,2) motor-space stiffness [N·m/rad]
        """
        q_rad = np.radians(q)

        J_th, J_ti, J_ba = self._jacs(q_rad)

        # 1st order — additive contributions
        K_motor = self.eta @ (J_th.T @ K_theta @ J_th +
                              J_ti.T @ K_tip   @ J_ti +
                              J_ba.T @ K_base  @ J_ba)

        if f_ext is not None:
            theta_ref = np.radians(theta_ref_deg)
            delta_theta, delta_x_tip, delta_x_base = self._deflections(q_rad, theta_ref, d_ref)

            # Geometric stiffness: outward term (same for all elements)
            H_tip_P  = np.tensordot(self.P,
                                    self.H_tip(q_rad[0], q_rad[1],
                                               self.rtip[0], self.rtip[1], self.rtip[2]),
                                    axes=([1], [0]))
            K_g_out = np.tensordot(f_ext, H_tip_P, axes=([0], [0]))   # (2,2)

            # Geometric stiffness: inward term — virtual torques from all springs
            tau_theta = K_theta @ delta_theta
            tau_tip   = K_tip   @ delta_x_tip
            tau_base  = K_base  @ delta_x_base

            H_theta = self.H_theta(q_rad[0], q_rad[1])
            H_tip   = self.H_tip( q_rad[0], q_rad[1],
                                  self.rtip[0],  self.rtip[1],  self.rtip[2])
            H_base  = self.H_base(q_rad[0], q_rad[1],
                                  self.rbase[0], self.rbase[1], self.rbase[2])

            tau_d = np.concatenate([tau_theta, tau_tip, tau_base])   # (9,)
            H_d   = np.vstack([H_theta, H_tip, H_base])              # (9,2,2)

            K_g_in = self.eta @ np.tensordot(tau_d, H_d, axes=([0], [0]))  # (2,2)

            K_motor += K_g_out - K_g_in

        return K_motor

    def tip_stiffness(self, q, K_theta, K_tip, K_base,
                      theta_ref_deg=None, d_ref=None, f_ext=None):
        """
        Cartesian tip stiffness matrix in the projected output space.

            K_x = ((P·J_x)*)^T · K_motor · (P·J_x)*

        Args:
            q:             (2,) current motor angles [deg]
            K_theta:       (3,3) finger-space stiffness [N/rad]
            K_tip:         (3,3) task-space stiffness at fingertip [N/m]
            K_base:        (3,3) task-space stiffness at MCP base [N/m]
            theta_ref_deg: (3,) joint-space reference [deg] (required when f_ext is given)
            d_ref:         dict with 'tip' (3,) and 'base' (3,) reference positions [m]
                           (required when f_ext is given)
            f_ext:         (3,) external force for CCT correction [N]. None → 1st order.

        Returns:
            K_x: (3,3) tip stiffness matrix [N/m]
        """
        J_pinv_tip_P = self._J_tip_P_pinv(np.radians(q))
        K_motor      = self.motor_stiffness(q, K_theta, K_tip, K_base,
                                            theta_ref_deg, d_ref, f_ext)
        return J_pinv_tip_P.T @ K_motor @ J_pinv_tip_P

    def tip_force(self, q, theta_ref_deg, d_ref, K_theta, K_tip, K_base):
        """
        Contact force at the fingertip from all virtual element contributions.

            f_x = ((P·J_x)*)^T · η · (J_θ^T·K_θ·Δθ + J_tip^T·K_tip·Δx_tip + J_base^T·K_base·Δx_base)

        Args:
            q:             (2,) current motor angles [deg]
            theta_ref_deg: (3,) joint-space reference [deg]
            d_ref:         dict with 'tip' (3,) and 'base' (3,) reference positions [m]
            K_theta:       (3,3) finger-space stiffness [N/rad]
            K_tip:         (3,3) task-space stiffness at fingertip [N/m]
            K_base:        (3,3) task-space stiffness at MCP base [N/m]

        Returns:
            f: (3,) tip force [N]
        """
        q_rad     = np.radians(q)
        theta_ref = np.radians(theta_ref_deg)

        J_th, J_ti, J_ba = self._jacs(q_rad)
        J_pinv_tip_P     = self._J_tip_P_pinv(q_rad)

        delta_theta, delta_x_tip, delta_x_base = self._deflections(q_rad, theta_ref, d_ref)

        tau = (J_th.T @ K_theta @ delta_theta +
               J_ti.T @ K_tip   @ delta_x_tip  +
               J_ba.T @ K_base  @ delta_x_base)

        return J_pinv_tip_P.T @ self.eta @ tau

    # ------------------------------------------------------------------
    # Sensitivities
    # ------------------------------------------------------------------

    def tip_sensitivity_K_theta(self, q, theta_ref_deg):
        """
        df_x / dvec(K_theta)^T  ∈ R^{3×9}  (finger-space spring).

        From  f_x = ((P·J_x)*)^T · η · J_θ^T · K_θ · Δθ,
        differentiating: C = Δθ^T ⊗ ((P·J_x)*)^T · η · J_θ^T

        Args:
            q:             (2,) motor angles [deg]
            theta_ref_deg: (3,) joint-space reference [deg]

        Returns:
            C: (3,9) sensitivity matrix
        """
        q_rad       = np.radians(q)
        theta_ref   = np.radians(theta_ref_deg)
        J_th, _, _   = self._jacs(q_rad)
        J_pinv_tip_P = self._J_tip_P_pinv(q_rad)
        delta_theta  = theta_ref - motor_to_joint(q_rad)
        A = J_pinv_tip_P.T @ self.eta @ J_th.T
        return np.kron(delta_theta.reshape(1, -1), A)

    def tip_sensitivity_K_tip(self, q, d_ref):
        """
        df_x / dvec(K_tip)^T  ∈ R^{3×9}  (task-space spring at fingertip).

        Args:
            q:     (2,) motor angles [deg]
            d_ref: dict with 'tip' (3,) reference position [m]

        Returns:
            C: (3,9) sensitivity matrix
        """
        q_rad        = np.radians(q)
        _, J_ti, _   = self._jacs(q_rad)
        J_pinv_tip_P = self._J_tip_P_pinv(q_rad)
        delta_x_tip  = d_ref['tip'] - FK_DIP(q_rad, self.rtip)
        A = J_pinv_tip_P.T @ self.eta @ J_ti.T
        return np.kron(delta_x_tip.reshape(1, -1), A)

    def tip_sensitivity_K_base(self, q, d_ref):
        """
        df_x / dvec(K_base)^T  ∈ R^{3×9}  (task-space spring at MCP base).

        Args:
            q:     (2,) motor angles [deg]
            d_ref: dict with 'base' (3,) reference position [m]

        Returns:
            C: (3,9) sensitivity matrix
        """
        q_rad        = np.radians(q)
        _, _, J_ba   = self._jacs(q_rad)
        J_pinv_tip_P = self._J_tip_P_pinv(q_rad)
        delta_x_base = d_ref['base'] - FK_MCP(q_rad, self.rbase)
        A = J_pinv_tip_P.T @ self.eta @ J_ba.T
        return np.kron(delta_x_base.reshape(1, -1), A)

    def ref_sensitivity(self, q, K_theta, K_tip, K_base):
        """
        df_x / dtheta_ref and df_x / dd_ref sensitivities.

        Args:
            q:      (2,) current motor angles [deg]
            K_theta: (3,3) finger-space stiffness [N/rad]
            K_tip:   (3,3) task-space stiffness at fingertip [N/m]
            K_base:  (3,3) task-space stiffness at MCP base [N/m]

        Returns:
            S_joint: (3,3) sensitivity df_x / dtheta_ref
            S_task:  dict with 'tip' and 'base' keys, each a (3,3) sensitivity
        """
        q_rad        = np.radians(q)
        J_th, J_ti, J_ba = self._jacs(q_rad)
        J_pinv_tip_P = self._J_tip_P_pinv(q_rad)

        S_joint = J_pinv_tip_P.T @ self.eta @ (J_th.T @ K_theta)
        S_task  = {
            'tip':  J_pinv_tip_P.T @ self.eta @ (J_ti.T @ K_tip),
            'base': J_pinv_tip_P.T @ self.eta @ (J_ba.T @ K_base),
        }
        return S_joint, S_task

    def stiffness_inversion(self, q, K_des):
        """
        Joint minimum-norm inversion for K_theta, K_tip, and K_base given K_des.

        vec(K_des) ≈ [M_th  M_tip  M_base] · [vec(K_theta); vec(K_tip); vec(K_base)]
        where M_i = (R_i^T ⊗ L_i),  L_i = J_pinv^T·η·J_i^T,  R_i = J_i·J_pinv.

        Args:
            q:     (2,) motor angles [deg]
            K_des: (3,3) desired tip stiffness [N/m]

        Returns:
            K_theta_ff: (3,3)
            K_tip_ff:   (3,3)
            K_base_ff:  (3,3)
        """
        q_rad        = np.radians(q)
        J_th, J_ti, J_ba = self._jacs(q_rad)
        J_pinv       = self._J_tip_P_pinv(q_rad)

        L_th   = J_pinv.T @ self.eta @ J_th.T
        R_th   = J_th @ J_pinv
        L_tip  = J_pinv.T @ self.eta @ J_ti.T
        R_tip  = J_ti @ J_pinv
        L_base = J_pinv.T @ self.eta @ J_ba.T
        R_base = J_ba @ J_pinv

        M_th   = np.kron(R_th.T, L_th)
        M_tip  = np.kron(R_tip.T, L_tip)
        M_base = np.kron(R_base.T, L_base)
        M_all  = np.hstack([M_th, M_tip, M_base])

        vec_all = np.linalg.pinv(M_all) @ K_des.flatten('F')
        K_theta_ff = vec_all[:9].reshape(3, 3, order='F')
        K_tip_ff   = vec_all[9:18].reshape(3, 3, order='F')
        K_base_ff  = vec_all[18:].reshape(3, 3, order='F')
        return K_theta_ff, K_tip_ff, K_base_ff

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def stiffness_descent(self, q, theta_ref_deg, d_ref, K_theta, K_tip, K_base,
                          f_meas, f_des, lr=1e-4):
        """
        One gradient descent step on all three stiffness matrices simultaneously.

        Minimises  L = ½ ||f_meas - f_des||²  w.r.t. K_theta, K_tip, K_base.
        Gradients are decoupled (f linear in each K):
            ∂L/∂vec(K_theta) = C_theta^T · (f_meas - f_des)
            ∂L/∂vec(K_tip)   = C_tip^T   · (f_meas - f_des)
            ∂L/∂vec(K_base)  = C_base^T  · (f_meas - f_des)

        Args:
            q:                      (2,) motor angles [deg]
            theta_ref_deg:          (3,) joint-space reference [deg]
            d_ref:                  dict with 'tip' (3,) and 'base' (3,) reference positions [m]
            K_theta, K_tip, K_base: (3,3) stiffness matrices
            f_meas, f_des:          (3,) measured and desired tip forces [N]
            lr:                     learning rate (scalar or 3-tuple for per-K rates)

        Returns:
            K_theta_new, K_tip_new, K_base_new: (3,3) updated stiffness matrices
        """
        lr_th, lr_ti, lr_ba = (lr, lr, lr) if np.isscalar(lr) else lr
        error = f_meas - f_des

        return (K_theta - lr_th * (self.tip_sensitivity_K_theta(q, theta_ref_deg).T @ error).reshape(3, 3, order='F'),
                K_tip   - lr_ti * (self.tip_sensitivity_K_tip(  q, d_ref        ).T @ error).reshape(3, 3, order='F'),
                K_base  - lr_ba * (self.tip_sensitivity_K_base( q, d_ref        ).T @ error).reshape(3, 3, order='F'))

    def ref_descent(self, q, theta_ref_deg, d_ref, K_theta, K_tip, K_base,
                    f_meas, f_des, lr=(1e-02, 1e-02)):
        """
        One gradient descent step on theta_ref_deg and d_ref to reduce force error.

        Specular counterpart of stiffness_descent:
          stiffness_descent : K_new             = K             - lr * C.T @ error
          ref_descent       : theta_ref_deg_new = theta_ref_deg - lr_th   * degrees(S_joint.T @ error)
                              d_ref_new         = d_ref         - lr_task * S_task.T  @ error

        Args:
            q:             (2,) current motor angles [deg]
            theta_ref_deg: (3,) joint-space reference [deg]
            d_ref:         dict with 'tip' and/or 'base' keys, each a (3,) task-space position [m]
            K_theta:       (3,3) finger-space stiffness [N/rad]
            K_tip:         (3,3) task-space stiffness at fingertip [N/m]
            K_base:        (3,3) task-space stiffness at MCP base [N/m]
            f_meas, f_des: (3,) measured and desired tip forces [N]
            lr:            learning rate — scalar or (lr_theta, lr_task) 2-tuple

        Returns:
            theta_ref_deg_new: (3,) updated joint-space reference [deg]
            d_ref_new:         dict with updated 'tip' and/or 'base' positions [m]
        """
        S_joint, S_task = self.ref_sensitivity(q, K_theta, K_tip, K_base)
        error = f_meas - f_des

        lr_th, lr_task = lr if isinstance(lr, (tuple, list)) else (lr, lr)

        theta_ref_deg_new = theta_ref_deg - lr_th * np.degrees(S_joint.T @ error)

        d_ref_new = dict(d_ref)
        if 'tip'  in d_ref: d_ref_new['tip']  = d_ref['tip']  - lr_task * (S_task['tip'].T  @ error)
        if 'base' in d_ref: d_ref_new['base'] = d_ref['base'] - lr_task * (S_task['base'].T @ error)

        return theta_ref_deg_new, d_ref_new

# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    q_base  = np.array([30.0, 20.0])
    q_ref   = np.array([50.0, 50.0])
    K_theta = np.diag([0.4, 0.3, 0.2])   # finger-space [N/rad]
    K_tip   = np.diag([1.0, 1.0, 1.0])   # task-space tip [N/m]
    K_base  = np.diag([0.5, 0.5, 0.5])   # task-space base [N/m]
    eta     = np.diag([0.8, 0.7])

    model = tip_stiffness_MixedSpace(n=None, eta=eta)

    # Lagrangian references from q_ref
    theta_ref_deg = np.degrees(motor_to_joint(np.radians(q_ref)))
    d_ref = {
        'tip':  FK_DIP(np.radians(q_ref), model.rtip),
        'base': FK_MCP(np.radians(q_ref), model.rbase),
    }

    # Ground truth force
    f_true = model.tip_force(q_base, theta_ref_deg, d_ref, K_theta, K_tip, K_base)
    print(f"True force at q_base: {f_true}")

    # ------------------------------------------------------------------
    # FD stiffness validation: K_tip · Δx ≈ -Δf
    # ------------------------------------------------------------------
    print("\n--- FD stiffness validation (mixed space) ---")
    np.random.seed(1)
    f0 = model.tip_force(q_base, theta_ref_deg, d_ref, K_theta, K_tip, K_base)
    K1 = model.tip_stiffness(q_base, K_theta, K_tip, K_base)
    K2 = model.tip_stiffness(q_base, K_theta, K_tip, K_base,
                              theta_ref_deg=theta_ref_deg, d_ref=d_ref, f_ext=f0)
    x0 = FK_DIP(np.radians(q_base), model.rtip)
    for eps in [0.05, 0.005, 0.0005]:
        errs_1, errs_2 = [], []
        for _ in range(6):
            dq = np.random.randn(2)
            dq = dq * (eps / np.linalg.norm(dq))
            dx = FK_DIP(np.radians(q_base + dq), model.rtip) - x0
            df = model.tip_force(q_base + dq, theta_ref_deg, d_ref, K_theta, K_tip, K_base) - f0
            scale = np.linalg.norm(df) + 1e-15
            errs_1.append(np.linalg.norm(K1 @ dx + df) / scale)
            errs_2.append(np.linalg.norm(K2 @ dx + df) / scale)
        print(f"  eps={eps:.4f}°  1st-order rel={np.mean(errs_1):.2e}"
              f"  2nd-order rel={np.mean(errs_2):.2e}"
              f"  (ratio {np.mean(errs_1) / np.mean(errs_2):.1f}x)")

    # ------------------------------------------------------------------
    # Finite-difference test for all three sensitivities
    # ------------------------------------------------------------------
    print("\n--- Finite-difference tests (mixed space) ---")
    np.random.seed(0)

    for name, K_var, get_C, K_others in [
        ("K_theta", K_theta,
         lambda: model.tip_sensitivity_K_theta(q_base, theta_ref_deg),
         dict(K_tip=K_tip, K_base=K_base)),
        ("K_tip",   K_tip,
         lambda: model.tip_sensitivity_K_tip(q_base, d_ref),
         dict(K_theta=K_theta, K_base=K_base)),
        ("K_base",  K_base,
         lambda: model.tip_sensitivity_K_base(q_base, d_ref),
         dict(K_theta=K_theta, K_tip=K_tip)),
    ]:
        C   = get_C()
        dK  = np.random.randn(3, 3) * 0.01
        kw0 = {**K_others, name: K_var}
        kw1 = {**K_others, name: K_var + dK}
        f0  = model.tip_force(q_base, theta_ref_deg, d_ref, **kw0)
        f1  = model.tip_force(q_base, theta_ref_deg, d_ref, **kw1)
        df_fd  = f1 - f0
        df_lin = C @ dK.flatten('F')
        print(f"\n{name}:")
        print(f"  FD   δf : {df_fd}")
        print(f"  Lin  δf : {df_lin}")
        print(f"  Error   : {np.linalg.norm(df_fd - df_lin):.2e}  (should be O(|dK|²))")

    # ------------------------------------------------------------------
    # stiffness_descent convergence
    # ------------------------------------------------------------------
    print("\n--- stiffness_descent convergence (mixed space) ---")
    f_des = np.array([0.0, 0.05, 0.03])
    K_th_opt, K_ti_opt, K_ba_opt = (np.diag([0.4, 0.3, 0.2]),
                     np.diag([1.0, 1.0, 1.0]),
                     np.diag([0.5, 0.5, 0.5]))
    MAX_ITERS = 10000
    for i in range(MAX_ITERS):
        f_meas = model.tip_force(q_base, theta_ref_deg, d_ref, K_th_opt, K_ti_opt, K_ba_opt)
        err = np.linalg.norm(f_meas - f_des)
        if i % 200 == 0:
            print(f"  iter {i:3d}  ||error|| = {err:.6f}")
        K_th_opt, K_ti_opt, K_ba_opt = model.stiffness_descent(
            q_base, theta_ref_deg, d_ref, K_th_opt, K_ti_opt, K_ba_opt,
            f_meas, f_des, lr=(5e-4, 5e-4, 5e-4))
    print(f"  Final ||error|| = {np.linalg.norm(model.tip_force(q_base, theta_ref_deg, d_ref, K_th_opt, K_ti_opt, K_ba_opt) - f_des):.6f}")

    # ------------------------------------------------------------------
    # ref_descent convergence
    # ------------------------------------------------------------------
    print("\n--- ref_descent convergence (mixed space) ---")
    f_des = np.array([0.0, 0.023, 0.037])
    q_ref_opt = np.array([50.0, 50.0])
    theta_ref_opt_deg = np.degrees(motor_to_joint(np.radians(q_ref_opt)))
    d_ref_opt = {
        'tip':  FK_DIP(np.radians(q_ref_opt), model.rtip),
        'base': FK_MCP(np.radians(q_ref_opt), model.rbase),
    }
    for i in range(MAX_ITERS):
        f_meas = model.tip_force(q_base, theta_ref_opt_deg, d_ref_opt, K_theta, K_tip, K_base)
        err = np.linalg.norm(f_meas - f_des)
        if i % 200 == 0:
            print(f"  iter {i:3d}  ||error|| = {err:.6f}")
        theta_ref_opt_deg, d_ref_opt = model.ref_descent(
            q_base, theta_ref_opt_deg, d_ref_opt, K_theta, K_tip, K_base,
            f_meas, f_des, lr=(5e-05, 1e-02))
    print(f"  Final ||error|| = {np.linalg.norm(model.tip_force(q_base, theta_ref_opt_deg, d_ref_opt, K_theta, K_tip, K_base) - f_des):.6f}")

    # --- stiffness_inversion check (mixed space) ---
    print("\n--- stiffness_inversion check (mixed space) ---")
    K_theta_target = np.diag([0.6, 0.4, 0.3])
    K_tip_target   = np.diag([1.5, 1.5, 1.5])
    K_base_target  = np.diag([0.8, 0.8, 0.8])
    K_des_stiff    = model.tip_stiffness(q_base, K_theta_target, K_tip_target, K_base_target)
    K_ff_theta, K_ff_tip, K_ff_base = model.stiffness_inversion(q_base, K_des_stiff)
    K_recon = model.tip_stiffness(q_base, K_ff_theta, K_ff_tip, K_ff_base)
    recon_err = np.linalg.norm(K_recon - K_des_stiff, 'fro')
    print(f"  ||K_recon - K_des||_F = {recon_err:.6f}")
