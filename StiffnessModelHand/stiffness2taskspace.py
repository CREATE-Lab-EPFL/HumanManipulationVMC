"""
Compute fingertip stiffness from VMC task-space (Cartesian) springs.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KinematicsHand.FK_Hand import *
from KinematicsHand.JacobiansHand import *
from KinematicsHand.HessiansHand import *
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS

FINGERS_ALL  = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGERS_4    = ['index', 'middle', 'ring', 'pinky']
ALL_POINTS   = ['thumb', 'index', 'middle', 'ring', 'pinky', 'palm']

DEFAULT_RATTACHMENTS = {**FINGER_TIP_OFFSETS, 'palm': np.zeros(3)}


class tip_stiffness_TaskSpace:
    """
    Compute fingertip stiffness matrix for VMC task-space (Cartesian) stiffness model.

    Springs are Cartesian (position-based) at up to six attachment points
    (five fingertips + palm). The contact normal at each fingertip is derived
    on the fly from FK — no fixed n is assumed.

    K_dict keys and dimensions (any subset):
        'thumb':  (3,3)  — Cartesian stiffness at thumb tip  [N/m]
        'index':  (3,3)  — Cartesian stiffness at index tip  [N/m]
        'middle': (3,3)  — Cartesian stiffness at middle tip [N/m]
        'ring':   (3,3)  — Cartesian stiffness at ring tip   [N/m]
        'pinky':  (3,3)  — Cartesian stiffness at pinky tip  [N/m]
        'palm':   (3,3)  — Cartesian stiffness at palm point [N/m]

    d_ref_dict keys: same subset. Each value is a (3,) reference position [m]
    in the world frame.

    Uses constant efficiency model:
        η = diag([η_CMC1, η_CMC2, η_MCP_thumb, η_IP,
                  η_spread, η_MCP_idx, η_PIP_idx, η_MCP_mid, η_PIP_mid,
                  η_MCP_rng, η_PIP_rng, η_MCP_pnk, η_PIP_pnk])
    """

    def __init__(self, rattachments=None, eta=None, mode='normal'):
        """
        Args:
            rattachments: dict of attachment offsets in each link's local frame [m].
                          Defaults to DEFAULT_RATTACHMENTS.
            eta:          (13,) motor efficiency vector. Defaults to ones (no efficiency loss).
            mode:         'normal' — P = n nᵀ (rank-1, force/stiffness along contact normal only);
                          'full'   — P = I₃   (full 3-D force and stiffness).
        """
        self.rattachments = {**DEFAULT_RATTACHMENTS, **(rattachments or {})}
        self.eta          = np.asarray(eta) if eta is not None else np.ones(13)
        self.mode         = mode

        self.jac = HandJacobians()
        self.hes = HandHessians()

    # ------------------------------------------------------------------
    # Private helpers — output point geometry (fingertip normal & Jacobian)
    # ------------------------------------------------------------------

    def _normal(self, finger, q):
        """
        Contact normal: ⊥ phalanx, y-axis of the final link frame.
        """
        if finger == 'thumb':
            R, _ = Frame_motor2thumb(q, 'IP')
        else:
            R, _ = Frame_motor2finger(q, finger, 'DIP')
        return R[:, 1]

    def _P(self, finger, q):
        """
        Contact projection matrix: n nᵀ in 'normal' mode, I₃ in 'full' mode.
        """
        if self.mode == 'full':
            return np.eye(3)
        n = self._normal(finger, q)
        return n[:, None] @ n[None, :]

    def _J_tip(self, finger, q):
        """
        Jacobian of (specific) fingertip position wrt motor angles.
        """
        r = self.rattachments[finger]
        if finger == 'thumb':
            return np.array(self.jac.get_thumb_jacobian('IP', q, r))
        return np.array(self.jac.get_finger_jacobian(finger, 'DIP', q, r))

    def _J_tip_P_pinv(self, finger, q):
        """
        Pseudo-inverse of P @ J_tip.
        """
        return np.linalg.pinv(self._P(finger, q) @ self._J_tip(finger, q))

    def _H_tip(self, finger, q):
        """
        (3,13,13) Hessian of (specific) fingertip position wrt motor angles.
        """
        r = self.rattachments[finger]
        if finger == 'thumb':
            return self.hes.get_thumb_hessian('IP', q, r)
        return self.hes.get_finger_hessian(finger, 'DIP', q, r)

    # ------------------------------------------------------------------
    # Private helpers — input springs (task-space attachment points)
    # ------------------------------------------------------------------

    def _pos(self, point, q):
        """
        Current world-frame position of attachment point [m].
        """
        r = self.rattachments[point]
        if point == 'thumb':
            return FK_motor2thumbPos(q, 'IP', r)
        if point == 'palm':
            return FK_motor2palm(r)[1]
        return FK_motor2fingerPos(q, point, 'DIP', r)

    def _J_pos(self, point, q):
        """
        (3,13) position Jacobian of attachment point wrt motor angles.
        """
        r = self.rattachments[point]
        if point == 'thumb':
            return np.array(self.jac.get_thumb_jacobian('IP', q, r))
        if point == 'palm':
            return np.array(self.jac.get_wrist_palm_jacobian(q, r))
        return np.array(self.jac.get_finger_jacobian(point, 'DIP', q, r))

    def _H_pos(self, point, q):
        """
        (3,13,13) Hessian of attachment point position wrt motor angles.
        """
        r = self.rattachments[point]
        if point == 'thumb':
            return self.hes.get_thumb_hessian('IP', q, r)
        if point == 'palm':
            return self.hes.get_wrist_palm_hessian(q, r)
        return self.hes.get_finger_hessian(point, 'DIP', q, r)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def motor_stiffness(self, q, K_dict):
        """
        (13,13) 1st-order motor stiffness from all task-space springs.

            K_motor = η · Σ_p  J_xp^T · K_p · J_xp

        Args:
            q:      (13,) current motor angles [rad]
            K_dict: dict of per-point (3,3) stiffness matrices

        Returns:
            K_motor: (13,13)
        """
        eta = np.diag(self.eta)
        S = np.zeros((13, 13))
        for point, K in K_dict.items():
            J = self._J_pos(point, q)
            S += J.T @ K @ J
        return eta @ S

    def tip_stiffness(self, finger, q, K_dict, d_ref_dict=None, f_ext=None):
        """
        (3,3) Cartesian stiffness at fingertip of the given finger.

        Args:
            finger:      'thumb', 'index', 'middle', 'ring', or 'pinky'
            q:           (13,) current motor angles [rad]
            K_dict:      dict of per-point (3,3) stiffness matrices
            d_ref_dict:  dict of (3,) reference positions [m] (required for f_ext)
            f_ext:       (3,) contact force for 2nd-order CCT correction [N]

        Returns:
            K_x: (3,3) tip stiffness [N/m]
        """
        K_motor = self.motor_stiffness(q, K_dict)

        if f_ext is not None and d_ref_dict is not None:
            P    = self._P(finger, q)
            H_xf = self._H_tip(finger, q)                          # (3,13,13)
            H_xf_P = np.tensordot(P, H_xf, axes=([1], [0]))        # (3,13,13)
            K_g_out = np.tensordot(f_ext, H_xf_P, axes=([0], [0])) # (13,13)

            # Inward CCT term: virtual forces from all springs weighted by H_xp
            tau_d = []
            H_d   = []
            for point, K in K_dict.items():
                delta_x = d_ref_dict[point] - self._pos(point, q)
                tau_d.append(K @ delta_x)
                H_d.append(self._H_pos(point, q))
            tau_d   = np.concatenate(tau_d)           # (3*n_pts,)
            H_d     = np.vstack(H_d)                  # (3*n_pts,15,15)
            eta_mat = np.diag(self.eta)
            K_g_in  = eta_mat @ np.tensordot(tau_d, H_d, axes=([0], [0]))

            K_motor = K_motor + K_g_out - K_g_in

        J_pinv = self._J_tip_P_pinv(finger, q)
        return J_pinv.T @ K_motor @ J_pinv

    def tip_force(self, finger, q, d_ref_dict, K_dict):
        """
        3-vector contact force at fingertip from all task-space springs.

            f_f = pinv(P_f · J_xf)^T · η · Σ_p  J_xp^T · K_p · Δx_p

        Args:
            finger:     'thumb', 'index', 'middle', 'ring', or 'pinky'
            q:          (13,) current motor angles [rad]
            d_ref_dict: dict of (3,) reference positions [m]
            K_dict:     dict of per-point (3,3) stiffness matrices

        Returns:
            f: (3,) tip force [N]
        """
        eta = np.diag(self.eta)
        tau = np.zeros(13)
        for point, K in K_dict.items():
            J       = self._J_pos(point, q)
            delta_x = d_ref_dict[point] - self._pos(point, q)
            tau    += J.T @ K @ delta_x
        J_pinv = self._J_tip_P_pinv(finger, q)
        return J_pinv.T @ eta @ tau

    def all_tip_stiffnesses(self, q, K_dict, d_ref_dict=None, f_ext_dict=None):
        """
        Dict of (3,3) stiffness matrices, one per fingertip.

        Args:
            q:           (13,) current motor angles [rad]
            K_dict:      dict of per-point (3,3) stiffness matrices
            d_ref_dict:  dict of (3,) reference positions (required for f_ext_dict)
            f_ext_dict:  dict of (3,) contact forces per finger for 2nd-order correction

        Returns:
            {finger: (3,3)} for all five fingers
        """
        f_ext_dict = f_ext_dict or {}
        return {
            f: self.tip_stiffness(f, q, K_dict, d_ref_dict, f_ext_dict.get(f))
            for f in FINGERS_ALL
        }

    def all_tip_forces(self, q, d_ref_dict, K_dict):
        """
        Dict of (3,) contact forces, one per fingertip.

        Returns:
            {finger: (3,)} for all five fingers
        """
        return {f: self.tip_force(f, q, d_ref_dict, K_dict) for f in FINGERS_ALL}

    def stiffness_inversion(self, finger, point, q, K_des):
        """
        Minimum-norm K_ff_point that best produces K_des via tip_stiffness.

        K_x += L @ K_p @ R  where  L = J_pinv^T · η · J_xp^T  (3×3),
                                    R = J_xp · J_pinv             (3×3).
        Returns K_ff_point = reshape( (R^T ⊗ L)^+ · vec(K_des) ).

        Args:
            finger:  output fingertip
            point:   spring attachment point
            q:       (13,) current motor angles [rad]
            K_des:   (3,3) desired tip stiffness [N/m]

        Returns:
            K_ff: (3,3) minimum-norm point stiffness that best produces K_des
        """
        eta_mat = np.diag(self.eta)
        J_pinv  = self._J_tip_P_pinv(finger, q)   # (13,3)
        J_p     = self._J_pos(point, q)            # (3,13)
        L = J_pinv.T @ eta_mat @ J_p.T             # (3,3)
        R = J_p @ J_pinv                            # (3,3)
        M = np.kron(R.T, L)                         # (9,9)
        return (np.linalg.pinv(M) @ K_des.flatten('F')).reshape(3, 3, order='F')

    def stiffness_inversion_all(self, finger, q, K_dict, K_des):
        """
        Joint minimum-norm inversion for all task-space points.

        vec(K_des) ≈ [M_1  M_2  ...  M_P] · [vec(K_1); vec(K_2); ...; vec(K_P)]
        where M_p = (R_p^T ⊗ L_p),  L_p = J_pinv^T·η·J_p^T,  R_p = J_p·J_pinv.

        Args:
            finger:  output fingertip
            q:       (13,) current motor angles [rad]
            K_dict:  dict of per-point stiffness matrices (ordering defines vec stack)
            K_des:   (3,3) desired tip stiffness [N/m]

        Returns:
            K_ff_dict: dict of minimum-norm point stiffness matrices
        """
        eta_mat = np.diag(self.eta)
        J_pinv  = self._J_tip_P_pinv(finger, q)

        M_blocks = []
        for point in K_dict.keys():
            J_p = self._J_pos(point, q)
            L = J_pinv.T @ eta_mat @ J_p.T
            R = J_p @ J_pinv
            M_blocks.append(np.kron(R.T, L))

        M_all = np.hstack(M_blocks) if M_blocks else np.zeros((9, 0))
        vec_all = np.linalg.pinv(M_all) @ K_des.flatten('F') if M_all.size else np.array([])

        K_ff_dict = {}
        offset = 0
        for point in K_dict.keys():
            block = vec_all[offset:offset + 9]
            K_ff_dict[point] = block.reshape(3, 3, order='F')
            offset += 9
        return K_ff_dict

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def stiffness_descent(self, finger, q, d_ref_dict, K_dict,
                          f_meas, f_des, lr=1e-4):
        """
        One gradient-descent step on all K_dict entries simultaneously.

            ∂L/∂vec(K_p) = C_p^T · (f_meas − f_des)

        Args:
            finger:         output fingertip
            q:              (13,) current motor angles [rad]
            d_ref_dict:     dict of (3,) reference positions [m]
            K_dict:         dict of per-point (3,3) stiffness matrices
            f_meas, f_des:  (3,) measured and desired tip forces [N]
            lr:             scalar or dict of per-point learning rates

        Returns:
            K_dict_new: updated copy of K_dict
        """
        eta    = np.diag(self.eta)
        J_pinv = self._J_tip_P_pinv(finger, q)
        error = f_meas - f_des
        K_new = {}
        for point, K in K_dict.items():
            lr_p  = lr[point] if isinstance(lr, dict) else lr
            J_p   = self._J_pos(point, q)
            delta = d_ref_dict[point] - self._pos(point, q)
            A     = J_pinv.T @ eta @ J_p.T
            grad  = A.T @ error.reshape(-1, 1) @ delta.reshape(1, -1)
            K_new[point] = K - lr_p * grad
        return K_new

    def ref_descent(self, finger, q, d_ref_dict, K_dict,
                    f_meas, f_des, lr=1e-8):
        """
        One gradient-descent step on d_ref_dict to reduce force error.

            x_ref_p^+ = x_ref_p − lr · S_p^T · (f_meas − f_des)

        Args:
            finger:         output fingertip
            q:              (13,) current motor angles [rad]
            d_ref_dict:     dict of (3,) reference positions [m]
            K_dict:         dict of per-point (3,3) stiffness matrices
            f_meas, f_des:  (3,) measured and desired tip forces [N]
            lr:             scalar or dict of per-point learning rates [m/N]

        Returns:
            d_ref_dict_new: updated copy of d_ref_dict
        """
        eta    = np.diag(self.eta)
        J_pinv = self._J_tip_P_pinv(finger, q)
        error = f_meas - f_des
        x_new  = {}
        for point, x_ref in d_ref_dict.items():
            lr_p = lr[point] if isinstance(lr, dict) else lr
            S    = J_pinv.T @ eta @ self._J_pos(point, q).T @ K_dict[point]
            x_new[point] = x_ref - lr_p * (S.T @ error)
        return x_new

# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import time

    np.random.seed(42)

    # Thumb and index flexed
    q_ref  = np.zeros(13)
    q_base = np.zeros(13)
    q_base[2] = np.deg2rad(30.0)   # thumb MCP
    q_base[3] = np.deg2rad(20.0)   # thumb IP
    q_base[5] = np.deg2rad(30.0)   # index MCP
    q_base[6] = np.deg2rad(20.0)   # index PIP

    k = 200.0   # [N/m]

    print("=" * 70)
    print("STIFFNESS2TASKSPACE — initializing Jacobians and Hessians ...")
    t0 = time.time()
    model = tip_stiffness_TaskSpace(mode='full')
    print(f"  done in {time.time() - t0:.1f} s")
    print("=" * 70)

    d_ref_dict = {p: model._pos(p, q_ref) for p in ALL_POINTS}
    K_dict     = {p: k * np.eye(3) for p in ALL_POINTS}

    # 1. Tip stiffnesses and forces at q_base
    print("\n1. Tip stiffnesses (eigenvalues) and forces at q_base:")
    for finger in FINGERS_ALL:
        K_x = model.tip_stiffness(finger, q_base, K_dict, d_ref_dict)
        f   = model.tip_force(finger, q_base, d_ref_dict, K_dict)
        eig = np.linalg.eigvalsh(K_x)
        print(f"  {finger:6s}  K_x eig = {np.round(eig,4)}  f = {np.round(f,5)}")

    # 2. FD stiffness validation: K_tip · Δx ≈ -Δf for row-space perturbations
    print("\n2. FD stiffness validation (thumb and index):")
    np.random.seed(1)
    for finger in ['thumb', 'index']:
        r    = model.rattachments[finger]
        J_tip = np.array(
            model.jac.get_thumb_jacobian('IP', q_base, r) if finger == 'thumb'
            else model.jac.get_finger_jacobian(finger, 'DIP', q_base, r)
        )
        f0 = model.tip_force(finger, q_base, d_ref_dict, K_dict)
        x0 = model._pos(finger, q_base)
        K1 = model.tip_stiffness(finger, q_base, K_dict, d_ref_dict)
        K2 = model.tip_stiffness(finger, q_base, K_dict, d_ref_dict, f_ext=f0)
        print(f"  {finger}:")
        for eps in [1e-3, 1e-4, 1e-5]:
            errs_1, errs_2 = [], []
            for _ in range(6):
                v  = np.random.randn(3)
                dq = J_tip.T @ v
                dq = dq * (eps / np.linalg.norm(dq))
                dx = model._pos(finger, q_base + dq) - x0
                df = model.tip_force(finger, q_base + dq, d_ref_dict, K_dict) - f0
                scale = np.linalg.norm(df) + 1e-15
                errs_1.append(np.linalg.norm(K1 @ dx + df) / scale)
                errs_2.append(np.linalg.norm(K2 @ dx + df) / scale)
            print(f"    eps={eps:.0e}  1st-order rel={np.mean(errs_1):.2e}"
                  f"  2nd-order rel={np.mean(errs_2):.2e}"
                  f"  (ratio {np.mean(errs_1) / np.mean(errs_2):.1f}x)")

    # 3. stiffness_descent convergence (thumb and index)
    print("\n3. stiffness_descent convergence ...")
    f_des = np.array([0.0, 0.02, 0.5])
    for finger in ['thumb', 'index']:
        K_opt = {p: K.copy() for p, K in K_dict.items()}
        MAX_ITERS = 30000
        for i in range(MAX_ITERS):
            f_meas = model.tip_force(finger, q_base, d_ref_dict, K_opt)
            err    = np.linalg.norm(f_meas - f_des)
            if i % 5000 == 0:
                print(f"  [{finger}] iter {i:3d}  ||error|| = {err:.6f}")
            # (no early exit) continue to run fixed MAX_ITERS iterations
            K_opt = model.stiffness_descent(finger, q_base, d_ref_dict, K_opt,
                                            f_meas, f_des, lr=50)
        final = np.linalg.norm(model.tip_force(finger, q_base, d_ref_dict, K_opt) - f_des)
        print(f"  [{finger}] Final  ||error|| = {final:.6f}")

    # 4. ref_descent convergence (thumb and index)
    print("\n4. ref_descent convergence ...")
    f_des = np.array([0.0, 0.02, 0.5])
    for finger in ['thumb', 'index']:
        x_ref_opt = {p: x.copy() for p, x in d_ref_dict.items()}
        for i in range(MAX_ITERS):
            f_meas = model.tip_force(finger, q_base, x_ref_opt, K_dict)
            err    = np.linalg.norm(f_meas - f_des)
            if i % 5000 == 0:
                print(f"  [{finger}] iter {i:4d}  ||error|| = {err:.6f}")
            # (no early exit) continue to run fixed MAX_ITERS iterations
            x_ref_opt = model.ref_descent(finger, q_base, x_ref_opt, K_dict,
                                          f_meas, f_des, lr=2e-7)
        final = np.linalg.norm(model.tip_force(finger, q_base, x_ref_opt, K_dict) - f_des)
        print(f"  [{finger}] Final  ||error|| = {final:.6f}")

    # 5. stiffness_inversion check (thumb and index)
    print("\n5. stiffness_inversion check:")
    for finger in ['thumb', 'index']:
        K_target    = {p: K.copy() * 1.2 for p, K in K_dict.items()}
        K_des_stiff = model.tip_stiffness(finger, q_base, K_target, d_ref_dict)
        K_ff_dict   = model.stiffness_inversion_all(finger, q_base, K_dict, K_des_stiff)
        K_recon     = model.tip_stiffness(finger, q_base, K_ff_dict, d_ref_dict)
        recon_err   = np.linalg.norm(K_recon - K_des_stiff, 'fro')
        print(f"  [{finger}] ||K_recon - K_des||_F = {recon_err:.6f}")
