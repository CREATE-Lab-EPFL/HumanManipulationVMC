"""
Compute fingertip stiffness from VMC joint-space springs.
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
SPREAD_NAMES = ['spread_index', 'spread_middle', 'spread_ring', 'spread_pinky']

DEFAULT_RTIPS = FINGER_TIP_OFFSETS


class tip_stiffness_JointSpace:
    """
    Compute fingertip stiffness matrix for VMC joint-space stiffness model.

    Springs live in joint-angle space (per group). The contact normal at each
    fingertip is derived on the fly from FK — no fixed n is assumed.

    K_dict keys and dimensions:
        'thumb':         (4,4)   — [CMC1, CMC2, MCP, IP]
        'spread_index':  (1,1)
        'spread_middle': (1,1)
        'spread_ring':   (1,1)
        'spread_pinky':  (1,1)
        'index':         (3,3)   — [MCP, PIP, DIP]
        'middle':        (3,3)
        'ring':          (3,3)
        'pinky':         (3,3)

    Uses constant efficiency model:
        η = diag([η_CMC1, η_CMC2, η_MCP_thumb, η_IP,
                  η_spread, η_MCP_idx, η_PIP_idx, η_MCP_mid, η_PIP_mid,
                  η_MCP_rng, η_PIP_rng, η_MCP_pnk, η_PIP_pnk])
    """

    def __init__(self, rtips=None, eta=None, mode='normal'):
        """
        Args:
            rtips: dict of tip offsets in DIP/IP local frame [m]. Defaults to
                   DEFAULT_RTIPS (z-direction, 0.0175 m per finger).
            eta:   (13,) motor efficiency vector. Defaults to ones (no efficiency loss).
            mode:  'normal' — P = n nᵀ (rank-1, force/stiffness along contact normal only);
                   'full'   — P = I₃   (full 3-D force and stiffness).
        """
        self.rtips = {**DEFAULT_RTIPS, **(rtips or {})}
        self.eta   = np.asarray(eta) if eta is not None else np.ones(13)
        self.mode  = mode

        self.jac = HandJacobians()
        self.hes = HandHessians()

    # ------------------------------------------------------------------
    # Private helpers — output point geometry
    # ------------------------------------------------------------------

    def _normal(self, finger, q):
        """Contact normal: ⊥ phalanx, y-axis of the final link frame."""
        if finger == 'thumb':
            R, _ = Frame_motor2thumb(q, 'IP')
        else:
            R, _ = Frame_motor2finger(q, finger, 'DIP')
        return R[:, 1]

    def _P(self, finger, q):
        """Contact projection matrix: n nᵀ in 'normal' mode, I₃ in 'full' mode."""
        if self.mode == 'full':
            return np.eye(3)
        n = self._normal(finger, q)
        return n[:, None] @ n[None, :]

    def _J_tip(self, finger, q):
        """Jacobian of (specific) fingertip position wrt motor angles."""
        rtip = self.rtips[finger]
        if finger == 'thumb':
            return np.array(self.jac.get_thumb_jacobian('IP', q, rtip))
        return np.array(self.jac.get_finger_jacobian(finger, 'DIP', q, rtip))

    def _J_tip_P_pinv(self, finger, q):
        """Pseudo-inverse of P @ J_tip."""
        return np.linalg.pinv(self._P(finger, q) @ self._J_tip(finger, q))

    def _H_tip(self, finger, q):
        """(3,13,13) Hessian of (specific) fingertip position wrt motor angles."""
        rtip = self.rtips[finger]
        if finger == 'thumb':
            return self.hes.get_thumb_hessian('IP', q, rtip)
        return self.hes.get_finger_hessian(finger, 'DIP', q, rtip)

    # ------------------------------------------------------------------
    # Private helpers — input springs (joint-angle groups)
    # ------------------------------------------------------------------

    def _J_angle(self, group, q):
        """(n,13) Jacobian of joint angles for group wrt all motor angles."""
        if group == 'thumb':
            return np.array(self.jac.get_angles_jacobian('thumb', q))
        if group.startswith('spread_'):
            return np.array(self.jac.get_spread_jacobian(group[7:], q))
        return np.array(self.jac.get_angles_jacobian(group, q))

    def _H_angle(self, group, q):
        """(n,13,13) Hessian of joint angles for group wrt all motor angles."""
        if group == 'thumb':
            return np.array(self.hes.get_angles_hessian('thumb', q))
        if group.startswith('spread_'):
            return np.array(self.hes.get_spread_hessian(group[7:], q))
        return np.array(self.hes.get_angles_hessian(group, q))

    def _group_theta(self, group, q):
        """Current joint angles for one group."""
        if group == 'thumb':
            return np.asarray(FK_motor2thumb(q))
        if group.startswith('spread_'):
            return np.array([FK_motor2spread(q, group[7:])])
        return np.asarray(FK_motor2finger(q, group))

    def _deflection(self, group, q, theta_ref_group):
        """Joint-angle deflection (theta_ref_group - current) for one group."""
        return np.asarray(theta_ref_group) - self._group_theta(group, q)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def motor_to_theta_deg(self, q, K_dict):
        """
        Convert motor angles to concatenated joint-space reference in degrees (K_dict ordering).

        Args:
            q:      (13,) motor angles [rad]
            K_dict: dict whose key order defines the concatenation order

        Returns:
            theta_deg: (n_total,) concatenated joint angles [deg]
        """
        return np.degrees(np.concatenate([self._group_theta(g, q) for g in K_dict.keys()]))

    def motor_stiffness(self, q, K_dict):
        """
        (13,13) 1st-order motor stiffness from all joint-space springs.

            K_motor = η · Σ_g  J_θg^T · K_θg · J_θg

        Args:
            q:      (13,) current motor angles [rad]
            K_dict: dict of per-group stiffness matrices

        Returns:
            K_motor: (13,13)
        """
        eta = np.diag(self.eta)
        S = np.zeros((13, 13))
        for group, K in K_dict.items():
            J = self._J_angle(group, q)
            S += J.T @ np.atleast_2d(K) @ J
        return eta @ S

    def tip_stiffness(self, finger, q, K_dict, f_ext=None, theta_ref_deg=None):
        """
        (3,3) Cartesian stiffness at fingertip of the given finger.

        Args:
            finger:        'thumb', 'index', 'middle', 'ring', or 'pinky'
            q:             (13,) current motor angles [rad]
            K_dict:        dict of per-group stiffness matrices
            f_ext:         (3,)  contact force for 2nd-order CCT correction [N].
            theta_ref_deg: (n_total,) concatenated joint-space reference [deg] (K_dict ordering);
                           required for the inward CCT term. If None, the inward term is omitted.

        Returns:
            K_x: (3,3) tip stiffness [N/m]
        """
        K_motor = self.motor_stiffness(q, K_dict)

        if f_ext is not None:
            P    = self._P(finger, q)
            H_xf = self._H_tip(finger, q)                          # (3,15,15)
            H_xf_P = np.tensordot(P, H_xf, axes=([1], [0]))        # (3,15,15)
            K_g_out = np.tensordot(f_ext, H_xf_P, axes=([0], [0])) # (15,15)

            if theta_ref_deg is not None:
                theta_ref = np.radians(theta_ref_deg)
                tau_d = []
                H_d   = []
                idx   = 0
                for group, K in K_dict.items():
                    J   = self._J_angle(group, q)
                    n   = J.shape[0]
                    delta = self._deflection(group, q, theta_ref[idx:idx + n])
                    tau_d.append(np.atleast_2d(K) @ delta)
                    H_d.append(self._H_angle(group, q))
                    idx += n
                tau_d   = np.concatenate(tau_d)
                H_d     = np.vstack(H_d)
                eta_mat = np.diag(self.eta)
                K_g_in  = eta_mat @ np.tensordot(tau_d, H_d, axes=([0], [0]))
                K_motor = K_motor + K_g_out - K_g_in
            else:
                K_motor = K_motor + K_g_out

        J_pinv = self._J_tip_P_pinv(finger, q)
        return J_pinv.T @ K_motor @ J_pinv

    def tip_force(self, finger, q, theta_ref_deg, K_dict):
        """
        3-vector contact force at fingertip from all joint-space springs.

            f_f = pinv(P_f · J_xf)^T · η · Σ_g  J_θg^T · K_θg · Δθ_g

        Args:
            finger:        'thumb', 'index', 'middle', 'ring', or 'pinky'
            q:             (13,) current motor angles [rad]
            theta_ref_deg: (n_total,) concatenated joint-space reference [deg] (K_dict ordering)
            K_dict:        dict of per-group stiffness matrices

        Returns:
            f: (3,) tip force [N]
        """
        theta_ref = np.radians(theta_ref_deg)
        eta = np.diag(self.eta)
        tau = np.zeros(13)
        idx = 0
        for group, K in K_dict.items():
            J     = self._J_angle(group, q)
            n     = J.shape[0]
            delta = self._deflection(group, q, theta_ref[idx:idx + n])
            tau  += J.T @ np.atleast_2d(K) @ delta
            idx  += n
        J_pinv = self._J_tip_P_pinv(finger, q)
        return J_pinv.T @ eta @ tau

    def all_tip_stiffnesses(self, q, K_dict, f_ext_dict=None, theta_ref_deg=None):
        """
        Dict of (3,3) stiffness matrices, one per fingertip.

        Args:
            q:             (13,) current motor angles [rad]
            K_dict:        dict of per-group stiffness matrices
            f_ext_dict:    dict of (3,) contact forces per finger for 2nd-order CCT correction
            theta_ref_deg: (n_total,) concatenated joint-space reference [deg] (K_dict ordering)

        Returns:
            {finger: (3,3)} for all five fingers
        """
        f_ext_dict = f_ext_dict or {}
        return {
            f: self.tip_stiffness(f, q, K_dict, f_ext_dict.get(f), theta_ref_deg=theta_ref_deg)
            for f in FINGERS_ALL
        }

    def all_tip_forces(self, q, theta_ref_deg, K_dict):
        """
        Dict of (3,) contact forces, one per fingertip.

        Returns:
            {finger: (3,)} for all five fingers
        """
        return {f: self.tip_force(f, q, theta_ref_deg, K_dict) for f in FINGERS_ALL}

    def stiffness_inversion(self, finger, group, q, K_des):
        """
        Minimum-norm K_ff_group that best produces K_des via tip_stiffness.

        K_x += L @ K_g @ R  where  L = J_pinv^T · η · J_g^T  (3×n_g),
                                    R = J_g · J_pinv             (n_g×3).
        Returns K_ff_group = reshape( (R^T ⊗ L)^+ · vec(K_des) ).

        Args:
            finger:  output fingertip
            group:   spring group
            q:       (15,) current motor angles [rad]
            K_des:   (3,3) desired tip stiffness [N/m]

        Returns:
            K_ff: (n_g, n_g) minimum-norm group stiffness that best produces K_des
        """
        eta_mat = np.diag(self.eta)
        J_pinv  = self._J_tip_P_pinv(finger, q)   # (15,3)
        J_g     = self._J_angle(group, q)          # (n_g,15)
        n_g     = J_g.shape[0]
        L = J_pinv.T @ eta_mat @ J_g.T             # (3,n_g)
        R = J_g @ J_pinv                            # (n_g,3)
        M = np.kron(R.T, L)                         # (9,n_g²)
        return (np.linalg.pinv(M) @ K_des.flatten('F')).reshape(n_g, n_g, order='F')

    def stiffness_inversion_all(self, finger, q, K_dict, K_des):
        """
        Joint minimum-norm inversion for all joint-space groups.

        vec(K_des) ≈ [M_1  M_2  ...  M_G] · [vec(K_1); vec(K_2); ...; vec(K_G)]
        where M_g = (R_g^T ⊗ L_g),  L_g = J_pinv^T·η·J_g^T,  R_g = J_g·J_pinv.

        Args:
            finger:  output fingertip
            q:       (15,) current motor angles [rad]
            K_dict:  dict of per-group stiffness matrices (ordering defines vec stack)
            K_des:   (3,3) desired tip stiffness [N/m]

        Returns:
            K_ff_dict: dict of minimum-norm group stiffness matrices
        """
        eta_mat = np.diag(self.eta)
        J_pinv  = self._J_tip_P_pinv(finger, q)

        M_blocks = []
        sizes = []
        for group in K_dict.keys():
            J_g = self._J_angle(group, q)
            n_g = J_g.shape[0]
            L = J_pinv.T @ eta_mat @ J_g.T
            R = J_g @ J_pinv
            M_blocks.append(np.kron(R.T, L))
            sizes.append(n_g)

        M_all = np.hstack(M_blocks) if M_blocks else np.zeros((9, 0))
        vec_all = np.linalg.pinv(M_all) @ K_des.flatten('F') if M_all.size else np.array([])

        K_ff_dict = {}
        offset = 0
        for group, n_g in zip(K_dict.keys(), sizes):
            block = vec_all[offset:offset + n_g * n_g]
            K_ff_dict[group] = block.reshape(n_g, n_g, order='F')
            offset += n_g * n_g
        return K_ff_dict

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def stiffness_descent(self, finger, q, theta_ref_deg, K_dict, f_meas, f_des, lr=1e-4):
        """
        One gradient-descent step on all K_dict entries simultaneously.

        Minimises  L = ½ ||f_meas - f_des||²  w.r.t. each K_g independently.
            ∂L/∂vec(K_g) = C_g^T · (f_meas - f_des)

        Args:
            finger:         output fingertip
            q:              (15,) current motor angles [rad]
            theta_ref_deg:  (n_total,) concatenated joint-space reference [deg]
            K_dict:         dict of per-group stiffness matrices
            f_meas, f_des:  (3,) measured and desired tip forces [N]
            lr:             scalar or dict of per-group learning rates

        Returns:
            K_dict_new: updated copy of K_dict
        """
        theta_ref = np.radians(theta_ref_deg)
        eta    = np.diag(self.eta)
        J_pinv = self._J_tip_P_pinv(finger, q)
        error = f_meas - f_des

        idx = 0
        K_new = {}
        for group, K in K_dict.items():
            lr_g = lr[group] if isinstance(lr, dict) else lr
            J_g  = self._J_angle(group, q)
            n_g  = J_g.shape[0]
            delta = self._deflection(group, q, theta_ref[idx:idx + n_g])
            A = J_pinv.T @ eta @ J_g.T
            grad = A.T @ error.reshape(-1, 1) @ delta.reshape(1, -1)
            K_new[group] = K - lr_g * grad
            idx += n_g
        return K_new

    def ref_descent(self, finger, q, theta_ref_deg, K_dict, f_meas, f_des, lr=1e-06):
        """
        One gradient-descent step on the concatenated joint-reference vector
        to reduce tip force error.

        Args:
            finger:        output fingertip
            q:             (13,) current motor angles [rad]
            theta_ref_deg: (n_total,) concatenated joint-reference vector [deg]
            K_dict:        dict of per-group stiffness matrices (ordering defines theta_ref)
            f_meas:        (3,) measured tip force [N]
            f_des:         (3,) desired tip force [N]
            lr:            learning rate — same numeric value as the [rad/N] lr

        Returns:
            theta_ref_deg_new: (n_total,) updated joint-reference vector [deg]
        """
        eta    = np.diag(self.eta)
        J_pinv = self._J_tip_P_pinv(finger, q)

        cols = []
        for group, K in K_dict.items():
            J_q = self._J_angle(group, q)
            cols.append(J_q.T @ np.atleast_2d(K))
        S_mat = np.hstack(cols) if cols else np.zeros((15, 0))

        S = J_pinv.T @ eta @ S_mat
        error = f_meas - f_des
        return theta_ref_deg - lr * np.degrees(S.T @ error)          # stay in degrees

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

    k = 0.4   # uniform stiffness [N·m/rad]
    K_dict = {
        'thumb':         k * np.eye(4),
        'spread_index':  k * np.eye(1),
        'spread_middle': k * np.eye(1),
        'spread_ring':   k * np.eye(1),
        'spread_pinky':  k * np.eye(1),
        'index':         k * np.eye(3),
        'middle':        k * np.eye(3),
        'ring':          k * np.eye(3),
        'pinky':         k * np.eye(3),
    }

    print("=" * 70)
    print("STIFFNESS2JOINTSPACE — initializing Jacobians and Hessians ...")
    t0 = time.time()
    model = tip_stiffness_JointSpace(mode='full')
    print(f"  done in {time.time() - t0:.1f} s")
    print("=" * 70)

    # Convert motor reference to joint-space reference in degrees
    theta_ref_deg = model.motor_to_theta_deg(q_ref, K_dict)

    def _tip_pos(finger, q):
        rtip = model.rtips[finger]
        if finger == 'thumb':
            return FK_motor2thumbPos(q, 'IP', rtip)
        return FK_motor2fingerPos(q, finger, 'DIP', rtip)

    # 1. Tip stiffnesses and forces at q_base
    print("\n1. Tip stiffnesses (eigenvalues) and forces at q_base:")
    for finger in FINGERS_ALL:
        K_x = model.tip_stiffness(finger, q_base, K_dict)
        f   = model.tip_force(finger, q_base, theta_ref_deg, K_dict)
        eig = np.linalg.eigvalsh(K_x)
        print(f"  {finger:6s}  K_x eig = {np.round(eig,4)}  f = {np.round(f,5)}")

    # 2. FD stiffness validation: K_tip · Δx ≈ -Δf for row-space perturbations
    print("\n2. FD stiffness validation (thumb and index):")
    np.random.seed(1)
    for finger in ['thumb', 'index']:
        f0  = model.tip_force(finger, q_base, theta_ref_deg, K_dict)
        x0  = _tip_pos(finger, q_base)
        rtip = model.rtips[finger]
        J_tip = np.array(
            model.jac.get_thumb_jacobian('IP', q_base, rtip) if finger == 'thumb'
            else model.jac.get_finger_jacobian(finger, 'DIP', q_base, rtip)
        )
        K1 = model.tip_stiffness(finger, q_base, K_dict)
        K2 = model.tip_stiffness(finger, q_base, K_dict, f_ext=f0, theta_ref_deg=theta_ref_deg)
        print(f"  {finger}:")
        for eps in [1e-4, 1e-5, 1e-6]:
            errs_1st, errs_2nd = [], []
            for _ in range(6):
                v  = np.random.randn(3)
                dq = J_tip.T @ v
                dq = dq * (eps / np.linalg.norm(dq))
                dx = _tip_pos(finger, q_base + dq) - x0
                df = model.tip_force(finger, q_base + dq, theta_ref_deg, K_dict) - f0
                scale = np.linalg.norm(df) + 1e-15
                errs_1st.append(np.linalg.norm(K1 @ dx + df) / scale)
                errs_2nd.append(np.linalg.norm(K2 @ dx + df) / scale)
            print(f"    eps={eps:.0e}  1st-order rel={np.mean(errs_1st):.2e}"
                  f"  2nd-order rel={np.mean(errs_2nd):.2e}"
                  f"  (ratio {np.mean(errs_1st)/np.mean(errs_2nd):.1f}x)")

    # 3. stiffness_descent convergence (thumb and index)
    print("\n3. stiffness_descent convergence ...")
    f_des = np.array([0.0, 0.02, 0.03])
    MAX_ITERS = 10000
    for finger in ['thumb', 'index']:
        K_opt = {g: K.copy() for g, K in K_dict.items()}
        for i in range(MAX_ITERS):
            f_meas = model.tip_force(finger, q_base, theta_ref_deg, K_opt)
            err    = np.linalg.norm(f_meas - f_des)
            if i % 200 == 0:
                print(f"  [{finger}] iter {i:4d}  ||error|| = {err:.6f}")
            K_opt = model.stiffness_descent(finger, q_base, theta_ref_deg, K_opt,
                                            f_meas, f_des, lr=3e-3)
        final = np.linalg.norm(model.tip_force(finger, q_base, theta_ref_deg, K_opt) - f_des)
        print(f"  [{finger}] Final  ||error|| = {final:.6f}")

    # 4. ref_descent convergence (thumb and index) — operate on theta_ref_deg
    print("\n4. ref_descent convergence ...")
    f_des = np.array([0.0, 0.02, 0.02])
    for finger in ['thumb', 'index']:
        theta_ref_opt_deg = model.motor_to_theta_deg(q_ref, K_dict)

        for i in range(MAX_ITERS):
            f_meas = model.tip_force(finger, q_base, theta_ref_opt_deg, K_dict)
            err    = np.linalg.norm(f_meas - f_des)
            if i % 200 == 0:
                print(f"  [{finger}] iter {i:4d}  ||error|| = {err:.6f}")
            theta_ref_opt_deg = model.ref_descent(finger, q_base, theta_ref_opt_deg, K_dict,
                                                  f_meas, f_des, lr=2e-3)
        final = np.linalg.norm(model.tip_force(finger, q_base, theta_ref_opt_deg, K_dict) - f_des)
        print(f"  [{finger}] Final  ||error|| = {final:.6f}")

    # 5. stiffness_inversion check (thumb and index)
    print("\n5. stiffness_inversion check:")
    for finger in ['thumb', 'index']:
        K_target    = {g: K.copy() * 1.2 for g, K in K_dict.items()}
        K_des_stiff = model.tip_stiffness(finger, q_base, K_target)
        K_ff_dict   = model.stiffness_inversion_all(finger, q_base, K_dict, K_des_stiff)
        K_recon     = model.tip_stiffness(finger, q_base, K_ff_dict)
        recon_err   = np.linalg.norm(K_recon - K_des_stiff, 'fro')
        print(f"  [{finger}] ||K_recon - K_des||_F = {recon_err:.6f}")
