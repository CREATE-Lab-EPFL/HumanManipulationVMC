"""
Compute fingertip stiffness from VMC virtual elements in mixed
(joint-space + task-space) coordinates.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KinematicsHand.FK_Hand import *
from KinematicsHand.JacobiansHand import *
from KinematicsHand.HessiansHand import *
from ModelIDHand.hand_params import FINGER_TIP_OFFSETS

FINGERS_ALL = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGERS_4   = ['index', 'middle', 'ring', 'pinky']
ALL_POINTS  = ['thumb', 'index', 'middle', 'ring', 'pinky', 'palm']

DEFAULT_RTIPS        = FINGER_TIP_OFFSETS
DEFAULT_RATTACHMENTS = {**FINGER_TIP_OFFSETS, 'palm': np.zeros(3)}


class tip_stiffness_MixedSpace:
    """
    Compute fingertip stiffness and force for VMC with virtual elements in both
    joint space and task space.

    Springs live simultaneously in joint-angle space and Cartesian space.
    Any subset of either group can be active. The contact normal at each
    fingertip is derived on the fly from FK — no fixed n is assumed.

    K_joint_dict keys and dimensions:
        'thumb':         (4,4)   — [CMC1, CMC2, MCP, IP]
        'spread_index':  (1,1)
        'spread_middle': (1,1)
        'spread_ring':   (1,1)
        'spread_pinky':  (1,1)
        'index':         (3,3)   — [MCP, PIP, DIP]
        'middle':        (3,3)
        'ring':          (3,3)
        'pinky':         (3,3)

    K_task_dict keys and dimensions (any subset):
        'thumb':  (3,3)  — Cartesian stiffness at thumb tip  [N/m]
        'index':  (3,3)  — Cartesian stiffness at index tip  [N/m]
        'middle': (3,3)  — Cartesian stiffness at middle tip [N/m]
        'ring':   (3,3)  — Cartesian stiffness at ring tip   [N/m]
        'pinky':  (3,3)  — Cartesian stiffness at pinky tip  [N/m]
        'palm':   (3,3)  — Cartesian stiffness at palm point [N/m]

    Uses constant efficiency model:
        η = diag([η_CMC1, η_CMC2, η_MCP_thumb, η_IP,
                  η_spread, η_MCP_idx, η_PIP_idx, η_MCP_mid, η_PIP_mid,
                  η_MCP_rng, η_PIP_rng, η_MCP_pnk, η_PIP_pnk])
    """

    def __init__(self, rtips=None, rattachments=None, eta=None, mode='normal'):
        """
        Args:
            rtips:         dict of tip offsets in DIP/IP local frame [m] for
                           normal computation. Defaults to DEFAULT_RTIPS.
            rattachments:  dict of attachment offsets for task-space springs [m].
                           Defaults to DEFAULT_RATTACHMENTS.
            eta:           (13,) motor efficiency vector. Defaults to ones (no efficiency loss).
            mode:          'normal' — P = n nᵀ (rank-1, force/stiffness along contact normal only);
                           'full'   — P = I₃   (full 3-D force and stiffness).
        """
        self.rtips        = {**DEFAULT_RTIPS,        **(rtips or {})}
        self.rattachments = {**DEFAULT_RATTACHMENTS, **(rattachments or {})}
        self.eta          = np.asarray(eta) if eta is not None else np.ones(13)
        self.mode         = mode

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
        r = self.rtips[finger]
        if finger == 'thumb':
            return np.array(self.jac.get_thumb_jacobian('IP', q, r))
        return np.array(self.jac.get_finger_jacobian(finger, 'DIP', q, r))

    def _J_tip_P_pinv(self, finger, q):
        return np.linalg.pinv(self._P(finger, q) @ self._J_tip(finger, q))

    def _H_tip(self, finger, q):
        r = self.rtips[finger]
        if finger == 'thumb':
            return self.hes.get_thumb_hessian('IP', q, r)
        return self.hes.get_finger_hessian(finger, 'DIP', q, r)

    # ------------------------------------------------------------------
    # Private helpers — joint-space springs
    # ------------------------------------------------------------------

    def _J_angle(self, group, q):
        if group == 'thumb':
            return np.array(self.jac.get_angles_jacobian('thumb', q))
        if group.startswith('spread_'):
            return np.array(self.jac.get_spread_jacobian(group[7:], q))
        return np.array(self.jac.get_angles_jacobian(group, q))

    def _group_angles(self, group, q):
        """Current joint angles for a group [rad]."""
        if group == 'thumb':
            return FK_motor2thumb(q)
        if group.startswith('spread_'):
            return np.array([FK_motor2spread(q, group[7:])])
        return FK_motor2finger(q, group)

    def _deflection_joint(self, group, q, theta_group_ref_rad):
        """Deflection for a joint group: theta_group_ref_rad - current_angles [rad]."""
        return theta_group_ref_rad - self._group_angles(group, q)

    # ------------------------------------------------------------------
    # Private helpers — task-space springs
    # ------------------------------------------------------------------

    def _pos(self, point, q):
        r = self.rattachments[point]
        if point == 'thumb':
            return FK_motor2thumbPos(q, 'IP', r)
        if point == 'palm':
            return FK_motor2palm(r)[1]
        return FK_motor2fingerPos(q, point, 'DIP', r)

    def _J_pos(self, point, q):
        r = self.rattachments[point]
        if point == 'thumb':
            return np.array(self.jac.get_thumb_jacobian('IP', q, r))
        if point == 'palm':
            return np.array(self.jac.get_wrist_palm_jacobian(q, r))
        return np.array(self.jac.get_finger_jacobian(point, 'DIP', q, r))

    def _H_pos(self, point, q):
        r = self.rattachments[point]
        if point == 'thumb':
            return self.hes.get_thumb_hessian('IP', q, r)
        if point == 'palm':
            return self.hes.get_wrist_palm_hessian(q, r)
        return self.hes.get_finger_hessian(point, 'DIP', q, r)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def motor_stiffness(self, q, K_joint_dict, K_task_dict):
        """
        13×13 1st-order motor stiffness from all virtual elements.

            K_motor = η · (Σ_g J_θg^T·K_θg·J_θg  +  Σ_p J_xp^T·K_p·J_xp)

        Args:
            q:             (13,) current motor angles [rad]
            K_joint_dict:  dict of per-group joint-space stiffness matrices
            K_task_dict:   dict of per-point (3,3) task-space stiffness matrices

        Returns:
            K_motor: (13,13)
        """
        eta = np.diag(self.eta)
        S   = np.zeros((13, 13))
        for group, K in K_joint_dict.items():
            J  = self._J_angle(group, q)
            S += J.T @ np.atleast_2d(K) @ J
        for point, K in K_task_dict.items():
            J  = self._J_pos(point, q)
            S += J.T @ K @ J
        return eta @ S

    def tip_stiffness(self, finger, q, K_joint_dict, K_task_dict,
                      d_ref_dict=None, f_ext=None):
        """
        3×3 Cartesian stiffness at fingertip of the given finger.

        Args:
            finger:        'thumb', 'index', 'middle', 'ring', or 'pinky'
            q:             (13,) current motor angles [rad]
            K_joint_dict:  dict of per-group joint-space stiffness matrices
            K_task_dict:   dict of per-point (3,3) task-space stiffness matrices
            d_ref_dict:    dict of (3,) reference positions [m]   (for 2nd order)
            f_ext:         (3,) contact force for 2nd-order CCT correction [N]

        Returns:
            K_x: (3,3) tip stiffness [N/m]
        """
        K_motor = self.motor_stiffness(q, K_joint_dict, K_task_dict)

        if f_ext is not None:
            P       = self._P(finger, q)
            H_xf    = self._H_tip(finger, q)                          # (3,15,15)
            H_xf_P  = np.tensordot(P, H_xf, axes=([1], [0]))          # (3,15,15)
            K_g_out = np.tensordot(f_ext, H_xf_P, axes=([0], [0]))    # (15,15)

            # Inward term: only from task-space springs (H_θg = 0 for joint springs)
            tau_d = []
            H_d   = []
            if d_ref_dict is not None:
                for point, K in K_task_dict.items():
                    delta_x = d_ref_dict[point] - self._pos(point, q)
                    tau_d.append(K @ delta_x)
                    H_d.append(self._H_pos(point, q))
            if tau_d:
                tau_d   = np.concatenate(tau_d)
                H_d     = np.vstack(H_d)
                eta_mat = np.diag(self.eta)
                K_g_in  = eta_mat @ np.tensordot(tau_d, H_d, axes=([0], [0]))
            else:
                K_g_in = 0.0

            K_motor = K_motor + K_g_out - K_g_in

        J_pinv = self._J_tip_P_pinv(finger, q)
        return J_pinv.T @ K_motor @ J_pinv

    def tip_force(self, finger, q, theta_ref_deg, d_ref_dict, K_joint_dict, K_task_dict):
        """
        3-vector contact force at fingertip from all virtual element contributions.

            f_f = pinv(P_f·J_xf)^T · η · (Σ_g J_θg^T·K_θg·Δθ_g
                                          + Σ_p J_xp^T·K_p·Δx_p)

        Args:
            finger:        'thumb', 'index', 'middle', 'ring', or 'pinky'
            q:             (13,) current motor angles [rad]
            theta_ref_deg: (n_total,) concatenated joint-space reference [deg],
                           ordered to match K_joint_dict iteration order
            d_ref_dict:    dict of (3,) reference positions [m]
            K_joint_dict:  dict of per-group joint-space stiffness matrices
            K_task_dict:   dict of per-point (3,3) task-space stiffness matrices

        Returns:
            f: (3,) tip force [N]
        """
        eta = np.diag(self.eta)
        tau = np.zeros(13)
        idx = 0
        for group, K in K_joint_dict.items():
            J   = self._J_angle(group, q)
            n_g = J.shape[0]
            theta_group = np.radians(theta_ref_deg[idx:idx + n_g])
            delta = self._deflection_joint(group, q, theta_group)
            tau  += J.T @ np.atleast_2d(K) @ delta
            idx  += n_g
        for point, K in K_task_dict.items():
            J       = self._J_pos(point, q)
            delta_x = d_ref_dict[point] - self._pos(point, q)
            tau    += J.T @ K @ delta_x
        J_pinv = self._J_tip_P_pinv(finger, q)
        return J_pinv.T @ eta @ tau

    def all_tip_stiffnesses(self, q, K_joint_dict, K_task_dict,
                            d_ref_dict=None, f_ext_dict=None):
        """
        Dict of 3×3 stiffness matrices, one per fingertip.

        Returns:
            {finger: (3,3)} for all five fingers
        """
        f_ext_dict = f_ext_dict or {}
        return {
            f: self.tip_stiffness(f, q, K_joint_dict, K_task_dict,
                                  d_ref_dict, f_ext_dict.get(f))
            for f in FINGERS_ALL
        }

    def all_tip_forces(self, q, theta_ref_deg, d_ref_dict, K_joint_dict, K_task_dict):
        """
        Dict of (3,) contact forces, one per fingertip.

        Returns:
            {finger: (3,)} for all five fingers
        """
        return {
            f: self.tip_force(f, q, theta_ref_deg, d_ref_dict, K_joint_dict, K_task_dict)
            for f in FINGERS_ALL
        }

    # ------------------------------------------------------------------
    # Gradient descent
    # ------------------------------------------------------------------

    def stiffness_descent(self, finger, q, theta_ref_deg, d_ref_dict,
                          K_joint_dict, K_task_dict, f_meas, f_des,
                          lr_joint=1e-4, lr_task=1e-4):
        """
        One gradient-descent step on all stiffness entries simultaneously.

            ∂L/∂vec(K_g) = C_g^T · error   (joint springs)
            ∂L/∂vec(K_p) = C_p^T · error   (task springs)

        Args:
            finger:        output fingertip
            q:             (15,) motor angles [rad]
            theta_ref_deg: (n_total,) concatenated joint reference [deg]
            d_ref_dict:    dict of (3,) reference positions [m]
            K_joint_dict:  dict of per-group joint-space stiffness matrices
            K_task_dict:   dict of per-point (3,3) task-space stiffness matrices
            f_meas, f_des: (3,) measured and desired tip forces [N]
            lr_joint:      scalar or dict learning rate for joint springs
            lr_task:       scalar or dict learning rate for task springs

        Returns:
            K_joint_new, K_task_new: updated copies
        """
        theta_ref = np.radians(theta_ref_deg)
        eta    = np.diag(self.eta)
        J_pinv = self._J_tip_P_pinv(finger, q)
        error = f_meas - f_des

        K_joint_new = {}
        idx = 0
        for group, K in K_joint_dict.items():
            lr_g = lr_joint[group] if isinstance(lr_joint, dict) else lr_joint
            J_g  = self._J_angle(group, q)
            n_g  = J_g.shape[0]
            delta = self._deflection_joint(group, q, theta_ref[idx:idx + n_g])
            A    = J_pinv.T @ eta @ J_g.T
            grad = A.T @ error.reshape(-1, 1) @ delta.reshape(1, -1)
            K_joint_new[group] = K - lr_g * grad
            idx += n_g

        K_task_new = {}
        for point, K in K_task_dict.items():
            lr_p = lr_task[point] if isinstance(lr_task, dict) else lr_task
            J_p  = self._J_pos(point, q)
            delta = d_ref_dict[point] - self._pos(point, q)
            A    = J_pinv.T @ eta @ J_p.T
            grad = A.T @ error.reshape(-1, 1) @ delta.reshape(1, -1)
            K_task_new[point] = K - lr_p * grad

        return K_joint_new, K_task_new

    def ref_descent(self, finger, q, theta_ref_deg, d_ref_dict,
                    K_joint_dict, K_task_dict, f_meas, f_des,
                    lr_joint=3.33e-05, lr_task=4.18e-08):
        """
        One gradient-descent step on theta_ref_deg and d_ref_dict.

            theta_ref_deg^+ = theta_ref_deg − lr_joint · degrees(S_joint^T · error)
            d_ref_p^+       = d_ref_p       − lr_task  · S_p^T · error

        Args:
            finger:        output fingertip
            q:             (15,) motor angles [rad]
            theta_ref_deg: (n_total,) concatenated joint reference [deg]
            d_ref_dict:    dict of (3,) reference positions [m]
            K_joint_dict:  dict of per-group joint-space stiffness matrices
            K_task_dict:   dict of per-point (3,3) task-space stiffness matrices
            f_meas, f_des: (3,) measured and desired tip forces [N]
            lr_joint:      learning rate for theta_ref_deg
            lr_task:       scalar or dict learning rate for d_ref [m/N]

        Returns:
            theta_ref_deg_new: (n_total,) updated joint reference [deg]
            d_ref_dict_new:    updated copy
        """
        eta    = np.diag(self.eta)
        J_pinv = self._J_tip_P_pinv(finger, q)

        cols = []
        for group, K in K_joint_dict.items():
            J_q = self._J_angle(group, q)
            cols.append(J_q.T @ np.atleast_2d(K))
        M_joint = np.hstack(cols) if cols else np.zeros((13, 0))
        S_joint = J_pinv.T @ eta @ M_joint

        S_task = {
            point: J_pinv.T @ eta @ self._J_pos(point, q).T @ K
            for point, K in K_task_dict.items()
        }
        error = f_meas - f_des

        theta_ref_deg_new = theta_ref_deg - lr_joint * np.degrees(S_joint.T @ error)
        d_ref_new = {}
        for point, d_ref in d_ref_dict.items():
            if point in S_task:
                lr_p = lr_task[point] if isinstance(lr_task, dict) else lr_task
                d_ref_new[point] = d_ref - lr_p * (S_task[point].T @ error)
            else:
                d_ref_new[point] = d_ref.copy()
        return theta_ref_deg_new, d_ref_new

    def stiffness_inversion_K_joint(self, finger, group, q, K_des):
        """
        Minimum-Frobenius-norm K_group such that K_x ≈ K_des (joint-space spring).

            vec(K_des) ≈ M · vec(K_group),   M = (J_g·J_pinv)^T ⊗ (J_pinv^T·η·J_g^T)
        """
        eta_mat = np.diag(self.eta)
        J_pinv  = self._J_tip_P_pinv(finger, q)   # (13,3)
        J_g     = self._J_angle(group, q)          # (n_g,13)
        n_g     = J_g.shape[0]
        L = J_pinv.T @ eta_mat @ J_g.T             # (3,n_g)
        R = J_g @ J_pinv                            # (n_g,3)
        M = np.kron(R.T, L)                         # (9,n_g²)
        return (np.linalg.pinv(M) @ K_des.flatten('F')).reshape(n_g, n_g, order='F')

    def stiffness_inversion_K_task(self, finger, point, q, K_des):
        """
        Minimum-Frobenius-norm K_point such that K_x ≈ K_des (task-space spring).

            vec(K_des) ≈ M · vec(K_point),   M = (J_p·J_pinv)^T ⊗ (J_pinv^T·η·J_p^T)
        """
        eta_mat = np.diag(self.eta)
        J_pinv  = self._J_tip_P_pinv(finger, q)   # (15,3)
        J_p     = self._J_pos(point, q)            # (3,15)
        L = J_pinv.T @ eta_mat @ J_p.T             # (3,3)
        R = J_p @ J_pinv                            # (3,3)
        M = np.kron(R.T, L)                         # (9,9)
        return (np.linalg.pinv(M) @ K_des.flatten('F')).reshape(3, 3, order='F')

    def stiffness_inversion_all(self, finger, q, K_joint_dict, K_task_dict, K_des):
        """
        Joint minimum-norm inversion for all joint and task springs.

        vec(K_des) ≈ [M_j1 ... M_jG  M_t1 ... M_tP] · [vec(K_j1); ...; vec(K_jG); vec(K_t1); ...]
        where M = (R^T ⊗ L), L = J_pinv^T·η·J^T, R = J·J_pinv.

        Args:
            finger:        output fingertip
            q:             (13,) current motor angles [rad]
            K_joint_dict:  dict of joint-space stiffness matrices
            K_task_dict:   dict of task-space stiffness matrices
            K_des:         (3,3) desired tip stiffness [N/m]

        Returns:
            K_joint_ff: dict of minimum-norm joint stiffness matrices
            K_task_ff:  dict of minimum-norm task stiffness matrices
        """
        eta_mat = np.diag(self.eta)
        J_pinv  = self._J_tip_P_pinv(finger, q)

        M_blocks = []
        joint_sizes = []
        for group in K_joint_dict.keys():
            J_g = self._J_angle(group, q)
            n_g = J_g.shape[0]
            L = J_pinv.T @ eta_mat @ J_g.T
            R = J_g @ J_pinv
            M_blocks.append(np.kron(R.T, L))
            joint_sizes.append(n_g)

        for point in K_task_dict.keys():
            J_p = self._J_pos(point, q)
            L = J_pinv.T @ eta_mat @ J_p.T
            R = J_p @ J_pinv
            M_blocks.append(np.kron(R.T, L))

        M_all = np.hstack(M_blocks) if M_blocks else np.zeros((9, 0))
        vec_all = np.linalg.pinv(M_all) @ K_des.flatten('F') if M_all.size else np.array([])

        K_joint_ff = {}
        offset = 0
        for group, n_g in zip(K_joint_dict.keys(), joint_sizes):
            block = vec_all[offset:offset + n_g * n_g]
            K_joint_ff[group] = block.reshape(n_g, n_g, order='F')
            offset += n_g * n_g

        K_task_ff = {}
        for point in K_task_dict.keys():
            block = vec_all[offset:offset + 9]
            K_task_ff[point] = block.reshape(3, 3, order='F')
            offset += 9

        return K_joint_ff, K_task_ff

# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    import time

    np.random.seed(42)

    # Thumb and index flexed (13 motors, wrist rigid)
    q_ref  = np.zeros(13)
    q_base = np.zeros(13)
    q_base[2] = np.deg2rad(30.0)   # thumb MCP
    q_base[3] = np.deg2rad(20.0)   # thumb IP
    q_base[5] = np.deg2rad(30.0)   # index MCP
    q_base[6] = np.deg2rad(20.0)   # index PIP

    k_joint = 0.4    # [N·m/rad]
    k_task  = 100.0  # [N/m]

    K_joint_dict = {
        'thumb':         k_joint * np.eye(4),
        'spread_index':  k_joint * np.eye(1),
        'spread_middle': k_joint * np.eye(1),
        'spread_ring':   k_joint * np.eye(1),
        'spread_pinky':  k_joint * np.eye(1),
        'index':         k_joint * np.eye(3),
        'middle':        k_joint * np.eye(3),
        'ring':          k_joint * np.eye(3),
        'pinky':         k_joint * np.eye(3),
    }

    print("=" * 70)
    print("STIFFNESS2MIXEDSPACE — initializing Jacobians and Hessians ...")
    t0 = time.time()
    model = tip_stiffness_MixedSpace(mode='full')
    print(f"  done in {time.time() - t0:.1f} s")
    print("=" * 70)

    d_ref_dict  = {p: model._pos(p, q_ref) for p in ALL_POINTS}
    K_task_dict = {p: k_task * np.eye(3) for p in ALL_POINTS}

    # Build initial theta_ref_deg from q_ref (ordered to match K_joint_dict)
    def _build_theta_ref_deg(q, K_jd):
        parts = []
        for group in K_jd.keys():
            if group == 'thumb':
                parts.append(FK_motor2thumb(q))
            elif group.startswith('spread_'):
                parts.append(np.array([FK_motor2spread(q, group[7:])]))
            else:
                parts.append(FK_motor2finger(q, group))
        return np.degrees(np.concatenate(parts))

    theta_ref_init_deg = _build_theta_ref_deg(q_ref, K_joint_dict)

    MAX_ITERS = 30000

    # 1. Tip stiffnesses and forces
    print("\n1. Tip stiffnesses (eigenvalues) and forces at q_base:")
    for finger in FINGERS_ALL:
        K_x = model.tip_stiffness(finger, q_base, K_joint_dict, K_task_dict)
        f   = model.tip_force(finger, q_base, theta_ref_init_deg, d_ref_dict,
                              K_joint_dict, K_task_dict)
        eig = np.linalg.eigvalsh(K_x)
        print(f"  {finger:6s}  K_x eig = {np.round(eig,3)}  f = {np.round(f,5)}")

    # 2. FD stiffness validation: K_tip · Δx ≈ -Δf for row-space perturbations
    print("\n2. FD stiffness validation (thumb and index):")
    np.random.seed(1)
    for finger in ['thumb', 'index']:
        r    = model.rtips[finger]
        J_tip = np.array(
            model.jac.get_thumb_jacobian('IP', q_base, r) if finger == 'thumb'
            else model.jac.get_finger_jacobian(finger, 'DIP', q_base, r)
        )
        f0 = model.tip_force(finger, q_base, theta_ref_init_deg, d_ref_dict,
                             K_joint_dict, K_task_dict)
        x0 = model._pos(finger, q_base)
        K1 = model.tip_stiffness(finger, q_base, K_joint_dict, K_task_dict)
        K2 = model.tip_stiffness(finger, q_base, K_joint_dict, K_task_dict,
                                  d_ref_dict=d_ref_dict, f_ext=f0)
        print(f"  {finger}:")
        for eps in [1e-3, 1e-4, 1e-5]:
            errs_1, errs_2 = [], []
            for _ in range(6):
                v  = np.random.randn(3)
                dq = J_tip.T @ v
                dq = dq * (eps / np.linalg.norm(dq))
                dx = model._pos(finger, q_base + dq) - x0
                df = model.tip_force(finger, q_base + dq, theta_ref_init_deg, d_ref_dict,
                                     K_joint_dict, K_task_dict) - f0
                scale = np.linalg.norm(df) + 1e-15
                errs_1.append(np.linalg.norm(K1 @ dx + df) / scale)
                errs_2.append(np.linalg.norm(K2 @ dx + df) / scale)
            print(f"    eps={eps:.0e}  1st-order rel={np.mean(errs_1):.2e}"
                  f"  2nd-order rel={np.mean(errs_2):.2e}"
                  f"  (ratio {np.mean(errs_1) / np.mean(errs_2):.1f}x)")

    # 3. stiffness_descent convergence (thumb and index)
    print("\n3. stiffness_descent convergence ...")
    f_des = np.array([0.0, 0.02, 0.03])
    for finger in ['thumb', 'index']:
        Kj_opt = {g: K.copy() for g, K in K_joint_dict.items()}
        Kt_opt = {p: K.copy() for p, K in K_task_dict.items()}
        for i in range(MAX_ITERS):
            f_meas = model.tip_force(finger, q_base, theta_ref_init_deg, d_ref_dict,
                                     Kj_opt, Kt_opt)
            err    = np.linalg.norm(f_meas - f_des)
            if i % 5000 == 0:
                print(f"  [{finger}] iter {i:3d}  ||error|| = {err:.6f}")
            Kj_opt, Kt_opt = model.stiffness_descent(
                finger, q_base, theta_ref_init_deg, d_ref_dict, Kj_opt, Kt_opt,
                f_meas, f_des, lr_joint=2e-4, lr_task=1e-3)
        final = np.linalg.norm(
            model.tip_force(finger, q_base, theta_ref_init_deg, d_ref_dict, Kj_opt, Kt_opt) - f_des)
        print(f"  [{finger}] Final  ||error|| = {final:.6f}")

    # 4. ref_descent convergence (thumb and index)
    print("\n4. ref_descent convergence ...")
    for finger in ['thumb', 'index']:
        f_des = np.array([0.0, 0.02, 0.03])
        theta_ref_opt_deg = _build_theta_ref_deg(q_ref, K_joint_dict)
        d_ref_opt = {p: x.copy() for p, x in d_ref_dict.items()}
        for i in range(MAX_ITERS):
            f_meas = model.tip_force(finger, q_base, theta_ref_opt_deg, d_ref_opt,
                                     K_joint_dict, K_task_dict)
            err    = np.linalg.norm(f_meas - f_des)
            if i % 5000 == 0:
                print(f"  [{finger}] iter {i:4d}  ||error|| = {err:.6f}")
            theta_ref_opt_deg, d_ref_opt = model.ref_descent(
                finger, q_base, theta_ref_opt_deg, d_ref_opt, K_joint_dict, K_task_dict,
                f_meas, f_des, lr_joint=3.33e-05, lr_task=4.18e-08)
        final = np.linalg.norm(
            model.tip_force(finger, q_base, theta_ref_opt_deg, d_ref_opt,
                            K_joint_dict, K_task_dict) - f_des)
        print(f"  [{finger}] Final  ||error|| = {final:.6f}")

    # 5. stiffness_inversion check (thumb and index)
    print("\n5. stiffness_inversion check:")
    for finger in ['thumb', 'index']:
        Kj_target   = {g: K.copy() * 1.2 for g, K in K_joint_dict.items()}
        Kt_target   = {p: K.copy() * 1.2 for p, K in K_task_dict.items()}
        K_des_stiff = model.tip_stiffness(finger, q_base, Kj_target, Kt_target)
        Kj_ff, Kt_ff = model.stiffness_inversion_all(finger, q_base, K_joint_dict, K_task_dict, K_des_stiff)
        K_recon = model.tip_stiffness(finger, q_base, Kj_ff, Kt_ff)
        recon_err = np.linalg.norm(K_recon - K_des_stiff, 'fro')
        print(f"  [{finger}] ||K_recon - K_des||_F = {recon_err:.6f}")
