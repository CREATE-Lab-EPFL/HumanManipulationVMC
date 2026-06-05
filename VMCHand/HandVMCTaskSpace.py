"""
VMC Controller for ADAPT Hand in Task Space.

Six virtual attachment points:
    thumb  — thumb fingertip (IP link)
    index  — index fingertip (DIP link)
    middle — middle fingertip (DIP link)
    ring   — ring fingertip (DIP link)
    pinky  — pinky fingertip (DIP link)
    palm   — palm frame origin (wrist)

Each point is governed by an independent virtual spring and damper.
Stiffness / damping can be scalars (isotropic), 3-vectors (per-axis),
or 3×3 matrices. Any spring/damper class from VMC_utils.VirtualModels
can be substituted.

Units:
    positions  [m]
    forces     [N]
    stiffness  [N/m]
    damping    [N·s/m]
    torques    [N·m]
"""

import numpy as np
from VMC_utils.VirtualModels import LinearSpring, LinearDamper
from KinematicsHand.FK_Hand import FK_motor2fingerPos, FK_motor2thumbPos, FK_motor2palm
from KinematicsHand.JacobiansHand import HandJacobians

_FINGERS = ['index', 'middle', 'ring', 'pinky']
_POINTS  = ['thumb', 'index', 'middle', 'ring', 'pinky', 'palm']


class VMC:
    """
    Virtual Model Controller for the ADAPT Hand in Task Space.

    Each of the 6 Cartesian points (fingertips + palm) is regulated by an
    independent spring-damper pair. Torques are computed via Jacobian transpose:

        tau = sum_p  J_p^T @ ( F_spring_p + F_damper_p )

    Usage
    -----
    vmc = VMC()
    vmc.set_stiffness(200.0)                  # isotropic, all points
    vmc.set_damping(5.0)
    vmc.targets['index'] = np.array([...])    # set index fingertip target
    tau = vmc.hand_torques(q_motor, q_dot_motor)
    """

    POINTS = _POINTS

    def __init__(self):

        # Virtual springs [N/m] — zero by default (no force until explicitly set)
        self.springs = {p: LinearSpring(np.zeros(3)) for p in _POINTS}

        # Virtual dampers [N·s/m]
        self.dampers = {p: LinearDamper(np.zeros(3)) for p in _POINTS}

        # Target positions in the hand base frame [m]
        self.targets = {p: np.zeros(3) for p in _POINTS}

        # Local attachment point in each link's own frame [m].
        # Defaults to [0, 0, 0] (link origin). Override to place the spring
        # exactly at the fingertip or another point of interest.
        self.attachment_points = {p: np.zeros(3) for p in _POINTS}

        self.jac = HandJacobians()

    # ------------------------------------------------------------------
    # Convenience setters
    # ------------------------------------------------------------------

    def set_stiffness(self, K, points=None):
        """
        Set spring stiffness on selected points (default: all).

        Args:
            K:      scalar, 3-vector, or 3×3 matrix [N/m]
            points: list of point names, or None for all
        """
        for p in (points or _POINTS):
            self.springs[p].stiffness = np.asarray(K, dtype=float)

    def set_damping(self, B, points=None):
        """
        Set damper damping on selected points (default: all).

        Args:
            B:      scalar, 3-vector, or 3×3 matrix [N·s/m]
            points: list of point names, or None for all
        """
        for p in (points or _POINTS):
            self.dampers[p].damping = np.asarray(B, dtype=float)

    # ------------------------------------------------------------------
    # Position / velocity helpers
    # ------------------------------------------------------------------

    def _thumb(self, q_motor, q_dot_motor):
        r   = self.attachment_points['thumb']
        J   = np.array(self.jac.get_thumb_jacobian('IP', q_motor, r))
        pos = FK_motor2thumbPos(q_motor, 'IP', r)
        return pos, J @ q_dot_motor, J

    def _finger(self, finger, q_motor, q_dot_motor):
        r   = self.attachment_points[finger]
        J   = np.array(self.jac.get_finger_jacobian(finger, 'DIP', q_motor, r))
        pos = FK_motor2fingerPos(q_motor, finger, 'DIP', r)
        return pos, J @ q_dot_motor, J

    def _palm(self, q_motor, q_dot_motor):
        r      = self.attachment_points['palm']
        J      = np.array(self.jac.get_wrist_palm_jacobian(q_motor, r))  # always zero
        _, pos = FK_motor2palm(r)
        return pos, J @ q_dot_motor, J

    # ------------------------------------------------------------------
    # Torque computation
    # ------------------------------------------------------------------

    def hand_torques(self, q_motor, q_dot_motor):
        """
        Compute motor torques from task-space virtual springs and dampers.

        Args:
            q_motor:     [15] motor angles    (rad)
            q_dot_motor: [15] motor velocities (rad/s)

        Returns:
            tau: [15] motor torques (N·m)
        """
        tau = np.zeros(15)

        # Thumb fingertip (IP link)
        pos, vel, J = self._thumb(q_motor, q_dot_motor)
        tau += J.T @ (self.springs['thumb'].compute_force(pos, self.targets['thumb'])
                    + self.dampers['thumb'].compute_force(vel))

        # Four-finger fingertips (DIP link)
        for finger in _FINGERS:
            pos, vel, J = self._finger(finger, q_motor, q_dot_motor)
            tau += J.T @ (self.springs[finger].compute_force(pos, self.targets[finger])
                        + self.dampers[finger].compute_force(vel))

        # Palm center
        pos, vel, J = self._palm(q_motor, q_dot_motor)
        tau += J.T @ (self.springs['palm'].compute_force(pos, self.targets['palm'])
                    + self.dampers['palm'].compute_force(vel))

        return tau
