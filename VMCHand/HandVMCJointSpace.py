## VMC Hand Controller - Joint Space
## ==================================
## Applies joint-space springs and dampers across all 22 DOFs.
## Stiffness and damping are configurable per joint via self.stiffness / self.damping.
## Use set_stiffness(K) / set_damping(B) to assign uniform values across all joints.

import numpy as np
from KinematicsHand.FK_Hand import FK_motor2wrist, FK_motor2thumb, FK_motor2finger, FK_motor2spread
from KinematicsHand.JacobiansHand import HandJacobians

_SPREAD_FINGERS = ['index', 'middle', 'ring', 'pinky']


class VMC:
    """
    Joint-space spring-damper controller for all 22 hand DOFs.

    Stiffness and damping are stored as numpy arrays inside self.stiffness and
    self.damping (keyed by joint group) so each joint can be tuned independently.
    Call set_stiffness(K) / set_damping(B) to assign the same value everywhere.
    """

    def __init__(self):

        # Per-joint stiffness [N·m/rad]  — 22 DOF total:
        #   wrist(2) + thumb(4) + spread×4(4) + index(3) + middle(3) + ring(3) + pinky(3)
        self.stiffness = {
            'wrist':         np.full(2, 0.4),   # [pitch, yaw]
            'thumb':         np.full(4, 0.4),   # [CMC1, CMC2, MCP, IP]
            'spread_index':  np.array([0.4]),
            'spread_middle': np.array([0.4]),
            'spread_ring':   np.array([0.4]),
            'spread_pinky':  np.array([0.4]),
            'index':         np.full(3, 0.4),   # [MCP, PIP, DIP]
            'middle':        np.full(3, 0.4),
            'ring':          np.full(3, 0.4),
            'pinky':         np.full(3, 0.4),
        }

        # Per-joint damping [N·m·s/rad]
        self.damping = {
            'wrist':         np.full(2, 0.05),
            'thumb':         np.full(4, 0.05),
            'spread_index':  np.array([0.05]),
            'spread_middle': np.array([0.05]),
            'spread_ring':   np.array([0.05]),
            'spread_pinky':  np.array([0.05]),
            'index':         np.full(3, 0.05),
            'middle':        np.full(3, 0.05),
            'ring':          np.full(3, 0.05),
            'pinky':         np.full(3, 0.05),
        }

        # Reference positions
        self.wrist = np.array(np.deg2rad([-20.0, 0.0]))
        self.thumb = np.deg2rad([60.0, 0.0, 60.0, 60.0])
        self.spread = {
            'index':  np.array([np.deg2rad(-10.0)]),
            'middle': np.array([np.deg2rad(0.0)]),
            'ring':   np.array([np.deg2rad(10.0)]),
            'pinky':  np.array([np.deg2rad(10.0)]),
        }
        self.index_target      = np.deg2rad([40.0, 90.0, 90.0])
        self.middle_target     = np.deg2rad([40.0, 90.0, 90.0])
        self.ring_pinky_target = np.deg2rad([40.0, 90.0, 90.0])

        self.jac = HandJacobians()

    # ------------------------------------------------------------------
    # Convenience setters
    # ------------------------------------------------------------------

    def set_stiffness(self, K):
        """Assign uniform stiffness K [N·m/rad] to every joint."""
        for arr in self.stiffness.values():
            arr[:] = K

    def set_damping(self, B):
        """Assign uniform damping B [N·m·s/rad] to every joint."""
        for arr in self.damping.values():
            arr[:] = B

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _joint_torque(self, theta, theta_target, J, q_dot_motor, stiffness, damping):
        """
        Motor torques for one joint group.

        tau = J^T @ ( K*(theta_target - theta) - B*(J @ q_dot_motor) )
        """
        tau_spring = stiffness * (theta_target - theta)
        tau_damper = damping   * (J @ q_dot_motor)
        return J.T @ (tau_spring - tau_damper)

    def hand_torques(self, q_motor, q_dot_motor):
        """
        Compute motor torques using per-joint spring-dampers.

        Args:
            q_motor:     [15] motor angles    (rad)
            q_dot_motor: [15] motor velocities (rad/s)

        Returns:
            tau: [15] motor torques (N·m)
        """
        tau = np.zeros(15)

        # Wrist
        J = np.array(self.jac.get_wrist_motor_jacobian(q_motor))
        tau += self._joint_torque(FK_motor2wrist(q_motor), self.wrist, J, q_dot_motor,
                                  self.stiffness['wrist'], self.damping['wrist'])

        # Thumb
        J = np.array(self.jac.get_angles_jacobian('thumb', q_motor))
        tau += self._joint_torque(FK_motor2thumb(q_motor), self.thumb, J, q_dot_motor,
                                  self.stiffness['thumb'], self.damping['thumb'])

        # Spread (one DOF per finger, all from same motor)
        for finger in _SPREAD_FINGERS:
            J = np.array(self.jac.get_spread_jacobian(finger, q_motor))
            theta = np.array([FK_motor2spread(q_motor, finger)])
            tau += self._joint_torque(theta, self.spread[finger], J, q_dot_motor,
                                      self.stiffness[f'spread_{finger}'],
                                      self.damping[f'spread_{finger}'])

        # Index finger
        J = np.array(self.jac.get_angles_jacobian('index', q_motor))
        tau += self._joint_torque(FK_motor2finger(q_motor, 'index'), self.index_target, J, q_dot_motor,
                                  self.stiffness['index'], self.damping['index'])

        # Middle finger
        J = np.array(self.jac.get_angles_jacobian('middle', q_motor))
        tau += self._joint_torque(FK_motor2finger(q_motor, 'middle'), self.middle_target, J, q_dot_motor,
                                  self.stiffness['middle'], self.damping['middle'])

        # Ring and pinky (shared reference)
        for finger in ['ring', 'pinky']:
            J = np.array(self.jac.get_angles_jacobian(finger, q_motor))
            tau += self._joint_torque(FK_motor2finger(q_motor, finger), self.ring_pinky_target, J, q_dot_motor,
                                      self.stiffness[finger], self.damping[finger])

        return tau
