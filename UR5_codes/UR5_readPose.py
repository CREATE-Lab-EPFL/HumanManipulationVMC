import numpy as np
import rtde_receive
from scipy.spatial.transform import Rotation

try:
    from UR5_codes.UR5_config import UR5_IP
except ImportError:
    from UR5_config import UR5_IP


class UR5Receiver:
    """
    Persistent RTDE receiver for continuous UR5 pose queries.
    Opens the connection once; call get_tcp_rotation_matrix() at any rate.

    >>> recv = UR5Receiver()
    >>> R_world2hand = recv.get_tcp_rotation_matrix().T   # in control loop
    >>> recv.disconnect()
    """

    def __init__(self, ip=UR5_IP):
        self._recv = rtde_receive.RTDEReceiveInterface(ip)

    def get_tcp_pose(self):
        """Return current TCP pose as [x, y, z, rx, ry, rz] (m, rad axis-angle)."""
        return np.array(self._recv.getActualTCPPose())

    def get_tcp_rotation_matrix(self):
        """Return 3x3 rotation matrix of TCP frame in world frame (maps TCP→world)."""
        return Rotation.from_rotvec(self.get_tcp_pose()[3:]).as_matrix()

    def disconnect(self):
        self._recv.disconnect()


# One-shot helpers (open and close a connection each call — use only at startup)
def get_tcp_pose(ip=UR5_IP):
    """Return current TCP pose as [x, y, z, rx, ry, rz] (m, rad axis-angle)."""
    return np.array(rtde_receive.RTDEReceiveInterface(ip).getActualTCPPose())


def get_tcp_rotation_matrix(ip=UR5_IP):
    """Return 3x3 rotation matrix of TCP frame in world frame (maps TCP→world)."""
    return Rotation.from_rotvec(get_tcp_pose(ip)[3:]).as_matrix()


if __name__ == "__main__":
    pose = get_tcp_pose()
    R    = get_tcp_rotation_matrix()
    print("TCP pose [x,y,z,rx,ry,rz]:", np.round(pose, 4))
    print("Rotation matrix (world→TCP):\n", np.round(R, 4))
