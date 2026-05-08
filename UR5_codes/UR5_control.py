import rtde_control
import rtde_receive
from UR5_config import (UR5_POSE, UR5_IP,
                         UR5_INIT_SPEED, UR5_INIT_ACCELERATION)

## Solve connections issues:
# sudo ip addr flush dev eno1
# sudo ip addr add 192.168.1.11/24 dev eno1
# sudo ip link set eno1 up
# OR: sudo ip addr flush dev eno1 && sudo ip addr add 192.168.1.11/24 dev eno1 && sudo ip link set eno1 up

class URController():

    def __init__(self):
        super().__init__()

        self.init_pose = UR5_POSE

        # UR5 interfaces
        self.ctrl = rtde_control.RTDEControlInterface(UR5_IP)
        self.recv = rtde_receive.RTDEReceiveInterface(UR5_IP)

        self.ctrl.setTcp([0,0,0,0,0,0])
        self.ctrl.endTeachMode()

    def move_arm(self):
        current_pose = self.recv.getActualTCPPose()
        print("Current TCP Pose:", current_pose)
        print("Target TCP Pose:", self.init_pose)

        # Move to the target init_pose
        self.ctrl.moveL(self.init_pose.tolist(), speed=UR5_INIT_SPEED, acceleration=UR5_INIT_ACCELERATION)
        print("UR5 movement complete!")


if __name__ == "__main__":

    ur_controller = URController()
    ur_controller.move_arm()
