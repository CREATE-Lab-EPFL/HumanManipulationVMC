from pynput import keyboard
import rtde_control
import rtde_receive
import numpy as np

## Password: createlab

class URTipControllerNode():

    def __init__(self):
        super().__init__()

        arm_IP = "192.168.1.10"
        self.ctrl = rtde_control.RTDEControlInterface(arm_IP)
        self.recv = rtde_receive.RTDEReceiveInterface(arm_IP)

        self.ctrl.setTcp([0,0,0,0,0,0])
        self.ctrl.endTeachMode()

        self.running = True

    def move_arm_keyboard(self):
        raw_pose = np.array(self.recv.getActualTCPPose())
        step = 0.001


        print("\nKeyboard control active:")
        print("W/S: +X / -X")
        print("A/D: +Y / -Y")
        print("Q/E: +Z / -Z")
        print("ESC: quit\n")

        def on_press(key):
            nonlocal raw_pose
            try:
                if key.char == "w":
                    raw_pose[0] += step
                elif key.char == "s":
                    raw_pose[0] -= step
                elif key.char == "a":
                    raw_pose[1] += step
                elif key.char == "d":
                    raw_pose[1] -= step
                elif key.char == "q":
                    raw_pose[2] += step
                elif key.char == "e":
                    raw_pose[2] -= step
                self.ctrl.moveL(raw_pose, speed=0.05, acceleration=0.1)
            except AttributeError:
                if key == keyboard.Key.esc:
                    self.running = False
                    return False  # stop listener

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


if __name__ == "__main__":
    ur_tip_controller = URTipControllerNode()
    ur_tip_controller.move_arm_keyboard()