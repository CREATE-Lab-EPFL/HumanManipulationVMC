import numpy as np

# Units of measure:
# - POSITIONS: [mm]
# - MASSES: [kg]
# - FORCES: [N]
# - GRAVITY: [9.81 m/s²]


class GravityCompensation:
    """
    Gravity compensation with arbitrary system orientation.
    
    Gravity is always -z in world frame.
    System may be oriented differently: provide transformation matrix.
    """
    
    def __init__(self, mass, R_world_to_system=np.eye(3), gravity=9.81):
        """
        Args:
            mass: M total body mass (kg)
            gravity: g gravitational acceleration (m/s²)
        """
        self.mass = mass
        self.gravity = gravity
        self.world_gravity = np.array([0.0, 0.0, -self.mass * self.gravity])  # -z direction
        self.R_world_to_system = R_world_to_system
    
    def compute_force(self):
        """
        Compute gravity compensation force in system frame.
        
        Args:
            R_world_to_system: 3x3 rotation matrix from world to system frame
                              Default is identity (system = world frame)
        
        Returns:
            F: [Fx, Fy, Fz] compensation force in system frame (N)
        """
        # Transform world gravity to system frame, then negate for compensation
        gravity_in_system = self.R_world_to_system @ self.world_gravity
        return -gravity_in_system


if __name__ == "__main__":

    # Finger frame: finger z-axis = world -z-axis (180° rotation around x)
    R = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])

    # Test
    gc = GravityCompensation(mass=0.5, R_world_to_system=R)
    
    # Default: world frame (identity matrix)
    F_world = gc.compute_force()
    print(f"World frame: {F_world} N")
    

    F_finger = gc.compute_force(R)
    print(f"Finger frame: {F_finger} N")