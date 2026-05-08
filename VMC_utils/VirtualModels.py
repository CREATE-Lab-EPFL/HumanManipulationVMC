import numpy as np

# Units of measure:
# - POSITIONS: [m]
# - FORCES: [N]
# - VELOCITIES: [m/s]
# - DAMPING: [N·s/m]
# - STIFFNESS: [N/m]
# Speed: sudo echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# Note:
# - Linear springs and dampers can produce unbounded forces.
# - Tanh springs and dampers saturate at specified max force per component.
# - Gaussian springs produce repulsive forces that decay with distance.
# - Constrained versions simulate carts, applying forces only along free directions.
# - Deadzone limit springs apply forces only when joint limits are exceeded.

# ============================================================================
# SPRINGS (position-based forces)
# ============================================================================


class LinearSpring:
    """Linear spring force: F = K * (target - current)"""

    def __init__(self, stiffness):
        """
        Args:
            stiffness: K spring stiffness - scalar, vector, or matrix
            dim: dimension of the space
        """
        self.stiffness = stiffness

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space
        
        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:      # scalar → diagonal matrix
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:    # vector → diagonal matrix
            return np.diag(gain)
        else:                   # already a matrix
            return gain
        
    def compute_force(self, pos_current, pos_target):
        """
        Args:
            pos_current: current position
            pos_target: target position

        Returns:
            F: force vector
        """
        error = np.asarray(pos_target) - np.asarray(pos_current)
        return self._to_matrix(self.stiffness, len(error)) @ error
    

class TanhSpring:
    """Tanh spring with component-wise saturation"""

    def __init__(self, stiffness, max_force):
        """
        Args:
            stiffness: K spring stiffness at origin - scalar, vector, or matrix
            max_force: F_max maximum force magnitude per component (N)
        """
        self.stiffness = stiffness
        self.max_force = max_force

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space
        
        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:
            return np.diag(gain)
        else:
            return gain

    def compute_force(self, pos_current, pos_target):
        """
        Args:
            pos_current: current position
            pos_target: target position

        Returns:
            F: force vector
        """
        error = np.asarray(pos_target) - np.asarray(pos_current)
        return self.max_force * np.tanh(self._to_matrix(self.stiffness, len(error)) @ error / self.max_force)


class ConstrainedLinearSpring:
    """
    Directional linear spring acting only along specified free direction(s).

    Implements a cart constraint where the spring force is projected along n,
    the direction orthogonal to the cart plane: F = P @ K @ P @ e,
    where P = n @ n.T.
    """

    def __init__(self, stiffness, n=np.array([[0], [0], [1]])):
        """
        Args:
            stiffness: K spring stiffness - scalar, vector, or matrix
            n: vector orthogonal to the cart plane (must be normalized).
               E.g. n = [1, 0, 0] means spring force acts along x only.
        """
        self.stiffness = stiffness
        n = np.asarray(n).reshape(-1, 1)
        n = n / np.linalg.norm(n)
        self.P = n @ n.T

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space

        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:
            return np.diag(gain)
        else:
            return gain

    def compute_force(self, pos_current, pos_target):
        """
        Compute spring force projected into the free subspace.

        F = P @ K @ P @ e

        Args:
            pos_current: current position
            pos_target: target position

        Returns:
            F: force vector (constrained directions are zeroed)
        """
        error = np.asarray(pos_target).flatten() - np.asarray(pos_current).flatten()
        K = self._to_matrix(self.stiffness, len(error))
        return self.P @ K @ self.P @ error


class ConstrainedTanhSpring:
    """
    Directional tanh spring acting only along specified free direction(s).

    Implements a cart constraint where the spring force is projected along n,
    the direction orthogonal to the cart plane:
    F = P @ F_max * tanh(P @ K @ P @ e / F_max), where P = n @ n.T.
    """

    def __init__(self, stiffness, max_force, n=np.array([[0], [0], [1]])):
        """
        Args:
            stiffness: K spring stiffness - scalar, vector, or matrix (N/m)
            max_force: F_max maximum force magnitude per component (N)
            n: vector orthogonal to the cart plane (must be normalized).
               E.g. n = [1, 0, 0] means spring force acts along x only.
        """
        self.stiffness = stiffness
        self.max_force = max_force
        n = np.asarray(n).reshape(-1, 1)
        n = n / np.linalg.norm(n)
        self.P = n @ n.T

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space
        
        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:
            return np.diag(gain)
        else:
            return gain

    def compute_force(self, pos_current, pos_target):
        """
        Compute tanh spring force projected into the free subspace.

        F = P @ F_max * tanh(P @ K @ P @ e / F_max)

        Args:
            pos_current: current position (m)
            pos_target: target position (m)

        Returns:
            F: force vector (N), constrained directions are zeroed
        """
        error = np.asarray(pos_target) - np.asarray(pos_current)
        K = self._to_matrix(self.stiffness, len(error))
        return self.P @ (self.max_force * np.tanh(self.P @ K @ self.P @ error / self.max_force))
    

class GaussianSpring:
    """Gaussian repulsive spring for obstacle avoidance"""

    def __init__(self, strength, sigma):
        """
        Args:
            strength: A force amplitude (N)
            sigma: spread of Gaussian (m)
        """
        self.strength = strength
        self.sigma = sigma

    def compute_force(self, pos_current, pos_obstacle):
        """
        Args:
            pos_current: [x, y, z] current position (m)
            pos_obstacle: [x, y, z] obstacle position (m)

        Returns:
            F: [Fx, Fy, Fz] repulsive force vector (N)
        """
        delta = pos_current - pos_obstacle
        distance_sq = np.dot(delta, delta)
        decay = np.exp(-distance_sq / (2 * self.sigma**2))
        return self.strength * decay * delta
    
    
class ConstrainedGaussianSpring:
    """
    Directional Gaussian repulsive spring acting only along specified free direction(s).

    Implements a cart constraint where the spring force is projected along n,
    the direction orthogonal to the cart plane, using P = n @ n.T.
    """

    def __init__(self, strength, sigma, n=np.array([[0], [0], [1]])):
        """
        Args:
            strength: A force amplitude (N)
            sigma: spread of Gaussian (m)
            n: vector orthogonal to the cart plane (must be normalized).
               E.g. n = [1, 0, 0] means spring force acts along x only.
        """
        self.strength = strength
        self.sigma = sigma
        n = np.asarray(n).reshape(-1, 1)
        n = n / np.linalg.norm(n)
        self.P = n @ n.T

    def compute_force(self, pos_current, pos_obstacle):
        """
        Compute Gaussian repulsive force projected along n.

        F = P @ (A * exp(-||P @ delta||^2 / (2*sigma^2)) * P @ delta)

        Args:
            pos_current: current position (m)
            pos_obstacle: obstacle position (m)

        Returns:
            F: repulsive force vector (N), force is along n direction only
        """
        delta = np.asarray(pos_current).flatten() - np.asarray(pos_obstacle).flatten()
        delta_proj = self.P @ delta
        distance_sq = np.dot(delta_proj, delta_proj)
        decay = np.exp(-distance_sq / (2 * self.sigma**2))
        return self.P @ (self.strength * decay * delta_proj)
    

class SigmoidSpring:
    """Sigmoid spring varying stiffness with distance to target."""

    def __init__(self, kmin, kmax, threshold, alpha, element_wise=True):
        """
        Args:
            kmin: minimum stiffness (below threshold)
            kmax: maximum stiffness (above threshold)
            threshold: distance at which transition occurs
            alpha: rate of change (steepness of transition)
            element_wise: if True, compute stiffness per element; if False, use norm of error
        """
        self.kmin = kmin
        self.kmax = kmax
        self.threshold = threshold
        self.alpha = alpha
        self.element_wise = element_wise
        self._last_stiffness = None

    def get_stiffness(self, pos_current, pos_target):
        """
        Compute the current stiffness based on distance to target.

        Args:
            pos_current: [x, y, z] current position (m)
            pos_target: [x, y, z] target position (m)

        Returns:
            k: stiffness value(s) - scalar or array depending on element_wise
        """
        error = pos_target - pos_current
        if self.element_wise:
            distance = np.abs(error)
        else:
            distance = np.linalg.norm(error)
        k = self.kmin + (self.kmax - self.kmin) / (1 + np.exp(-self.alpha * (distance - self.threshold)))
        self._last_stiffness = k
        return k

    def compute_force(self, pos_current, pos_target):
        """
        Args:
            pos_current: [x, y, z] current position (m)
            pos_target: [x, y, z] target position (m)

        Returns:
            F: [Fx, Fy, Fz] force vector (N)
        """
        error = pos_target - pos_current
        k = self.get_stiffness(pos_current, pos_target)
        return k * error
    

class PolynomialSpring:
    """Polynomial spring varying stiffness with distance to target."""

    def __init__(self, stiffness, order, dist_norm, element_wise=True):
        """
        Args:
            stiffness: base stiffness (N/m)
            order: polynomial order
            element_wise: if True, compute stiffness per element; if False, use norm of error
        """
        self.stiffness = stiffness
        self.order = order
        self.element_wise = element_wise
        self.dist_norm = dist_norm
        self._last_stiffness = None

    def get_stiffness(self, pos_current, pos_target):
        """
        Compute the current stiffness based on distance to target.

        Args:
            pos_current: [x, y, z] current position (m)
            pos_target: [x, y, z] target position (m)

        Returns:
            k: stiffness value(s) - scalar or array depending on element_wise
        """
        error = pos_target - pos_current
        if self.element_wise:
            distance = np.abs(error)
        else:
            distance = np.linalg.norm(error)
        k = self.stiffness * (distance / self.dist_norm) ** self.order
        self._last_stiffness = k
        return k

    def compute_force(self, pos_current, pos_target):
        """
        Args:
            pos_current: [x, y, z] current position (m)
            pos_target: [x, y, z] target position (m)

        Returns:
            F: [Fx, Fy, Fz] force vector (N)
        """
        error = pos_target - pos_current
        k = self.get_stiffness(pos_current, pos_target)
        return k * error


class DeadzoneLimitSpring:
    """Deadzone spring for joint limits (ReLU-like behavior)."""
    
    def __init__(self, stiffness, angle_lower, angle_upper):
        """
        Args:
            stiffness: K spring stiffness (N·m/rad)
            angle_lower: lower joint limit (rad)
            angle_upper: upper joint limit (rad)
        """
        self.stiffness = stiffness
        self.angle_lower = angle_lower
        self.angle_upper = angle_upper
    
    def compute_force(self, angle_current):
        """
        Args:
            angle_current: current joint angle (rad)
        
        Returns:
            F: scalar torque pushing back toward valid range (N·m)
        """
        if angle_current < self.angle_lower:
            return self.stiffness * (self.angle_lower - angle_current)
        elif angle_current > self.angle_upper:
            return self.stiffness * (self.angle_upper - angle_current)
        else:
            return 0.0


# ============================================================================
# DAMPERS (velocity-based forces)
# ============================================================================

class LinearDamper:
    """Linear damper force: F = -D @ v_current"""
    
    def __init__(self, damping):
        """
        Args:
            damping: D damping coefficient - scalar, vector, or matrix (N·s/m)
        """
        self.damping = damping

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space
        
        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:
            return np.diag(gain)
        else:
            return gain
    
    def compute_force(self, velocity_current):
        """
        Args:
            velocity_current: current velocity (m/s)
        
        Returns:
            F: damping force vector (N)
        """
        velocity = np.asarray(velocity_current)
        return -self._to_matrix(self.damping, len(velocity)) @ velocity
    


class TanhDamper:
    """Tanh damper with component-wise saturation"""

    def __init__(self, damping, max_force):
        """
        Args:
            damping: D damping coefficient - scalar, vector, or matrix (N·s/m)
            max_force: F_max maximum damping force per component (N)
        """
        self.damping = damping
        self.max_force = max_force

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space
        
        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:
            return np.diag(gain)
        else:
            return gain

    def compute_force(self, velocity_current):
        """
        Args:
            velocity_current: current velocity (m/s)

        Returns:
            F: damping force vector (N)
        """
        velocity = np.asarray(velocity_current)
        return -self.max_force * np.tanh(self._to_matrix(self.damping, len(velocity)) @ velocity / self.max_force)
    

class ConstrainedLinearDamper:
    """
    Directional linear damper acting only along specified free direction(s).

    Implements a cart constraint where the damping force is projected along n,
    the direction orthogonal to the cart plane: F = -P @ D @ P @ v,
    where P = n @ n.T.
    """

    def __init__(self, damping, n=np.array([[0], [0], [1]])):
        """
        Args:
            damping: D damping coefficient - scalar, vector, or matrix (N·s/m)
            n: vector orthogonal to the cart plane (must be normalized).
               E.g. n = [1, 0, 0] means damping force acts along x only.
        """
        self.damping = damping
        n = np.asarray(n).reshape(-1, 1)
        n = n / np.linalg.norm(n)
        self.P = n @ n.T

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space

        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:
            return np.diag(gain)
        else:
            return gain

    def compute_force(self, velocity_current):
        """
        Compute damping force projected into the free subspace.

        F = -P @ D @ P @ v

        Args:
            velocity_current: current velocity (m/s)

        Returns:
            F: damping force vector (N), constrained directions are zeroed
        """
        velocity = np.asarray(velocity_current).flatten()
        D = self._to_matrix(self.damping, len(velocity))
        return -self.P @ D @ self.P @ velocity


class ConstrainedTanhDamper:
    """
    Directional tanh damper acting only along specified free direction(s).

    Implements a cart constraint where the damping force is projected along n,
    the direction orthogonal to the cart plane:
    F = -P @ F_max * tanh(P @ D @ P @ v / F_max), where P = n @ n.T.
    """

    def __init__(self, damping, max_force, n=np.array([[0], [0], [1]])):
        """
        Args:
            damping: D damping coefficient - scalar, vector, or matrix (N·s/m)
            max_force: F_max maximum damping force per component (N)
            n: vector orthogonal to the cart plane (must be normalized).
               E.g. n = [1, 0, 0] means damping force acts along x only.
        """
        self.damping = damping
        self.max_force = max_force
        n = np.asarray(n).reshape(-1, 1)
        n = n / np.linalg.norm(n)
        self.P = n @ n.T

    def _to_matrix(self, gain, dim):
        """
        Transforms gain (scalar, vector, or matrix) into a proper matrix.

        Args:
            gain: scalar, 1D array (diagonal), or 2D array (matrix)
            dim: dimension of the space

        Returns:
            2D numpy array (matrix)
        """
        gain = np.asarray(gain)
        if gain.ndim == 0:
            return float(gain) * np.eye(dim)
        elif gain.ndim == 1:
            return np.diag(gain)
        else:
            return gain

    def compute_force(self, velocity_current):
        """
        Compute tanh damping force projected into the free subspace.

        F = -P @ F_max * tanh(P @ D @ P @ v / F_max)

        Args:
            velocity_current: current velocity (m/s)

        Returns:
            F: damping force vector (N), constrained directions are zeroed
        """
        velocity = np.asarray(velocity_current).flatten()
        D = self._to_matrix(self.damping, len(velocity))
        return -self.P @ (self.max_force * np.tanh(self.P @ D @ self.P @ velocity / self.max_force))