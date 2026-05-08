# VMC_utils — Virtual Model Primitives

Shared building blocks for all VMC controllers. Every experiment imports from here.

## Files

### `VirtualModels.py`

Implements the primitive virtual elements used to compose VMC controllers:

| Class | Description |
|-------|-------------|
| `LinearSpring` | Hooke's law: F = K · (target − current) |
| `TanhSpring` | Saturating spring with component-wise force limit |
| `SigmoidSpring` | Non-linear spring with sigmoid-varying stiffness |
| `PolynomialSpring` | Non-linear spring with polynomial-varying stiffness |
| `GaussianSpring` | Gaussian repulsive field |
| `DeadzoneLimitSpring` | Passive joint-limit spring with deadzone (no force inside limits) |
| `ConstrainedLinearSpring` | Linear spring projected onto specified free direction(s) |
| `ConstrainedTanhSpring` | Saturating spring projected onto specified free direction(s) |
| `ConstrainedGaussianSpring` | Gaussian repulsive spring projected onto specified free direction(s) |
| `LinearDamper` | Viscous damping: F = B · velocity |
| `TanhDamper` | Saturating damper with component-wise force limit |
| `ConstrainedLinearDamper` | Linear damper projected onto specified free direction(s) |
| `ConstrainedTanhDamper` | Saturating damper projected onto specified free direction(s) |

### `GravityCompensation.py`

Gravity torque computation for arbitrary system orientations. Transforms the
gravity vector from world frame to system frame using a rotation matrix, then
computes joint torques via the Jacobian transpose.
