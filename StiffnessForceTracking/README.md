# StiffnessForceTracking — Experimental Area

Model-based closed-loop control over the full force-displacement space.
Independent optimization pathways provide control over both the operating
point (force) and the response slope (stiffness):

- **Force+position control** (`force_position_control`): gradient descent on K_d,
  keeping d_ref fixed. Adapting K changes the slope of the force-displacement
  curve, moving both the force level and the stiffness simultaneously.
- **Force+stiffness control** (`force_stiffness_control`): gradient descent on d_ref,
  keeping K_d fixed. Adapting d_ref moves the operating point while K_d (stiffness)
  is a free independent parameter, so force and stiffness are decoupled.

Both controllers compute gradients from the stiffness model and use measured
tip force from the load cell as the tracking signal.

Controller structure:
- A stiffness model maps joint-space stiffness and reference displacement to
    predicted tip force.
- The gradient of the force error drives updates to either K_d or d_ref.
- Scalar baselines use the same update structure but without the model terms.

Each method is paired with a scalar (model-free) baseline using the same learning
rate structure for direct comparison. All timing and learning-rate parameters are
defined in `experiment_config.py`.

## Files

### Closed-loop (load-cell feedback)

| File | Platform | Description |
|------|----------|-------------|
| `experiment_config.py` | — | Shared config: force levels, timing, learning rates |
| `FORCE_CONTROL.py` | — | Batch runner: executes the controller variants sequentially |
| `force_position_control.py` | finger | Force+position control via K gradient descent (model-based) |
| `force_position_control_scalar.py` | finger | Scalar k·I baseline for force+position control |
| `force_stiffness_control.py` | finger | Force+stiffness control via d_ref gradient descent (model-based) |
| `force_stiffness_control_scalar.py` | finger | Scalar θ_ref baseline for force+stiffness control |
| `plot_force_position_control.ipynb` | — | Plot force position control experiment |
| `plot_force_stiffness_control.ipynb` | — | Plot force stiffness control experiment |

### Open-loop (model-predicted force feedback)

Both variants use the same gradient descent as their closed-loop counterparts, but
the feedback signal driving the gradient is the model-predicted tip force
(`tip_force(q, q_ref, K)`) rather than the load cell. The load cell is still
connected and its readings are logged, but they play no role in control.

These variants isolate model-driven behavior by removing measurement feedback
from the update step while keeping identical logging and timing.

| File | Platform | Description |
|------|----------|-------------|
| `force_position_control_openloop.py` | finger | K-descent driven by model-predicted force |
| `force_stiffness_control_openloop.py` | finger | d_ref-descent driven by model-predicted force |
| `plot_openloop_tracking.ipynb` | — | Plot openloop tracking experiment |

