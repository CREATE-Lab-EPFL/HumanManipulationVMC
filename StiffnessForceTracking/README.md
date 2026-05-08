# StiffnessForceTracking — Experimental Area 4

Model-based closed-loop control over the full force-displacement space.
Two independent optimization pathways provide control over both the operating
point (force) and the response slope (stiffness):

- **Force+position control** (`force_position_control`): gradient descent on K_d,
  keeping d_ref fixed. Adapting K changes the slope of the F-d curve, moving both
  the force level and the stiffness simultaneously.
- **Force+stiffness control** (`force_stiffness_control`): gradient descent on d_ref,
  keeping K_d fixed. Adapting d_ref moves the operating point while K_d (stiffness)
  is a free independent parameter — force and stiffness are decoupled.

Each method is paired with a scalar (model-free) baseline using the same learning
rate structure for direct comparison. All timing and learning-rate parameters are
defined in `experiment_config.py`.

## Files

### Closed-loop (load-cell feedback)

| File | Platform | Description |
|------|----------|-------------|
| `experiment_config.py` | — | Shared config: force levels, timing, learning rates |
| `FORCE_CONTROL.py` | — | Batch runner: executes all four controllers sequentially |
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

| File | Platform | Description |
|------|----------|-------------|
| `force_position_control_openloop.py` | finger | K-descent driven by model-predicted force |
| `force_stiffness_control_openloop.py` | finger | d_ref-descent driven by model-predicted force |
| `plot_openloop_tracking.ipynb` | — | Plot openloop tracking experiment |

