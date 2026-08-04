# StiffnessForceTracking — Experimental Area

Model-based closed-loop control over the full force-displacement space, via two independent gradient-descent pathways on the stiffness model (joint-space stiffness + reference displacement → predicted tip force):

- **Force+position control** (`force_position_control`) — gradient descent on `K_d` with `d_ref` fixed. Moves both force level and stiffness together by changing the slope of the force-displacement curve.
- **Force+stiffness control** (`force_stiffness_control`) — gradient descent on `d_ref` with `K_d` fixed. Moves the operating point while stiffness stays an independent, freely-set parameter.

Both use measured tip force from the load cell as the tracking signal, and are each paired with a scalar (model-free) baseline using the same learning-rate structure for direct comparison. All timing/learning-rate parameters live in `experiment_config.py`.

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

Same gradient descent as the closed-loop counterparts, but driven by the model-predicted tip force (`tip_force(q, q_ref, K)`) instead of the load cell — the load cell stays connected and logged, but plays no role in control. Isolates model-driven behavior from measurement feedback while keeping identical logging and timing.

| File | Platform | Description |
|------|----------|-------------|
| `force_position_control_openloop.py` | finger | K-descent driven by model-predicted force |
| `force_stiffness_control_openloop.py` | finger | d_ref-descent driven by model-predicted force |
| `plot_openloop_tracking.ipynb` | — | Plot openloop tracking experiment |
