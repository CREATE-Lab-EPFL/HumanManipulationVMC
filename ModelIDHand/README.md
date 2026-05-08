# ModelIDHand — ADAPT Hand System Identification

`hand_params.py` is the **single source of truth** — all hand modules import from here.

## Motor layout (software order)

| idx | motor | idx | motor |
|-----|-------|-----|-------|
| 0–1 | wrist (differential) | 6 | spread (all fingers) |
| 2–5 | thumb CMC1, CMC2, MCP, IP | 7–14 | index/middle/ring/pinky MCP+PIP |

## Files

| File | Purpose |
|------|---------|
| `hand_params.py` | Transmissions, URDF kinematics, masses, COGs, joint limits |
| `motor_config.py` | Hardware↔software motor ordering and named slices (wrist, thumb, spread, fingers) |
| `hand_viz.py` | Real-time 3D visualizer — see below |
| `hand_node.py` | ROS2 node for sysid experiments |
| `hand_fitting.ipynb` | Fits transmission models for all hand joints from joint data; saves to hand_params.py |

## `hand_viz.py` — 3D visualizer

```bash
python -m ModelIDHand.hand_viz   # runs built-in demos
```

```python
from ModelIDHand.hand_viz import plot_hand, HandVisualizer, random_trajectory, MOTOR_LIMITS, STYLES

# Static single pose
plot_hand(q_motor)
plot_hand(q_motor, style="thick")           # named preset
plot_hand(q_motor, style={"linestyle": ":", "linewidth": 3, "marker": "D"})  # custom dict

# Random trajectory
viz = HandVisualizer(style="dotted")
viz.run(random_trajectory(n_waypoints=20, duration=10.0), duration=10.0)

# Live source — any callable(t_elapsed) → q_motor(15,)
viz.run(lambda t: latest_q)               # ROS callback, runs until window closed
viz.run(lambda t: data[int(t * fps)])     # pre-recorded array

# Manual loop
while running:
    viz.update(get_motor_angles())
```

### Finger styles

The `style` parameter applies to all finger/thumb chains. Structural elements (wrist, spokes, palm outline) always use the fixed gray style.

| Name | Description |
|------|-------------|
| `"default"` | solid line, round joints |
| `"thick"` | heavier line and larger markers |
| `"dotted"` | dotted line |
| `"dashed"` | dashed line |
| `"segmented"` | dash-dot line, square markers |
| `"minimal"` | thin solid line, no markers |

A plain dict of matplotlib line kwargs can be passed instead of a name (any key accepted by `ax.plot`).

`MOTOR_LIMITS` is a `(15, 2)` array of `[min, max]` motor bounds derived from `JOINT_LIMITS`.
