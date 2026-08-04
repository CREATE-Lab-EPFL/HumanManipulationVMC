# ModelIDHand — ADAPT Hand System Identification

`hand_params.py` is the **single source of truth** — all hand modules import from here.

## Motor layout (software order)

The 13 torque-controlled motors are indexed 0–12. Wrist motors (hardware IDs 13, 14) run in position mode and are excluded from this table.

| idx | motor |
|-----|-------|
| 0–3 | thumb CMC1, CMC2, MCP, IP |
| 4   | spread (drives all four finger spread joints) |
| 5–6 | index MCP, PIP |
| 7–8 | middle MCP, PIP |
| 9–10 | ring MCP, PIP |
| 11–12 | pinky MCP, PIP |

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
python -m ModelIDHand.hand_viz   # built-in demos
```

```python
from ModelIDHand.hand_viz import plot_hand, HandVisualizer, random_trajectory, MOTOR_LIMITS, STYLES

plot_hand(q_motor, style="thick")                  # static pose, named style preset
HandVisualizer(style="dotted").run(random_trajectory(n_waypoints=20, duration=10.0))
HandVisualizer().run(lambda t: latest_q)            # live source, e.g. a ROS callback
```

`style` accepts a preset name (below) or a dict of matplotlib line kwargs. Structural elements (wrist, spokes, palm outline) always use a fixed gray style. `MOTOR_LIMITS` is a `(13, 2)` array of `[min, max]` motor bounds derived from `JOINT_LIMITS`.

| Style | Description |
|-------|-------------|
| `"default"` | solid line, round joints |
| `"thick"` | heavier line, larger markers |
| `"dotted"` / `"dashed"` | dotted / dashed line |
| `"segmented"` | dash-dot line, square markers |
| `"minimal"` | thin solid line, no markers |
