# FusionPlotting / Client

Python-side scripts that write joint targets to the bridge file read by the
Fusion add-in.

---

## Files

| File | Purpose |
|------|---------|
| `JointClient.py` | Core client class — reads/writes `~/FusionBridge/commands.json`. |
| `_fusion_utils.py` | FK helpers: converts 13-motor vectors to Fusion joint-angle dicts. |
| `PosesFinger.py` | **Hardcoded figure poses for the single-finger model.** Edit here. |
| `PosesHand.py` | **Hardcoded figure poses for the full hand model.** Edit here. |
| `ExampleFinger.py` | Minimal usage example (finger). |
| `ExampleHand.py` | Minimal usage example (hand). |

---

## Sending a pose

Run the script — it shows a numbered menu and you pick the pose interactively:

```powershell
python PosesFinger.py
# or
python PosesHand.py
```

```
Available poses:
  [0] flat
  [1] bent
  [2] force_gradient
Select pose number: 1

Sending pose 'bent':
  MCP: 10.0 deg
  ...
```

The script prints the joint values, sends them to Fusion, and reads back the
values Fusion confirms after applying them.

> **Important:** Fusion 360 must be open and the correct model file (finger or
> hand) must be the **active document** when the script runs.  If any other
> file is in the foreground the add-in will not find the joints and nothing
> will move.

---

## Adding a new figure

Add an entry to the `FIGURES` dict in the relevant file:

```python
# PosesHand.py
FIGURES = {
    "my_new_pose": {
        "wrist_pitch":  0.0,
        "thumb_CMC1":  30.0,
        # ... all joints
    },
}
```

Then set `POSE = "my_new_pose"` and run.

---

## Joint names

### Finger model (`PosesFinger.py`)

| Name | Joint |
|------|-------|
| `MCP` | Metacarpophalangeal |
| `PIP` | Proximal interphalangeal |
| `DIP` | Distal interphalangeal |

Positive = flexion, degrees.

### Hand model (`PosesHand.py`)

| Name | Sign convention |
|------|----------------|
| `wrist_pitch`, `wrist_yaw` | Always 0.0 (position-controlled) |
| `thumb_CMC1`, `thumb_CMC2`, `thumb_MCP`, `thumb_IP` | + = flexion |
| `index_spread` | − = adduction toward middle |
| `ring_spread`, `pinky_spread` | + = adduction toward middle |
| `index_MCP`, `index_PIP` | + = flexion |
| `middle_MCP`, `middle_PIP` | + = flexion |
| `ring_MCP`, `ring_PIP` | + = flexion |
| `pinky_MCP`, `pinky_PIP` | + = flexion |

All angles in degrees.  The add-in (`FusionConventions.py`) handles any sign
flips before forwarding to Fusion — do not manually negate values here.
