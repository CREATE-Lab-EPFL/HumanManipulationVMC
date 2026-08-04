# FusionPlotting / Client

Python-side scripts that write joint targets to the bridge file read by the Fusion add-in.

## Files

| File | Purpose |
|------|---------|
| `JointClient.py` | Core client class — reads/writes `~/FusionBridge/commands.json` |
| `_fusion_utils.py` | FK helpers: converts 13-motor vectors to Fusion joint-angle dicts |
| `PosesFinger.py` | Hardcoded figure poses for the single-finger model — edit here to add poses |
| `PosesHand.py` | Hardcoded figure poses for the full hand model — edit here to add poses |
| `ExampleFinger.py` | Minimal usage example (finger) |
| `ExampleHand.py` | Minimal usage example (hand) |

## Sending a pose

```powershell
python PosesFinger.py   # or PosesHand.py
```

Prints a numbered menu of the poses in `FIGURES`; pick one, and the script sends it to Fusion and reads back the confirmed joint values.

> Fusion 360 must be open with the correct model (finger or hand) as the **active document** — otherwise the add-in can't find the joints.

## Adding a new figure

Add an entry to `FIGURES` in the relevant file, then set `POSE = "my_new_pose"` and run:

```python
FIGURES = {
    "my_new_pose": {"wrist_pitch": 0.0, "thumb_CMC1": 30.0, ...},  # all joints
}
```

## Joint names

**Finger** (`PosesFinger.py`): `MCP`, `PIP`, `DIP` — positive = flexion, degrees.

**Hand** (`PosesHand.py`), all in degrees:

| Name | Sign convention |
|------|----------------|
| `wrist_pitch`, `wrist_yaw` | Always 0.0 (position-controlled) |
| `thumb_CMC1/CMC2/MCP/IP` | + = flexion |
| `index_spread` | − = adduction toward middle |
| `ring_spread`, `pinky_spread` | + = adduction toward middle |
| `<finger>_MCP/PIP` | + = flexion |

Sign flips are handled by the add-in (`FusionConventions.py`) — do not negate values manually.
