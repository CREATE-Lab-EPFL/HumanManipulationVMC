# FusionPlotting

Drive Autodesk Fusion 360 joints from an external Python script via a shared JSON bridge file. Built for the finger and ADAPT Hand CAD models under **CREATE Lab > Bioinspired Robotic Hands > Lorenzo Vignoli** in Fusion — used to generate pose figures from the CAD models.

## Layout

```
FusionPlotting/
├── FusionAddin/FusionJointBridge/   # copy into Fusion's AddIns directory
├── Client/                          # Python-side scripts — see Client/README.md
├── Shared/commands.json             # bridge-file template
└── requirements.txt
```

Both sides resolve to the same file with no configuration: `~/FusionBridge/commands.json` (`C:\Users\<you>\FusionBridge\commands.json` on Windows). `Shared/commands.json` is only a template — the client creates the real file on first use.

## Setup (Windows)

1. Copy `FusionAddin/FusionJointBridge/` to `%AppData%\Autodesk\Autodesk Fusion 360\API\AddIns\`.
2. In Fusion: **Tools → Add-Ins → My Add-Ins → FusionJointBridge → Run** (tick **Run on Startup** to persist).
3. Find joint names in the Fusion browser (**Joints** tree, hover for tooltip) and set them in `Client/ExampleFinger.py` / `ExampleHand.py`.

## Running

No install needed — the client is pure standard library.

```powershell
python FusionPlotting/Client/PosesFinger.py   # hardcoded figure poses, finger
python FusionPlotting/Client/PosesHand.py     # hardcoded figure poses, hand
```

Or from your own script:

```python
from FusionPlotting.Client.JointClient import JointClient

client = JointClient()
client.write_targets({"MCP": 45.0, "PIP": 30.0}, angle_unit="degrees")
print(client.read_current())
```

See [Client/README.md](Client/README.md) for the full client reference.

## Units & bridge file

The client declares units (`"degrees"`/`"radians"`, `"mm"`/`"cm"`) in the `"units"` block; the add-in converts to Fusion's internal units (radians, cm). All conversion logic lives in `FusionAddin/FusionJointBridge/Units.py`.

```jsonc
{
  "units":   {"angle": "degrees", "length": "mm"},
  "targets": {"MCP": 30.0, "PIP": 20.0},    // client writes; add-in applies every POLL_INTERVAL (default 0.2 s)
  "current": {"MCP": 29.8, "PIP": 19.6}     // add-in writes back after applying
}
```

## Troubleshooting

- **Joint not moving** — name must exactly match the Fusion browser (case-sensitive); check `~/FusionBridge/addin.log`.
- **No values read back** — the add-in must be running and the target design must be the active document.
- **"Unsupported joint type"** — only revolute, slider, and cylindrical joints are handled.
