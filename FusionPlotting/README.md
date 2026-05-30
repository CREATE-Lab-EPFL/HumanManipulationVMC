# FusionPlotting

Drive Autodesk Fusion 360 joints from an external Python script via a shared
JSON bridge file.  Designed for the finger and ADAPT Hand designs under
**CREATE Lab > Bioinspired Robotic Hands > Lorenzo Vignoli** in Fusion.

---

## Folder layout

```
FusionPlotting/
├── FusionAddin/
│   └── FusionJointBridge/      ← copy this folder into Fusion's AddIns directory
│       ├── FusionJointBridge.py
│       ├── FusionJointBridge.manifest
│       ├── JointHandler.py
│       └── Units.py            ← all unit conversions live here
├── Client/
│   ├── JointClient.py          ← reusable client class
│   ├── ExampleFinger.py        ← drives the single-finger model
│   └── ExampleHand.py          ← drives the ADAPT Hand model
├── Shared/
│   └── commands.json           ← template; first run copies it to ~/FusionBridge/
├── requirements.txt
└── README.md
```

---

## Bridge file location

Both the add-in and the client resolve to the same path without any
configuration:

```
C:\Users\<you>\FusionBridge\commands.json
```

The `Shared/commands.json` in this repo is just the template.  `JointClient`
creates the real file on first use.

---

## Installing the add-in on Windows

1. Copy the entire `FusionAddin/FusionJointBridge/` folder to:

   ```
   C:\Users\<you>\AppData\Roaming\Autodesk\Autodesk Fusion 360\API\AddIns\
   ```

   The result should look like:

   ```
   …\AddIns\FusionJointBridge\FusionJointBridge.py
   …\AddIns\FusionJointBridge\FusionJointBridge.manifest
   …\AddIns\FusionJointBridge\JointHandler.py
   …\AddIns\FusionJointBridge\Units.py
   ```

2. In Fusion 360: **Tools → Add-Ins → My Add-Ins → FusionJointBridge → Run**.

   A message box confirms the add-in started and shows the bridge file path.

3. To start automatically with Fusion, tick **Run on Startup** in the same dialog.

---

## Finding joint names

The add-in matches joints by the name shown in the **Fusion browser** (the
left-side tree).  To find exact names:

- Expand **Joints** in the browser.
- Hover over a joint — the tooltip shows its name.
- Or right-click → **Properties**.

Update `FINGER_JOINTS` in `Client/ExampleFinger.py` and `HAND_JOINTS` in
`Client/ExampleHand.py` to match.

---

## Running the client

No installation needed — the client uses only the Python standard library.

```powershell
# From the repo root or from Client/
python FusionPlotting/Client/ExampleFinger.py
python FusionPlotting/Client/ExampleHand.py
```

Or in your own script:

```python
from FusionPlotting.Client.JointClient import JointClient

client = JointClient()
client.write_targets({"MCP": 45.0, "PIP": 30.0}, angle_unit="degrees")

import time; time.sleep(0.5)
print(client.read_current())
```

---

## Units

The `"units"` block in `commands.json` declares what the client is writing.
The add-in reads this block and converts to Fusion's internal units before
applying.

| Field    | Allowed values     | Fusion internal |
|----------|--------------------|-----------------|
| `angle`  | `"degrees"`, `"radians"` | radians   |
| `length` | `"mm"`, `"cm"`     | cm              |

All conversions happen in `FusionAddin/FusionJointBridge/Units.py` and
nowhere else.

---

## Bridge file schema

```jsonc
{
  "units": {
    "angle":  "degrees",   // declared by the client; add-in converts from this
    "length": "mm"
  },
  "targets": {             // client writes — add-in applies every 200 ms
    "MCP": 30.0,
    "PIP": 20.0
  },
  "current": {             // add-in writes after applying — client reads back
    "MCP": 29.8,
    "PIP": 19.6
  },
  "last_updated": "2026-05-30T12:00:00+00:00"
}
```

---

## Tuning the poll rate

Edit `POLL_INTERVAL` (seconds) near the top of
`FusionAddin/FusionJointBridge/FusionJointBridge.py`.  Default is **0.2 s**
(5 Hz).  Lower values increase responsiveness but add more Fusion main-thread
wake-ups.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Joint not moving | Check that the name in the JSON exactly matches the Fusion browser name (case-sensitive). Errors are logged to `~/FusionBridge/addin.log`. |
| "No values read back" | Make sure the add-in is running (**Tools → Add-Ins → Run**) and the design with those joints is the active document. |
| `addin.log` shows "Unsupported joint type" | The joint is not revolute or slider. Cylindrical is also handled; others are not. |
| Changes not visible | Try a manual viewport orbit — Fusion sometimes needs user input to redraw. |
