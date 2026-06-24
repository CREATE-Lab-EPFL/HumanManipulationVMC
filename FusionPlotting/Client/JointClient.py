"""
Client-side interface to the Fusion bridge file.

Write joint targets:
    client = JointClient()
    client.write_targets({"MCP": 30.0, "PIP": 20.0}, angle_unit="degrees")

Read back current values that Fusion reported:
    current = client.read_current()   # {"MCP": 29.8, "PIP": 19.5}
"""

import json
import pathlib
from datetime import datetime, timezone
from typing import Optional

# Agreed location: both the add-in and this client resolve to the same file
# without any configuration.
BRIDGE_PATH = pathlib.Path.home() / "FusionBridge" / "commands.json"

_DEFAULT_UNITS: dict[str, str] = {"angle": "degrees", "length": "mm"}


class JointClient:
    def __init__(self, bridge_path: Optional[pathlib.Path] = None):
        self.bridge_path = pathlib.Path(bridge_path or BRIDGE_PATH)
        self._ensure_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_targets(
        self,
        targets: dict[str, float],
        *,
        angle_unit: str = "degrees",
        length_unit: str = "mm",
    ) -> None:
        """Overwrite the targets section and declare the active units."""
        data = self._read()
        data["units"]        = {"angle": angle_unit, "length": length_unit}
        data["targets"]      = {k: self._wire(k, v) for k, v in targets.items()}
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._write(data)

    def read_current(self) -> dict[str, float]:
        """Return the last values Fusion wrote back after applying targets."""
        raw = self._read().get("current", {})
        return {k: self._wire(k, v) for k, v in raw.items()}

    def read_units(self) -> dict[str, str]:
        """Return the unit declaration currently in the bridge file."""
        return self._read().get("units", _DEFAULT_UNITS)

    def clear_targets(self) -> None:
        """Remove all targets (Fusion will stop moving joints)."""
        self.write_targets({})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wire(name: str, value: float) -> float:
        # DIP joints in the Fusion model have the opposite sign to MCP/PIP.
        # Negate on the wire so callers always use positive = flexion.
        return -float(value) if name.endswith("_DIP") else float(value)

    def _ensure_file(self) -> None:
        self.bridge_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.bridge_path.exists():
            self._write({
                "units":        _DEFAULT_UNITS,
                "targets":      {},
                "current":      {},
                "last_updated": "",
            })

    def _read(self) -> dict:
        with open(self.bridge_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        # Atomic write: write to a temp file then replace to avoid partial reads.
        tmp = self.bridge_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.bridge_path)
