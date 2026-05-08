"""
Run all four StiffnessForceTracking controllers sequentially.

Executes N_RUNS of each controller in order:
  1. force_position_control        (model-based K gradient descent)
  2. force_position_control_scalar (scalar K update)
  3. force_stiffness_control       (model-based theta_ref gradient descent)
  4. force_stiffness_control_scalar (scalar theta_ref update)

Total: 4 × N_RUNS data collections.
Run from the repo root: python -m StiffnessForceTracking.FORCE_CONTROL
"""

import subprocess
import sys
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')

CONTROLLERS = [
    'force_position_control',
    'force_position_control_scalar',
    'force_stiffness_control',
    'force_stiffness_control_scalar',
]


def collected_data_flag(ctrl):
    """Read COLLECTED_DATA value directly from the script source."""
    path = os.path.join(_HERE, f'{ctrl}.py')
    with open(path) as f:
        for line in f:
            m = re.match(r'^\s*COLLECTED_DATA\s*=\s*(True|False)', line)
            if m:
                return m.group(1) == 'True'
    return False   # default: run if flag not found


for i, ctrl in enumerate(CONTROLLERS, 1):
    if collected_data_flag(ctrl):
        print(f'  Skipping {ctrl} — COLLECTED_DATA = True')
        continue

    print(f'\n{"=" * 60}')
    print(f'  Controller {i}/{len(CONTROLLERS)}: {ctrl}')
    print(f'{"=" * 60}\n')
    result = subprocess.run(
        [sys.executable, '-m', f'StiffnessForceTracking.{ctrl}'],
        cwd=_ROOT,
    )
    if result.returncode not in (0, -2):  # -2 = SIGINT (Ctrl+C interrupt)
        print(f'\nERROR: {ctrl} exited with code {result.returncode}. Aborting.')
        sys.exit(result.returncode)
    print(f'\n=== {ctrl} complete ===')

print('\n\nAll controllers done.')
