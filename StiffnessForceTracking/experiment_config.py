"""
Shared configuration for all StiffnessForceTracking experiments.

All experiment scripts import from here so that reference signal timing and
learning rates are defined in one place.

Naming convention:
  force_position_control  — gradient descent on K   (force + position controlled)
  force_stiffness_control — gradient descent on d_ref (force + stiffness controlled, K free)
"""

# =============================================================================
# Reference signal
# =============================================================================
FORCE_LEVELS        = [0.5, 0.8]         # N — square wave alternation
STEP_HOLD           = 30.0              # s — duration per level
N_CYCLES            = 3                 # number of full cycles to run
INITIAL_SETTLE_TIME = 5.0              # s — settle before tracking begins
N_PREP              = 5                 # number of press-release preparation cycles
PREP_HOLD           = 5.0              # s — duration of each preparation cycle

TRACK_DURATION = len(FORCE_LEVELS) * STEP_HOLD * N_CYCLES  # 180 s

# =============================================================================
# Learning rates
LR_FORCE_POS   = 1e-4   # force_position_control / force_position_control_scalar
LR_FORCE_STIFF = 5e-4   # force_stiffness_control / force_stiffness_control_scalar
BENDING_DAMPING = 0.003   # [N·m·s/rad] — VMC joint-space damping

# =============================================================================
# Open-loop step-response config
# =============================================================================
FORCE_LEVELS_OL  = [0.4, 0.7, 1.0]   # N — staircase levels (one pass per run)
STEP_HOLD_OL     = 60.0               # s — duration per level
TRACK_DURATION_OL = len(FORCE_LEVELS_OL) * STEP_HOLD_OL  # 180 s
