# StiffnessModelFinger — Tip Stiffness Mapping

Maps virtual element stiffness to fingertip stiffness in Cartesian task space:

```
K_x = ((P·J_x)*)^T · η · J_d^T · K_d · J_d · (P·J_x)*
```

Two approximation levels are provided (Salisbury 1st order and full CCT 2nd order).
All three classes expose the same interface: `tip_force`, `tip_stiffness`,
`stiffness_descent`, `ref_descent`.

## Files

### `stiffness2fingerspace.py` — `tip_stiffness_FingerSpace`

Virtual springs at the joints: `d = [θ_MCP, θ_PIP, θ_DIP]`.
Stiffness `K` in [N/rad]. One reference vector `q_ref` (motor angles).

### `stiffness2taskspace.py` — `tip_stiffness_TaskSpace`

Virtual springs at Cartesian points: `d = [x_tip, x_base]`.
Stiffnesses `K_tip`, `K_base` in [N/m]. Two independent springs at the
fingertip and MCP base.

### `stiffness2mixedspace.py` — `tip_stiffness_MixedSpace`

General formulation combining both: `d = [θ_MCP, θ_PIP, θ_DIP, x_tip, x_base]`.
Accepts `K_theta` (finger-space) + `K_tip` + `K_base` (task-space) simultaneously.
Virtual contributions compose additively through the Principle of Virtual Work.
Any subset can be disabled by setting the corresponding K to zero.

## Quick test commands

Run a file's self-tests (FD checks + convergence) from the repository root:

```bash
python3 -u StiffnessModelFinger/stiffness2fingerspace.py
python3 -u StiffnessModelFinger/stiffness2taskspace.py
python3 -u StiffnessModelFinger/stiffness2mixedspace.py
```

