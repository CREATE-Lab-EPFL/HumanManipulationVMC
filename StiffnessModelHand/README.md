# StiffnessModelHand — Fingertip Stiffness Mapping

Maps virtual element stiffness to fingertip stiffness in Cartesian space for
all five fingers of the ADAPT Hand (15 motors, 5 fingertips):

```
K_x_f = pinv(P_f · J_xf)^T · η · K_motor · pinv(P_f · J_xf)
```

The projection matrix `P_f` is selected via the `mode` constructor argument:

| `mode`     | `P_f`         | Rank | Effect                                      |
|------------|---------------|------|---------------------------------------------|
| `'normal'` | `n_f n_f^T`   | 1    | Force/stiffness along contact normal only   |
| `'full'`   | `I₃`          | 3    | Full 3-D Cartesian force and stiffness      |

In `'normal'` mode, `n_f` is the y-axis of the DIP/IP link frame (perpendicular
to the phalanx), computed from FK at each call — no fixed direction is assumed.

Two approximation levels: Salisbury 1st order and CCT 2nd order.
All three classes expose the same interface per fingertip:
`tip_force`, `tip_stiffness`, `all_tip_forces`, `all_tip_stiffnesses`,
`stiffness_descent`, `ref_descent`.

---

## Files

### `stiffness2jointspace.py` — `tip_stiffness_JointSpace`

Virtual springs in joint-angle space, matching `HandVMCJointSpace`.

Spring groups and stiffness matrix shapes:

| Key               | Shape  | DOF                          |
|-------------------|--------|------------------------------|
| `'wrist'`         | (2,2)  | [pitch, yaw]                 |
| `'thumb'`         | (4,4)  | [CMC1, CMC2, MCP, IP]        |
| `'spread_index'`  | (1,1)  |                              |
| `'spread_middle'` | (1,1)  |                              |
| `'spread_ring'`   | (1,1)  |                              |
| `'spread_pinky'`  | (1,1)  |                              |
| `'index'`         | (3,3)  | [MCP, PIP, DIP]              |
| `'middle'`        | (3,3)  |                              |
| `'ring'`          | (3,3)  |                              |
| `'pinky'`         | (3,3)  |                              |

Motor stiffness: `K_motor = η · Σ_g J_θg^T · K_θg · J_θg`  (15×15)

2nd-order CCT: only the outward geometric term survives — all joint-to-motor
mappings are linear so `H_θg = 0` and the inward term is zero.

### `stiffness2taskspace.py` — `tip_stiffness_TaskSpace`

Virtual springs at Cartesian attachment points, matching `HandVMCTaskSpace`.

Spring points and their FK link:

| Key       | Link      |
|-----------|-----------|
| `'thumb'` | IP        |
| `'index'` | DIP       |
| `'middle'`| DIP       |
| `'ring'`  | DIP       |
| `'pinky'` | DIP       |
| `'palm'`  | wrist     |

Each `K_dict[point]` is a (3,3) stiffness matrix [N/m].
Each `x_ref_dict[point]` is a (3,) reference position [m] in the world frame.

Motor stiffness: `K_motor = η · Σ_p J_xp^T · K_p · J_xp`  (15×15)

2nd-order CCT: both outward and inward terms are non-trivial (position Hessians
are non-zero).

### `stiffness2mixedspace.py` — `tip_stiffness_MixedSpace`

General formulation combining both spring types additively via PVW:

```
d = [θ_wrist, θ_thumb, θ_spread×4, θ_index, θ_middle, θ_ring, θ_pinky,
     x_thumb, x_index, x_middle, x_ring, x_pinky, x_palm]
```

Accepts `K_joint_dict` (joint-space) and `K_task_dict` (task-space) simultaneously.
Any subset can be disabled by omitting its keys.

Motor stiffness:
```
K_motor = η · (Σ_g J_θg^T·K_θg·J_θg  +  Σ_p J_xp^T·K_p·J_xp)
```

2nd-order CCT: outward term from H_xf (same as above); inward term only from
task-space springs (joint springs contribute zero).

---

## Key differences from StiffnessModelFinger

| Feature             | Finger                          | Hand                              |
|---------------------|---------------------------------|-----------------------------------|
| Motor DOF           | 2                               | 15                                |
| Motor stiffness     | (2×2)                           | (15×15)                           |
| Tip stiffness       | single finger                   | per fingertip (5 fingers)         |
| Projection `P`      | `n nᵀ` (`n=None` → `I₃`)       | `mode='normal'`/`'full'`          |
| Contact normal `n`  | fixed at construction           | computed from FK at each call     |
| Joint groups        | one finger only                 | all 10 groups (wrist to fingers)  |
| Task points         | tip + base (2)                  | up to 6 (fingertips + palm)       |

---

## Quick test commands

```bash
python3 -u StiffnessModelHand/stiffness2jointspace.py
python3 -u StiffnessModelHand/stiffness2taskspace.py
python3 -u StiffnessModelHand/stiffness2mixedspace.py
```
