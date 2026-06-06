"""
Utility for converting recorded motor-angle CSV data to Fusion joint-angle dicts.

Current 13-motor software order (wrist excluded — position-controlled):
  q[0..3]  = thumb CMC1, CMC2, MCP, IP
  q[4]     = spread
  q[5..6]  = index MCP, PIP
  q[7..8]  = middle MCP, PIP
  q[9..10] = ring MCP, PIP
  q[11..12]= pinky MCP, PIP
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from KinematicsHand.FK_Hand import (
    FK_motor2thumb, FK_motor2finger, FK_motor2spread,
)


def q13_to_joints_deg(q13):
    """Convert 13-motor software-order vector to Fusion joint dict [deg].

    Args:
        q13: (13,) array in current software order (no wrist)

    Returns:
        dict mapping Fusion joint names to angles in degrees
    """
    q = np.asarray(q13, dtype=float)
    assert q.shape == (13,), f"Expected (13,) motor vector, got {q.shape}"

    thumb  = FK_motor2thumb(q)
    s_idx  = FK_motor2spread(q, "index")
    s_ring = FK_motor2spread(q, "ring")
    s_pnk  = FK_motor2spread(q, "pinky")
    idx    = FK_motor2finger(q, "index")
    mid    = FK_motor2finger(q, "middle")
    ring   = FK_motor2finger(q, "ring")
    pinky  = FK_motor2finger(q, "pinky")

    return {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   float(np.degrees(thumb[0])),
        "thumb_CMC2":   float(np.degrees(thumb[1])),
        "thumb_MCP":    float(np.degrees(thumb[2])),
        "thumb_IP":     float(np.degrees(thumb[3])),
        "index_spread": float(np.degrees(s_idx)),
        "ring_spread":  float(np.degrees(s_ring)),
        "pinky_spread": float(np.degrees(s_pnk)),
        "index_MCP":    float(np.degrees(idx[0])),
        "index_PIP":    float(np.degrees(idx[1])),
        "middle_MCP":   float(np.degrees(mid[0])),
        "middle_PIP":   float(np.degrees(mid[1])),
        "ring_MCP":     float(np.degrees(ring[0])),
        "ring_PIP":     float(np.degrees(ring[1])),
        "pinky_MCP":    float(np.degrees(pinky[0])),
        "pinky_PIP":    float(np.degrees(pinky[1])),
    }


def act_cols_to_joints_deg(row):
    """Convert PoseControl *_act_rad columns (already FK-computed) to Fusion joint dict [deg].

    Args:
        row: pandas Series or dict with keys like 'thumb_CMC1_act_rad', etc.
    """
    def d(key): return float(np.degrees(row[key]))
    return {
        "wrist_pitch":  0.0,
        "wrist_yaw":    0.0,
        "thumb_CMC1":   d("thumb_CMC1_act_rad"),
        "thumb_CMC2":   d("thumb_CMC2_act_rad"),
        "thumb_MCP":    d("thumb_MCP_act_rad"),
        "thumb_IP":     d("thumb_IP_act_rad"),
        "index_spread": d("spread_index_act_rad"),
        "ring_spread":  d("spread_ring_act_rad"),
        "pinky_spread": d("spread_pinky_act_rad"),
        "index_MCP":    d("index_MCP_act_rad"),
        "index_PIP":    d("index_PIP_act_rad"),
        "middle_MCP":   d("middle_MCP_act_rad"),
        "middle_PIP":   d("middle_PIP_act_rad"),
        "ring_MCP":     d("ring_MCP_act_rad"),
        "ring_PIP":     d("ring_PIP_act_rad"),
        "pinky_MCP":    d("pinky_MCP_act_rad"),
        "pinky_PIP":    d("pinky_PIP_act_rad"),
    }


def avg_q13(df, q_prefix="q_motor_{}_rad"):
    """Return mean 13-motor vector from a filtered DataFrame.

    Args:
        df:       filtered DataFrame
        q_prefix: column name template; use '{}' for the motor index (0..12)
    """
    cols = [q_prefix.format(i) for i in range(13)]
    return df[cols].mean().values


def print_joints(label, joints):
    print(f"\n{label}:")
    for k, v in joints.items():
        print(f"  {k}: {v:.1f} deg")
