"""
Interactive fingertip contact-normal visualizer for the full hand.

Shows the 3D hand skeleton with one quiver arrow per fingertip representing
the contact normal n = tip_stiffness_JointSpace._normal(finger, q).
Sliders on the right panel control all 13 motor angles.

Run:
    python StiffnessModelHand/normal_viz_gui.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ModelIDHand.hand_viz import (
    _WRIST_PTS, _thumb_pts, _finger_pts,
    _COL, _NAMES, _equal_aspect, MOTOR_LIMITS,
)
from KinematicsHand.FK_Hand import FK_motor2fingerPos, FK_motor2thumbPos
from StiffnessModelHand.stiffness2jointspace import tip_stiffness_JointSpace

# ---------------------------------------------------------------------------

_FINGERS_ALL = ["thumb", "index", "middle", "ring", "pinky"]

_NORMAL_LEN = 0.040   # quiver arrow length [m]

_MOTOR_LABELS = [
    "Thumb CMC1  [0]", "Thumb CMC2  [1]",
    "Thumb MCP   [2]", "Thumb IP    [3]",
    "Spread      [4]",
    "Index MCP   [5]", "Index PIP   [6]",
    "Middle MCP  [7]", "Middle PIP  [8]",
    "Ring MCP    [9]", "Ring PIP   [10]",
    "Pinky MCP  [11]", "Pinky PIP  [12]",
]


_THUMB_TIP_OFFSET = np.array([-0.025, 0.0, 0.0])   # matches hand_viz._TTIP


def _tip_world(finger, q, rtips):
    """World-frame fingertip position.
    Thumb uses [-0.025,0,0] (phalanx runs along -x in IP frame).
    Other fingers use stiffness model rtip offset (along +z in DIP frame).
    """
    if finger == "thumb":
        return FK_motor2thumbPos(q, "IP", _THUMB_TIP_OFFSET)
    return FK_motor2fingerPos(q, finger, "DIP", rtips[finger])


# ---------------------------------------------------------------------------

class NormalVizGUI:
    """
    Interactive visualizer: full hand skeleton + contact normal quivers.

    The normals are computed by tip_stiffness_JointSpace._normal(), exactly as
    used by the VMC stiffness model.  Fixing _normal() there instantly updates
    what this GUI shows.
    """

    def __init__(self):
        print("Initialising stiffness model (Jacobians + Hessians) …", end=" ", flush=True)
        self._model = tip_stiffness_JointSpace()
        print("done.")
        self._rtips = self._model.rtips
        self._q     = np.zeros(13)

        # ---- figure --------------------------------------------------------
        self._fig = plt.figure(figsize=(16, 9))
        self._fig.patch.set_facecolor("#f7f7f7")
        self._fig.suptitle(
            "Fingertip contact normals",
            fontsize=10,
        )

        # 3-D panel (left 60 %)
        self._ax = self._fig.add_axes([0.01, 0.03, 0.59, 0.93], projection="3d")
        self._ax.set_xlabel("X [m]", fontsize=8)
        self._ax.set_ylabel("Y [m]", fontsize=8)
        self._ax.set_zlabel("Z [m]", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.view_init(elev=20, azim=-55)

        # build hand skeleton lines
        self._build_hand()

        # build normal quiver arrows
        self._quivers: dict = {}
        self._draw_quivers()

        # freeze axis limits so sliders don't zoom the view
        _equal_aspect(self._ax)
        self._xlim = self._ax.get_xlim3d()
        self._ylim = self._ax.get_ylim3d()
        self._zlim = self._ax.get_zlim3d()

        # ---- normal-component readout labels --------------------------------
        self._setup_readout()

        # ---- sliders --------------------------------------------------------
        self._build_sliders()

        plt.ion()
        plt.show()

    # -----------------------------------------------------------------------
    # Hand skeleton
    # -----------------------------------------------------------------------

    def _build_hand(self):
        q   = self._q
        _sk = dict(color="dimgray", linestyle="-", linewidth=1.5, zorder=2)
        kw  = dict(linestyle="-", linewidth=2.0, marker="o", markersize=3, zorder=3)

        wp = _WRIST_PTS
        tp = _thumb_pts(q)
        fp = {n: _finger_pts(q, n) for n in _NAMES}
        po = wp[-1]

        self._ln_w,  = self._ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], **_sk)
        self._ln_t,  = self._ax.plot(
            tp[:, 0], tp[:, 1], tp[:, 2], color=_COL["thumb"], **kw)
        self._ln_f   = {
            n: self._ax.plot(
                fp[n][:, 0], fp[n][:, 1], fp[n][:, 2], color=_COL[n], **kw)[0]
            for n in _NAMES}
        self._ln_sp  = [
            self._ax.plot(
                [po[0], fp[n][0, 0]],
                [po[1], fp[n][0, 1]],
                [po[2], fp[n][0, 2]], **_sk)[0]
            for n in _NAMES]
        self._ln_tsp, = self._ax.plot(
            [po[0], tp[0, 0]], [po[1], tp[0, 1]], [po[2], tp[0, 2]], **_sk)
        bases = np.array([tp[0]] + [fp[n][0] for n in _NAMES])
        self._ln_out, = self._ax.plot(
            bases[:, 0], bases[:, 1], bases[:, 2],
            "--", color="silver", linewidth=1.0, zorder=1)

        self._ax.legend(
            handles=[
                plt.Line2D([0], [0], color=_COL[n], lw=2, label=n.capitalize())
                for n in ["thumb"] + _NAMES
            ],
            loc="upper left", fontsize=7,
        )

    def _update_hand(self):
        q  = self._q
        wp = _WRIST_PTS; tp = _thumb_pts(q)
        fp = {n: _finger_pts(q, n) for n in _NAMES}
        po = wp[-1]

        def upd(ln, pts):
            ln.set_data(pts[:, 0], pts[:, 1])
            ln.set_3d_properties(pts[:, 2])

        def upd2(ln, a, b):
            ln.set_data([a[0], b[0]], [a[1], b[1]])
            ln.set_3d_properties([a[2], b[2]])

        upd(self._ln_w, wp);  upd(self._ln_t, tp)
        for i, n in enumerate(_NAMES):
            upd(self._ln_f[n], fp[n])
            upd2(self._ln_sp[i], po, fp[n][0])
        upd2(self._ln_tsp, po, tp[0])
        upd(self._ln_out, np.array([tp[0]] + [fp[n][0] for n in _NAMES]))

    # -----------------------------------------------------------------------
    # Normal quivers
    # -----------------------------------------------------------------------

    def _draw_quivers(self):
        q = self._q
        for f in _FINGERS_ALL:
            o = _tip_world(f, q, self._rtips)
            n = self._model._normal(f, q)
            qv = self._ax.quiver(
                o[0], o[1], o[2],
                n[0], n[1], n[2],
                length=_NORMAL_LEN, normalize=True,
                color=_COL[f], linewidth=2.5,
                arrow_length_ratio=0.28, zorder=6,
            )
            self._quivers[f] = qv

    def _update_quivers(self):
        for qv in self._quivers.values():
            qv.remove()
        self._quivers.clear()
        self._draw_quivers()

    # -----------------------------------------------------------------------
    # Normal-component readout (small text panel below the 3-D axis)
    # -----------------------------------------------------------------------

    def _setup_readout(self):
        ax_txt = self._fig.add_axes([0.01, 0.00, 0.59, 0.04])
        ax_txt.axis("off")
        self._readout_ax = ax_txt
        self._readout_txt = ax_txt.text(
            0.01, 0.5, self._readout_str(),
            transform=ax_txt.transAxes,
            fontsize=7.5, fontfamily="monospace",
            verticalalignment="center",
        )

    def _readout_str(self):
        q = self._q
        parts = []
        for f in _FINGERS_ALL:
            n = self._model._normal(f, q)
            parts.append(f"{f[0].upper()}:[{n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}]")
        return "  ".join(parts)

    def _update_readout(self):
        self._readout_txt.set_text(self._readout_str())

    # -----------------------------------------------------------------------
    # Sliders
    # -----------------------------------------------------------------------

    def _build_sliders(self):
        h   = 0.026    # slider height
        gap = 0.0365   # vertical step per slider
        top = 0.958
        lft = 0.635
        wid = 0.345

        self._sliders = []

        # Reset button at the very bottom
        ax_btn = self._fig.add_axes([lft + wid * 0.25, 0.005, wid * 0.50, 0.028])
        btn = Button(ax_btn, "Reset all", color="#fffacd", hovercolor="#ffd700")
        btn.label.set_fontsize(8)
        btn.on_clicked(self._on_reset)
        self._btn_reset = btn   # keep reference

        for i in range(13):
            y    = top - i * gap
            lo   = float(MOTOR_LIMITS[i, 0])
            hi   = float(MOTOR_LIMITS[i, 1])
            axsl = self._fig.add_axes([lft, y, wid, h])
            sl   = Slider(
                axsl, _MOTOR_LABELS[i], lo, hi,
                valinit=0.0, color="steelblue", track_color="#d8d8d8",
            )
            sl.label.set_fontsize(7.5)
            sl.valtext.set_fontsize(7.5)

            def _cb(val, idx=i, s=sl):
                self._q[idx] = s.val
                self._update_hand()
                self._update_quivers()
                self._update_readout()
                self._ax.set_xlim3d(*self._xlim)
                self._ax.set_ylim3d(*self._ylim)
                self._ax.set_zlim3d(*self._zlim)
                self._fig.canvas.draw_idle()

            sl.on_changed(_cb)
            self._sliders.append(sl)

    def _on_reset(self, _event):
        for sl in self._sliders:
            sl.reset()

    # -----------------------------------------------------------------------

    def show(self):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    gui = NormalVizGUI()
    gui.show()
