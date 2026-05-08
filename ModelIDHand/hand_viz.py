import sys, os
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from KinematicsHand.FK_Hand import (
    FK_motor2palm, FK_motor2fingerPos, FK_motor2thumbPos, FK_motor2wrist, rot_axis,
)
from ModelIDHand.hand_params import (
    WRIST, FINGER_BASE_ORIGINS, JOINT_LIMITS, FINGER_TRANSMISSIONS,
    THUMB_TRANSMISSION, SPREAD_ANGLE_CORRECTION,
)

_Z3    = np.zeros(3)
_FTIP  = np.array([0.0, 0.0, 0.018])
_TTIP  = np.array([-0.025, 0.0, 0.0])
_NAMES = ["index", "middle", "ring", "pinky"]
_COL   = dict(thumb="tab:blue", index="tab:orange", middle="tab:green",
              ring="tab:red",   pinky="tab:purple")

STYLES = {
    "default":   dict(linestyle="-",  linewidth=2.0, marker="o", markersize=4),
    "thick":     dict(linestyle="-",  linewidth=4.5, marker="o", markersize=6),
    "dotted":    dict(linestyle=":",  linewidth=2.5, marker="o", markersize=4),
    "dashed":    dict(linestyle="--", linewidth=2.0, marker="o", markersize=4),
    "segmented": dict(linestyle="-.", linewidth=2.0, marker="s", markersize=5),
    "minimal":   dict(linestyle="-",  linewidth=1.0),
}


def _build_motor_limits():
    T, F = THUMB_TRANSMISSION, FINGER_TRANSMISSIONS
    mp   = JOINT_LIMITS["wrist_pitch"][1] / WRIST["spur_ratio"]
    smax = JOINT_LIMITS["pinky_spread"][1] / (abs(SPREAD_ANGLE_CORRECTION) * 1.5)
    lim  = lambda j, r: [JOINT_LIMITS[j][0] * r, JOINT_LIMITS[j][1] * r]
    rows = [
        [-mp, mp], [-mp, mp],
        lim("thumb_CMC1", T["CMC1_pulley"] / T["r_motor"]),
        lim("thumb_CMC2", T["CMC2_pulley"] / T["r_motor"]),
        lim("thumb_MCP",  T["MCP_c"]       / T["r_motor"]),
        lim("thumb_IP",   T["IP_c"]        / T["r_motor"]),
        [0.0, smax],
    ]
    for n in _NAMES:
        rows += [lim(f"{n}_MCP", F[n]["r_motor"] / F[n]["r_pulley"]),
                 lim(f"{n}_PIP", F[n]["c_param"] / F[n]["r_motor"])]
    return np.array(rows)

MOTOR_LIMITS = _build_motor_limits()


def random_trajectory(n_waypoints: int, duration: float, seed=None):
    """Return callable q(t) → (15,) with cosine-eased interpolation."""
    rng = np.random.default_rng(seed)
    lo, hi = MOTOR_LIMITS[:, 0], MOTOR_LIMITS[:, 1]
    wps = rng.uniform(lo, hi, size=(n_waypoints, 15))
    sd  = duration / (n_waypoints - 1)
    def q_at_t(t):
        t   = float(np.clip(t, 0.0, duration))
        seg = min(int(t / sd), n_waypoints - 2)
        a   = (1.0 - np.cos(np.pi * (t - seg * sd) / sd)) / 2.0
        return (1.0 - a) * wps[seg] + a * wps[seg + 1]
    return q_at_t


def _wrist_pts(q):
    _, yaw = FK_motor2wrist(q)
    R = rot_axis(WRIST["yaw_axis"], yaw)
    return np.array([_Z3, WRIST["yaw_origin"].copy(),
                     WRIST["yaw_origin"] + R @ WRIST["pitch_origin"],
                     FK_motor2palm(q, _Z3)[1]])

def _thumb_pts(q):
    p = lambda j, o=_Z3: FK_motor2thumbPos(q, j, o)
    return np.array([FK_motor2palm(q, FINGER_BASE_ORIGINS["thumb"])[1],
                     p("CMC1"), p("CMC2"), p("MCP"), p("IP"), p("IP", _TTIP)])

def _finger_pts(q, n):
    p = lambda j, o=_Z3: FK_motor2fingerPos(q, n, j, o)
    return np.array([FK_motor2palm(q, FINGER_BASE_ORIGINS[n])[1],
                     p("Spread"), p("MCP"), p("PIP"), p("DIP"), p("DIP", _FTIP)])


def _equal_aspect(ax):
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    c = lims.mean(axis=1); r = 0.5 * (lims[:, 1] - lims[:, 0]).max()
    ax.set_xlim3d(c[0]-r, c[0]+r); ax.set_ylim3d(c[1]-r, c[1]+r); ax.set_zlim3d(c[2]-r, c[2]+r)

def _resolve(style):
    """Resolve style name or dict to plot kwargs."""
    return STYLES[style] if isinstance(style, str) else (style or STYLES["default"])

def _seg(ax, a, b):
    ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], "-", color="dimgray", linewidth=1.5, zorder=2)


def plot_hand(q_motor, ax=None, title=None, style="default"):
    """
    Static 3D hand schematic for a single pose.

    Parameters
    ----------
    q_motor : array-like, shape (15,)
    ax      : Axes3D, optional
    title   : str, optional
    style   : str or dict
        Named preset (see STYLES) or a dict of matplotlib line kwargs.
        Applies to finger/thumb chains only.
    """
    q = np.asarray(q_motor, dtype=np.float64)
    assert q.shape == (15,)
    if ax is None:
        fig = plt.figure(figsize=(8, 9)); ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    kw  = _resolve(style)
    drw = lambda pts, c: ax.plot(pts[:,0], pts[:,1], pts[:,2], color=c, zorder=3, **kw)

    wp = _wrist_pts(q); tp = _thumb_pts(q)
    fp = {n: _finger_pts(q, n) for n in _NAMES}

    ax.plot(wp[:,0], wp[:,1], wp[:,2], "-", color="dimgray", linewidth=1.5, zorder=3)
    drw(tp, _COL["thumb"])
    for n in _NAMES:
        drw(fp[n], _COL[n]); _seg(ax, wp[-1], fp[n][0])
    _seg(ax, wp[-1], tp[0])
    bases = np.array([tp[0]] + [fp[n][0] for n in _NAMES])
    ax.plot(bases[:,0], bases[:,1], bases[:,2], "--", color="silver", linewidth=1.0, zorder=1)

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    _equal_aspect(ax)
    ax.legend(handles=[plt.Line2D([0],[0], color=_COL[n], lw=2, label=n.capitalize())
                       for n in ["thumb"]+_NAMES], loc="upper left", fontsize=8)
    if title: ax.set_title(title)
    return fig, ax


class HandVisualizer:
    """
    Real-time 3D hand visualizer. Call update() in your control loop; renders at render_hz.

    >>> viz = HandVisualizer(style="thick")
    >>> viz.run(random_trajectory(20, 10.0), duration=10.0)
    >>> viz.run(lambda t: latest_q)           # live source, runs until window closed
    """

    def __init__(self, render_hz=60.0, title="Hand visualizer", style="default"):
        self._period = 1.0 / render_hz
        self._last   = 0.0
        self._kw     = _resolve(style)

        self._fig = plt.figure(figsize=(8, 9))
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._ax.set_xlabel("X [m]"); self._ax.set_ylabel("Y [m]"); self._ax.set_zlabel("Z [m]")
        self._fig.suptitle(title)
        self._build(np.zeros(15))
        _equal_aspect(self._ax)
        plt.tight_layout(); plt.ion(); plt.show()

    def _mkln(self, pts, color):
        ln, = self._ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, zorder=3, **self._kw)
        return ln

    def _build(self, q):
        wp = _wrist_pts(q); tp = _thumb_pts(q)
        fp = {n: _finger_pts(q, n) for n in _NAMES}
        po = wp[-1]
        _sk = dict(color="dimgray", linestyle="-", linewidth=1.5, zorder=2)

        self._ln_w,  = self._ax.plot(wp[:,0], wp[:,1], wp[:,2], **_sk)
        self._ln_t   = self._mkln(tp, _COL["thumb"])
        self._ln_f   = {n: self._mkln(fp[n], _COL[n]) for n in _NAMES}
        self._ln_sp  = [self._ax.plot([po[0],fp[n][0,0]], [po[1],fp[n][0,1]], [po[2],fp[n][0,2]], **_sk)[0]
                        for n in _NAMES]
        self._ln_tsp,= self._ax.plot([po[0],tp[0,0]], [po[1],tp[0,1]], [po[2],tp[0,2]], **_sk)
        bases = np.array([tp[0]] + [fp[n][0] for n in _NAMES])
        self._ln_out,= self._ax.plot(bases[:,0], bases[:,1], bases[:,2],
                                     "--", color="silver", linewidth=1.0, zorder=1)
        self._ax.legend(handles=[plt.Line2D([0],[0], color=_COL[n], lw=2, label=n.capitalize())
                                  for n in ["thumb"]+_NAMES], loc="upper left", fontsize=8)

    @staticmethod
    def _upd(ln, pts):
        ln.set_data(pts[:,0], pts[:,1]); ln.set_3d_properties(pts[:,2])

    @staticmethod
    def _upd2(ln, a, b):
        ln.set_data([a[0],b[0]], [a[1],b[1]]); ln.set_3d_properties([a[2],b[2]])

    def update(self, q_motor):
        """Push a new motor configuration (15,) and redraw if render period elapsed."""
        q  = np.asarray(q_motor, dtype=np.float64)
        wp = _wrist_pts(q); tp = _thumb_pts(q)
        fp = {n: _finger_pts(q, n) for n in _NAMES}
        po = wp[-1]

        self._upd(self._ln_w, wp); self._upd(self._ln_t, tp)
        for i, n in enumerate(_NAMES):
            self._upd(self._ln_f[n], fp[n]); self._upd2(self._ln_sp[i], po, fp[n][0])
        self._upd2(self._ln_tsp, po, tp[0])
        self._upd(self._ln_out, np.array([tp[0]] + [fp[n][0] for n in _NAMES]))

        now = time.perf_counter()
        if now - self._last >= self._period:
            self._fig.canvas.draw(); self._fig.canvas.flush_events(); self._last = now

    def run(self, q_source, duration=None):
        """Loop calling q_source(t_elapsed) until duration or window close. Returns (n_frames, elapsed)."""
        t0 = time.perf_counter(); n = 0
        try:
            while plt.fignum_exists(self._fig.number):
                t = time.perf_counter() - t0
                if duration and t >= duration: break
                self.update(q_source(t)); n += 1
        except KeyboardInterrupt:
            pass
        elapsed = time.perf_counter() - t0
        print(f"  {n} FK evals in {elapsed:.1f} s  →  {n/elapsed:.0f} Hz")
        return n, elapsed


if __name__ == "__main__":
    N, DUR, SEED, HZ = 15, 12.0, 42, 60

    q_at_t = random_trajectory(n_waypoints=N, duration=DUR, seed=SEED)
    viz = HandVisualizer(render_hz=HZ, title=f"Random — {N} waypoints, {DUR:.0f} s")
    viz.run(q_at_t, duration=DUR)
    plt.close("all")

    lo, hi = MOTOR_LIMITS[:, 0], MOTOR_LIMITS[:, 1]
    c, a   = (lo+hi)/2, (hi-lo)/2
    phase  = np.linspace(0, np.pi, 15)
    viz2 = HandVisualizer(render_hz=HZ, title="Live source demo")
    viz2.run(lambda t: c + a * np.sin(2*np.pi/4.0*t + phase), duration=8.0)
    plt.close("all")
