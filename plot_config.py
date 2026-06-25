"""
Shared plotting helpers for all experiments.

Usage in every notebook:
    import sys, os
    sys.path.insert(0, '..')        # 1-level-deep notebooks
    # or
    sys.path.insert(0, '../..')     # 2-level-deep notebooks
    from plot_config import COLORS, FIG_W_SINGLE, FIG_W_DOUBLE, set_font_size

Setup block (copy into each notebook, override as needed):
    FIG_W     = 10    # inches — change freely
    FIG_H     = 6     # inches — change freely
    GRID      = False
    FONT_SIZE = 18    # pt — change to rescale all text uniformly
    set_font_size(FONT_SIZE)
"""

import os as _os
import numpy as np
import matplotlib.pyplot as plt

# Ensure MiKTeX is on PATH so text.usetex works even when Jupyter is launched
# from a shortcut or IDE that didn't inherit the updated system PATH.
_MIKTEX_BIN = r'C:\Users\loren\AppData\Local\Programs\MiKTeX\miktex\bin\x64'
if _os.path.isdir(_MIKTEX_BIN):
    _path = _os.environ.get('PATH', '')
    if _MIKTEX_BIN not in _path:
        _os.environ['PATH'] = _path + _os.pathsep + _MIKTEX_BIN

# Apply the shared mplstyle automatically when this module is imported.
_REPO = _os.path.dirname(_os.path.abspath(__file__))
plt.style.use(_os.path.join(_REPO, 'plot_config.mplstyle'))

# ── Nature Communications column widths (inches) ───────────────────────────────
FIG_W_SINGLE = 3.46   # 88 mm  — single column
FIG_W_DOUBLE = 7.09   # 180 mm — double column
FIG_W_1P5    = 4.72   # 120 mm — 1.5 column

# ── Okabe-Ito palette (same order as prop_cycle in plot_config.mplstyle) ──────
# Use COLORS[i] or 'Ci' in matplotlib — they reference the same palette.
COLORS = [
    '#4DBBD5',  # C0 blue/cyan
    '#E64B35',  # C1 red
    '#00A087',  # C2 teal
    '#3C5488',  # C3 navy
    '#F39B7F',  # C4 coral
    '#8491B4',  # C5 lavender
    '#91D1C2',  # C6 light teal
    '#DC0000',  # C7 bright red
    '#7E6148',  # C8 brown
    '#949494',  # C9 gray
]

def set_font_size(size):
    """Override all text sizes uniformly. Call after import in each notebook setup cell."""
    plt.rcParams.update({
        'font.size':             size,
        'axes.labelsize':        size,
        'axes.titlesize':        size,
        'xtick.labelsize':       size,
        'ytick.labelsize':       size,
        'legend.fontsize':       size * 0.75,
        'legend.title_fontsize': size * 0.75,
    })


# ── Style constants for bespoke polar/radar charts ────────────────────────────
RADAR_RC = {
    'grid_color':     '0.80',
    'grid_lw':        0.4,
    'grid_ls':        ':',
    'spine_color':    '#cccccc',
    'tick_pad':       12,
    'fill_alpha_des': 0.12,
    'fill_alpha_trk': 0.18,
}


def draw_radar(
    spokes,
    traces,
    save_path=None,
    *,
    ax=None,
    vmax=None,
    r_lim=1.25,
    r_label=1.30,
    fs_spokes=None,
    lw=2.0,
    alpha_fill=0.15,
    markersize=5,
):
    """
    Draw a radar (spider) chart.

    Parameters
    ----------
    spokes : list[str], length N
        Spoke labels.
    traces : list[dict]
        Each entry must have 'label', 'values' (array length N), 'color'.
        Optional per-trace keys: 'lw', 'ls', 'alpha' (override global defaults).
    save_path : str, optional
        Save the figure here (only when ax is None and a new figure is created).
    ax : polar Axes, optional
        Draw into an existing polar axes instead of creating a new figure.
    vmax : array-like(N), optional
        Per-spoke normalization maxima.  Computed from traces if None.
    """
    if fs_spokes is None:
        fs_spokes = plt.rcParams.get('axes.labelsize', 18)

    N = len(spokes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ac = np.append(angles, angles[0])

    if vmax is None:
        vmax = np.ones(N)
        for i in range(N):
            finite = [float(np.array(t['values'])[i]) for t in traces
                      if np.isfinite(float(np.array(t['values'])[i]))]
            vmax[i] = max(finite) if finite else 1.0
    vmax = np.where(np.asarray(vmax, dtype=float) < 1e-12, 1.0,
                    np.asarray(vmax, dtype=float))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7),
                               subplot_kw=dict(projection='polar'))

    for t in traces:
        v  = np.array(t['values'], dtype=float) / vmax
        vc = np.append(v, v[0])
        ax.plot(ac, vc,
                color=t['color'],
                linewidth=t.get('lw', lw),
                linestyle=t.get('ls', '-'),
                marker='o',
                markersize=markersize,
                label=t.get('label', ''))
        ax.fill(ac, vc, color=t['color'], alpha=t.get('alpha', alpha_fill),
                edgecolor='none')

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, label in zip(angles, spokes):
        xd = np.cos(angle)
        yd = np.sin(angle)
        ha = 'left'   if xd >  0.1 else ('right'  if xd < -0.1 else 'center')
        va = 'bottom' if yd >  0.1 else ('top'    if yd < -0.1 else 'center')
        ax.text(angle, r_label, label, ha=ha, va=va,
                fontsize=fs_spokes, fontweight='bold')

    ax.set_ylim(0, r_lim)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', length=0)

    if own_fig:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                  ncol=len(traces),
                  frameon=False)
        plt.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig, ax

    return ax
