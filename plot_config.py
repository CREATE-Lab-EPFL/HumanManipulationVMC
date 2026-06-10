"""
Shared plotting helpers for all experiments.

Usage in every notebook:
    import sys; sys.path.insert(0, '../..')   # or '..' for 1-level-deep notebooks
    from plot_config import draw_radar         # mplstyle is applied automatically
"""

import os as _os
import numpy as np
import matplotlib.pyplot as plt

# Apply the shared mplstyle automatically when this module is imported.
_REPO = _os.path.dirname(_os.path.abspath(__file__))
plt.style.use(_os.path.join(_REPO, 'plot_config.mplstyle'))


# Style constants for bespoke polar charts (supplement rcParams).
# Font/line values are intentionally read from rcParams so they track the shared mplstyle.
RADAR_RC = {
    'grid_color':     '0.80',
    'grid_lw':        0.4,    # finer than the default for polar readability
    'grid_ls':        ':',    # dotted looks cleaner on polar axes
    'spine_color':    '#cccccc',
    'tick_pad':       24,
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
    fs_spokes=18,
    lw=2.5,
    alpha_fill=0.15,
    markersize=6,
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
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))

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
        ax.fill(ac, vc, color=t['color'], alpha=t.get('alpha', alpha_fill))

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
                  ncol=len(traces), fontsize=fs_spokes * 0.8, frameon=False)
        plt.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig, ax

    return ax
