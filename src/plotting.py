"""plotting.py — єдиний публікаційний стиль для фігур."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "shewhart": "#d62728",
    "ewma": "#9467bd",
    "page": "#1f77b4",
    "gsa2": "#2ca02c",
    "gsa3": "#17becf",
    "gsa_frac": "#8c564b",
    "oracle": "#7f7f7f",
    "winsor": "#bcbd22",
    "sign": "#e377c2",
    "mc": "#ff7f0e",
    "analytic": "#1f77b4",
}

MARKERS = {
    "shewhart": "o", "ewma": "s", "page": "^", "gsa2": "D",
    "gsa3": "v", "gsa_frac": "P", "oracle": "x",
    "winsor": "*", "sign": "X",
}


def setup():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.fontsize": 9.5,
        "legend.framealpha": 0.92,
        "lines.linewidth": 1.9,
        "lines.markersize": 6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
    })


def newfig(w=7.2, h=4.6):
    setup()
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax
